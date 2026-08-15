import test from 'node:test';
import assert from 'node:assert/strict';

import worker from '../src/worker.js';

const origin = 'https://thevikram123.github.io';

function env(overrides = {}) {
  return {
    ALLOWED_ORIGINS: origin,
    GROQ_VISION_MODEL: 'qwen/qwen3.6-27b',
    GROQ_REASON_MODEL: 'openai/gpt-oss-120b',
    GROQ_LIGHT_MODEL: 'openai/gpt-oss-20b',
    GROQ_API_KEY: { get: async () => 'test-only-key' },
    NBC_KV: {
      get: async () => JSON.stringify({
        nodes: [], pages: {}, adj: {}, communities: {},
      }),
    },
    REASON_RATE_LIMITER: { limit: async () => ({ success: true }) },
    VISION_RATE_LIMITER: { limit: async () => ({ success: true }) },
    PLAN_RATE_LIMITER: { limit: async () => ({ success: true }) },
    FLOORPLAN_SERVICE_URL: 'https://floorplan.test',
    FLOORPLAN_SERVICE_TOKEN: 'test-service-token',
    ...overrides,
  };
}

function post(path, body) {
  return new Request(`https://worker.test${path}`, {
    method: 'POST',
    headers: { Origin: origin, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

test('health reports model, index, secret, and limiter bindings', async () => {
  const response = await worker.fetch(
    new Request('https://worker.test/health'),
    env(),
  );
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.groqBindingConfigured, true);
  assert.equal(body.nbcIndexConfigured, true);
  assert.equal(body.rateLimitsConfigured, true);
  assert.equal(body.floorplanConfigured, true);
});

test('floorplan conversion fails closed when its limiter is missing', async () => {
  const form = new FormData();
  form.append('file', new Blob(['plan'], { type: 'image/png' }), 'plan.png');
  const response = await worker.fetch(new Request('https://worker.test/plan/convert', {
    method: 'POST', headers: { Origin: origin }, body: form,
  }), env({ PLAN_RATE_LIMITER: undefined }));
  assert.equal(response.status, 503);
  assert.equal((await response.json()).error, 'plan rate limiter is not configured');
});

test('floorplan conversion calls protected Qwen service then GPT-OSS NBCS assessment', async (t) => {
  let processorRequest;
  let groqPayload;
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (String(url) === 'https://floorplan.test/convert') {
      processorRequest = { url: String(url), options };
      return Response.json({
        buildingProfile: { occupancy: 'Business', buildingHeightM: 12 },
        topology: {
          units: 'mm',
          rooms: Array.from({ length: 100 }, (_, i) => ({
            id: `room-${i}`, type: 'OFFICE', label: `Office ${i}`,
            boundary: Array.from({ length: 100 }, (_, p) => [p, p + i]),
          })),
          room_graph: { nodes: [], edges: [] },
        },
        guidance: { reviewStatus: 'approved' }, metrics: { rooms: 0 }, artifacts: {},
        oversizedArtifactMustNotReachGroq: 'x'.repeat(100_000),
      });
    }
    groqPayload = JSON.parse(options.body);
    return Response.json({
      choices: [{ message: { content: JSON.stringify({
        planSummary: 'Three reconstructed rooms reviewed.',
        score: null,
        findings: [{
          check: 'Exit count', status: 'cannot_verify', severity: 'major',
          observed: 'No designated exits', required: 'Verify against NBCS',
          measurementEvidence: 'No exit labels', clauseId: '', page: null,
          rationale: 'The plan does not designate exits.',
        }],
        citedClauses: [], limitations: ['Exit purpose is not labelled'],
      }) } }],
    });
  });
  const form = new FormData();
  form.append('file', new Blob(['plan'], { type: 'image/png' }), 'plan.png');
  const response = await worker.fetch(new Request('https://worker.test/plan/convert', {
    method: 'POST', headers: { Origin: origin }, body: form,
  }), env());
  assert.equal(response.status, 200);
  assert.equal(processorRequest.url, 'https://floorplan.test/convert');
  assert.equal(processorRequest.options.headers['X-FireShield-Service-Token'], 'test-service-token');
  assert.equal(processorRequest.options.headers['X-FireShield-Groq-Key'], 'test-only-key');
  assert.match(processorRequest.options.headers['Content-Type'], /^multipart\/form-data; boundary=/);
  assert.equal(response.headers.get('access-control-allow-origin'), origin);
  const body = await response.json();
  assert.equal(body.compliance.findings[0].status, 'cannot_verify');
  assert.equal(typeof body.compliance.score, 'number');
  assert.equal(body.compliance.score, 50);
  assert.equal(body.compliance.scoreBasis, 'verified_geometry');
  assert.ok(body.compliance.scoreConfidence > 0);
  assert.equal(groqPayload.model, 'openai/gpt-oss-120b');
  assert.match(groqPayload.messages[1].content, /Business/);
  assert.ok(groqPayload.messages[1].content.length <= 12_000);
  assert.doesNotMatch(groqPayload.messages[1].content, /oversizedArtifactMustNotReachGroq/);
  assert.equal(groqPayload.response_format.type, 'json_schema');
  assert.equal(groqPayload.response_format.json_schema.strict, true);
  assert.equal(groqPayload.reasoning_effort, 'low');
});

test('unapproved geometry produces image-semantic context instead of trusted measurements', async (t) => {
  let groqPayload;
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (String(url) === 'https://floorplan.test/convert') {
      return Response.json({
        buildingProfile: { occupancy: 'Business' },
        guidance: {
          reviewStatus: 'needs_correction', specificationStatus: 'usable',
          specificationConfidence: 0.91, specificationSummary: 'Commercial office and fire stair visible.',
        },
        topology: {
          units: 'mm', mm_per_px: 10,
          rooms: [{ id: 'bad-room', type: 'OFFICE', boundary: [[987654, 987654]] }],
          room_graph: { nodes: [], edges: [] },
        },
        commercialModel: { floor: {
          spaces: [{ id: 'S1', name: 'Office', type: 'OFFICE', evidence: { status: 'measured' } }],
          circulation: { stairs: [{ id: 'ST1', evidence: { status: 'candidate' } }] },
          fireLifeSafety: { equipment: [] },
        } },
        metrics: { rooms: 1 }, artifacts: {},
      });
    }
    groqPayload = JSON.parse(options.body);
    return Response.json({ choices: [{ message: { content: JSON.stringify({
      planSummary: 'Image-only assessment.', score: null, findings: [], citedClauses: [],
      limitations: ['Geometry not verified'],
    }) } }] });
  });
  const form = new FormData();
  form.append('file', new Blob(['plan'], { type: 'image/png' }), 'plan.png');
  const response = await worker.fetch(new Request('https://worker.test/plan/convert', {
    method: 'POST', headers: { Origin: origin }, body: form,
  }), env());
  assert.equal(response.status, 200);
  const context = JSON.parse(groqPayload.messages[1].content);
  assert.equal(context.plan.assessmentMode, 'image_semantic');
  assert.equal(context.plan.mmPerPx, null);
  assert.deepEqual(context.plan.rooms, []);
  assert.doesNotMatch(groqPayload.messages[1].content, /987654/);
  assert.equal(context.plan.imageSemanticEvidence.visibleSpaces[0].name, 'Office');
  const body = await response.clone().json();
  assert.equal(body.compliance.score, 50);
  assert.equal(body.compliance.scoreBasis, 'image_semantic');
  assert.equal(body.compliance.assessmentStatus, 'provisional');
});

test('chat fails closed when the reasoning limiter is missing', async () => {
  const response = await worker.fetch(
    post('/groq/chat', { messages: [{ role: 'user', content: 'hello' }] }),
    env({ REASON_RATE_LIMITER: undefined }),
  );
  assert.equal(response.status, 503);
  assert.equal((await response.json()).error, 'rate limiter is not configured');
});

test('reasoning reserves five model-call units and returns 429 before Groq', async () => {
  let calls = 0;
  const limiter = { limit: async () => ({ success: ++calls < 4 }) };
  const response = await worker.fetch(
    post('/groq/reason', { buildingProfile: {}, detected: [], docs: [] }),
    env({ REASON_RATE_LIMITER: limiter }),
  );
  assert.equal(response.status, 429);
  assert.equal(calls, 4);
  assert.equal(response.headers.get('retry-after'), '60');
});

test('disallowed origins are rejected before consuming a limiter unit', async () => {
  let calls = 0;
  const request = new Request('https://worker.test/groq/chat', {
    method: 'POST',
    headers: { Origin: 'https://evil.example', 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages: [{ role: 'user', content: 'hello' }] }),
  });
  const response = await worker.fetch(request, env({
    REASON_RATE_LIMITER: {
      limit: async () => {
        calls++;
        return { success: true };
      },
    },
  }));
  assert.equal(response.status, 403);
  assert.equal(calls, 0);
});
