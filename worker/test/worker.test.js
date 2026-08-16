import test from 'node:test';
import assert from 'node:assert/strict';

import worker, { __testing__ } from '../src/worker.js';

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

test('the model picker switches to whichever pool actually has headroom', (t) => {
  const { pickAvailableModel, groqQuota } = __testing__;
  const HEAVY = 'openai/gpt-oss-120b';
  const LIGHT = 'openai/gpt-oss-20b';
  // groqQuota is shared module state (read by every real request too) —
  // always leave it as found so this doesn't leak into other tests.
  t.after(() => { delete groqQuota[HEAVY]; delete groqQuota[LIGHT]; });

  // Nothing observed yet for either model: prefer the requested one.
  assert.equal(pickAvailableModel(LIGHT, HEAVY, 900), LIGHT);
  assert.equal(pickAvailableModel(HEAVY, LIGHT, 1600), HEAVY);

  // The preferred (light) pool is known-tight, the other has room: switch.
  groqQuota[LIGHT] = { remainingTokens: 5, resetAt: Date.now() + 30_000 };
  assert.equal(pickAvailableModel(LIGHT, HEAVY, 900), HEAVY);

  // Both known-tight: pick whichever has more remaining, rather than refuse.
  groqQuota[HEAVY] = { remainingTokens: 50, resetAt: Date.now() + 30_000 };
  assert.equal(pickAvailableModel(LIGHT, HEAVY, 900), HEAVY);

  // The tight window has since rolled over: treat it as available again.
  groqQuota[LIGHT] = { remainingTokens: 5, resetAt: Date.now() - 1 };
  assert.equal(pickAvailableModel(LIGHT, HEAVY, 900), LIGHT);
});

test('summariseToolLoop removes every tool_calls turn and its paired replies', () => {
  const { summariseToolLoop } = __testing__;
  const messages = [
    { role: 'system', content: 'S' },
    { role: 'user', content: 'context' },
    {
      role: 'assistant', content: '',
      tool_calls: [{ id: 'c1', function: { name: 'query_nbc', arguments: JSON.stringify({ question: 'sprinkler coverage' }) } }],
    },
    { role: 'tool', tool_call_id: 'c1', content: 'sprinklers required above 15m' },
    { role: 'assistant', content: '', tool_calls: [] }, // empty turn, no content, no tool call
    {
      role: 'assistant', content: '',
      tool_calls: [{ id: 'c2', function: { name: 'query_nbc', arguments: JSON.stringify({ question: 'exit width' }) } }],
    },
    { role: 'tool', tool_call_id: 'c2', content: 'minimum 1000mm' },
  ];
  const { base, summary } = summariseToolLoop(messages);

  // Nothing carrying tool_calls, and no orphaned tool replies, may remain —
  // this is what a tool_choice:'none' call must never see.
  assert.ok(!base.some((m) => Array.isArray(m.tool_calls) && m.tool_calls.length));
  assert.ok(!base.some((m) => m.role === 'tool'));
  assert.deepEqual(base, [{ role: 'system', content: 'S' }, { role: 'user', content: 'context' }]);

  // What was learned must survive as plain text, not be silently discarded.
  assert.match(summary, /sprinkler coverage/);
  assert.match(summary, /sprinklers required above 15m/);
  assert.match(summary, /exit width/);
  assert.match(summary, /minimum 1000mm/);
});

test('summariseToolLoop on a loop that never called a tool changes nothing', () => {
  const { summariseToolLoop } = __testing__;
  const messages = [{ role: 'system', content: 'S' }, { role: 'user', content: 'context' }];
  const { base, summary } = summariseToolLoop(messages);
  assert.deepEqual(base, messages);
  assert.equal(summary, '');
});

test('the token trimmer shrinks history without orphaning tool_call pairings', () => {
  const { fitMessagesToTokenBudget, estimateTokens } = __testing__;
  // A realistic blown-up tool loop: big context + several bulky NBC lookups.
  const messages = [
    { role: 'system', content: 'S'.repeat(400) },
    { role: 'user', content: 'C'.repeat(10_000) },
    { role: 'assistant', content: '', tool_calls: [{ id: 'call_1', function: { name: 'query_nbc', arguments: '{}' } }] },
    { role: 'tool', tool_call_id: 'call_1', content: 'R'.repeat(6000) },
    { role: 'assistant', content: '', tool_calls: [{ id: 'call_2', function: { name: 'query_nbc', arguments: '{}' } }] },
    { role: 'tool', tool_call_id: 'call_2', content: 'R'.repeat(6000) },
    { role: 'user', content: 'Now output the final assessment.' },
  ];
  const budget = 2000;
  const before = messages.reduce((n, m) => n + estimateTokens(m.content), 0);
  assert.ok(before > budget, 'fixture must actually exceed the budget');

  const fitted = fitMessagesToTokenBudget(messages, budget);
  const after = fitted.reduce((n, m) => n + estimateTokens(m.content), 0);
  assert.ok(after <= budget, `expected <= ${budget} tokens, got ${after}`);

  // Structure must survive: every tool message keeps its id, and every
  // tool_call issued by an assistant still has a matching tool reply.
  // Orphaned tool_call_ids are a hard Groq API error, so this is the
  // property that matters more than the exact byte savings.
  assert.equal(fitted.length, messages.length);
  const calledIds = fitted.flatMap((m) => (m.tool_calls || []).map((c) => c.id));
  const repliedIds = fitted.filter((m) => m.role === 'tool').map((m) => m.tool_call_id);
  assert.deepEqual(calledIds.sort(), repliedIds.sort());
  // The system prompt is never sacrificed.
  assert.equal(fitted[0].content, messages[0].content);
});

test('an under-budget conversation is passed through untouched', () => {
  const { fitMessagesToTokenBudget } = __testing__;
  const messages = [
    { role: 'system', content: 'short system' },
    { role: 'user', content: 'short context' },
  ];
  assert.deepEqual(fitMessagesToTokenBudget(messages, 5000), messages);
});

test('a Groq 413 retries with a compacted payload instead of waiting', async (t) => {
  const seen = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    const payload = JSON.parse(options.body);
    seen.push(payload);
    if (seen.length === 1) {
      // 413 carries no usable Retry-After — waiting can never fix it.
      return new Response(JSON.stringify({
        error: { code: 'rate_limit_exceeded', message: 'Request too large ... Limit 8000, Requested 8849' },
      }), { status: 413, headers: {} });
    }
    return Response.json({ choices: [{ message: { content: 'ok after shrink' } }] });
  });
  // groqChat caps each message at 6000 chars, so use enough messages that the
  // payload genuinely exceeds the compaction budget — otherwise the trimmer
  // correctly passes it through and the test proves nothing.
  const bulky = Array.from({ length: 8 }, () => ({ role: 'user', content: 'x'.repeat(6000) }));
  const response = await worker.fetch(post('/groq/chat', { messages: bulky }), env());
  assert.equal(response.status, 200);
  assert.equal(seen.length, 2, 'should retry exactly once');
  // The retry must actually be smaller, not a blind repeat of the same request.
  const firstChars = JSON.stringify(seen[0].messages).length;
  const retryChars = JSON.stringify(seen[1].messages).length;
  assert.ok(retryChars < firstChars, `retry must shrink (${retryChars} !< ${firstChars})`);
  assert.ok(seen[1].max_completion_tokens < seen[0].max_completion_tokens);
});

test('rate-limit headers are exposed cross-origin so the client can read them', async () => {
  // Custom response headers are invisible to browser JS on a cross-origin
  // fetch unless the server explicitly exposes them via CORS. Without this,
  // the Dart client's dynamic self-throttling silently sees nothing.
  const response = await worker.fetch(
    new Request('https://worker.test/health', { headers: { Origin: origin } }),
    env(),
  );
  const exposed = response.headers.get('access-control-expose-headers') || '';
  assert.match(exposed, /Retry-After/i);
  assert.match(exposed, /X-RateLimit-Remaining-Tokens/i);
  assert.match(exposed, /X-RateLimit-Reset-Tokens/i);
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
          units: 'mm', mm_per_px: 10,
          walls: [{ id: 'w0', start: [0, 0], end: [100, 0] }],
          rooms: Array.from({ length: 100 }, (_, i) => ({
            id: `room-${i}`, type: 'OFFICE', label: `Office ${i}`,
            boundary: Array.from({ length: 100 }, (_, p) => [p, p + i]),
          })),
          room_graph: { nodes: [], edges: [] },
        },
        visionAdvisory: { status: 'usable', confidence: 0.8, summary: 'Office floor with a fire stair.' },
        metrics: { walls: 1, rooms: 100 }, artifacts: {},
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

test('plan compliance can query_nbc for a specific measured condition before concluding', async (t) => {
  const groqCalls = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (String(url) === 'https://floorplan.test/convert') {
      return Response.json({
        buildingProfile: { occupancy: 'Business' },
        topology: {
          units: 'mm', mm_per_px: 10,
          walls: [{ id: 'w0', start: [0, 0], end: [900, 0] }],
          rooms: [{ id: 'r0', type: 'CORRIDOR', boundary: [[0, 0], [900, 0], [900, 100], [0, 100]] }],
          room_graph: { nodes: [], edges: [] },
        },
        metrics: { walls: 1, rooms: 1 }, artifacts: {},
      });
    }
    const payload = JSON.parse(options.body);
    groqCalls.push(payload);
    if (groqCalls.length === 1) {
      // First hop: the model asks for the exact corridor-width threshold
      // instead of reasoning over a fixed static bucket.
      return Response.json({
        choices: [{ message: {
          content: '',
          tool_calls: [{
            id: 'call_1',
            function: { name: 'query_nbc', arguments: JSON.stringify({ question: 'minimum corridor width Business occupancy' }) },
          }],
        } }],
      });
    }
    // Second hop (no further tool calls) and the forced final strict-schema
    // call both land here; either way, the final call must carry the findings.
    return Response.json({
      choices: [{ message: { content: JSON.stringify({
        planSummary: 'Corridor width checked against the queried threshold.',
        score: 80,
        findings: [{
          check: 'Corridor width', status: 'compliant', severity: 'minor',
          observed: 'Corridor measured 900mm wide', required: 'Per queried clause',
          measurementEvidence: '900mm', clauseId: '', page: null,
          rationale: 'Measured corridor width meets the queried minimum.',
        }],
        citedClauses: [], limitations: [],
      }) } }],
    });
  });
  const form = new FormData();
  form.append('file', new Blob(['plan'], { type: 'image/png' }), 'plan.png');
  const response = await worker.fetch(new Request('https://worker.test/plan/convert', {
    method: 'POST', headers: { Origin: origin }, body: form,
  }), env());
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.compliance.findings[0].status, 'compliant');
  // One lookup hop, then the forced strict-schema final call. Kept at one hop
  // because each extra hop is another full-history gpt-oss call, which on the
  // 8000 TPM/min free tier caused 429s and tripled the wait.
  assert.equal(groqCalls.length, 2);
  assert.ok(groqCalls[0].tools.some((tool) => tool.function.name === 'query_nbc'));
  // Tool-selection hops run on the light model so they don't burn the same
  // 8000 TPM/min pool the final 120b verdict call (and any concurrent
  // site/photo audit reasonCompliance call) draws on.
  assert.equal(groqCalls[0].model, 'openai/gpt-oss-20b');
  const final = groqCalls[groqCalls.length - 1];
  assert.equal(final.model, 'openai/gpt-oss-120b');
  assert.equal(final.tool_choice, 'none');
  assert.equal(final.response_format.type, 'json_schema');
  // No message reaching a tool_choice:'none' call may carry tool_calls —
  // Groq rejected such a request live with tool_use_failed ("Tool choice is
  // none, but model called a tool") even with no `tools` field present,
  // apparently the model continuing the pattern from its own prior turn.
  assert.ok(!final.messages.some((m) => Array.isArray(m.tool_calls) && m.tool_calls.length));
  assert.ok(!final.messages.some((m) => m.role === 'tool'));
  // What the lookup found must still reach the model, as a plain-text summary.
  assert.ok(final.messages.some((m) => m.role === 'user' && String(m.content).includes('NBCS lookup results')));
  assert.ok(final.messages.some((m) => String(m.content).includes('minimum corridor width')));
});

test('a non-approving Qwen advisory never strips the deterministic geometry', async (t) => {
  let groqPayload;
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (String(url) === 'https://floorplan.test/convert') {
      return Response.json({
        buildingProfile: { occupancy: 'Business' },
        // Separate Qwen advisory stream that did NOT approve — must not gate geometry.
        visionAdvisory: {
          status: 'insufficient', confidence: 0.91,
          summary: 'Commercial office and fire stair visible.',
          spaces: [{ label: 'Office', type: 'OFFICE' }],
          openings: [{ type: 'door', center: [25, 0] }, { type: 'window', center: [400, 0] }],
        },
        topology: {
          units: 'mm', mm_per_px: 10,
          walls: [
            { id: 'w0', start: [0, 0], end: [500, 0], wall_type: 'external', thickness_mm: 200 },
          ],
          rooms: [{ id: 'office', type: 'OFFICE', boundary: [[0, 0], [50, 0], [50, 50], [0, 50]] }],
          room_graph: { nodes: [], edges: [] },
        },
        metrics: { walls: 1, rooms: 1 }, artifacts: {},
      });
    }
    groqPayload = JSON.parse(options.body);
    return Response.json({ choices: [{ message: { content: JSON.stringify({
      planSummary: 'Geometry-based assessment.', score: null, findings: [], citedClauses: [],
      limitations: [],
    }) } }] });
  });
  const form = new FormData();
  form.append('file', new Blob(['plan'], { type: 'image/png' }), 'plan.png');
  const response = await worker.fetch(new Request('https://worker.test/plan/convert', {
    method: 'POST', headers: { Origin: origin }, body: form,
  }), env());
  assert.equal(response.status, 200);
  const context = JSON.parse(groqPayload.messages[1].content);
  // Deterministic geometry is fed to GPT-OSS regardless of the Qwen verdict.
  assert.equal(context.plan.assessmentMode, 'verified_geometry');
  assert.equal(context.plan.mmPerPx, 10);
  assert.equal(context.plan.rooms.length, 1);
  // Qwen rides along as a separate advisory input, not a gate.
  assert.equal(context.plan.visionAdvisory.status, 'insufficient');
  assert.equal(context.plan.visionAdvisory.spaces[0].label, 'Office');
  // Regression: the actual wall segments must reach the model, not just a
  // count — a plan with real traced walls previously still read as "no wall
  // geometry" because plan.walls was never included in the payload at all.
  assert.equal(context.plan.walls.length, 1);
  assert.deepEqual(context.plan.walls[0].start, [0, 0]);
  assert.deepEqual(context.plan.walls[0].end, [500, 0]);
  // Qwen's openings read substitutes for the (often-disabled) deterministic
  // door/window detector, with counts precomputed rather than left for the
  // model to tally from a raw array.
  assert.equal(context.plan.visionOpenings.doorCount, 1);
  assert.equal(context.plan.visionOpenings.windowCount, 1);
  const body = await response.clone().json();
  assert.equal(body.compliance.scoreBasis, 'verified_geometry');
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

test('a Groq 429 with a short Retry-After is retried once instead of failing the request', async (t) => {
  let calls = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    calls++;
    if (calls === 1) {
      return new Response(JSON.stringify({ error: { code: 'rate_limit_exceeded', message: 'slow down' } }), {
        status: 429, headers: { 'retry-after': '0.01' },
      });
    }
    return Response.json({ choices: [{ message: { content: 'ok after retry' } }] });
  });
  const response = await worker.fetch(
    post('/groq/chat', { messages: [{ role: 'user', content: 'hello' }] }),
    env(),
  );
  assert.equal(response.status, 200);
  assert.equal((await response.json()).content, 'ok after retry');
  assert.equal(calls, 2);
});

test('a Groq 429 with no usable Retry-After surfaces immediately, no retry', async (t) => {
  let calls = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    calls++;
    return new Response(JSON.stringify({ error: { code: 'rate_limit_exceeded', message: 'slow down' } }), {
      status: 429, headers: {},
    });
  });
  const response = await worker.fetch(
    post('/groq/chat', { messages: [{ role: 'user', content: 'hello' }] }),
    env(),
  );
  assert.equal(response.status, 429);
  assert.equal(calls, 1);
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

test('groqReason uses Mistral as primary when MISTRAL_API_KEY is configured and succeeds', async (t) => {
  const seenUrls = [];
  const seenPayloads = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    seenUrls.push(String(url));
    const payload = JSON.parse(options.body);
    seenPayloads.push(payload);

    if (String(url).includes('api.mistral.ai')) {
      return Response.json({
        choices: [{
          message: {
            content: JSON.stringify({
              occupancySummary: 'Business occupancy (Mistral)',
              score: 95,
              findings: [],
              citedClauses: []
            })
          }
        }]
      });
    }

    if (String(url).includes('api.groq.com')) {
      return Response.json({ choices: [{ message: { content: 'no tools', tool_calls: [] } }] });
    }
    return Response.json({});
  });

  const response = await worker.fetch(
    post('/groq/reason', { buildingProfile: {}, detected: [], docs: [] }),
    env({
      MISTRAL_API_KEY: 'test-mistral-key',
    }),
  );

  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.occupancySummary, 'Business occupancy (Mistral)');
  assert.equal(body.score, 95);

  const mistralIndex = seenUrls.findIndex(u => u.includes('api.mistral.ai'));
  assert.ok(mistralIndex >= 0, 'Mistral should be called');
  const mistralPayload = seenPayloads[mistralIndex];
  assert.equal(mistralPayload.model, 'mistral-medium-latest');
  assert.deepEqual(mistralPayload.response_format, { type: 'json_object' });
});

test('groqReason falls back to Groq when MISTRAL_API_KEY is configured but Mistral fails', async (t) => {
  const seenUrls = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    seenUrls.push(String(url));

    if (String(url).includes('api.mistral.ai')) {
      return new Response(JSON.stringify({ error: { message: 'Mistral internal error' } }), {
        status: 500,
        headers: {},
      });
    }

    if (String(url).includes('api.groq.com')) {
      const payload = JSON.parse(options.body);
      if (payload.tool_choice === 'none') {
        return Response.json({
          choices: [{
            message: {
              content: JSON.stringify({
                occupancySummary: 'Business occupancy (Groq fallback)',
                score: 85,
                findings: [],
                citedClauses: []
              })
            }
          }]
        });
      }
      return Response.json({ choices: [{ message: { content: 'no tools', tool_calls: [] } }] });
    }
    return Response.json({});
  });

  const response = await worker.fetch(
    post('/groq/reason', { buildingProfile: {}, detected: [], docs: [] }),
    env({
      MISTRAL_API_KEY: 'test-mistral-key',
    }),
  );

  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.occupancySummary, 'Business occupancy (Groq fallback)');
  assert.equal(body.score, 85);

  assert.ok(seenUrls.includes('https://api.mistral.ai/v1/chat/completions'), 'Mistral should have been tried');
  const groqCalls = seenUrls.filter(u => u.includes('api.groq.com'));
  assert.equal(groqCalls.length, 2, 'Should call Groq twice (1 tool hop + 1 fallback verdict)');
});

test('groqReason goes straight to Groq when MISTRAL_API_KEY is not configured', async (t) => {
  const seenUrls = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    seenUrls.push(String(url));

    if (String(url).includes('api.groq.com')) {
      const payload = JSON.parse(options.body);
      if (payload.tool_choice === 'none') {
        return Response.json({
          choices: [{
            message: {
              content: JSON.stringify({
                occupancySummary: 'Business occupancy (Groq straight)',
                score: 80,
                findings: [],
                citedClauses: []
              })
            }
          }]
        });
      }
      return Response.json({ choices: [{ message: { content: 'no tools', tool_calls: [] } }] });
    }
    return Response.json({});
  });

  const response = await worker.fetch(
    post('/groq/reason', { buildingProfile: {}, detected: [], docs: [] }),
    env(),
  );

  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.occupancySummary, 'Business occupancy (Groq straight)');
  assert.equal(body.score, 80);

  assert.ok(!seenUrls.includes('https://api.mistral.ai/v1/chat/completions'), 'Mistral should not be called');
  const groqCalls = seenUrls.filter(u => u.includes('api.groq.com'));
  assert.equal(groqCalls.length, 2, 'Should call Groq twice');
});
