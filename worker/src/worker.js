// FireShield secure Groq gateway.
//
// Trust boundary (see "Cloudflare worker and groq skill.md"):
//   GitHub Pages PWA  ──HTTPS, no key──►  this Worker  ──Bearer key──►  api.groq.com
//
// The browser never sees GROQ_API_KEY. This Worker validates origin, rate-limits,
// reads the key from Secrets Store, and adds it server-side. It exposes:
//   GET  /health            liveness + binding presence (never the key)
//   POST /groq/chat         text chat (AI Assistant)
//   POST /groq/vision       qwen vision → structured equipment/scene JSON
//   POST /groq/reason       gpt-oss compliance, querying the NBC graph as a tool
//   POST /nbc/query         direct graph query (debug / Regulations phase)

import { queryNbc } from './nbc_graph.js';

// Compatibility tombstone for the short-lived Cloudflare Containers attempt.
// Existing Durable Object metadata still references this export. Keeping the
// class avoids an irreversible delete migration; no binding routes requests to
// it and it stores no state.
export class FloorplanContainer {
  async fetch() {
    return json({ error: 'legacy floor-plan container retired' }, 410);
  }
}

const GROQ_BASE = 'https://api.groq.com/openai/v1';
const MAX_JSON_BYTES = 48_000;
const MAX_VISION_BYTES = 22 * 1024 * 1024; // 5 images * ~4MB + JSON overhead
const MAX_TOOL_HOPS = 4;
// One lookup round, not three. Each extra hop is another gpt-oss call carrying
// the whole growing history, which on an 8000 TPM/min free tier both exhausted
// the budget (429s with ~45s cooldowns) and tripled the wait for a result. One
// round still lets the model fetch the clauses it needs before concluding.
const PLAN_MAX_TOOL_HOPS = 1;
// Groq's free-tier TPM counts input tokens PLUS max_completion_tokens against
// the same 8000/min budget, so a single request can be rejected outright with
// 413 "Request too large" — no waiting or model-switching can rescue it, the
// payload itself must fit. The tool-calling loop is what blew past this: each
// hop appends an assistant message plus tool results, so by the final call the
// accumulated history plus the reserved completion budget exceeded 8000.
// Well under the hard 8000 cap: the budget is shared with whatever else the
// session just did, and the 429s showed ~7900 already consumed before a plan
// assessment even started.
const GROQ_TPM_BUDGET = 5200;
// The verdict must fit a complete JSON object (summary, findings, citations,
// limitations). Squeezing this to 1100 truncated the generation mid-object and
// failed schema validation, so it is sized to finish the response while input
// plus output still stays under the 8000/min cap.
const PLAN_FINAL_MAX_TOKENS = 1600;
const REASON_HOP_MAX_TOKENS = 700;
const REASON_FINAL_MAX_TOKENS = 1600;
const TOOL_RESULT_MAX_CHARS = 1500;

// Groq bills whole tokens; ~4 chars/token is the standard rough estimate and
// only needs to be conservative enough to keep us under the cap.
function estimateTokens(text) {
  return Math.ceil(String(text || '').length / 4);
}

function messageTokens(message) {
  return estimateTokens(message?.content || '')
    + estimateTokens(JSON.stringify(message?.tool_calls || ''))
    + 4; // per-message role/format overhead
}

/// Shrink an accumulated tool-calling conversation to fit `maxInputTokens`.
/// Message STRUCTURE is preserved (never drop an assistant message carrying
/// tool_calls, or its matching tool replies — orphaned tool_call_ids are a
/// hard API error). Instead the oldest tool results, which are bulky NBC
/// Collapse a tool-calling loop into plain text before a tool_choice:'none'
/// call. Observed live: Groq rejected the final call with tool_use_failed
/// ("Tool choice is none, but model called a tool") even though that request
/// carried no `tools` field at all — the model still attempted a tool call,
/// apparently continuing the pattern from an earlier assistant turn still
/// present in its own message history. Removing every tool_calls-bearing
/// assistant turn (and its paired tool replies) from the context the final
/// call sees removes the pattern to continue, regardless of the exact cause;
/// what was learned from those lookups is preserved as a plain-text summary
/// so the verdict is still grounded in them.
function summariseToolLoop(messages) {
  const base = [];
  const findings = [];
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i];
    if (message.role === 'assistant' && Array.isArray(message.tool_calls) && message.tool_calls.length) {
      const args = message.tool_calls.map((call) => {
        try { return JSON.parse(call.function?.arguments || '{}'); } catch { return {}; }
      });
      let index = 0;
      while (i + 1 < messages.length && messages[i + 1].role === 'tool') {
        i++;
        const question = String(args[index]?.question || args[index]?.seed_terms || '(lookup)');
        findings.push(`Q: ${question.slice(0, 200)}\nA: ${String(messages[i].content || '').slice(0, 800)}`);
        index++;
      }
      continue; // drop the assistant tool_calls turn itself
    }
    if (message.role === 'assistant' && !(message.content || '').trim()) continue; // empty turn, nothing to keep
    base.push(message);
  }
  return { base, summary: findings.join('\n\n') };
}

/// lookups already reflected in later reasoning, get truncated first.
function fitMessagesToTokenBudget(messages, maxInputTokens) {
  const fitted = messages.map((message) => ({ ...message }));
  const total = () => fitted.reduce((sum, message) => sum + messageTokens(message), 0);
  if (total() <= maxInputTokens) return fitted;

  // Pass 1: truncate tool results oldest-first, leaving a marker so the model
  // knows the lookup happened rather than silently seeing nothing.
  for (let i = 0; i < fitted.length && total() > maxInputTokens; i++) {
    if (fitted[i].role !== 'tool') continue;
    const budgetChars = 400;
    if ((fitted[i].content || '').length > budgetChars) {
      fitted[i].content = `${fitted[i].content.slice(0, budgetChars)} …[truncated to fit token budget]`;
    }
  }
  // Pass 2: still too big — drop the bulk of older tool payloads entirely,
  // keeping the message (and its tool_call_id) so pairing stays valid.
  for (let i = 0; i < fitted.length && total() > maxInputTokens; i++) {
    if (fitted[i].role !== 'tool') continue;
    fitted[i].content = '[earlier NBCS lookup omitted to fit the token budget]';
  }
  // Pass 3: the evidence context itself is the last thing to shrink.
  if (total() > maxInputTokens) {
    const contextIndex = fitted.findIndex((message) => message.role === 'user');
    if (contextIndex >= 0) {
      const overBy = (total() - maxInputTokens) * 4;
      const content = fitted[contextIndex].content || '';
      fitted[contextIndex].content = content.slice(0, Math.max(1200, content.length - overBy - 200));
    }
  }
  return fitted;
}
const NBC_QUERY_TOOL = {
  type: 'function',
  function: {
    name: 'query_nbc',
    description:
      'Query the NBCS 2026 Part F fire-safety graph for the requirement text, '
      + 'protection matrix (Table 7), spacing/coverage rules and IS references '
      + 'relevant to a system, occupancy, clause, or a specific measured condition '
      + '(e.g. an exact corridor width, exit count, or travel distance from the '
      + 'geometry). Call it for every system or measurement your evidence implicates '
      + 'before concluding.',
    parameters: {
      type: 'object',
      properties: {
        question: { type: 'string', description: 'natural-language regulation question' },
        seed_terms: { type: 'string', description: 'optional extra keywords to focus the search' },
      },
      required: ['question'],
    },
  },
};
const VISUAL_GUIDANCE_QUERIES = {
  extinguisher: 'fire extinguisher requirement inspection installation Table 7',
  sprinkler: 'automatic wet sprinkler requirement sprinkler heads inspection',
  smoke_detector: 'automatic smoke detection alarm system point type smoke detector',
  heat_detector: 'heat detector automatic fire detection alarm system',
  manual_call_point: 'manual call point break glass fire alarm',
  alarm_sounder_strobe: 'fire alarm notification visual alarm strobe voice evacuation',
  alarm_panel: 'fire alarm panel annunciation fire command centre',
  exit_sign: 'escape lighting exit signage means of egress',
  emergency_light: 'emergency lighting escape route duration illumination',
  fire_door: 'fire door fire rating self closing means of egress',
  door_closer: 'fire door self closing device door closer',
  hydrant_hose_reel: 'hose reel wet riser yard hydrant Table 7',
  landing_valve: 'landing valve wet riser firefighting shaft',
  kitchen_suppression: 'wet chemical kitchen suppression commercial cooking area',
  fire_pump: 'fire pumps water supply firefighting installation',
  fire_department_connection: 'fire brigade inlet connection dividing breeching',
  evacuation_map: 'floor plan stairway marking sign evacuation information',
};

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const origin = request.headers.get('Origin') || '';
    const cors = corsHeaders(origin, env);
    try {
      if (request.method === 'OPTIONS') {
        if (!isAllowedOrigin(origin, env)) return json({ error: 'origin not allowed' }, 403, cors);
        return new Response(null, { status: 204, headers: cors });
      }
      if (url.pathname === '/health' && request.method === 'GET') {
        return json({
          ok: true,
          service: 'fireshield groq gateway',
          groqBindingConfigured: Boolean(env.GROQ_API_KEY),
          nbcIndexConfigured: Boolean(env.NBC_BUCKET || env.NBC_KV),
          rateLimitsConfigured: Boolean(env.REASON_RATE_LIMITER && env.VISION_RATE_LIMITER),
          floorplanConfigured: Boolean(
            env.FLOORPLAN_SERVICE_URL && env.FLOORPLAN_SERVICE_TOKEN && env.PLAN_RATE_LIMITER,
          ),
          models: {
            vision: env.GROQ_VISION_MODEL,
            reason: env.GROQ_REASON_MODEL,
            lightFallback: env.GROQ_LIGHT_MODEL,
          },
        }, 200, cors);
      }

      if (!isAllowedOrigin(origin, env)) return json({ error: 'origin not allowed' }, 403, cors);

      if (url.pathname === '/plan/convert' && request.method === 'POST') {
        const limited = await enforcePlanRateLimit(env, cors);
        if (limited) return limited;
        const modelLimited = await enforcePlanModelRateLimits(env, cors);
        if (modelLimited) return modelLimited;
        const groqKey = await readSecret(env.GROQ_API_KEY);
        if (!groqKey) return json({ error: 'GROQ_API_KEY is not configured' }, 503, cors);
        return convertFloorplan(request, env, cors, groqKey);
      }

      const routes = {
        '/groq/chat': groqChat,
        '/groq/vision': groqVision,
        '/groq/reason': groqReason,
        '/nbc/query': nbcQueryRoute,
      };
      const handler = routes[url.pathname];
      if (!handler || request.method !== 'POST') return json({ error: 'not found' }, 404, cors);

      // /nbc/query needs no Groq key.
      if (url.pathname === '/nbc/query') return handler(request, env, cors);

      const rateLimitResponse = await enforceGroqRateLimit(url.pathname, env, cors);
      if (rateLimitResponse) return rateLimitResponse;

      const groqKey = await readSecret(env.GROQ_API_KEY);
      if (!groqKey) return json({ error: 'GROQ_API_KEY is not configured' }, 503, cors);
      return handler(request, env, cors, groqKey);
    } catch (error) {
      console.error(JSON.stringify({
        message: 'gateway request failed', path: url.pathname,
        error: error instanceof Error ? error.message : String(error),
      }));
      return json({ error: 'gateway request failed' }, 500, cors);
    }
  },
};

async function convertFloorplan(request, env, cors, groqKey) {
  if (tooLarge(request, 25 * 1024 * 1024)) {
    return json({ error: 'floor plan exceeds the 25 MB limit' }, 413, cors);
  }
  const serviceUrl = String(env.FLOORPLAN_SERVICE_URL || '').replace(/\/$/, '');
  const serviceToken = await readSecret(env.FLOORPLAN_SERVICE_TOKEN);
  if (!serviceUrl || !serviceToken) {
    return json({ error: 'floor plan processor is not configured' }, 503, cors);
  }
  let upstream;
  try {
    upstream = await fetch(`${serviceUrl}/convert`, {
      method: 'POST',
      headers: {
        'Content-Type': request.headers.get('Content-Type') || 'application/octet-stream',
        'X-FireShield-Service-Token': serviceToken,
        'X-FireShield-Groq-Key': groqKey,
      },
      body: request.body,
      signal: AbortSignal.timeout(180_000),
    });
  } catch (error) {
    console.error(JSON.stringify({
      message: 'floor plan processor unavailable',
      error: error instanceof Error ? error.message : String(error),
    }));
    return json({ error: 'floor plan processor unavailable' }, 502, cors);
  }
  let converted;
  try {
    converted = await upstream.json();
  } catch {
    return json({ error: 'floor plan processor returned invalid JSON' }, 502, cors);
  }
  if (!upstream.ok) return json(converted, upstream.status, cors);

  const compliance = await assessFloorplan(converted, env, groqKey);
  if (compliance.error) {
    return json({
      ...converted,
      partial: true,
      compliance: {
        assessmentStatus: 'unavailable',
        planSummary: 'Plan conversion completed, but the NBCS reasoning provider did not return an assessment.',
        score: 50,
        scoreBasis: 'evidence_availability',
        scoreConfidence: 0.05,
        findings: [],
        citedClauses: [],
        limitations: [compliance.error],
        // Client can dynamically self-throttle its retry using Groq's own
        // cooldown instead of a fixed guess.
        retryAfterSeconds: compliance.retryAfterSeconds ?? null,
      },
    }, 200, { ...cors, ...compliance.headers });
  }
  return json({ ...converted, compliance: compliance.value }, 200, { ...cors, ...compliance.headers });
}

async function assessFloorplan(converted, env, apiKey) {
  // A fixed set of generic topic buckets can't answer a question tailored to
  // this building's actual measured geometry (a specific corridor width, exit
  // count, travel distance). Give gpt-oss the same query_nbc tool the site/photo
  // audit already uses so it looks up the exact clause for what it measures,
  // instead of reasoning only over whatever six static queries happened to return.
  const plan = compactPlanForReasoning(converted);
  const messages = [
    { role: 'system', content: PLAN_REASON_SYSTEM },
    { role: 'user', content: fitPlanReasoningContext(plan, []) },
  ];

  // Same split as groqReason: tool-selection hops prefer gpt-oss-20b (its own
  // TPM pool) so they don't eat into the 8000 TPM/min budget the final
  // gpt-oss-120b verdict call below needs. Re-picked every hop so a currently
  // tight 20b pool (e.g. from a concurrent site/photo audit hop) doesn't get
  // blindly retried — it automatically switches to whichever model has room.
  for (let hop = 0; hop < PLAN_MAX_TOOL_HOPS; hop++) {
    const hopModel = pickAvailableModel(
      env.GROQ_LIGHT_MODEL || env.GROQ_REASON_MODEL, env.GROQ_REASON_MODEL, 900);
    const data = await callGroq(apiKey, {
      model: hopModel,
      messages: fitMessagesToTokenBudget(messages, GROQ_TPM_BUDGET - 900),
      tools: [NBC_QUERY_TOOL],
      tool_choice: 'auto',
      temperature: 0.2,
      max_completion_tokens: 900,
      reasoning_effort: 'low',
      include_reasoning: false,
    });
    if (data.error) return data;

    const msg = data.json?.choices?.[0]?.message;
    if (!msg) return { error: 'GPT-OSS returned an empty response', status: 502, headers: {} };
    messages.push({ role: 'assistant', content: msg.content || '', tool_calls: msg.tool_calls });

    if (!msg.tool_calls || msg.tool_calls.length === 0) break;
    for (const call of msg.tool_calls) {
      let args = {};
      try { args = JSON.parse(call.function.arguments || '{}'); } catch { /* ignore */ }
      let result;
      try {
        result = await queryNbc(env, String(args.question || ''), {
          seedTerms: String(args.seed_terms || ''), k: 3, hops: 1,
        });
      } catch (e) {
        result = { error: 'nbc query failed', detail: String(e.message || e) };
      }
      messages.push({
        role: 'tool',
        tool_call_id: call.id,
        content: JSON.stringify(result).slice(0, 4000),
      });
    }
  }

  // Prefer 120b for the actual verdict; switch to 20b only if 120b's pool is
  // the one currently tight. Both models support JSON Schema Mode and tool
  // use (per their Groq model pages), so either can serve any step here.
  const finalModel = pickAvailableModel(env.GROQ_REASON_MODEL, env.GROQ_LIGHT_MODEL, PLAN_FINAL_MAX_TOKENS);
  const { base: planBase, summary: planFindings } = summariseToolLoop(messages);
  const finalMessages = fitMessagesToTokenBudget(
    [
      ...planBase,
      {
        role: 'user',
        content: (planFindings ? `NBCS lookup results:\n${planFindings}\n\n` : '')
          + 'Now output the final compliance assessment as JSON only, matching the schema.',
      },
    ],
    GROQ_TPM_BUDGET - PLAN_FINAL_MAX_TOKENS,
  );
  const finalPayload = {
    model: finalModel,
    messages: finalMessages,
    tool_choice: 'none',
    temperature: 0.2,
    max_completion_tokens: PLAN_FINAL_MAX_TOKENS,
    reasoning_effort: 'low',
    include_reasoning: false,
    response_format: PLAN_RESPONSE_FORMAT,
  };
  let finalData = await callGroq(apiKey, finalPayload);
  // One reinforced retry on a schema-validation miss, mirroring the vision
  // path. Without this a single malformed generation discarded an otherwise
  // complete assessment and surfaced as "provider did not return an assessment".
  if (finalData.error && finalData.code === 'json_validate_failed') {
    // The usual cause is the generation being cut off mid-object, so the retry
    // both asks for a compact answer and raises the completion room rather than
    // repeating the same budget that truncated.
    finalData = await callGroq(apiKey, {
      ...finalPayload,
      max_completion_tokens: PLAN_FINAL_MAX_TOKENS + 400,
      messages: [...finalMessages, {
        role: 'user',
        content: 'Your previous reply failed schema validation, most likely because it was cut '
          + 'off. Return exactly one COMPLETE JSON object with every required field present. '
          + 'Keep it compact: at most six findings and short rationales. Use null (not a missing '
          + 'field) for clauseId or page when a finding has no clause to cite.',
      }],
    });
  }
  if (finalData.error) return finalData;
  const value = safeJson(finalData.json?.choices?.[0]?.message?.content);
  if (!value || !Array.isArray(value.findings)) {
    return { error: 'GPT-OSS returned an invalid plan assessment', status: 502, headers: {} };
  }
  // Forward gpt-oss-120b's own remaining-quota numbers so the client can pace
  // its *next* Groq-dependent call instead of guessing a fixed delay.
  return { value: finalisePlanAssessment(value, plan), headers: finalData.headers };
}

function finalisePlanAssessment(value, plan) {
  // clauseId/page are nullable in the schema (a cannot_verify finding has no
  // clause to cite). Normalise to '' here so the app and stored history keep
  // seeing plain strings.
  if (Array.isArray(value.findings)) {
    value.findings = value.findings.map((finding) => ({
      ...finding,
      clauseId: finding?.clauseId ?? '',
    }));
  }
  if (Array.isArray(value.citedClauses)) {
    value.citedClauses = value.citedClauses.map((clause) => ({
      ...clause,
      id: clause?.id ?? '',
      title: clause?.title ?? '',
    }));
  }
  const findings = Array.isArray(value.findings) ? value.findings : [];
  const severityWeight = { minor: 1, major: 2, critical: 3 };
  const statusScore = { compliant: 100, gap: 35, critical_gap: 0, cannot_verify: 50 };
  let weightedScore = 0;
  let totalWeight = 0;
  let verifiable = 0;
  for (const finding of findings) {
    const weight = severityWeight[finding.severity] || 1;
    weightedScore += (statusScore[finding.status] ?? 50) * weight;
    totalWeight += weight;
    if (finding.status !== 'cannot_verify') verifiable++;
  }
  const calculated = totalWeight ? weightedScore / totalWeight : 50;
  const supplied = typeof value.score === 'number' ? value.score : Number.NaN;
  const score = Number.isFinite(supplied) ? supplied : calculated;
  const evidenceFraction = findings.length ? verifiable / findings.length : 0;
  const geometryVerified = plan.assessmentMode === 'verified_geometry';
  const hasGeometry = plan.assessmentMode !== 'image_semantic';
  const advisoryConfidence = Number(plan.visionAdvisory?.confidence) || 0;
  // Geometry (scaled or unscaled) is the primary evidence; the Qwen advisory
  // stream only nudges confidence, it is never the sole basis when a DXF exists.
  const confidence = hasGeometry
    ? 0.45 + 0.25 * evidenceFraction + (geometryVerified ? 0.15 : 0) + 0.10 * advisoryConfidence
    : 0.15 + 0.35 * evidenceFraction + 0.30 * advisoryConfidence;
  return {
    ...value,
    assessmentStatus: geometryVerified && evidenceFraction >= 0.5 ? 'complete' : 'provisional',
    score: Math.round(Math.max(0, Math.min(score, 100)) * 10) / 10,
    scoreBasis: plan.assessmentMode,
    scoreConfidence: Math.round(Math.max(0.05, Math.min(confidence, 1)) * 100) / 100,
  };
}

function fitPlanReasoningContext(plan, guidance, maxChars = 12_000) {
  const context = { plan, nbcsGuidance: guidance };
  let serialized = JSON.stringify(context);
  if (serialized.length <= maxChars) return serialized;

  // Preserve measurements and citations first; polygon detail and lower-ranked
  // retrieval hits are reduced before whole entities are omitted.
  context.plan.rooms = context.plan.rooms.map(({ boundary, ...room }) => room);
  context.plan.exteriorBoundary = context.plan.exteriorBoundary.slice(0, 20);
  context.plan.roomGraph.nodes = context.plan.roomGraph.nodes.slice(0, 50);
  context.plan.roomGraph.edges = context.plan.roomGraph.edges.slice(0, 80);
  context.nbcsGuidance = context.nbcsGuidance.map((group) => ({
    ...group,
    results: (group.results || []).slice(0, 2).map((result) => ({
      ...result,
      rationale: String(result.rationale || '').slice(0, 160),
      snippet: String(result.snippet || '').slice(0, 220),
    })),
  }));
  serialized = JSON.stringify(context);
  if (serialized.length <= maxChars) return serialized;

  const sourceCounts = {
    walls: context.plan.walls.length,
    rooms: context.plan.rooms.length,
    doors: context.plan.doors.length,
    windows: context.plan.windows.length,
    objects: context.plan.objects.length,
  };
  // Walls are the core geometry evidence (see the walls-omission bug this
  // fixed: a plan with real traced walls must never reach the model looking
  // like it has none) — trim last and lightest, everything else first.
  context.plan.rooms = context.plan.rooms.slice(0, 25);
  context.plan.doors = context.plan.doors.slice(0, 40);
  context.plan.windows = context.plan.windows.slice(0, 40);
  context.plan.objects = context.plan.objects.slice(0, 40);
  context.plan.roomGraph.nodes = context.plan.roomGraph.nodes.slice(0, 25);
  context.plan.roomGraph.edges = context.plan.roomGraph.edges.slice(0, 40);
  context.plan.contextCoverage = {
    sourceCounts,
    includedCounts: {
      walls: context.plan.walls.length,
      rooms: context.plan.rooms.length,
      doors: context.plan.doors.length,
      windows: context.plan.windows.length,
      objects: context.plan.objects.length,
    },
    note: 'Lower-priority entity detail omitted to stay within the Groq token budget.',
  };
  serialized = JSON.stringify(context);
  if (serialized.length <= maxChars) return serialized;

  // Walls still get cut before being zeroed entirely, but only here, and only
  // down to a bound — never to nothing, since assessmentMode/hasGeometry
  // already told the model geometry exists and it must not contradict that.
  context.plan.walls = context.plan.walls.slice(0, 60);
  context.plan.rooms = [];
  context.plan.doors = [];
  context.plan.windows = [];
  context.plan.objects = [];
  context.plan.roomGraph = { nodes: [], edges: [] };
  context.nbcsGuidance = context.nbcsGuidance.map((group) => ({
    question: group.question,
    results: (group.results || []).slice(0, 1),
  }));
  return JSON.stringify(context);
}

// Bound the site-assessment reasoning prompt the same way fitPlanReasoningContext
// bounds the floor-plan one. Unbounded observedEquipmentGuidance (up to 8 types x
// several NBC results each, with page-text snippets) was the "prompt too big"
// stall reported after the CLIPSeg step: gpt-oss-120b would sit far longer than
// expected processing a context that dwarfed the actual detections it described.
function fitReasonContext(context, maxChars = 10_000) {
  let serialized = JSON.stringify(context);
  if (serialized.length <= maxChars) return serialized;

  // Trim guidance depth first — it's supporting citation material, not evidence.
  context.observedEquipmentGuidance = (context.observedEquipmentGuidance || []).map((group) => ({
    ...group,
    results: (group.results || []).slice(0, 2).map((result) => ({
      ...result,
      rationale: String(result.rationale || '').slice(0, 120),
      snippet: String(result.snippet || '').slice(0, 160),
      relations: undefined,
    })),
  }));
  serialized = JSON.stringify(context);
  if (serialized.length <= maxChars) return serialized;

  // Then reduce breadth: fewer guided types, fewer documents.
  context.observedEquipmentGuidance = context.observedEquipmentGuidance.slice(0, 4);
  context.documents = (context.documents || []).slice(0, 10);
  serialized = JSON.stringify(context);
  if (serialized.length <= maxChars) return serialized;

  // Final bounded form: keep every detected item (that's the actual evidence)
  // but drop citation guidance to a single top result per type.
  context.observedEquipmentGuidance = context.observedEquipmentGuidance.map((group) => ({
    type: group.type,
    results: (group.results || []).slice(0, 1),
  }));
  context.documents = (context.documents || []).slice(0, 5);
  return JSON.stringify(context);
}

function compactPlanForReasoning(converted) {
  const topology = converted?.topology || {};
  const advisory = converted?.visionAdvisory || {};
  const metrics = converted?.metrics || {};
  const point = (value) => Array.isArray(value) ? value.slice(0, 2).map(Number) : value;

  // The deterministic DXF is the geometry authority and is ALWAYS reasoned over.
  // Whether printed scale was recovered decides only if absolute NBC dimensions
  // (widths, travel distances in metres) can be checked — never whether geometry
  // is included at all. Qwen never gates this.
  const wallCount = Number(metrics.walls)
    || (Array.isArray(topology.walls) ? topology.walls.length : 0);
  const mmPerPx = Number(topology.mm_per_px);
  const hasScale = Number.isFinite(mmPerPx) && mmPerPx > 0;
  const hasGeometry = wallCount > 0;
  const assessmentMode = hasGeometry
    ? (hasScale ? 'verified_geometry' : 'geometry_unscaled')
    : 'image_semantic';

  return {
    assessmentMode,
    buildingProfile: {
      occupancy: String(converted?.buildingProfile?.occupancy || '').slice(0, 100),
      buildingHeightM: Number(converted?.buildingProfile?.buildingHeightM) || null,
      floors: Number(converted?.buildingProfile?.floors) || null,
      floorAreaM2: Number(converted?.buildingProfile?.floorAreaM2
        || converted?.commercialModel?.building?.floorAreaM2) || null,
      sprinklered: converted?.buildingProfile?.sprinklered === true,
    },
    // Separate Qwen advisory stream: semantic observations that enrich the
    // reasoning context. It is INPUT, not a filter — it cannot remove geometry.
    visionAdvisory: {
      status: String(advisory.status || 'not_run'),
      confidence: Number(advisory.confidence) || 0,
      summary: String(advisory.summary || '').slice(0, 500),
      spaces: Array.isArray(advisory.spaces) ? advisory.spaces.slice(0, 80) : [],
      openings: Array.isArray(advisory.openings) ? advisory.openings.slice(0, 80) : [],
      elements: Array.isArray(advisory.elements) ? advisory.elements.slice(0, 80) : [],
    },
    metrics: { ...metrics, geometryVerified: hasScale && hasGeometry },
    units: topology.units,
    mmPerPx: hasScale ? mmPerPx : null,
    imageSize: topology.image_size,
    exteriorBoundary: Array.isArray(topology.exterior_boundary)
      ? topology.exterior_boundary.slice(0, 60).map(point) : [],
    // The actual traced wall segments — this is the geometry the system
    // prompt tells gpt-oss to reason over. It was missing entirely before:
    // only the wallCount (via metrics) reached the model, so a plan with 37
    // correctly-traced walls still read as "no wall geometry" to the model.
    walls: Array.isArray(topology.walls) ? topology.walls.slice(0, 120).map((wall) => ({
      id: wall.id, start: point(wall.start), end: point(wall.end),
      wall_type: wall.wall_type, thickness_mm: wall.thickness_mm,
    })) : [],
    rooms: Array.isArray(topology.rooms) ? topology.rooms.slice(0, 50).map((room) => ({
      id: room.id, type: room.type, label: room.label,
      area_mm2: room.area_mm2,
      boundary: Array.isArray(room.boundary) ? room.boundary.slice(0, 24).map(point) : [],
    })) : [],
    doors: Array.isArray(topology.doors) ? topology.doors.slice(0, 80).map((door) => ({
      id: door.id, wall_id: door.wall_id, width_mm: door.width_mm, center: point(door.center),
    })) : [],
    windows: Array.isArray(topology.windows) ? topology.windows.slice(0, 80).map((window) => ({
      id: window.id, wall_id: window.wall_id, width_mm: window.width_mm, center: point(window.center),
    })) : [],
    // The deterministic door/window detector needs the CubiCasa model, which
    // is disabled at runtime (Render memory limit) — plan.doors/windows are
    // then genuinely empty, not a bug. Qwen's specify() call already reads
    // door/window openings in that same single pass (no extra Groq call, no
    // extra TPM cost) — surface it as real evidence with counts computed here
    // rather than leaving gpt-oss to count array entries itself.
    visionOpenings: (() => {
      const items = (Array.isArray(advisory.openings) ? advisory.openings : [])
        .slice(0, 80)
        .map((item) => ({
          type: String(item?.type || '').slice(0, 20),
          center: point(item?.center),
          // Which spaces the opening joins, and whether it reaches outdoors.
          // An external door is an egress point; an internal one is not, and
          // a bare count cannot distinguish them.
          connects: Array.isArray(item?.connects)
            ? item.connects.slice(0, 2).map((name) => String(name).slice(0, 80)) : [],
          isExternal: item?.isExternal === true,
        }));
      const doors = items.filter((item) => item.type === 'door');
      return {
        doorCount: doors.length,
        windowCount: items.filter((item) => item.type === 'window').length,
        externalDoorCount: doors.filter((item) => item.isExternal).length,
        items,
      };
    })(),
    objects: Array.isArray(topology.objects) ? topology.objects.slice(0, 80).map((object) => ({
      id: object.id, type: object.type, center: point(object.center), confidence: object.confidence,
    })) : [],
    roomGraph: {
      nodes: Array.isArray(topology.room_graph?.nodes) ? topology.room_graph.nodes.slice(0, 80)
        .map((node) => typeof node === 'object' ? {
          id: String(node.id || '').slice(0, 80), type: String(node.type || '').slice(0, 80),
        } : String(node).slice(0, 80)) : [],
      edges: Array.isArray(topology.room_graph?.edges) ? topology.room_graph.edges.slice(0, 120)
        .map((edge) => typeof edge === 'object' ? {
          source: String(edge.source || '').slice(0, 80), target: String(edge.target || '').slice(0, 80),
          type: String(edge.type || '').slice(0, 80),
        } : String(edge).slice(0, 160)) : [],
    },
  };
}

async function enforcePlanRateLimit(env, cors) {
  if (!env.PLAN_RATE_LIMITER) return json({ error: 'plan rate limiter is not configured' }, 503, cors);
  const result = await env.PLAN_RATE_LIMITER.limit({ key: 'floorplan-conversion' });
  if (result.success) return null;
  return json({ error: 'floor plan conversion rate limit exceeded', retryAfterSeconds: 60 }, 429, {
    ...cors,
    'Retry-After': '60',
    'X-RateLimit-Policy': '5;w=60',
  });
}

async function enforcePlanModelRateLimits(env, cors) {
  const reservations = [
    [env.VISION_RATE_LIMITER, 'qwen/qwen3.6-27b'],
    [env.REASON_RATE_LIMITER, 'openai/gpt-oss-120b'],
  ];
  for (const [limiter, key] of reservations) {
    if (!limiter) return json({ error: 'AI rate limiter is not configured' }, 503, cors);
    const result = await limiter.limit({ key });
    if (!result.success) {
      return json({ error: 'AI request rate limit exceeded', retryAfterSeconds: 60 }, 429, {
        ...cors,
        'Retry-After': '60',
        'X-RateLimit-Policy': '25;w=60',
      });
    }
  }
  return null;
}

// ── Groq: text chat ─────────────────────────────────────────────────────────
async function groqChat(request, env, cors, apiKey) {
  if (tooLarge(request, MAX_JSON_BYTES)) return json({ error: 'request too large' }, 413, cors);
  let body;
  try { body = await request.json(); } catch { return json({ error: 'invalid JSON' }, 400, cors); }
  const messages = (Array.isArray(body.messages) ? body.messages.slice(-20) : [])
    .filter((m) => ['system', 'user', 'assistant'].includes(m?.role))
    .map((m) => ({ role: m.role, content: String(m.content || '').slice(0, 6000) }));
  if (!messages.length) return json({ error: 'messages are required' }, 400, cors);

  // Ground the assistant in the same page-anchored NBCS 2026 index used by the
  // compliance engine. If lookup fails, the model is told not to invent rules.
  const latestQuestion = [...messages].reverse().find((m) => m.role === 'user')?.content || '';
  let nbcContext = { results: [] };
  try {
    nbcContext = await queryNbc(env, latestQuestion, { k: 8, hops: 1 });
  } catch (error) {
    console.error(JSON.stringify({
      message: 'chat NBC context lookup failed',
      error: error instanceof Error ? error.message : String(error),
    }));
  }

  const data = await callGroq(apiKey, {
    model: env.GROQ_REASON_MODEL,
    messages: [
      { role: 'system', content: CHAT_SYSTEM },
      {
        role: 'system',
        content: `Relevant NBCS 2026 index results (cite page numbers when used):\n${JSON.stringify(nbcContext.results).slice(0, 8000)}`,
      },
      ...messages,
    ],
    temperature: 0.4,
    max_completion_tokens: 1024,
    include_reasoning: false,
  });
  if (data.error) {
    return json({ error: data.error, retryAfterSeconds: data.retryAfterSeconds ?? null },
      data.status, { ...cors, ...data.headers });
  }
  return json({ content: data.json?.choices?.[0]?.message?.content || '' }, 200, { ...cors, ...data.headers });
}

// ── Groq: vision (equipment / scene detection) ───────────────────────────────
async function groqVision(request, env, cors, apiKey) {
  if (tooLarge(request, MAX_VISION_BYTES)) return json({ error: 'request too large' }, 413, cors);
  let body;
  try { body = await request.json(); } catch { return json({ error: 'invalid JSON' }, 400, cors); }

  const images = Array.isArray(body.images) ? body.images.slice(0, 5) : [];
  if (!images.length) return json({ error: 'images are required' }, 400, cors);
  for (const dataUrl of images) {
    if (typeof dataUrl !== 'string' || !/^data:image\/(png|jpe?g|webp);base64,/.test(dataUrl)) {
      return json({ error: 'each image must be a base64 data URL (png/jpeg/webp)' }, 400, cors);
    }
  }

  const evidenceContext = sanitiseEvidenceContext(body.evidenceContext);

  const instruction = `${String(body.prompt || DEFAULT_VISION_PROMPT).slice(0, 4000)}\n`
    + `Evidence capture context: ${JSON.stringify(evidenceContext)}. `
    + 'Use this only to locate the scene; do not treat the auditor notes as proof of equipment or condition.';
  const content = [
    { type: 'text', text: instruction },
    ...images.map((url) => ({ type: 'image_url', image_url: { url } })),
  ];

  const visionPayload = {
    model: env.GROQ_VISION_MODEL,
    messages: [
      { role: 'system', content: VISION_SYSTEM },
      { role: 'user', content },
    ],
    temperature: 0.2,
    max_completion_tokens: 1200,
    // Qwen 3.6 defaults toward its thinking mode, which can consume the entire
    // completion budget on hidden reasoning tokens for a complex/alarming scene
    // and return empty content (json_validate_failed with no failed_generation).
    // Force non-thinking mode so tokens go to the JSON response itself.
    reasoning_effort: 'none',
    response_format: { type: 'json_object' },
  };
  let data = await callGroq(apiKey, visionPayload);
  // Groq JSON Object Mode can emit invalid JSON (json_validate_failed). The docs
  // prescribe a validate-and-retry; do exactly one reinforced retry.
  if (data.error && data.code === 'json_validate_failed') {
    data = await callGroq(apiKey, {
      ...visionPayload,
      messages: [
        { role: 'system', content: VISION_SYSTEM },
        { role: 'user', content },
        {
          role: 'user',
          content: 'Your previous reply failed JSON validation. Return exactly one '
            + 'valid JSON object of the requested shape, with no markdown, comments '
            + 'or trailing text.',
        },
      ],
    });
  }
  if (data.error) {
    return json({ error: data.error, retryAfterSeconds: data.retryAfterSeconds ?? null },
      data.status, { ...cors, ...data.headers });
  }
  const parsed = safeJson(data.json?.choices?.[0]?.message?.content);
  return json({ detections: parsed?.detections ?? parsed ?? [] }, 200, { ...cors, ...data.headers });
}

// ── Groq: compliance reasoning with live NBC graph tool ──────────────────────
async function groqReason(request, env, cors, apiKey) {
  if (tooLarge(request, MAX_JSON_BYTES)) return json({ error: 'request too large' }, 413, cors);
  let body;
  try { body = await request.json(); } catch { return json({ error: 'invalid JSON' }, 400, cors); }

  const profile = body.buildingProfile || {};
  const detected = Array.isArray(body.detected) ? body.detected.slice(0, 40) : [];
  const docs = Array.isArray(body.docs) ? body.docs.slice(0, 30) : [];
  const evidenceContext = sanitiseEvidenceContext(body.evidenceContext);
  const observedTypes = [...new Set(detected.map((item) => String(item?.type || '')))]
    .filter((type) => VISUAL_GUIDANCE_QUERIES[type])
    .slice(0, 8);
  const observedGuidance = await Promise.all(observedTypes.map(async (type) => {
    try {
      const result = await queryNbc(env, VISUAL_GUIDANCE_QUERIES[type], { k: 3, hops: 1 });
      return { type, results: result.results };
    } catch (error) {
      return { type, error: 'NBCS guidance lookup failed' };
    }
  }));

  const messages = [
    { role: 'system', content: REASON_SYSTEM },
    { role: 'user', content: fitReasonContext({
      buildingProfile: profile,
      evidenceContext,
      detectedEquipment: detected,
      observedEquipmentGuidance: observedGuidance,
      documents: docs,
    }) },
  ];

  // Tool-calling loop: let gpt-oss pull the exact clauses it needs. Deciding
  // *which* clause to look up is a light task — prefer gpt-oss-20b (a separate
  // TPM pool) and reserve gpt-oss-120b for the final synthesis below.
  // Re-picked every hop (not fixed once) so if 20b's own pool is the one
  // currently tight — e.g. a concurrent floor-plan compliance hop just used
  // it — this automatically falls through to whichever model actually has
  // headroom right now, rather than blindly retrying the one that's tight.
  for (let hop = 0; hop < MAX_TOOL_HOPS; hop++) {
    const hopModel = pickAvailableModel(
      env.GROQ_LIGHT_MODEL || env.GROQ_REASON_MODEL, env.GROQ_REASON_MODEL, REASON_HOP_MAX_TOKENS);
    const data = await callGroq(apiKey, {
      model: hopModel,
      messages: fitMessagesToTokenBudget(messages, GROQ_TPM_BUDGET - REASON_HOP_MAX_TOKENS),
      tools: [NBC_QUERY_TOOL],
      tool_choice: 'auto',
      temperature: 0.3,
      max_completion_tokens: REASON_HOP_MAX_TOKENS,
      include_reasoning: false,
    });
    if (data.error) {
      return json({ error: data.error, retryAfterSeconds: data.retryAfterSeconds ?? null },
        data.status, { ...cors, ...data.headers });
    }

    const msg = data.json?.choices?.[0]?.message;
    if (!msg) return json({ error: 'empty model response' }, 502, cors);
    // Never echo hidden reasoning back to the client.
    messages.push({ role: 'assistant', content: msg.content || '', tool_calls: msg.tool_calls });

    if (!msg.tool_calls || msg.tool_calls.length === 0) break;

    for (const call of msg.tool_calls) {
      let args = {};
      try { args = JSON.parse(call.function.arguments || '{}'); } catch { /* ignore */ }
      let result;
      try {
        result = await queryNbc(env, String(args.question || ''), {
          seedTerms: String(args.seed_terms || ''),
        });
      } catch (e) {
        result = { error: 'nbc query failed', detail: String(e.message || e) };
      }
      messages.push({
        role: 'tool',
        tool_call_id: call.id,
        content: JSON.stringify(result).slice(0, TOOL_RESULT_MAX_CHARS),
      });
    }
  }

  // Force a final structured verdict grounded in the tool results. This is
  // the step that actually needs 120b's reasoning strength, so it's the
  // preferred model — but if 120b's pool is the one currently tight and 20b
  // has room, switch automatically rather than fail a request 20b could
  // still answer (degraded, but real, beats a hard 429 mid-demo).
  const finalModel = pickAvailableModel(
    env.GROQ_REASON_MODEL, env.GROQ_LIGHT_MODEL, REASON_FINAL_MAX_TOKENS);
  const { base: reasonBase, summary: reasonFindings } = summariseToolLoop(messages);
  const finalData = await callGroq(apiKey, {
    model: finalModel,
    messages: fitMessagesToTokenBudget(
      [
        ...reasonBase,
        {
          role: 'user',
          content: (reasonFindings ? `NBCS lookup results:\n${reasonFindings}\n\n` : '') + FINAL_INSTRUCTION,
        },
      ],
      GROQ_TPM_BUDGET - REASON_FINAL_MAX_TOKENS,
    ),
    tool_choice: 'none',
    temperature: 0.2,
    max_completion_tokens: REASON_FINAL_MAX_TOKENS,
    include_reasoning: false,
    response_format: { type: 'json_object' },
  });
  if (finalData.error) {
    return json({ error: finalData.error, retryAfterSeconds: finalData.retryAfterSeconds ?? null },
      finalData.status, { ...cors, ...finalData.headers });
  }
  const verdict = safeJson(finalData.json?.choices?.[0]?.message?.content) || {};
  return json(verdict, 200, { ...cors, ...finalData.headers });
}

// ── Direct graph query (debug / Regulations phase) ───────────────────────────
async function nbcQueryRoute(request, env, cors) {
  if (tooLarge(request, MAX_JSON_BYTES)) return json({ error: 'request too large' }, 413, cors);
  let body;
  try { body = await request.json(); } catch { return json({ error: 'invalid JSON' }, 400, cors); }
  const question = String(body.question || '').slice(0, 500);
  if (!question) return json({ error: 'question is required' }, 400, cors);
  try {
    const result = await queryNbc(env, question, {
      seedTerms: String(body.seed_terms || '').slice(0, 300),
      k: Math.min(Number(body.k) || 8, 15),
      hops: Math.min(Number(body.hops) || 1, 2),
    });
    return json(result, 200, cors);
  } catch (e) {
    console.error(JSON.stringify({ message: 'nbc query failed', error: String(e.message || e) }));
    return json({ error: 'nbc index unavailable' }, 503, cors);
  }
}

// ── Groq call helper ─────────────────────────────────────────────────────────
// A plan assessment now completes in 8-9s in normal conditions (down from
// ~45s+ before the single-hop change) — a 50s wait ceiling on a 429 would
// have reintroduced exactly that near-minute latency on any collision. Kept
// short enough to absorb a brief overlap without becoming the slow path.
const MAX_RATE_LIMIT_WAIT_MS = 12_000;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Best-effort, per-isolate memory of each model's last-known Groq quota.
// Cloudflare may run multiple isolates across edge locations, so this isn't
// a perfectly global view — but within one isolate (the common case for a
// single demo session hitting the same nearby edge) it lets every call pick
// whichever of gpt-oss-120b/gpt-oss-20b currently has headroom, instead of
// a fixed model assignment that can still collide if that specific model's
// pool is the one already under pressure.
const groqQuota = {};

function parseGroqDurationSeconds(raw) {
  if (!raw) return null;
  const match = /^(?:(\d+)m)?(\d+(?:\.\d+)?)(m?s)$/.exec(String(raw).trim());
  if (!match) {
    const plain = parseFloat(raw);
    return Number.isFinite(plain) ? plain : null;
  }
  const minutes = parseFloat(match[1] || '0') || 0;
  const value = parseFloat(match[2] || '0') || 0;
  const seconds = match[3] === 'ms' ? value / 1000 : value;
  return minutes * 60 + seconds;
}

function recordGroqQuota(model, headers) {
  if (!model || !headers) return;
  const remaining = Number(headers['x-ratelimit-remaining-tokens']);
  if (!Number.isFinite(remaining)) return;
  const resetSeconds = parseGroqDurationSeconds(headers['x-ratelimit-reset-tokens']);
  groqQuota[model] = {
    remainingTokens: remaining,
    resetAt: resetSeconds != null ? Date.now() + resetSeconds * 1000 : (groqQuota[model]?.resetAt ?? 0),
  };
}

function groqHasHeadroom(model, estimatedTokens) {
  const state = groqQuota[model];
  if (!state) return true; // never observed -> assume available
  if (Date.now() >= state.resetAt) return true; // window has likely rolled over
  return state.remainingTokens >= estimatedTokens;
}

/// Picks whichever of two models currently has headroom for roughly
/// [estimatedTokens]. [preferred] wins when both look fine (it's the model
/// suited to the task); otherwise falls through to [fallback], and if both
/// look tight, picks whichever has more remaining as a last resort — the
/// call still goes out and the existing 429 retry-after backoff covers it.
function pickAvailableModel(preferred, fallback, estimatedTokens) {
  if (!fallback || fallback === preferred) return preferred;
  if (groqHasHeadroom(preferred, estimatedTokens)) return preferred;
  if (groqHasHeadroom(fallback, estimatedTokens)) return fallback;
  const preferredRemaining = groqQuota[preferred]?.remainingTokens ?? 0;
  const fallbackRemaining = groqQuota[fallback]?.remainingTokens ?? 0;
  return fallbackRemaining > preferredRemaining ? fallback : preferred;
}

async function callGroq(apiKey, payload, { retriedAfterRateLimit = false } = {}) {
  const upstream = await fetch(`${GROQ_BASE}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ stream: false, ...payload }),
  });
  let body;
  try { body = await upstream.json(); } catch { body = null; }
  if (!upstream.ok) {
    const detail = body?.error && typeof body.error === 'object' ? body.error : {};
    const code = String(detail.code || detail.type || '').slice(0, 100);
    const message = String(detail.message || '').slice(0, 500);
    // Groq's 429 for a short-lived per-minute token/request budget is a normal,
    // expected condition (multiple audit phases can land in the same window) —
    // not a hard failure. Honour its own Retry-After and retry once rather than
    // bubbling a 429 straight to the UI.
    if (upstream.status === 429 && !retriedAfterRateLimit) {
      const waitMs = Math.round(parseFloat(upstream.headers.get('retry-after') || '0') * 1000);
      if (waitMs > 0 && waitMs <= MAX_RATE_LIMIT_WAIT_MS) {
        await sleep(waitMs);
        return callGroq(apiKey, payload, { retriedAfterRateLimit: true });
      }
    }
    // 413 "Request too large" is NOT a wait-and-retry condition — the payload
    // exceeds the whole per-minute budget, so waiting changes nothing. Retry
    // once with the history aggressively compacted and a smaller completion
    // reservation, which is the only thing that can actually make it fit.
    if (upstream.status === 413 && !retriedAfterRateLimit && Array.isArray(payload.messages)) {
      const shrunkCompletion = Math.max(600, Math.floor((payload.max_completion_tokens || 1200) / 2));
      const shrunk = {
        ...payload,
        max_completion_tokens: shrunkCompletion,
        messages: fitMessagesToTokenBudget(payload.messages, Math.floor(GROQ_TPM_BUDGET / 2)),
      };
      return callGroq(apiKey, shrunk, { retriedAfterRateLimit: true });
    }
    // Server-log only diagnostic: never returned to the client. Helps debug
    // json_validate_failed (Groq includes the model's raw non-JSON output here).
    console.error(JSON.stringify({ message: 'Groq call failed', status: upstream.status, model: payload.model, code }));
    const providerDetail = [code, message].filter(Boolean).join(': ');
    const errorHeaders = groqRateLimitHeaders(upstream.headers);
    recordGroqQuota(payload.model, errorHeaders);
    return {
      error: `Groq call failed (${upstream.status})${providerDetail ? `: ${providerDetail}` : ''}`,
      status: upstream.status,
      code,
      retryAfterSeconds: parseFloat(upstream.headers.get('retry-after') || '0') || null,
      headers: errorHeaders,
    };
  }
  // Forward quota headers on success too (not just on 429). This is what lets
  // the client self-throttle its *next* call using Groq's own numbers —
  // x-ratelimit-remaining-tokens / -reset-tokens — instead of a fixed guess.
  const successHeaders = groqRateLimitHeaders(upstream.headers);
  recordGroqQuota(payload.model, successHeaders);
  return { json: body, headers: successHeaders };
}

// ── Prompts ──────────────────────────────────────────────────────────────────
const VISION_SYSTEM =
  'You are a fire-safety equipment inspector analysing site photos. Report only '
  + 'what is visibly present. Do not infer compliance. Respond with strict JSON.';

const CHAT_SYSTEM =
  'You are FireShield AI, a concise fire and life-safety assistant grounded in '
  + 'NBCS 2026 Part F (India). Use only the supplied NBCS index results for '
  + 'specific regulatory claims. Cite the source page when a result supports '
  + 'your answer. If the supplied results are insufficient, say that the rule '
  + 'could not be verified instead of inventing a clause, distance or threshold.';

const DEFAULT_VISION_PROMPT =
  'Identify fire-safety equipment in these images. For each item return an object '
  + '{type, count, condition, label, confidence}. `type` must be one of: '
  + 'extinguisher, sprinkler, smoke_detector, heat_detector, manual_call_point, '
  + 'alarm_sounder_strobe, alarm_panel, exit_sign, emergency_light, fire_door, '
  + 'door_closer, hydrant_hose_reel, landing_valve, kitchen_suppression, '
  + 'fire_pump, fire_department_connection, evacuation_map, other. '
  + '`count` = how many of that type are visible across the images. `condition` = '
  + 'short note (e.g. "gauge in green", "obstructed", "expired tag", "unknown"). '
  + '`label` = any readable text/rating (e.g. "ABC 6kg", "120 min"). '
  + '`confidence` = 0..1. Return {"detections":[...]} only.';

const REASON_SYSTEM =
  'You are a fire and life-safety compliance auditor working strictly to NBCS 2026 '
  + 'Part F (India). You are given a building profile and equipment observed in site '
  + 'photos. Determine which fire-safety systems are MANDATORY for this occupancy, '
  + 'height and area, then use the query_nbc tool to fetch the exact requirement '
  + '(counts, spacing, coverage, Table 7 protection level, IS references) for each. '
  + 'Compare required vs observed. Never invent a requirement or a pass: if the photos '
  + 'do not establish a fact, mark it cannot_verify. A missing visual detection is not '
  + 'proof that equipment is absent unless the evidence explicitly documents complete '
  + 'coverage of the relevant floor/area. CLIPSeg supplies object-location evidence only; '
  + 'Qwen condition notes may be used as observations but not as measurements or certificates. '
  + 'For every observed equipment class, query the applicable inspection/placement guidance. '
  + 'Cite the clause id and page from '
  + 'the tool results. Call query_nbc once per system before concluding.';

const FINAL_INSTRUCTION =
  'Now output the final compliance assessment as JSON only, no prose, in this shape: '
  + '{"occupancySummary": string, "score": number (0-100 overall compliance), '
  + '"findings": [{"system": string, "status": "compliant"|"gap"|"critical_gap"|"cannot_verify", '
  + '"severity": "minor"|"major"|"critical", "observed": string, "required": string, '
  + '"clauseId": string, "page": number, "rationale": string}], '
  + '"citedClauses": [{"id": string, "title": string, "page": number}]}. '
  + 'Base every finding only on the query_nbc tool results already gathered.';

const PLAN_REASON_SYSTEM =
  'You are FireShield AI assessing a building plan against NBCS 2026 Part F (India). '
  + 'The `plan` geometry (walls, rooms, doors, windows, exteriorBoundary, roomGraph) was extracted '
  + 'deterministically from the drawing and is the authoritative geometric evidence — always reason over it. '
  + '`visionAdvisory` is a SEPARATE Qwen visual stream: use it only as supporting context for space labels and '
  + 'visible safety features; it is advisory and never overrides or invalidates the geometry. '
  + '`plan.doors`/`plan.windows` come from a deterministic door/window detector that is often disabled — they '
  + 'may legitimately be empty even when doors/windows are visibly present in the drawing. In that case, use '
  + '`plan.visionOpenings` (Qwen\'s read: {doorCount, windowCount, externalDoorCount, '
  + 'items:[{type,center,connects,isExternal}]}) as the egress evidence instead of treating an empty '
  + 'plan.doors/windows as proof none exist. `connects` names the spaces each opening joins and `isExternal` '
  + 'marks the ones leading outdoors, so use externalDoorCount and the connects graph to assess exit count, '
  + 'exit separation/remoteness, and whether any space is a dead end with only one way out. Because it is '
  + 'advisory, still mark absolute door/window WIDTHS as cannot_verify unless plan.doors/windows independently '
  + 'confirms a measured width — but exit count, presence and connectivity must be assessed from visionOpenings '
  + 'rather than marked cannot_verify for no reason. '
  + 'Identify which fire-safety systems and life-safety checks this specific geometry implicates (exit count and '
  + 'width, travel distance, corridor width, compartmentation, detection/sprinkler coverage, refuge area, etc.), '
  + 'then use the query_nbc tool to fetch the exact NBCS requirement for each — including a specific measured '
  + 'value from the geometry when relevant (e.g. query the minimum corridor width for this occupancy, then '
  + 'compare it against the measured width). Never invent a clause id, page, or numeric threshold: only cite '
  + 'what a tool result returned. Call query_nbc for every check before concluding; a check with no tool result '
  + 'to support it must be cannot_verify. '
  + 'Never infer occupancy, door purpose, exit designation, fire rating, or protection equipment the structured '
  + 'plan does not establish. When assessmentMode is verified_geometry, printed scale was recovered: reason over '
  + 'absolute widths, areas and travel distances. When assessmentMode is geometry_unscaled, the geometry is '
  + 'trustworthy but scale was not recovered: reason over counts, connectivity, relative geometry and layout, '
  + 'and mark only absolute-dimension checks (metre widths, travel distances) as cannot_verify. When '
  + 'assessmentMode is image_semantic, no geometry was extracted: rely on visionAdvisory observations. '
  + 'A missing detected object is not proof of absence. Return JSON only: {"planSummary":string,"score":number,'
  + '"findings":[{"check":string,"status":"compliant"|"gap"|"critical_gap"|"cannot_verify",'
  + '"severity":"minor"|"major"|"critical","observed":string,"required":string,"measurementEvidence":string,'
  + '"clauseId":string,"page":number|null,"rationale":string}],'
  + '"citedClauses":[{"id":string,"title":string,"page":number}],"limitations":[string]}. '
  + 'Always assign a numeric best-available score from 0 to 100 using all findings. Treat cannot_verify as '
  + 'neutral uncertainty, never as a pass. The gateway separately calculates scoreBasis and scoreConfidence.';

const PLAN_RESPONSE_FORMAT = {
  type: 'json_schema',
  json_schema: {
    name: 'fire_plan_compliance_assessment',
    strict: true,
    schema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        planSummary: { type: 'string' },
        score: { type: 'number' },
        findings: {
          type: 'array',
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              check: { type: 'string' },
              status: { type: 'string', enum: ['compliant', 'gap', 'critical_gap', 'cannot_verify'] },
              severity: { type: 'string', enum: ['minor', 'major', 'critical'] },
              observed: { type: 'string' },
              required: { type: 'string' },
              measurementEvidence: { type: 'string' },
              // A cannot_verify finding legitimately has no clause to cite, and
              // the model emits null for it. Declaring this non-nullable made a
              // single such finding fail the WHOLE assessment with
              // json_validate_failed. Groq's structured-output docs prescribe
              // union-with-null for exactly this; nulls are normalised to ''
              // in finalisePlanAssessment so downstream shape is unchanged.
              clauseId: { type: ['string', 'null'] },
              page: { type: ['number', 'null'] },
              rationale: { type: 'string' },
            },
            required: [
              'check', 'status', 'severity', 'observed', 'required',
              'measurementEvidence', 'clauseId', 'page', 'rationale',
            ],
          },
        },
        citedClauses: {
          type: 'array',
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              id: { type: 'string' },
              title: { type: 'string' },
              // Same nullability trap as clauseId: a cited entry can lack a page.
              page: { type: ['number', 'null'] },
            },
            required: ['id', 'title', 'page'],
          },
        },
        limitations: { type: 'array', items: { type: 'string' } },
      },
      required: [
        'planSummary', 'score', 'findings', 'citedClauses', 'limitations',
      ],
    },
  },
};

// ── Shared helpers (from the Cloudflare-Groq skill) ──────────────────────────
async function readSecret(binding) {
  if (!binding) return '';
  if (typeof binding === 'string') return binding;
  return binding.get();
}
async function enforceGroqRateLimit(path, env, cors) {
  const isVision = path === '/groq/vision';
  const limiter = isVision ? env.VISION_RATE_LIMITER : env.REASON_RATE_LIMITER;
  if (!limiter) {
    return json({ error: 'rate limiter is not configured' }, 503, cors);
  }

  // A reasoning request can make four tool-loop calls plus one final verdict.
  // Reserve those calls against the shared gpt-oss budget before doing work.
  const units = path === '/groq/reason' ? MAX_TOOL_HOPS + 1 : 1;
  const key = isVision ? 'qwen/qwen3.6-27b' : 'openai/gpt-oss-120b';
  for (let i = 0; i < units; i++) {
    const result = await limiter.limit({ key });
    if (!result.success) {
      console.warn(JSON.stringify({ message: 'rate limited', path, model: key }));
      return json({
        error: 'AI request rate limit exceeded',
        retryAfterSeconds: 60,
      }, 429, {
        ...cors,
        'Retry-After': '60',
        'X-RateLimit-Policy': '25;w=60',
      });
    }
  }
  return null;
}
function groqRateLimitHeaders(headers) {
  const names = [
    'retry-after',
    'x-ratelimit-limit-requests',
    'x-ratelimit-limit-tokens',
    'x-ratelimit-remaining-requests',
    'x-ratelimit-remaining-tokens',
    'x-ratelimit-reset-requests',
    'x-ratelimit-reset-tokens',
  ];
  const forwarded = {};
  for (const name of names) {
    const value = headers.get(name);
    if (value) forwarded[name] = value;
  }
  return forwarded;
}
function tooLarge(request, max) {
  return Number(request.headers.get('content-length') || 0) > max;
}
function safeJson(str) {
  if (!str) return null;
  try { return JSON.parse(str); } catch { return null; }
}
function sanitiseEvidenceContext(value) {
  const raw = value && typeof value === 'object' ? value : {};
  return {
    level: String(raw.level || '').slice(0, 80),
    zone: String(raw.zone || '').slice(0, 160),
    purpose: String(raw.purpose || '').slice(0, 80),
    coverage: ['spot_check', 'complete_area'].includes(raw.coverage) ? raw.coverage : 'spot_check',
    notes: String(raw.notes || '').slice(0, 500),
    imageCount: Math.max(0, Math.min(Number(raw.imageCount) || 0, 5)),
  };
}
function allowedOrigins(env) {
  return String(env.ALLOWED_ORIGINS || '').split(',').map((v) => v.trim()).filter(Boolean);
}
function isAllowedOrigin(origin, env) {
  return Boolean(origin) && allowedOrigins(env).includes(origin);
}
function corsHeaders(origin, env) {
  const headers = {
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    // Custom response headers are invisible to browser JS on a cross-origin
    // call unless explicitly exposed. Without this, the Dart client's dynamic
    // rate-limit self-throttling silently sees no headers at all.
    'Access-Control-Expose-Headers':
      'Retry-After, X-RateLimit-Limit-Requests, X-RateLimit-Limit-Tokens, '
      + 'X-RateLimit-Remaining-Requests, X-RateLimit-Remaining-Tokens, '
      + 'X-RateLimit-Reset-Requests, X-RateLimit-Reset-Tokens',
    'Access-Control-Max-Age': '86400',
    Vary: 'Origin',
    'X-Content-Type-Options': 'nosniff',
    'Referrer-Policy': 'no-referrer',
  };
  if (isAllowedOrigin(origin, env)) headers['Access-Control-Allow-Origin'] = origin;
  return headers;
}
function json(value, status, headers = {}) {
  return new Response(JSON.stringify(value), {
    status, headers: { ...headers, 'Content-Type': 'application/json; charset=utf-8' },
  });
}

// Test-only surface for the dynamic model-availability picker. groqQuota is
// exported by reference so tests can seed/clear it directly without going
// through the full HTTP+mock-fetch machinery (and without leaking state into
// unrelated tests in the same process).
export const __testing__ = {
  pickAvailableModel, groqQuota, fitMessagesToTokenBudget, estimateTokens, summariseToolLoop,
};
