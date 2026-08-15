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

const GROQ_BASE = 'https://api.groq.com/openai/v1';
const MAX_JSON_BYTES = 48_000;
const MAX_VISION_BYTES = 22 * 1024 * 1024; // 5 images * ~4MB + JSON overhead
const MAX_TOOL_HOPS = 4;

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
          models: { vision: env.GROQ_VISION_MODEL, reason: env.GROQ_REASON_MODEL },
        }, 200, cors);
      }

      if (!isAllowedOrigin(origin, env)) return json({ error: 'origin not allowed' }, 403, cors);

      const routes = {
        '/groq/chat': groqChat,
        '/groq/vision': groqVision,
        '/groq/reason': groqReason,
        '/nbc/query': nbcQueryRoute,
      };
      const handler = routes[url.pathname];
      if (!handler || request.method !== 'POST') return json({ error: 'not found' }, 404, cors);

      // Per-caller rate limit (best-effort; skips if binding absent).
      const actor = request.headers.get('CF-Connecting-IP') || 'anonymous';
      if (env.GROQ_RATE_LIMITER) {
        const limited = await env.GROQ_RATE_LIMITER.limit({ key: `${url.pathname}:${actor}` });
        if (!limited.success) return json({ error: 'rate limit exceeded' }, 429, cors);
      }

      // /nbc/query needs no Groq key.
      if (url.pathname === '/nbc/query') return handler(request, env, cors);

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

// ── Groq: text chat ─────────────────────────────────────────────────────────
async function groqChat(request, env, cors, apiKey) {
  if (tooLarge(request, MAX_JSON_BYTES)) return json({ error: 'request too large' }, 413, cors);
  let body;
  try { body = await request.json(); } catch { return json({ error: 'invalid JSON' }, 400, cors); }
  const messages = (Array.isArray(body.messages) ? body.messages.slice(-20) : [])
    .filter((m) => ['system', 'user', 'assistant'].includes(m?.role))
    .map((m) => ({ role: m.role, content: String(m.content || '').slice(0, 6000) }));
  if (!messages.length) return json({ error: 'messages are required' }, 400, cors);

  const data = await callGroq(apiKey, {
    model: env.GROQ_REASON_MODEL,
    messages,
    temperature: 0.4,
    max_completion_tokens: 1024,
    include_reasoning: false,
  });
  if (data.error) return json({ error: data.error }, data.status, cors);
  return json({ content: data.json?.choices?.[0]?.message?.content || '' }, 200, cors);
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

  const instruction = String(body.prompt || DEFAULT_VISION_PROMPT).slice(0, 4000);
  const content = [
    { type: 'text', text: instruction },
    ...images.map((url) => ({ type: 'image_url', image_url: { url } })),
  ];

  const data = await callGroq(apiKey, {
    model: env.GROQ_VISION_MODEL,
    messages: [
      { role: 'system', content: VISION_SYSTEM },
      { role: 'user', content },
    ],
    temperature: 0.2,
    max_completion_tokens: 1200,
    response_format: { type: 'json_object' },
  });
  if (data.error) return json({ error: data.error }, data.status, cors);
  const parsed = safeJson(data.json?.choices?.[0]?.message?.content);
  return json({ detections: parsed?.detections ?? parsed ?? [] }, 200, cors);
}

// ── Groq: compliance reasoning with live NBC graph tool ──────────────────────
async function groqReason(request, env, cors, apiKey) {
  if (tooLarge(request, MAX_JSON_BYTES)) return json({ error: 'request too large' }, 413, cors);
  let body;
  try { body = await request.json(); } catch { return json({ error: 'invalid JSON' }, 400, cors); }

  const profile = body.buildingProfile || {};
  const detected = Array.isArray(body.detected) ? body.detected.slice(0, 40) : [];
  const docs = Array.isArray(body.docs) ? body.docs.slice(0, 30) : [];

  const tools = [{
    type: 'function',
    function: {
      name: 'query_nbc',
      description:
        'Query the NBCS 2026 Part F fire-safety graph for the requirement text, '
        + 'protection matrix (Table 7), spacing/coverage rules and IS references '
        + 'relevant to a system, occupancy or clause. Call once per system you must '
        + 'assess (extinguishers, sprinklers, detection/alarm, hydrant/wet-riser, '
        + 'exit signage & emergency lighting, fire doors, refuge/compartmentation).',
      parameters: {
        type: 'object',
        properties: {
          question: { type: 'string', description: 'natural-language regulation question' },
          seed_terms: { type: 'string', description: 'optional extra keywords to focus the search' },
        },
        required: ['question'],
      },
    },
  }];

  const messages = [
    { role: 'system', content: REASON_SYSTEM },
    { role: 'user', content: JSON.stringify({ buildingProfile: profile, detectedEquipment: detected, documents: docs }) },
  ];

  // Tool-calling loop: let gpt-oss pull the exact clauses it needs.
  for (let hop = 0; hop < MAX_TOOL_HOPS; hop++) {
    const data = await callGroq(apiKey, {
      model: env.GROQ_REASON_MODEL,
      messages,
      tools,
      tool_choice: 'auto',
      temperature: 0.3,
      max_completion_tokens: 1500,
      include_reasoning: false,
    });
    if (data.error) return json({ error: data.error }, data.status, cors);

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
        content: JSON.stringify(result).slice(0, 6000),
      });
    }
  }

  // Force a final structured verdict grounded in the tool results.
  const finalData = await callGroq(apiKey, {
    model: env.GROQ_REASON_MODEL,
    messages: [...messages, { role: 'user', content: FINAL_INSTRUCTION }],
    tool_choice: 'none',
    temperature: 0.2,
    max_completion_tokens: 2000,
    include_reasoning: false,
    response_format: { type: 'json_object' },
  });
  if (finalData.error) return json({ error: finalData.error }, finalData.status, cors);
  const verdict = safeJson(finalData.json?.choices?.[0]?.message?.content) || {};
  return json(verdict, 200, cors);
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
async function callGroq(apiKey, payload) {
  const upstream = await fetch(`${GROQ_BASE}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ stream: false, ...payload }),
  });
  let body;
  try { body = await upstream.json(); } catch { body = null; }
  if (!upstream.ok) {
    console.error(JSON.stringify({ message: 'Groq call failed', status: upstream.status, model: payload.model }));
    return { error: `Groq call failed (${upstream.status})`, status: upstream.status };
  }
  return { json: body };
}

// ── Prompts ──────────────────────────────────────────────────────────────────
const VISION_SYSTEM =
  'You are a fire-safety equipment inspector analysing site photos. Report only '
  + 'what is visibly present. Do not infer compliance. Respond with strict JSON.';

const DEFAULT_VISION_PROMPT =
  'Identify fire-safety equipment in these images. For each item return an object '
  + '{type, count, condition, label, confidence}. `type` must be one of: '
  + 'extinguisher, sprinkler, detector, manual_call_point, alarm_panel, '
  + 'exit_sign, emergency_light, fire_door, hydrant_hose_reel, fire_pump, other. '
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
  + 'do not establish a fact, mark it cannot_verify. Cite the clause id and page from '
  + 'the tool results. Call query_nbc once per system before concluding.';

const FINAL_INSTRUCTION =
  'Now output the final compliance assessment as JSON only, no prose, in this shape: '
  + '{"occupancySummary": string, "score": number (0-100 overall compliance), '
  + '"findings": [{"system": string, "status": "compliant"|"gap"|"critical_gap"|"cannot_verify", '
  + '"severity": "minor"|"major"|"critical", "observed": string, "required": string, '
  + '"clauseId": string, "page": number, "rationale": string}], '
  + '"citedClauses": [{"id": string, "title": string, "page": number}]}. '
  + 'Base every finding only on the query_nbc tool results already gathered.';

// ── Shared helpers (from the Cloudflare-Groq skill) ──────────────────────────
async function readSecret(binding) {
  if (!binding) return '';
  if (typeof binding === 'string') return binding;
  return binding.get();
}
function tooLarge(request, max) {
  return Number(request.headers.get('content-length') || 0) > max;
}
function safeJson(str) {
  if (!str) return null;
  try { return JSON.parse(str); } catch { return null; }
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
    'Access-Control-Allow-Headers': 'Content-Type',
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
