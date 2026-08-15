# FireShield secure Groq gateway (Cloudflare Worker)

Proxies Groq **vision** (`qwen/qwen3.6-27b`) and **reasoning** (`openai/gpt-oss-120b`)
so `GROQ_API_KEY` never reaches the browser. The reasoning route queries the
**NBCS 2026 Part F** graph live via a `query_nbc` tool (`src/nbc_graph.js`).

Follow `../../../../Cloudflare worker and groq skill.md` for the security rules.
**Run `wrangler whoami` before every mutation and stop if it shows the wrong account.**

## Routes
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + binding presence (never the key) |
| POST | `/groq/chat` | text chat (AI Assistant) |
| POST | `/groq/vision` | `{images:[dataURL…], prompt?}` → `{detections:[…]}` |
| POST | `/groq/reason` | `{buildingProfile, detected, docs}` → compliance verdict JSON |
| POST | `/nbc/query` | `{question, seed_terms?, k?, hops?}` → graph results (no Groq key) |

## One-time setup (PowerShell, isolated profile)

```powershell
Set-Location "<this worker folder>"
$env:XDG_CONFIG_HOME = "$PWD\.cloudflare-correct-profile"
New-Item -ItemType Directory -Force -Path $env:XDG_CONFIG_HOME | Out-Null
npm.cmd install
npm.cmd exec wrangler -- login
npm.cmd exec wrangler -- whoami     # confirm the intended account
```

1. **Secrets Store** — reuse the existing `GROQ_API_KEY`. Find the store id
   (metadata only, never the value) and paste it into `wrangler.jsonc` `<STORE_ID>`:
   ```powershell
   npm.cmd exec wrangler -- secrets-store store list --remote
   npm.cmd exec wrangler -- secrets-store secret list <STORE_ID> --remote
   ```
2. **R2 bucket** for the NBC index:
   ```powershell
   npm.cmd exec wrangler -- r2 bucket create fireshield-nbc-index
   ```
3. **Build + upload the index** (regenerate whenever the graph changes):
   ```powershell
   npm.cmd run index:build
   npm.cmd run index:upload
   ```
4. Set `<OWNER>` in `ALLOWED_ORIGINS` (your GitHub Pages origin), and confirm the
   rate-limit `namespace_id` (`53020`) is unused on the account.

## Deploy + verify

```powershell
npm.cmd exec wrangler -- whoami
npm.cmd exec wrangler -- deploy --dry-run    # inspect bindings: GROQ_API_KEY, NBC_BUCKET, models
npm.cmd exec wrangler -- deploy
```

```powershell
# health
Invoke-RestMethod -Uri "https://fireshield-groq-gateway.<SUBDOMAIN>.workers.dev/health"

# graph query (no key needed server-side)
$h = @{ Origin = "https://<OWNER>.github.io"; "Content-Type" = "application/json" }
$b = @{ question = "sprinkler requirement for hospital above 45 m" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "https://fireshield-groq-gateway.<SUBDOMAIN>.workers.dev/nbc/query" -Headers $h -Body $b
```

Expect: allowed origin → 200; unlisted origin → 403; burst → 429; missing key → 503
(explicit, never stale content). Tail logs: `npm.cmd run tail`.

Put only the deployed URL in the app (`lib/fireshield/services/fs_config.dart` via
`--dart-define=FIRESHIELD_WORKER_URL=…`). Never put the Groq key in the frontend.
