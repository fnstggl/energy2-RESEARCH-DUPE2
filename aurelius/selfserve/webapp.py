"""Local web UI for Aurelius Replay.

A single-page app served from 127.0.0.1 with zero external assets and zero
outbound calls — the page itself is the privacy statement. Flow:

1. drop a log file → ``POST /api/inspect`` → mapping plan + detected class,
2. review/override mappings (each override is an explicit
   ``canonical=source`` pair, same as the CLI ``--map``),
3. ``POST /api/run`` → readiness + descriptive + replay + report link,
4. ``GET /api/runs/{id}/report`` → the self-contained HTML report.

Uploads and run artifacts live under a local directory (default
``aurelius_replay_runs/`` in the working directory) and never leave it.
FastAPI + uvicorn are existing project dependencies (``aurelius/api``).

Note: no ``from __future__ import annotations`` here — FastAPI must evaluate
the endpoint annotations (``UploadFile``) eagerly inside the app factory.
"""

import json
import shutil
import uuid
from pathlib import Path
from typing import Optional

from . import TOOL_NAME, __version__
from .fingerprint import write_fingerprint
from .pipeline import inspect_file, run_pipeline
from .report import write_report
from .sniff import TableReadError

_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TOOL__</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background:#050505; color:#f0f0f0;
  font:15px/1.6 "Helvetica Neue",Helvetica,Arial,sans-serif; }
.mono { font-family:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace; }
.wrap { max-width:920px; margin:0 auto; padding:48px 24px 96px; }
.wordmark { font-family:ui-monospace,monospace; font-size:13px;
  letter-spacing:.35em; color:#b49b62; }
h1 { font-size:26px; font-weight:500; margin:14px 0 6px; }
.sub { color:rgba(240,240,240,.6); max-width:64ch; }
.privacy { font-family:ui-monospace,monospace; font-size:12px;
  color:rgba(240,240,240,.7); border:1px solid rgba(255,255,255,.11);
  padding:10px 14px; margin:22px 0; }
#drop { border:1px dashed rgba(255,255,255,.25); padding:44px 20px;
  text-align:center; color:rgba(240,240,240,.6); margin:24px 0;
  cursor:pointer; transition:border-color .15s; }
#drop.hover { border-color:#b49b62; color:#b49b62; }
button { background:none; border:1px solid #b49b62; color:#b49b62;
  font-family:ui-monospace,monospace; font-size:12px; letter-spacing:.14em;
  text-transform:uppercase; padding:10px 22px; cursor:pointer; }
button:disabled { opacity:.35; cursor:default; }
table { border-collapse:collapse; width:100%; margin:14px 0; font-size:13px; }
th { font-family:ui-monospace,monospace; font-size:11px; letter-spacing:.12em;
  text-transform:uppercase; color:rgba(240,240,240,.55); text-align:left;
  font-weight:400; border-bottom:1px solid rgba(255,255,255,.22);
  padding:7px 9px; }
td { border-bottom:1px solid rgba(255,255,255,.08); padding:7px 9px; }
select { background:#0a0a0a; color:#f0f0f0;
  border:1px solid rgba(255,255,255,.2); font-family:ui-monospace,monospace;
  font-size:12px; padding:4px 6px; }
.tier { display:inline-block; font-family:ui-monospace,monospace;
  font-size:11px; letter-spacing:.18em; text-transform:uppercase;
  padding:4px 10px; border:1px solid; }
.section { margin-top:40px; display:none; }
.err { color:#c96f6d; margin:12px 0; }
.ok { color:#7fae6a; } .warn { color:#b49b62; } .fail { color:#9b3f3d; }
a { color:#b49b62; }
.spin { color:rgba(240,240,240,.55); font-family:ui-monospace,monospace;
  font-size:12px; letter-spacing:.14em; }
</style></head>
<body><div class="wrap">
<div class="wordmark">AURELIUS&nbsp;REPLAY</div>
<h1>Replay your own scheduler history</h1>
<p class="sub">Point this at a historical scheduler / serving log export
(CSV, TSV or JSON-lines; gzip ok). It reruns the same arrivals through the
Aurelius decision layer and reports what would have changed — read-only,
against history, on this machine.</p>
<div class="privacy">LOCAL ONLY — this page is served from 127.0.0.1, makes
no outbound requests, and writes results only to the local run directory.
Directional analysis — not production savings.</div>

<div id="drop">drop a log file here, or click to choose
<input type="file" id="file" style="display:none"></div>
<div id="error" class="err"></div>

<div id="mapsec" class="section">
  <h1>1 · Confirm the mapping</h1>
  <p class="sub">Detected workload class:
    <span id="wclass" class="mono"></span> ·
    <span id="wevid" class="sub"></span></p>
  <table id="maptable"></table>
  <button id="runbtn">run replay</button>
  <span id="spin" class="spin"></span>
</div>

<div id="ressec" class="section">
  <h1>2 · Result</h1>
  <p><span class="tier" id="tier"></span></p>
  <p class="sub" id="headline" style="margin-top:14px"></p>
  <ul id="checks" style="margin:14px 0 14px 20px"></ul>
  <p style="margin-top:18px">
    <a id="rlink" target="_blank">open the full report</a> ·
    <a id="flink" target="_blank">schema fingerprint (shareable)</a></p>
</div>

<script>
const drop = document.getElementById('drop');
const fileEl = document.getElementById('file');
const err = document.getElementById('error');
let uploadId = null, mapping = null;

drop.onclick = () => fileEl.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('hover'); };
drop.ondragleave = () => drop.classList.remove('hover');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('hover');
  if (e.dataTransfer.files.length) inspect(e.dataTransfer.files[0]); };
fileEl.onchange = () => { if (fileEl.files.length) inspect(fileEl.files[0]); };

async function inspect(f) {
  err.textContent = ''; uploadId = null;
  drop.textContent = 'reading ' + f.name + ' …';
  const fd = new FormData(); fd.append('file', f);
  const r = await fetch('/api/inspect', { method:'POST', body: fd });
  const j = await r.json();
  drop.textContent = f.name;
  if (!r.ok) { err.textContent = j.detail || 'could not read file'; return; }
  uploadId = j.upload_id; mapping = j.mapping;
  renderMapping(j);
  document.getElementById('mapsec').style.display = 'block';
  document.getElementById('ressec').style.display = 'none';
}

function renderMapping(j) {
  document.getElementById('wclass').textContent = j.mapping.workload_class;
  document.getElementById('wevid').textContent = j.mapping.class_evidence;
  const t = document.getElementById('maptable');
  const cols = ['<option value="">— not present —</option>']
    .concat(j.columns.map(c => `<option>${esc(c)}</option>`)).join('');
  let h = '<tr><th>canonical field</th><th>your column</th>' +
          '<th>match</th><th>transform</th><th>samples</th></tr>';
  for (const m of j.mapping.mappings) {
    const sel = `<select data-canonical="${esc(m.canonical)}">` +
      cols.replace(`<option>${esc(m.source ?? '')}</option>`,
                   `<option selected>${esc(m.source ?? '')}</option>`) +
      '</select>';
    h += `<tr><td class="mono">${esc(m.canonical)}</td><td>${sel}</td>` +
         `<td class="mono ${m.matched_by === 'fuzzy' ? 'warn' : ''}">` +
         `${esc(m.matched_by)}</td>` +
         `<td class="mono">${esc(m.transform)}</td>` +
         `<td class="mono" style="color:rgba(240,240,240,.5)">` +
         `${esc((m.sample_values || []).slice(0,2).join(', '))}</td></tr>`;
  }
  t.innerHTML = h;
}

document.getElementById('runbtn').onclick = async () => {
  const overrides = {};
  for (const s of document.querySelectorAll('select[data-canonical]')) {
    if (s.value) overrides[s.dataset.canonical] = s.value;
  }
  document.getElementById('spin').textContent = 'replaying history …';
  document.getElementById('runbtn').disabled = true;
  const r = await fetch('/api/run', { method:'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ upload_id: uploadId, overrides,
      workload_class: mapping.workload_class }) });
  const j = await r.json();
  document.getElementById('spin').textContent = '';
  document.getElementById('runbtn').disabled = false;
  if (!r.ok) { err.textContent = j.detail || 'run failed'; return; }
  showResult(j);
};

function showResult(j) {
  const tier = document.getElementById('tier');
  tier.textContent = j.readiness.tier;
  const col = { ready:'#7fae6a', degraded:'#b49b62',
                insufficient:'#9b3f3d' }[j.readiness.tier] || '#9b3f3d';
  tier.style.color = col; tier.style.borderColor = col;
  const o = (j.replay_summary || {}).outcome || {};
  document.getElementById('headline').textContent = j.replay_summary
    ? `aurelius (constraint_aware) vs ${j.replay_summary.headline_baseline}: `
      + `${o.constraint_aware_vs_headline} `
      + `(${(o.margin_pct >= 0 ? '+' : '') + o.margin_pct}% goodput/$) — `
      + 'directional only, not production savings'
    : (j.replay_skipped_reason || 'replay skipped');
  const ul = document.getElementById('checks'); ul.innerHTML = '';
  for (const c of j.readiness.checks) {
    const li = document.createElement('li');
    li.className = c.level === 'pass' ? 'ok' : c.level;
    li.textContent = `[${c.level.toUpperCase()}] ${c.message}`;
    ul.appendChild(li);
  }
  document.getElementById('rlink').href = `/api/runs/${j.run_id}/report`;
  document.getElementById('flink').href = `/api/runs/${j.run_id}/fingerprint`;
  document.getElementById('ressec').style.display = 'block';
}

function esc(s) { const d = document.createElement('div');
  d.textContent = String(s); return d.innerHTML; }
</script>
</div></body></html>
"""


def create_app(run_dir: Path | str = "aurelius_replay_runs"):
    """Build the FastAPI app (imported lazily so the CLI works without it)."""
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from pydantic import BaseModel

    run_root = Path(run_dir)
    run_root.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title=TOOL_NAME, version=__version__, docs_url=None,
                  redoc_url=None, openapi_url=None)

    class RunRequest(BaseModel):
        upload_id: str
        overrides: dict[str, str] = {}
        workload_class: Optional[str] = None
        fleet_gpus: Optional[int] = None
        gpus_per_node: int = 8
        gpu_type: Optional[str] = None

    def _upload_path(upload_id: str) -> Path:
        safe = "".join(ch for ch in upload_id if ch.isalnum())
        d = run_root / "uploads" / safe
        files = list(d.glob("*")) if d.is_dir() else []
        if not files:
            raise HTTPException(404, "unknown upload_id — upload again")
        return files[0]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _PAGE.replace("__TOOL__", TOOL_NAME)

    @app.get("/api/health")
    def health() -> dict:
        return {"tool": TOOL_NAME, "version": __version__, "local_only": True}

    @app.post("/api/inspect")
    async def api_inspect(file: UploadFile = File(...)):
        upload_id = uuid.uuid4().hex[:12]
        dest_dir = run_root / "uploads" / upload_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = Path(file.filename or "upload.csv").name
        dest = dest_dir / name
        with dest.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        try:
            table, plan = inspect_file(dest)
        except TableReadError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {
            "upload_id": upload_id,
            "columns": table.columns,
            "mapping": plan.to_dict(),
        }

    @app.post("/api/run")
    def api_run(req: RunRequest):
        src = _upload_path(req.upload_id)
        cls = req.workload_class
        if cls not in ("serving", "gpu_jobs"):
            cls = None
        artifacts = run_pipeline(
            src,
            workload_class=cls,
            overrides={k: v for k, v in req.overrides.items() if v},
            fleet_gpus=req.fleet_gpus,
            gpus_per_node=req.gpus_per_node,
            gpu_type=req.gpu_type,
        )
        if not artifacts.ok:
            raise HTTPException(422, artifacts.error or "could not analyze")
        out_dir = run_root / artifacts.run_id
        write_report(artifacts, out_dir / "report.html")
        write_fingerprint(artifacts.fingerprint,
                          out_dir / "schema_fingerprint.json")
        (out_dir / "summary.json").write_text(
            json.dumps(artifacts.to_dict(), indent=2, default=str))
        return JSONResponse(artifacts.to_dict())

    @app.get("/api/runs/{run_id}/report")
    def api_report(run_id: str):
        safe = "".join(ch for ch in run_id if ch.isalnum())
        path = run_root / safe / "report.html"
        if not path.exists():
            raise HTTPException(404, "no report for this run id")
        return FileResponse(path, media_type="text/html")

    @app.get("/api/runs/{run_id}/fingerprint")
    def api_fingerprint(run_id: str):
        safe = "".join(ch for ch in run_id if ch.isalnum())
        path = run_root / safe / "schema_fingerprint.json"
        if not path.exists():
            raise HTTPException(404, "no fingerprint for this run id")
        return FileResponse(path, media_type="application/json",
                            filename="aurelius_schema_fingerprint.json")

    return app
