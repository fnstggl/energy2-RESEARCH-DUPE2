"""Self-contained HTML report for a self-serve replay run.

One file, no external assets, no scripts — it can be attached to an internal
email or opened offline, and it renders the same everywhere. Styling follows
the Aurelius site language (off-black, hairline borders, mono labels, single
gold accent).

Claim discipline: the report is generated text, so it is subject to the
``docs/RESULTS.md`` §8 claim rules. ``write_report`` scans the rendered HTML
for the forbidden claim substrings (outside negation context) and raises
before writing — the same contract the repo's reporting tests enforce.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from . import TOOL_NAME, __version__
from .pipeline import RunArtifacts

# docs/RESULTS.md §8 — forbidden outside an explicit negation context.
_FORBIDDEN_ABSOLUTE = (
    "guaranteed savings",
    "enterprise-ready autonomous optimization",
    "hyperscaler-validated economics",
    "production-proven",
)
_NEGATION_ONLY = ("production savings",)   # allowed only as "not production savings"

WORDING_BANNER = (
    "Historical replay on your own logs through the Aurelius simulator. "
    "Directional only — not production savings. "
    "Requires live telemetry calibration."
)


class ClaimRuleError(RuntimeError):
    """Rendered report contains wording the results standard forbids."""


def check_claim_rules(text: str) -> None:
    low = text.lower()
    for phrase in _FORBIDDEN_ABSOLUTE:
        if phrase in low:
            raise ClaimRuleError(f"forbidden phrase in report: {phrase!r}")
    for phrase in _NEGATION_ONLY:
        for m in re.finditer(re.escape(phrase), low):
            before = low[max(0, m.start() - 12):m.start()]
            if not before.rstrip().endswith("not"):
                raise ClaimRuleError(
                    f"phrase {phrase!r} used outside a negation context"
                )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _e(x: object) -> str:
    return html.escape(str(x))


def _num(x: object, digits: int = 2) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return _e(x)
    if v != v:  # NaN
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:,.{digits}f}"


def _pct(x: object) -> str:
    return "—" if x is None else f"{float(x):+,.1f}%"


def _tier_color(tier: str) -> str:
    return {"ready": "#7fae6a", "degraded": "#b49b62",
            "insufficient": "#9b3f3d"}.get(tier, "#9b3f3d")


_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #050505; color: #f0f0f0;
  font: 15px/1.6 "Helvetica Neue", Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased; }
.mono { font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo,
  Consolas, monospace; }
.wrap { max-width: 980px; margin: 0 auto; padding: 48px 24px 96px; }
header { border-bottom: 1px solid rgba(255,255,255,.11);
  padding-bottom: 28px; margin-bottom: 12px; }
.wordmark { font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 13px; letter-spacing: .35em; color: #b49b62; }
h1 { font-size: 30px; font-weight: 500; letter-spacing: -.01em;
  margin-top: 14px; }
.meta { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px;
  color: rgba(240,240,240,.55); margin-top: 10px; }
.banner { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px;
  color: rgba(240,240,240,.7); border: 1px solid rgba(255,255,255,.11);
  padding: 10px 14px; margin: 22px 0 8px; }
section { margin-top: 56px; }
.eyebrow { font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 12px; letter-spacing: .3em; color: rgba(240,240,240,.5);
  border-top: 1px solid rgba(255,255,255,.11); padding-top: 14px; }
.eyebrow b { color: #b49b62; font-weight: 400; }
h2 { font-size: 21px; font-weight: 500; margin: 10px 0 18px; }
p.note { color: rgba(240,240,240,.75); max-width: 72ch; margin: 10px 0; }
table { border-collapse: collapse; width: 100%; margin: 14px 0;
  font-size: 13.5px; }
th { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px;
  letter-spacing: .12em; text-transform: uppercase;
  color: rgba(240,240,240,.55); text-align: left; font-weight: 400;
  border-bottom: 1px solid rgba(255,255,255,.22); padding: 8px 10px; }
td { border-bottom: 1px solid rgba(255,255,255,.08); padding: 8px 10px;
  vertical-align: top; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.hl td { background: rgba(180,155,98,.08); }
.badge { display: inline-block; font-family: "IBM Plex Mono", monospace;
  font-size: 11px; letter-spacing: .18em; text-transform: uppercase;
  padding: 4px 10px; border: 1px solid; margin-left: 10px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,
  1fr)); gap: 1px; background: rgba(255,255,255,.11);
  border: 1px solid rgba(255,255,255,.11); margin: 16px 0; }
.card { background: #050505; padding: 16px 14px; }
.card .k { font-family: "IBM Plex Mono", monospace; font-size: 11px;
  letter-spacing: .14em; text-transform: uppercase;
  color: rgba(240,240,240,.5); }
.card .v { font-size: 22px; margin-top: 6px; font-variant-numeric:
  tabular-nums; }
.card .s { font-size: 12px; color: rgba(240,240,240,.5); margin-top: 2px; }
.callout { border-left: 2px solid #b49b62; padding: 12px 18px; margin: 18px 0;
  color: rgba(240,240,240,.85); }
.refusal { border-left: 2px solid #9b3f3d; padding: 12px 18px; margin: 18px 0; }
ul { margin: 10px 0 10px 20px; color: rgba(240,240,240,.8); }
li { margin: 4px 0; }
.ok { color: #7fae6a; } .warn { color: #b49b62; } .fail { color: #9b3f3d; }
footer { margin-top: 72px; border-top: 1px solid rgba(255,255,255,.11);
  padding-top: 18px; font-family: "IBM Plex Mono", monospace; font-size: 11.5px;
  color: rgba(240,240,240,.5); line-height: 1.8; }
"""


def _cards(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div class="card"><div class="k">{_e(k)}</div>'
        f'<div class="v">{v}</div><div class="s">{_e(s)}</div></div>'
        for k, v, s in items
    )
    return f'<div class="cards">{cells}</div>'


def _derived_fields(plan) -> dict[str, str]:
    """Canonical fields that have no direct column but are derived per-row."""
    mapped = {m.canonical for m in plan.mappings if m.source is not None}
    derived: dict[str, str] = {}
    if "gres" in mapped and "gpu_count" not in mapped:
        derived["gpu_count"] = "parsed from the TRES/GRES string"
    if "gres" in mapped and "gpu_type" not in mapped:
        derived["gpu_type"] = "parsed from the TRES/GRES string (when typed)"
    if {"start_time", "end_time"} <= mapped and "duration" not in mapped:
        derived["duration"] = "end_time − start_time"
    if {"submit_time", "start_time"} <= mapped and "queue_wait" not in mapped:
        derived["queue_wait"] = "start_time − submit_time"
    return derived


def _mapping_section(a: RunArtifacts) -> str:
    plan = a.mapping
    derived = _derived_fields(plan)
    rows = []
    for m in plan.mappings:
        if m.source is None and m.matched_by == "absent":
            if m.canonical in derived:
                rows.append(
                    f"<tr><td class='mono'>{_e(m.canonical)}</td>"
                    f"<td class='mono' style='color:rgba(240,240,240,.55)'>"
                    f"(derived)</td><td>derived</td>"
                    f"<td class='mono'>{_e(derived[m.canonical])}</td>"
                    f"<td></td></tr>"
                )
            continue
        badge = {"exact_synonym": "auto", "fuzzy": "fuzzy — confirm",
                 "user": "user", "derived": "derived"}.get(m.matched_by,
                                                           m.matched_by)
        samples = ", ".join(_e(s[:28]) for s in m.sample_values[:3])
        rows.append(
            f"<tr><td class='mono'>{_e(m.canonical)}</td>"
            f"<td class='mono'>{_e(m.source)}</td>"
            f"<td>{_e(badge)}</td><td class='mono'>{_e(m.transform)}</td>"
            f"<td class='mono' style='color:rgba(240,240,240,.55)'>{samples}"
            f"</td></tr>"
        )
    absent = [m.canonical for m in plan.mappings
              if m.source is None and m.matched_by == "absent"
              and m.canonical not in derived]
    unmapped = ", ".join(_e(c) for c in plan.unmapped_source_columns[:20])
    absent_txt = ", ".join(_e(x) for x in absent)
    warn_html = "".join(
        f"<li class='warn'>{_e(w)}</li>" for w in plan.warnings
    )
    return f"""
<section>
  <div class="eyebrow"><b>01</b>&nbsp;&nbsp;WHAT WE READ</div>
  <h2>Schema mapping — {_e(a.source_file)}</h2>
  <p class="note">Workload class: <span class="mono">{_e(plan.workload_class)}
  </span> ({_e(plan.class_evidence)}).</p>
  <table>
    <tr><th>canonical field</th><th>your column</th><th>match</th>
        <th>transform</th><th>sample values</th></tr>
    {''.join(rows)}
  </table>
  {f'<p class="note">Not present (fine if you do not have them): <span class="mono">{absent_txt}</span></p>' if absent else ''}
  {f'<p class="note">Ignored columns: <span class="mono">{unmapped}</span></p>' if unmapped else ''}
  {f'<ul>{warn_html}</ul>' if warn_html else ''}
</section>"""


def _readiness_section(a: RunArtifacts) -> str:
    r = a.readiness
    checks = "".join(
        f"<li class='{c.level if c.level != 'pass' else 'ok'}'>"
        f"<span class='mono'>[{_e(c.level.upper())}]</span> {_e(c.message)}</li>"
        for c in r.checks
    )
    levers = "".join(
        "<tr><td class='mono'>{}</td><td class='{}'>{}</td><td>{}</td>"
        "<td>{}</td></tr>".format(
            _e(lv.name), "ok" if lv.available else "warn",
            "ON" if lv.available else "OFF", _e(lv.reason),
            _e(lv.unlock) if not lv.available else "",
        )
        for lv in r.levers
    )
    return f"""
<section>
  <div class="eyebrow"><b>02</b>&nbsp;&nbsp;READINESS</div>
  <h2>Can this log support a replay?
    <span class="badge" style="color:{_tier_color(r.tier)};
    border-color:{_tier_color(r.tier)}">{_e(r.tier)}</span></h2>
  <ul>{checks}</ul>
  {f"<table><tr><th>lever</th><th>state</th><th>why</th><th>to unlock</th></tr>{levers}</table>" if levers else ""}
</section>"""


def _descriptive_section(a: RunArtifacts) -> str:
    d = a.descriptive
    if not d or d.get("empty"):
        return ""
    if d.get("class") == "serving":
        arr = d.get("arrival", {})
        tok = d.get("tokens", {})
        cards = _cards([
            ("requests", _num(d.get("requests"), 0), f"{d.get('span_hours')} h span"),
            ("mean load", _num(arr.get("mean_rpm"), 1), "requests / minute"),
            ("peak load", _num(arr.get("peak_rpm"), 0),
             f"{_num(arr.get('burstiness_peak_over_mean'), 1)}× the mean"),
            ("idle minutes", f"{_num(arr.get('idle_share_pct'), 1)}%",
             "minutes with zero arrivals"),
            ("failure rate", f"{_num(d.get('failure_rate_pct'), 2)}%",
             "from your status column"),
        ])
        model_rows = "".join(
            f"<tr><td class='mono'>{_e(m['value'])}</td>"
            f"<td class='num'>{_num(m['count'], 0)}</td>"
            f"<td class='num'>{_num(m['share_pct'], 1)}%</td></tr>"
            for m in d.get("model_mix", [])
        )
        note = d.get("burst_note")
        return f"""
<section>
  <div class="eyebrow"><b>03</b>&nbsp;&nbsp;YOUR CLUSTER, AS LOGGED</div>
  <h2>Measured from your own log — no simulation</h2>
  {cards}
  <table>
    <tr><th></th><th class="num">p50</th><th class="num">p95</th>
        <th class="num">p99</th><th class="num">max</th></tr>
    <tr><td>prompt tokens</td><td class="num">{_num(tok.get('prompt', {}).get('p50'), 0)}</td>
        <td class="num">{_num(tok.get('prompt', {}).get('p95'), 0)}</td>
        <td class="num">{_num(tok.get('prompt', {}).get('p99'), 0)}</td>
        <td class="num">{_num(tok.get('prompt', {}).get('max'), 0)}</td></tr>
    <tr><td>output tokens</td><td class="num">{_num(tok.get('output', {}).get('p50'), 0)}</td>
        <td class="num">{_num(tok.get('output', {}).get('p95'), 0)}</td>
        <td class="num">{_num(tok.get('output', {}).get('p99'), 0)}</td>
        <td class="num">{_num(tok.get('output', {}).get('max'), 0)}</td></tr>
  </table>
  {f"<table><tr><th>model</th><th class='num'>requests</th><th class='num'>share</th></tr>{model_rows}</table>" if model_rows else ""}
  {f'<div class="callout">{_e(note)}</div>' if note else ''}
</section>"""

    # gpu_jobs
    cards = _cards([
        ("jobs", _num(d.get("jobs"), 0), f"{d.get('span_days')} days of submissions"),
        ("gpu-hours demanded", _num(d.get("total_gpu_hours_demanded"), 0),
         "sum of gpus × runtime"),
        ("peak concurrent demand", _num(d.get("peak_concurrent_gpu_demand"), 0),
         "GPUs, from the log's own timeline"),
        ("queue wait p95",
         _num((d.get("queue_wait_hours") or {}).get("p95"), 1),
         "hours (measured)" if d.get("queue_wait_hours") else "not in log"),
        ("failure rate", f"{_num(d.get('failure_rate_pct'), 1)}%",
         "from your status column"),
    ])
    size_mix = d.get("gpu_request", {}).get("size_class_mix", {})
    size_rows = "".join(
        f"<tr><td class='mono'>{_e(k)} GPUs</td>"
        f"<td class='num'>{_num(v, 0)}</td></tr>"
        for k, v in size_mix.items()
    )
    wait_note = d.get("wait_note")
    return f"""
<section>
  <div class="eyebrow"><b>03</b>&nbsp;&nbsp;YOUR CLUSTER, AS LOGGED</div>
  <h2>Measured from your own log — no simulation</h2>
  {cards}
  <table>
    <tr><th></th><th class="num">p50</th><th class="num">p95</th>
        <th class="num">p99</th><th class="num">max</th></tr>
    <tr><td>job duration (hours)</td>
        <td class="num">{_num((d.get('duration_hours') or {}).get('p50'))}</td>
        <td class="num">{_num((d.get('duration_hours') or {}).get('p95'))}</td>
        <td class="num">{_num((d.get('duration_hours') or {}).get('p99'))}</td>
        <td class="num">{_num((d.get('duration_hours') or {}).get('max'))}</td></tr>
  </table>
  {f"<table><tr><th>job size</th><th class='num'>jobs</th></tr>{size_rows}</table>" if size_rows else ""}
  {f'<div class="callout">{_e(wait_note)}</div>' if wait_note else ''}
</section>"""


_POLICY_LABELS = {
    "constraint_aware": "aurelius (constraint_aware)",
    "fifo": "fifo (sanity baseline)",
}

# Policy families, mirroring the benchmark docs: baselines are what an
# operator runs today; variants are Aurelius-side alternates shown for
# completeness and never used as the headline.
_BASELINE_POLICIES = (
    "fifo", "sla_aware", "queue_aware", "cache_affinity_baseline",
    "first_fit", "best_fit", "first_fit_decreasing", "greedy_packing",
    "topology_aware", "utilization_aware",
)
_SERVING_TIMEOUT_GATE_PCT = 10.0   # docs/RESULTS.md SLA gate


def _policy_safety(p: dict, serving: bool) -> tuple[bool, str]:
    """SAFE/UNSAFE per the documented gates (serving: timeout ≤ 10%;
    jobs: no starvation events)."""
    if serving:
        t = p.get("timeout_rate_pct_mean")
        if t is None:
            return True, "—"
        ok = float(t) <= _SERVING_TIMEOUT_GATE_PCT
        return ok, ("SAFE" if ok else f"UNSAFE — timeout {float(t):.1f}%")
    starve = p.get("starvation_events") or 0
    ok = starve == 0
    return ok, ("SAFE" if ok else f"starvation ×{int(starve)}")


def _replay_section(a: RunArtifacts) -> str:
    if a.replay_summary is None:
        reason = a.replay_skipped_reason or "replay was not run"
        return f"""
<section>
  <div class="eyebrow"><b>04</b>&nbsp;&nbsp;COUNTERFACTUAL REPLAY</div>
  <h2>Replay refused — by design</h2>
  <div class="refusal"><p class="note">{_e(reason)}</p>
  <p class="note">A number computed on data that cannot support it would be
  worse than no number. Section 02 lists exactly what to add; the schema
  fingerprint (section 05) lets us help without seeing your logs.</p></div>
</section>"""

    s = a.replay_summary
    pol = s.get("policies", {})
    outcome = s.get("outcome", {})
    headline = s.get("headline_baseline", "")
    margin = outcome.get("margin_pct")
    fifo_margin = outcome.get("fifo_margin_pct")
    verdict = outcome.get("constraint_aware_vs_headline", "")

    serving = "n_requests" in s
    base = pol.get(headline, {})
    ca = pol.get("constraint_aware", {})
    kpi_key = "sla_safe_goodput_per_infra_dollar"
    hours_key = "active_gpu_hours" if serving else "provisioned_gpu_hours"
    gpu_delta = None
    if base.get(hours_key) and ca.get(hours_key):
        gpu_delta = 100.0 * (ca[hours_key] - base[hours_key]) / base[hours_key]

    if serving:
        header = ("<tr><th>policy</th><th>family</th>"
                  "<th class='num'>goodput / $</th>"
                  "<th class='num'>Δ vs baseline</th>"
                  "<th class='num'>gpu-hours</th><th class='num'>p99 ms</th>"
                  "<th>safety</th></tr>")
    else:
        header = ("<tr><th>policy</th><th>family</th>"
                  "<th class='num'>goodput / $</th>"
                  "<th class='num'>Δ vs baseline</th>"
                  "<th class='num'>gpu-hours</th>"
                  "<th class='num'>queue p95 h</th>"
                  "<th>safety</th></tr>")

    base_kpi = base.get(kpi_key) or 0
    ordered = sorted(
        pol.items(),
        key=lambda kv: (kv[0] not in _BASELINE_POLICIES, kv[0] != "fifo",
                        kv[0] != headline, kv[0] != "constraint_aware"),
    )
    rows = []
    for name, p in ordered:
        delta = (100.0 * (p.get(kpi_key, 0) - base_kpi) / base_kpi
                 if base_kpi else None)
        cls = ' class="hl"' if name == "constraint_aware" else ""
        label = _POLICY_LABELS.get(name, name)
        family = ("baseline" if name in _BASELINE_POLICIES
                  else "aurelius variant" if name != "constraint_aware"
                  else "aurelius")
        safe, safety_txt = _policy_safety(p, serving)
        safety_cell = (f"<td class='{'ok' if safe else 'fail'} mono'>"
                       f"{_e(safety_txt)}</td>")
        if serving:
            extra = (f"<td class='num'>{_num(p.get('active_gpu_hours'))}</td>"
                     f"<td class='num'>{_num(p.get('latency_p99_ms'), 0)}</td>"
                     f"{safety_cell}")
        else:
            extra = (f"<td class='num'>{_num(p.get('provisioned_gpu_hours'))}"
                     f"</td>"
                     f"<td class='num'>"
                     f"{_num((p.get('queue_wait_s_p95') or 0) / 3600.0, 1)}</td>"
                     f"{safety_cell}")
        rows.append(
            f"<tr{cls}><td class='mono'>{_e(label)}</td>"
            f"<td class='mono' style='color:rgba(240,240,240,.55)'>"
            f"{_e(family)}</td>"
            f"<td class='num'>{_num(p.get(kpi_key))}</td>"
            f"<td class='num'>{_pct(delta) if name != headline else '·'}</td>"
            f"{extra}</tr>"
        )

    if verdict in ("ALPHA_WIN", "WIN"):
        head_txt = (
            f"Replaying your history, the Aurelius policy scored "
            f"<b>{_pct(margin)}</b> on SLA-safe goodput per infrastructure "
            f"dollar vs <span class='mono'>{_e(headline)}</span> — the "
            "strongest realistic safe baseline in the comparison set"
        )
    elif verdict in ("TIE", "SAFE_TIE"):
        head_txt = (
            "On this log the Aurelius policy TIED the strongest safe "
            f"baseline (<span class='mono'>{_e(headline)}</span>) — an honest "
            "result: your history sits close to its safe frontier on the "
            "levers this log exposes"
        )
    else:
        head_txt = (
            f"On this log the Aurelius policy scored {_pct(margin)} vs "
            f"<span class='mono'>{_e(headline)}</span> — reported as-is"
        )

    cards = [("vs strongest safe baseline", _pct(margin), _e(headline))]
    if gpu_delta is not None:
        cards.append(("gpu-hours", _pct(gpu_delta), "aurelius vs baseline"))
    if fifo_margin is not None:
        cards.append(("vs naive fifo", _pct(fifo_margin), "sanity check only"))
    cards.append(("scale", _num(s.get("n_requests") or s.get("n_jobs"), 0),
                  "requests replayed" if serving else "jobs replayed"))

    return f"""
<section>
  <div class="eyebrow"><b>04</b>&nbsp;&nbsp;COUNTERFACTUAL REPLAY</div>
  <h2>Same arrivals, same work — only the scheduling decisions change</h2>
  <p class="note">{head_txt}.</p>
  {_cards(cards)}
  <table>{header}{''.join(rows)}</table>
  <p class="note">Every policy sees identical arrivals, identical work and an
  identical cost basis; the physics and the KPI are the locked Aurelius
  evaluation stack used for the public-trace benchmarks. The FIFO row is a
  sanity floor, not a comparison we would quote. The headline number is
  always <span class="mono">aurelius (constraint_aware)</span> vs the
  strongest realistic SAFE baseline — "aurelius variant" rows are alternate
  operating points shown for completeness (some trade SLA safety margin for
  throughput) and are never the quoted result.</p>
</section>"""


def _assumptions_section(a: RunArtifacts) -> str:
    items = []
    if a.fleet_assumption:
        f = a.fleet_assumption
        items.append(
            f"Fleet: {f['nodes']} nodes × {f['gpus_per_node']} GPUs "
            f"({f['gpu_model']}) — {f['note']}."
        )
    items.append(
        "Cost basis: public default GPU-hour rates from the Aurelius "
        "economics module — a pilot substitutes your actual procurement "
        "rates before any savings figure is quoted."
    )
    if a.readiness:
        for lv in a.readiness.levers:
            if not lv.available and lv.unlock:
                items.append(f"Lever off — {lv.name}: {lv.reason}. "
                             f"Unlock: {lv.unlock}.")
    lis = "".join(f"<li>{_e(i)}</li>" for i in items)
    return f"""
<section>
  <div class="eyebrow"><b>05</b>&nbsp;&nbsp;ASSUMPTIONS, OPENLY</div>
  <h2>What this replay assumes — and how to tighten it</h2>
  <ul>{lis}</ul>
</section>"""


def _next_section(a: RunArtifacts) -> str:
    return """
<section>
  <div class="eyebrow"><b>06</b>&nbsp;&nbsp;NEXT</div>
  <h2>If the direction is interesting</h2>
  <p class="note">1 — A <span class="mono">schema fingerprint</span> was
  written next to this report (column names, shapes and parse diagnostics —
  no log rows, no identifier values). If anything above is mismapped or a
  lever is off, send it to us and we ship a mapping for your exact export
  format.</p>
  <p class="note">2 — The credible number comes from a shadow pilot: Aurelius
  records the decisions it would make against your live telemetry,
  recommendation-only, and is graded against what actually happened. This
  replay is the reason to start one; the shadow result is the figure a
  decision is made on.</p>
</section>"""


def render_html(a: RunArtifacts) -> str:
    from datetime import datetime, timezone

    when = datetime.fromtimestamp(a.created_unix, tz=timezone.utc)
    if not a.ok:
        body = f"""
<section><div class="eyebrow"><b>01</b>&nbsp;&nbsp;COULD NOT READ FILE</div>
<h2>Nothing was analyzed</h2>
<div class="refusal"><p class="note">{_e(a.error)}</p>
<p class="note">Supported inputs: CSV / TSV / JSON-lines (optionally
gzipped) with a header row. Nothing was transmitted anywhere.</p></div>
</section>"""
    else:
        body = (
            _mapping_section(a)
            + _readiness_section(a)
            + _descriptive_section(a)
            + _replay_section(a)
            + _assumptions_section(a)
            + _next_section(a)
        )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(TOOL_NAME)} — {_e(a.source_file)}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<header>
  <div class="wordmark">AURELIUS&nbsp;REPLAY</div>
  <h1>Counterfactual scheduling report</h1>
  <div class="meta">run {_e(a.run_id)} · {_e(a.source_file)} ·
  {when.strftime('%Y-%m-%d %H:%M UTC')} · v{_e(__version__)} ·
  generated locally — no data left this machine</div>
  <div class="banner">{_e(WORDING_BANNER)}</div>
</header>
{body}
<footer>
  {_e(TOOL_NAME)} v{_e(__version__)} — read-only historical replay.
  Simulator benchmark result on your own logs. Directional only — not
  production savings. Requires live telemetry calibration. No production
  system was touched; no data was transmitted. Methodology:
  docs/RESULTS.md (canonical KPI: SLA-safe goodput per infrastructure
  dollar; headline vs the strongest realistic safe baseline, never vs FIFO).
</footer>
</div></body></html>"""


def write_report(a: RunArtifacts, out_path: str | Path) -> Path:
    html_text = render_html(a)
    check_claim_rules(html_text)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text)
    return out
