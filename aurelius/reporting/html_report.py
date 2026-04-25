"""HTML report generator for Aurelius savings reports.

Produces self-contained HTML with embedded base64 matplotlib charts.
Requires jinja2 and matplotlib (both in dev dependencies).
"""

from __future__ import annotations

import base64
import io
import logging

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

try:
    from jinja2 import BaseLoader, Environment
    _JINJA2_AVAILABLE = True
except ImportError:
    _JINJA2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Jinja2 HTML template (self-contained, no external resources)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Aurelius Savings Report</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           margin: 0; padding: 20px; background: #f9fafb; color: #111827; }
    .container { max-width: 1100px; margin: 0 auto; }
    h1 { color: #1f2937; font-size: 2em; }
    h2 { color: #374151; font-size: 1.4em; border-bottom: 2px solid #e5e7eb;
         padding-bottom: 6px; margin-top: 32px; }
    h3 { color: #4b5563; font-size: 1.05em; }
    .hero { display: flex; gap: 16px; margin: 20px 0; flex-wrap: wrap; }
    .card { background: white; border-radius: 8px; padding: 20px; flex: 1;
            min-width: 160px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .card .value { font-size: 2.1em; font-weight: 700; color: #059669; }
    .card .label { font-size: 0.9em; color: #6b7280; margin-top: 4px; }
    .card .ci { font-size: 0.75em; color: #9ca3af; margin-top: 2px; }
    .card.warn .value { color: #dc2626; }
    table { border-collapse: collapse; width: 100%; background: white;
            border-radius: 8px; overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-top: 10px; }
    th { background: #f3f4f6; padding: 10px 14px; text-align: left;
         font-size: 0.85em; color: #374151; }
    td { padding: 10px 14px; font-size: 0.9em; border-top: 1px solid #f3f4f6; }
    tr:hover td { background: #fafafa; }
    .chart { background: white; border-radius: 8px; padding: 16px;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 16px 0; }
    .chart img { width: 100%; max-width: 900px; display: block; margin: 0 auto; }
    .methodology { background: white; border-radius: 8px; padding: 20px;
                   box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 16px 0; }
    .methodology p { font-size: 0.9em; color: #4b5563; line-height: 1.6; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
             font-size: 0.8em; background: #d1fae5; color: #065f46; }
    footer { text-align: center; color: #9ca3af; font-size: 0.8em; margin-top: 40px; }
  </style>
</head>
<body>
<div class="container">
  <h1>Aurelius Savings Report</h1>
  <p>
    Generated: <strong>{{ report.generated_at }}</strong> &nbsp;|&nbsp;
    Folds: <strong>{{ report.n_folds }}</strong> &nbsp;|&nbsp;
    Baseline: <strong>{{ report.primary_baseline }}</strong>
    <span class="badge">leakage-free</span>
  </p>

  <h2>Summary</h2>
  <div class="hero">
    <div class="card">
      <div class="value">{{ "%.1f"|format(totals.cost_savings_pct|default(0)) }}%</div>
      <div class="label">Cost Savings vs Baseline</div>
      <div class="ci">
        95% CI: {{ "%.1f"|format(ci_cost_pct.lower_95) }}% –
        {{ "%.1f"|format(ci_cost_pct.upper_95) }}%
      </div>
    </div>
    <div class="card">
      <div class="value">${{ "%.0f"|format(totals.cost_savings_usd|default(0)) }}</div>
      <div class="label">Total Cost Savings (USD)</div>
      <div class="ci">
        95% CI: ${{ "%.0f"|format(ci_cost_usd.lower_95) }} –
        ${{ "%.0f"|format(ci_cost_usd.upper_95) }}
      </div>
    </div>
    <div class="card">
      <div class="value">{{ "%.1f"|format(totals.carbon_reduction_pct|default(0)) }}%</div>
      <div class="label">Carbon Reduction</div>
      <div class="ci">
        {{ "%.4f"|format(totals.carbon_reduction_tonnes|default(0)) }} tonnes CO&#8322;
      </div>
    </div>
    <div class="card{% if (totals.latency_violation_rate_pct|default(0)) > 2 %} warn{% endif %}">
      <div class="value">{{ totals.latency_violations|default(0) }}</div>
      <div class="label">
        SLA Violations ({{ "%.1f"|format(totals.latency_violation_rate_pct|default(0)) }}%)
      </div>
    </div>
    <div class="card">
      <div class="value">{{ "%.1f"|format(totals.utilization_pct|default(0)) }}%</div>
      <div class="label">Job Utilization</div>
      <div class="ci">Avg delay: {{ "%.1f"|format(totals.avg_queue_delay_hours|default(0)) }}h</div>
    </div>
  </div>

  {% if chart_savings %}
  <h2>Cost: Aurelius vs Baseline per Fold</h2>
  <div class="chart">
    <img src="data:image/png;base64,{{ chart_savings }}" alt="Cost savings per fold">
  </div>
  {% endif %}

  {% if chart_baselines %}
  <h2>Savings vs All Baselines (95% CI)</h2>
  <div class="chart">
    <img src="data:image/png;base64,{{ chart_baselines }}" alt="Savings vs baselines">
  </div>
  {% endif %}

  <h2>Per-Fold Results</h2>
  <table>
    <tr>
      <th>Fold</th><th>Eval Period</th><th>Jobs</th>
      <th>Optimizer $</th><th>Baseline $</th>
      <th>Savings $</th><th>Savings %</th>
      <th>Carbon &#916; gCO&#8322;</th><th>SLA Violations</th>
    </tr>
    {% for f in report.fold_results %}
    <tr>
      <td>{{ f.fold_index }}</td>
      <td>{{ f.eval_start[:10] }} – {{ f.eval_end[:10] }}</td>
      <td>{{ f.eval_jobs }}</td>
      <td>${{ "%.2f"|format(f.optimizer_cost_usd) }}</td>
      <td>${{ "%.2f"|format(f.baseline_cost_usd) }}</td>
      <td>${{ "%.2f"|format(f.cost_savings_usd) }}</td>
      <td>{{ "%.1f"|format(f.cost_savings_pct) }}%</td>
      <td>{{ "%.0f"|format(f.carbon_reduction_gco2) }}</td>
      <td>{{ f.latency_violations }}</td>
    </tr>
    {% endfor %}
  </table>

  <h2>Baseline Comparison</h2>
  <table>
    <tr>
      <th>Baseline</th><th>Savings % (est.)</th>
      <th>Lower 95% CI</th><th>Upper 95% CI</th><th>Folds</th>
    </tr>
    {% for name, data in report.baseline_comparison.items() %}
    <tr>
      <td>{{ name }}</td>
      <td>{{ "%.1f"|format(data.cost_savings_pct.estimate) }}%</td>
      <td>{{ "%.1f"|format(data.cost_savings_pct.lower_95) }}%</td>
      <td>{{ "%.1f"|format(data.cost_savings_pct.upper_95) }}%</td>
      <td>{{ data.n_folds }}</td>
    </tr>
    {% endfor %}
  </table>

  <h2>Methodology</h2>
  <div class="methodology">
    {% for key, text in report.methodology.items() %}
    <h3>{{ key | replace("_", " ") | title }}</h3>
    <p>{{ text }}</p>
    {% endfor %}
  </div>

  <footer>
    Aurelius Energy Optimization &mdash; report generated {{ report.generated_at }}
  </footer>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------

def _chart_savings_by_fold(fold_results: list[dict]) -> str:
    """Render a grouped bar chart of optimizer vs baseline cost per fold.

    Returns a base64-encoded PNG string, or "" if no data.
    """
    if not fold_results:
        return ""

    opt_costs = [f["optimizer_cost_usd"] for f in fold_results]
    bl_costs = [f["baseline_cost_usd"] for f in fold_results]
    x = list(range(len(fold_results)))
    fold_labels = [str(f["fold_index"]) for f in fold_results]

    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(x) * 0.9 + 2), 4))
    ax.bar([xi - width / 2 for xi in x], bl_costs, width,
           label="Baseline", color="#6b7280", alpha=0.85)
    ax.bar([xi + width / 2 for xi in x], opt_costs, width,
           label="Aurelius", color="#059669", alpha=0.85)
    ax.set_xlabel("Fold Index")
    ax.set_ylabel("Energy Cost (USD)")
    ax.set_title("Cost Comparison: Aurelius vs Baseline per Fold")
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _chart_baseline_comparison(baseline_comparison: dict) -> str:
    """Render a horizontal bar chart of savings % vs each baseline with 95% CI.

    Returns a base64-encoded PNG string, or "" if no data.
    """
    if not baseline_comparison:
        return ""

    names = list(baseline_comparison.keys())
    estimates = [baseline_comparison[n]["cost_savings_pct"]["estimate"] for n in names]
    lowers = [baseline_comparison[n]["cost_savings_pct"]["lower_95"] for n in names]
    uppers = [baseline_comparison[n]["cost_savings_pct"]["upper_95"] for n in names]

    xerr_low = [max(0.0, est - lo) for est, lo in zip(estimates, lowers)]
    xerr_high = [max(0.0, hi - est) for est, hi in zip(estimates, uppers)]

    height = max(3.0, len(names) * 0.7)
    fig, ax = plt.subplots(figsize=(9, height))
    colors = ["#059669" if e >= 0 else "#dc2626" for e in estimates]
    ax.barh(
        range(len(names)), estimates,
        xerr=[xerr_low, xerr_high],
        color=colors, alpha=0.85,
        capsize=4, error_kw={"linewidth": 1.5},
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Cost Savings % (positive = Aurelius cheaper than baseline)")
    ax.set_title("Aurelius Savings vs All Baselines (95% CI)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class _CIProxy:
    """Attribute-access wrapper for CI dict values (for Jinja2 templates)."""

    def __init__(self, d: dict) -> None:
        self.estimate: float = float(d.get("estimate") or 0.0)
        self.lower_95: float = float(d.get("lower_95") or 0.0)
        self.upper_95: float = float(d.get("upper_95") or 0.0)


def render_html_report(savings_report: dict) -> str:
    """Render a self-contained HTML savings report.

    Args:
        savings_report: Output of ``SavingsReport.generate()``.

    Returns:
        A UTF-8 HTML string with embedded base64 charts and no external
        dependencies — suitable for saving to a file or returning from an API.

    Raises:
        ImportError: If jinja2 is not installed.
    """
    if not _JINJA2_AVAILABLE:
        raise ImportError(
            "jinja2 is required for HTML reporting. "
            "Install it with: pip install jinja2"
        )

    totals = savings_report.get("totals", {})
    ci = savings_report.get("confidence_intervals", {})
    fold_results = savings_report.get("fold_results", [])

    chart_savings = _chart_savings_by_fold(fold_results)
    chart_baselines = _chart_baseline_comparison(
        savings_report.get("baseline_comparison", {})
    )

    # autoescape=True: baseline names and other dict keys come from
    # internal BacktestRound data, but we treat all strings as untrusted
    # to guard against unexpected content injection in the HTML output.
    env = Environment(loader=BaseLoader(), autoescape=True)
    template = env.from_string(_HTML_TEMPLATE)

    html = template.render(
        report=savings_report,
        totals=totals,
        ci_cost_pct=_CIProxy(ci.get("cost_savings_pct_per_fold", {})),
        ci_cost_usd=_CIProxy(ci.get("cost_savings_usd_per_fold", {})),
        chart_savings=chart_savings,
        chart_baselines=chart_baselines,
    )
    return html
