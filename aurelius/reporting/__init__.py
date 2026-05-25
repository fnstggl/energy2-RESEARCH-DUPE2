"""Aurelius reporting module — savings reports and HTML output."""

try:
    from aurelius.reporting.savings_report import ConfidenceInterval, SavingsReport
except ImportError:
    # pandas/numpy not installed — savings reports unavailable
    class SavingsReport:  # type: ignore[no-redef]
        pass
    class ConfidenceInterval:  # type: ignore[no-redef]
        pass

try:
    from aurelius.reporting.html_report import render_html_report
except ImportError:
    # matplotlib/jinja2 not installed — HTML reports unavailable
    def render_html_report(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("render_html_report requires matplotlib and jinja2")

__all__ = ["SavingsReport", "ConfidenceInterval", "render_html_report"]
