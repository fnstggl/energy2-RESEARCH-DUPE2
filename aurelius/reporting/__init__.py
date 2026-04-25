"""Aurelius reporting module — savings reports and HTML output."""

from aurelius.reporting.html_report import render_html_report
from aurelius.reporting.savings_report import ConfidenceInterval, SavingsReport

__all__ = ["SavingsReport", "ConfidenceInterval", "render_html_report"]
