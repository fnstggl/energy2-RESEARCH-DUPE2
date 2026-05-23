#!/usr/bin/env python3
"""Render the Aurelius enterprise briefing documents to PDF.

Produces four executive-shareable PDFs with title pages, restrained
formatting, clean tables, diagram placeholders and page-aware structure.
Pure reportlab (no system dependencies).
"""

from datetime import date
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Flowable, KeepTogether,
)
from reportlab.platypus.doctemplate import NextPageTemplate

OUT_DIR = "enterprisedocs/pdf"

# ---------------------------------------------------------------- palette ----
INK     = HexColor("#1A2433")   # near-black slate, body text
ACCENT  = HexColor("#1F3A5F")   # deep navy slate, headings
MUTED   = HexColor("#5B6675")   # secondary text
HAIR    = HexColor("#C7CFDA")   # hairlines / borders
PANEL   = HexColor("#F4F6F9")   # table header / panel fill
RULE    = HexColor("#1F3A5F")   # title-page rule

PAGE_W, PAGE_H = LETTER
LM = RM = 0.95 * inch
TM = 0.95 * inch
BM = 0.95 * inch
USABLE_W = PAGE_W - LM - RM

DOC_DATE = date(2026, 5, 23).strftime("%B %Y")
CONFIDENTIAL = "Confidential — for evaluation purposes"

# ---------------------------------------------------------------- styles -----
def _styles():
    s = getSampleStyleSheet()
    styles = {}
    styles["Title"] = ParagraphStyle(
        "TitleX", parent=s["Normal"], fontName="Helvetica-Bold",
        fontSize=27, leading=32, textColor=ACCENT, spaceAfter=0)
    styles["Subtitle"] = ParagraphStyle(
        "SubtitleX", parent=s["Normal"], fontName="Helvetica",
        fontSize=13, leading=18, textColor=MUTED, spaceBefore=10)
    styles["DocType"] = ParagraphStyle(
        "DocType", parent=s["Normal"], fontName="Helvetica-Bold",
        fontSize=10.5, leading=14, textColor=ACCENT, spaceAfter=4)
    styles["Meta"] = ParagraphStyle(
        "Meta", parent=s["Normal"], fontName="Helvetica",
        fontSize=9.5, leading=14, textColor=MUTED)
    styles["H1"] = ParagraphStyle(
        "H1X", parent=s["Normal"], fontName="Helvetica-Bold",
        fontSize=15, leading=19, textColor=ACCENT,
        spaceBefore=18, spaceAfter=7, keepWithNext=True)
    styles["H2"] = ParagraphStyle(
        "H2X", parent=s["Normal"], fontName="Helvetica-Bold",
        fontSize=11.5, leading=15, textColor=INK,
        spaceBefore=12, spaceAfter=4, keepWithNext=True)
    styles["Body"] = ParagraphStyle(
        "BodyX", parent=s["Normal"], fontName="Helvetica",
        fontSize=10, leading=15, textColor=INK,
        spaceAfter=8, alignment=TA_LEFT)
    styles["Lead"] = ParagraphStyle(
        "LeadX", parent=s["Normal"], fontName="Helvetica",
        fontSize=11.5, leading=17, textColor=INK, spaceAfter=10)
    styles["Bullet"] = ParagraphStyle(
        "BulletX", parent=s["Normal"], fontName="Helvetica",
        fontSize=10, leading=15, textColor=INK,
        leftIndent=15, bulletIndent=2, spaceAfter=4)
    styles["TblHead"] = ParagraphStyle(
        "TblHead", parent=s["Normal"], fontName="Helvetica-Bold",
        fontSize=9, leading=12, textColor=ACCENT)
    styles["TblCell"] = ParagraphStyle(
        "TblCell", parent=s["Normal"], fontName="Helvetica",
        fontSize=9, leading=12.5, textColor=INK)
    styles["TblCellB"] = ParagraphStyle(
        "TblCellB", parent=s["Normal"], fontName="Helvetica-Bold",
        fontSize=9, leading=12.5, textColor=INK)
    styles["Caption"] = ParagraphStyle(
        "Caption", parent=s["Normal"], fontName="Helvetica-Oblique",
        fontSize=8.5, leading=12, textColor=MUTED, alignment=TA_CENTER,
        spaceBefore=6)
    styles["Note"] = ParagraphStyle(
        "Note", parent=s["Normal"], fontName="Helvetica-Oblique",
        fontSize=9, leading=13, textColor=MUTED, spaceAfter=8)
    return styles

ST = _styles()

# ---------------------------------------------------------------- helpers ----
def esc(t):
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

def P(text, style="Body"):
    return Paragraph(esc(text) if "<" not in text else text, ST[style])

def Lead(text):
    return Paragraph(text, ST["Lead"])

def H1(text):
    return Paragraph(esc(text), ST["H1"])

def H2(text):
    return Paragraph(esc(text), ST["H2"])

def Note(text):
    return Paragraph(text, ST["Note"])

def Bullets(items):
    return [Paragraph(esc(i), ST["Bullet"], bulletText="–") for i in items]

def Sp(h=6):
    return Spacer(1, h)


class Divider(Flowable):
    """A thin full-width rule used to separate an appendix from the body."""
    def __init__(self, space_before=18, space_after=2):
        super().__init__()
        self.width = USABLE_W
        self.space_before = space_before
        self.space_after = space_after

    def wrap(self, aw, ah):
        return (self.width, self.space_before + self.space_after)

    def draw(self):
        c = self.canv
        c.setStrokeColor(HAIR)
        c.setLineWidth(0.6)
        y = self.space_after
        c.line(0, y, self.width, y)


class DiagramPlaceholder(Flowable):
    """A restrained bordered panel used as an inline diagram placeholder."""
    def __init__(self, caption, lines, height=120):
        super().__init__()
        self.caption = caption
        self.lines = lines
        self.width = USABLE_W
        self.height = height

    def wrap(self, aw, ah):
        return (self.width, self.height + 16)

    def draw(self):
        c = self.canv
        c.saveState()
        c.setStrokeColor(HAIR)
        c.setFillColor(PANEL)
        c.setLineWidth(0.8)
        c.roundRect(0, 16, self.width, self.height, 6, stroke=1, fill=1)
        c.setFillColor(MUTED)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(10, self.height + 2, "FIGURE")
        # centered schematic text
        c.setFillColor(ACCENT)
        n = len(self.lines)
        line_h = 15
        start_y = 16 + self.height / 2 + (n - 1) * line_h / 2 - 4
        for i, ln in enumerate(self.lines):
            font = "Helvetica-Bold" if i == 0 else "Helvetica"
            size = 10 if i == 0 else 9
            c.setFont(font, size)
            c.setFillColor(ACCENT if i == 0 else MUTED)
            c.drawCentredString(self.width / 2, start_y - i * line_h, ln)
        c.restoreState()


def _tbl_cell(text, header=False, bold=False):
    if header:
        return Paragraph(esc(str(text)), ST["TblHead"])
    return Paragraph(esc(str(text)), ST["TblCellB"] if bold else ST["TblCell"])

def Tbl(rows, col_widths, bold_first_col=False):
    data = []
    for r_i, row in enumerate(rows):
        cells = []
        for c_i, val in enumerate(row):
            header = (r_i == 0)
            bold = (not header and bold_first_col and c_i == 0)
            cells.append(_tbl_cell(val, header=header, bold=bold))
        data.append(cells)
    t = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), PANEL),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -2), 0.4, HAIR),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, HAIR),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    t.setStyle(TableStyle(style))
    return t


# ---------------------------------------------------------- page furniture ---
def _title_page(canvas, doc):
    canvas.saveState()
    # top rule
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(2)
    canvas.line(LM, PAGE_H - TM, LM + 60, PAGE_H - TM)
    # footer brand
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(LM, BM - 14, "AURELIUS")
    canvas.drawRightString(PAGE_W - RM, BM - 14, CONFIDENTIAL)
    canvas.restoreState()

def _make_later(doc_title):
    def _later(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(HAIR)
        canvas.setLineWidth(0.5)
        canvas.line(LM, BM - 8, PAGE_W - RM, BM - 8)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(LM, BM - 20, "AURELIUS  ·  " + doc_title)
        canvas.drawCentredString(PAGE_W / 2, BM - 20, CONFIDENTIAL)
        canvas.drawRightString(PAGE_W - RM, BM - 20, str(canvas.getPageNumber()))
        # header hairline + running title
        canvas.setStrokeColor(HAIR)
        canvas.line(LM, PAGE_H - TM + 10, PAGE_W - RM, PAGE_H - TM + 10)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(PAGE_W - RM, PAGE_H - TM + 15, doc_title.upper())
        canvas.restoreState()
    return _later


def title_block(title, subtitle, doctype):
    return [
        Spacer(1, 2.4 * inch),
        Paragraph(doctype.upper(), ST["DocType"]),
        Spacer(1, 10),
        Paragraph(esc(title), ST["Title"]),
        Paragraph(esc(subtitle), ST["Subtitle"]),
        Spacer(1, 2.6 * inch),
        Paragraph("Aurelius", ParagraphStyle(
            "brand", fontName="Helvetica-Bold", fontSize=11, textColor=ACCENT)),
        Paragraph(
            "AI infrastructure orchestration and cost optimization", ST["Meta"]),
        Spacer(1, 6),
        Paragraph("Document version 1.0  ·  " + DOC_DATE, ST["Meta"]),
        NextPageTemplate("body"),
        PageBreak(),
    ]


def build(filename, doc_title, title, subtitle, doctype, body):
    path = f"{OUT_DIR}/{filename}"
    doc = BaseDocTemplate(
        path, pagesize=LETTER,
        leftMargin=LM, rightMargin=RM, topMargin=TM, bottomMargin=BM,
        title=title, author="Aurelius")
    frame = Frame(LM, BM, USABLE_W, PAGE_H - TM - BM, id="main",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([
        PageTemplate(id="title", frames=[frame], onPage=_title_page),
        PageTemplate(id="body", frames=[frame], onPage=_make_later(doc_title)),
    ])
    story = title_block(title, subtitle, doctype)
    story.extend(body)
    doc.build(story)
    print("wrote", path)


# ================================================================ CONTENT ====

def doc_executive():
    b = []
    b.append(H1("Overview"))
    b.append(Lead(
        "Aurelius is an infrastructure orchestration layer that reduces GPU "
        "operating cost through SLA-aware workload placement, time-shifting, "
        "and market-aware scheduling. It decides when and where flexible "
        "compute runs so the same work lands in lower-cost hours and regions, "
        "without changing the work itself."))
    b.append(P(
        "Aurelius is a decision layer, not an execution platform. It produces "
        "placement and timing decisions that an existing scheduler carries out. "
        "It does not take custody of workloads, requires no privileged access to "
        "begin, and operates in a read-only posture by default."))

    b.append(H1("The opportunity"))
    b.append(P(
        "GPU compute is the dominant variable cost in AI infrastructure, and the "
        "energy that powers it is priced in markets that move hour to hour and "
        "differ region to region. Wholesale electricity prices routinely vary by "
        "a factor of two or more across a day and across interconnects, yet most "
        "schedulers place jobs as they arrive, in a default region, regardless of "
        "when or where the work would be cheapest to run."))
    b.append(P(
        "A large share of GPU work — training, fine-tuning, batch inference, "
        "data processing, maintenance — carries real scheduling slack that goes "
        "unused. The result is a recurring, structural overspend that is invisible "
        "on any single invoice and difficult to address manually at fleet scale."))

    b.append(H1("How it works"))
    b.append(P(
        "For each schedulable job, Aurelius forecasts near-term wholesale price "
        "movements across the relevant markets and selects the region and start "
        "window that minimizes energy cost, subject to the job's deadline, SLA "
        "class, and the capacity of each region. Savings derive from three "
        "mechanisms, in order of contribution: time-shifting flexible jobs into "
        "lower-priced hours; routing across regional price spreads; and "
        "rescheduling eligible long-running jobs when a materially better price "
        "emerges. Because all three depend on flexibility, savings scale with the "
        "share of the fleet that can tolerate delay or relocation."))
    b.append(Sp(2))
    b.append(DiagramPlaceholder(
        "Decision pipeline. Aurelius owns the decision; the customer's scheduler owns execution.",
        ["Market + workload signals",
         "Ingestion   »   Forecasting   »   Optimization   »   Safety gate",
         "Decision  »  Reporting (replay / shadow)  or  scoped execution"],
        height=96))
    b.append(Paragraph(
        "Decision pipeline. Aurelius owns the decision; the customer’s scheduler owns execution.",
        ST["Caption"]))

    b.append(H1("Validation status"))
    b.append(P(
        "Tier 1 region and time placement has been validated in leakage-free "
        "historical replay on real day-ahead prices from CAISO, PJM, and ERCOT. "
        "Measured against a strong baseline that always routes to the cheapest "
        "region at submission using live prices, the observed mean cost reduction "
        "across seven workload classes was 25.0% on Q1 2026 data and 22.8% on "
        "Summer 2025 data. These are historical-replay observations, not "
        "guarantees; customer-specific savings are confirmed in shadow mode "
        "before any commitment."))
    b.append(Sp(2))
    b.append(Tbl([
        ["Workload class", "Observed reduction (p50)", "Driver"],
        ["Background maintenance", "~40%", "Fully flexible, freely reschedulable"],
        ["Data processing", "~38%", "High flexibility, short duration"],
        ["LLM batch inference", "~34%", "Batch, tolerant of ~24h delay"],
        ["Scheduled batch", "~25%", "Moderate flexibility"],
        ["Training", "~15%", "Long, often non-interruptible"],
        ["Fine-tuning", "~13%", "Moderate; sensitive to volatility"],
        ["Real-time inference", "~10%", "Cannot be delayed"],
    ], [1.7 * inch, 1.7 * inch, 2.9 * inch], bold_first_col=True))
    b.append(Note(
        "Savings track flexibility. A fleet weighted toward batch and maintenance "
        "work sits at the upper end; a fleet weighted toward latency-hard "
        "inference sits at the lower end."))

    b.append(H1("Why this is safe to adopt"))
    b.append(P(
        "Three design choices make Aurelius suitable for production adoption. "
        "Decisions are advisory and reversible: Aurelius emits placement and "
        "timing decisions and the existing scheduler executes them, so there is "
        "no custody of workloads. Behavior degrades to the status quo: when a "
        "forecast is missing or low-confidence, the decision falls back "
        "deterministically to the customer's current behavior, and the optimizer "
        "cannot produce an outcome worse than the baseline it is measured against. "
        "And savings are evidenced before they are claimed: shadow mode compares "
        "predicted savings to realized settlement prices on the customer's own "
        "workload before any decision is acted on."))

    b.append(H1("Engagement model"))
    b.append(P(
        "A pilot proceeds through three non-invasive phases. The decision to move "
        "forward rests on the shadow-mode result measured on the customer's own "
        "footprint, not on the historical benchmark."))
    b.append(Sp(2))
    b.append(Tbl([
        ["Phase", "What it does", "Customer exposure"],
        ["Offline replay", "Re-runs historical workload against historical prices",
         "None — read-only analysis"],
        ["Shadow mode", "Records live decisions; compares to realized prices",
         "None — no workloads executed"],
        ["Controlled execution", "Acts on decisions for selected flexible workloads",
         "Opt-in, policy-gated, reversible"],
    ], [1.5 * inch, 3.1 * inch, 1.7 * inch], bold_first_col=True))

    b.append(H1("Scope of what is validated"))
    b.append(P(
        "Validated today: Tier 1 region and time placement across the three "
        "largest U.S. wholesale markets. Available but dependent on customer "
        "integration: cluster and queue-aware placement, GPU and node-level "
        "placement, broader carbon coverage, and European markets. These are "
        "described as roadmap, not as current capability, and none is a "
        "prerequisite for a first pilot."))

    b.append(H1("Recommended next step"))
    b.append(P(
        "Provide a recent workload trace covering 30 to 90 days and confirm the "
        "compute regions and SLA classes involved. The first deliverable is an "
        "offline-replay projection, followed by a shadow-mode validation on the "
        "actual footprint. Both phases are read-only and require no change to "
        "production scheduling."))
    return b


def doc_technical():
    b = []
    b.append(H1("Purpose"))
    b.append(Lead(
        "This overview describes the engineering structure of Aurelius for "
        "platform and infrastructure architecture review: the components, the "
        "data flow between them, the boundary of what the system controls, and "
        "its behavior under failure."))
    b.append(P(
        "Aurelius is a decision pipeline. Market and workload data enter through "
        "ingestion; a forecaster produces calibrated price expectations; an "
        "optimization engine selects placement and timing under constraints; a "
        "safety gate vets each decision; and the result is either reported "
        "(replay and shadow modes) or handed to an execution adapter (controlled "
        "execution). Every stage is deterministic given its inputs and a fixed "
        "seed, and every stage degrades to the customer's baseline behavior "
        "rather than to an unsafe state."))
    b.append(Sp(2))
    b.append(DiagramPlaceholder(
        "Component and data flow",
        ["Ingestion  »  Forecasting  »  Optimization  »  Safety gate",
         "optional signals: carbon  ·  queue  ·  GPU health",
         "Reporting (replay, shadow)      Execution adapters (dry-run default)"],
        height=104))
    b.append(Paragraph("Component and data flow.", ST["Caption"]))

    b.append(H1("Components"))

    b.append(H2("Data ingestion"))
    b.append(P(
        "Ingestion normalizes every external signal into a canonical, "
        "timestamped, per-region schema. Wholesale prices come from direct "
        "ISO and TSO connectors returning hourly day-ahead and real-time "
        "settlement prices. A provenance layer tags every record with its source "
        "and market type and enforces an admissibility gate, so that sandbox or "
        "synthetic data is structurally barred from any savings path. Workload "
        "traces are ingested from a customer export; a minimal set of fields is "
        "sufficient to begin, with per-class defaults applied for anything "
        "omitted. Optional signals — carbon intensity, queue state, and GPU "
        "telemetry — enter through the same schema and are joined to price data "
        "by region and hour."))

    b.append(H2("Forecasting"))
    b.append(P(
        "The forecaster produces calibrated quantile predictions of future "
        "prices per region and horizon. The validated production model is a "
        "gradient-boosted quantile model with lagged price, seasonality, and "
        "volatility-regime features for spike detection. Calibration of the "
        "prediction interval is treated as a first-class metric alongside point "
        "accuracy, because the safety gate depends on the interval being "
        "trustworthy. A deterministic baseline forecaster is retained as a "
        "fallback and as a benchmark reference."))

    b.append(H2("Optimization engine"))
    b.append(P(
        "The optimizer assigns each job a region and start window to minimize a "
        "weighted objective combining energy cost with optional terms for carbon, "
        "forecast risk, SLA penalty, queue delay, GPU health, and data transfer. "
        "It does so subject to hard constraints: each job's earliest-start and "
        "deadline window, its allowed and forbidden regions, per-region capacity, "
        "and minimum power level. The risk term penalizes scheduling into "
        "high-uncertainty periods, so a wide forecast interval naturally biases "
        "the optimizer toward the safer, baseline-like choice. Optional signal "
        "weights default to zero, so a deployment supplying only prices behaves "
        "as a price-only optimizer; other signals activate only when both data "
        "and a cost weight are provided."))

    b.append(H2("Policy and safety gate"))
    b.append(P(
        "Before any decision is emitted, the safety gate evaluates the forecast "
        "interval against a workload-specific downside threshold — most "
        "conservative for real-time inference, most permissive for training. If "
        "the projected downside exceeds the threshold, or if no valid interval is "
        "available, the gate blocks the optimized decision and the system falls "
        "back to the current-price baseline. The gate is fail-closed: absence of "
        "evidence defers to the baseline rather than proceeding. Hard constraints "
        "are enforced by the optimizer and re-checked at the gate; a decision "
        "that would violate them is never produced."))

    b.append(H2("Shadow mode"))
    b.append(P(
        "Shadow mode is the mechanism for customer-specific validation. A "
        "single-pass runner makes the decisions the optimizer would make live, "
        "training only on price data preceding the decision time, and records one "
        "decision per job: the chosen region and start, the forecast, the "
        "predicted cost, and the baseline cost. Real-time settlement prices are "
        "never visible at decision time. After the settlement window closes, a "
        "separate step fills in the realized cost from settlement data, and a "
        "report compares predicted to realized savings. This separation — decide, "
        "then realize from independent data — is what makes the result credible "
        "rather than self-graded."))

    b.append(H2("Scheduler integration"))
    b.append(P(
        "Execution adapters translate decisions into actions for common "
        "schedulers, plus a replay interface. All adapters default to dry-run, "
        "log every attempted action for audit, support a global kill switch, and "
        "require a signed policy bundle to enter live mode. These adapters are "
        "unit-tested against mocks; they have not yet been validated against live "
        "production infrastructure, and resource-mapping heuristics are "
        "deployment-specific and should be reviewed before controlled execution."))

    b.append(H1("Control boundary"))
    b.append(P(
        "Aurelius reads market data and a workload description, and writes "
        "decisions. In replay and shadow modes it writes only to its own report "
        "files and never contacts the customer's scheduler. In controlled "
        "execution it submits to the scheduler through an adapter, but only for "
        "workloads explicitly in scope and only under an active policy. The "
        "system never holds workload data or model weights, and the only "
        "credentials it needs are read-only market-data keys plus, for controlled "
        "execution, whatever the chosen adapter requires."))
    b.append(P(
        "The boundary is deliberate: Aurelius owns the decision, the customer's "
        "platform owns the execution. This keeps the blast radius of any fault "
        "contained to a suboptimal-but-valid placement, never an unsafe action."))

    b.append(H1("Failure modes and fallback"))
    b.append(P(
        "Every failure resolves to the customer's existing behavior or a strictly "
        "safe subset of it. There is no path in which Aurelius produces a "
        "placement worse than the baseline it is measured against, because the "
        "baseline is always available as the fallback."))
    b.append(Sp(2))
    b.append(Tbl([
        ["Condition", "Behavior"],
        ["Forecast unavailable or invalid",
         "Safety gate blocks; fall back to current-price baseline"],
        ["Forecast interval too wide / high downside",
         "Optimized decision blocked; baseline used"],
        ["Market feed missing for a region",
         "Region excluded from routing; remaining regions used"],
        ["Optional signal absent",
         "Corresponding objective term is zero; price-only behavior"],
        ["Hard constraint unsatisfiable",
         "Job placed at constrained baseline; flagged in report"],
        ["Execution adapter error (live mode)",
         "Logged; kill switch available; no retry into unsafe state"],
    ], [2.5 * inch, 3.8 * inch], bold_first_col=True))

    b.append(H1("Reproducibility"))
    b.append(P(
        "Forecasting and optimization are deterministic under a fixed random "
        "seed. Benchmarks pin the data window, baseline set, fold structure, and "
        "seed, and archive their outputs, so any reported figure can be "
        "regenerated. The validation methodology and its controls are documented "
        "in the Benchmark Appendix."))
    return b


def doc_benchmark():
    b = []
    b.append(H1("Purpose"))
    b.append(Lead(
        "This appendix is the technical validation reference. It describes how "
        "Aurelius is benchmarked, the controls that keep the results honest, the "
        "baselines it is measured against, and the limitations of the current "
        "validation."))

    b.append(H1("What is measured"))
    b.append(P(
        "The benchmark measures the energy-cost reduction Aurelius achieves on a "
        "workload relative to a baseline scheduling policy, using real wholesale "
        "electricity prices. The headline metric is percentage cost reduction "
        "versus a current-price-only baseline, reported per workload class and as "
        "a mean across classes. Secondary metrics — carbon impact, SLA "
        "violations, migration count, downside events, and per-fold variance — are "
        "reported alongside."))

    b.append(H1("Data"))
    b.append(Tbl([
        ["Market", "Region", "Signal"],
        ["CAISO", "US West", "Day-ahead and real-time prices"],
        ["PJM", "US East", "Day-ahead and real-time prices"],
        ["ERCOT", "US South", "Day-ahead and real-time prices"],
    ], [1.6 * inch, 1.6 * inch, 3.0 * inch], bold_first_col=True))
    b.append(Sp(4))
    b.append(P(
        "Validation uses two real historical windows: Q1 2026 (higher winter "
        "volatility) and Summer 2025 (more stable), with no missing price hours "
        "in the reported configurations. All economic claims use real, "
        "unrandomized data from the source markets; synthetic and sandbox data "
        "are structurally barred from any savings path."))

    b.append(H1("Walk-forward validation"))
    b.append(P(
        "Evaluation is leakage-free walk-forward, not in-sample. For each fold "
        "the forecaster is trained only on price data preceding the fold's "
        "evaluation window, then evaluated on the held-out window. The validated "
        "configuration uses 30-day training windows and five walk-forward folds. "
        "The same temporal split is applied to every optional signal, so a future "
        "reading cannot influence a past decision."))

    b.append(H1("Leakage prevention"))
    b.append(P(
        "Leakage prevention is treated as a correctness property and is "
        "independently tested. Forecaster training data is filtered to the "
        "pre-evaluation window per fold; real-time settlement prices are never "
        "visible to the forecaster or optimizer at decision time; optional "
        "signals use a last-known-before-decision lookup; and synthetic "
        "provenance raises an error if it reaches a benchmark path."))
    b.append(Note(
        "During development, a predict-time feature was inadvertently zero-filling "
        "the future window, manufacturing an artificial signal that inflated "
        "apparent savings. The defect was found in review, the inflated result was "
        "discarded and never archived, and the corrected figure is what is "
        "reported. The harness is designed so that this class of error surfaces "
        "rather than ships."))

    b.append(H1("Baselines"))
    b.append(Tbl([
        ["Baseline", "Definition", "Role"],
        ["Current-price-only",
         "Always route to the cheapest region at submission, using live prices",
         "Primary — strong, realistic"],
        ["Upper-bound diagnostic",
         "Optimizer with perfect future price knowledge",
         "Diagnostic ceiling, not a baseline"],
        ["Naive single-region",
         "No optimization",
         "Reference only; not used for claims"],
    ], [1.5 * inch, 3.1 * inch, 1.7 * inch], bold_first_col=True))
    b.append(Sp(4))
    b.append(P(
        "The headline metric is always versus the current-price-only baseline. It "
        "is the strongest realistic baseline — it has perfect current-price "
        "information and represents what a sophisticated manual operator achieves "
        "— so improvement over it is attributable to forecasting, cross-region "
        "routing, and rescheduling rather than to a weak comparison."))

    b.append(H1("Results summary"))
    b.append(Tbl([
        ["Window", "Configuration", "Mean reduction"],
        ["Q1 2026 (CAISO / PJM / ERCOT)",
         "Quantile forecaster, 30-day windows, 5 folds", "25.0%"],
        ["Summer 2025 (CAISO / PJM / ERCOT)",
         "Quantile forecaster, 30-day windows, 5–7 folds", "22.8%"],
    ], [2.5 * inch, 2.6 * inch, 1.2 * inch], bold_first_col=True))
    b.append(Sp(4))
    b.append(P(
        "Savings persist across both seasonal windows, which is the relevant "
        "robustness check. Per-workload figures appear in the ROI methodology and "
        "the Executive Summary."))

    b.append(H1("Upper-bound diagnostic"))
    b.append(P(
        "The upper-bound diagnostic quantifies how much of the available "
        "opportunity the optimizer captures. A small gap means the optimizer is "
        "near the structural limit; a large gap means forecasting is the binding "
        "constraint."))
    b.append(Sp(2))
    b.append(Tbl([
        ["Workload", "Validated (p50)", "Upper-bound diagnostic", "Gap"],
        ["LLM batch inference", "~34%", "~43%", "~9 pts"],
        ["Training", "~15%", "~30%", "~15 pts"],
        ["Fine-tuning", "~13%", "~47%", "~33 pts"],
    ], [2.0 * inch, 1.4 * inch, 1.9 * inch, 1.0 * inch], bold_first_col=True))
    b.append(Sp(4))
    b.append(P(
        "For batch inference the optimizer is near-optimal. For training and "
        "fine-tuning the larger gaps are driven by winter price volatility that "
        "the forecaster only partly anticipates; on stable summer data these gaps "
        "narrow to single digits. This is disclosed because it bounds the "
        "realistic upside and explains the variation between seasons."))

    b.append(H1("Limitations"))
    b.append(P(
        "The current validation is bounded, and the bounds are stated rather than "
        "implied."))
    b.extend(Bullets([
        "Coverage is three U.S. markets; European and Asia-Pacific markets are not yet validated.",
        "Two seasonal windows of roughly 90 days each are validated; full-year persistence is not yet demonstrated.",
        "Published figures use workloads generated from per-class profiles; customer-specific results require the customer's own trace, confirmed in shadow mode.",
        "Carbon-aware results are limited to CAISO marginal-emissions coverage on the available data plan.",
        "Queue-aware and GPU-health signals have been validated only against synthetic fixtures, which are excluded from savings claims.",
        "Experimental forecaster variants did not consistently beat the production model in the windows tested and are not the default.",
    ]))
    return b


def doc_pilot():
    b = []
    b.append(H1("Scope of a first pilot"))
    b.append(Lead(
        "A first pilot targets Tier 1 optimization — choosing the region and "
        "start window for flexible workloads. This is the validated capability "
        "and requires no privileged access to customer infrastructure."))
    b.append(P(
        "Cluster and queue-aware placement (Tier 2) and GPU and node-level "
        "placement (Tier 3) can be layered on later if the customer exposes the "
        "necessary scheduler and telemetry data; they are not part of the initial "
        "pilot's success criteria."))

    b.append(H1("Prerequisites"))
    b.append(P(
        "The pilot can begin with a single data export and two short "
        "confirmations. The workload trace describes the shape of jobs — what "
        "kind, how long, how flexible — not their contents."))
    b.append(Sp(2))
    b.append(Tbl([
        ["Prerequisite", "Form", "Purpose"],
        ["Workload trace", "Recent jobs, 30–90 days", "Establishes the real workload mix"],
        ["Compute regions", "List mapped to markets", "Selects the wholesale price feeds"],
        ["SLA / flexibility", "Delay tolerance per type", "Bounds the optimizer's headroom"],
        ["Current monthly GPU spend", "Cost figure", "Sets the magnitude of the projection"],
    ], [1.9 * inch, 1.9 * inch, 2.4 * inch], bold_first_col=True))
    b.append(Sp(4))
    b.append(H2("Optional inputs"))
    b.append(Tbl([
        ["Optional input", "Enables"],
        ["Historical energy invoices", "Cross-check of price-data alignment"],
        ["Queue depth / wait-time export", "Cluster and queue-aware placement (Tier 2)"],
        ["GPU telemetry", "GPU and node-level placement (Tier 3)"],
        ["Marginal-emissions data license", "Carbon-aware optimization beyond CAISO"],
    ], [2.7 * inch, 3.5 * inch], bold_first_col=True))

    b.append(H1("Deployment modes"))
    b.append(P(
        "A pilot proceeds through three modes of increasing engagement. Most of "
        "the evidence is produced in the first two; the third is optional and "
        "scoped to flexible workloads."))
    b.append(H2("Offline replay"))
    b.append(P(
        "Aurelius re-runs the historical workload trace against historical market "
        "prices and reports the cost reduction the optimizer would have achieved. "
        "This is a read-only analysis that touches no live system and produces an "
        "initial, workload-specific projection."))
    b.append(H2("Shadow mode"))
    b.append(P(
        "Aurelius records the decisions it would make against live market data — "
        "for each job, the chosen region and start window, the forecast, and the "
        "predicted savings — without executing any workload. After the settlement "
        "window closes, realized prices are compared to predicted savings, per job "
        "and per workload class. This is the primary source of credible, "
        "customer-specific economic evidence."))
    b.append(H2("Controlled execution"))
    b.append(P(
        "Optionally, and only after shadow-mode results support it, Aurelius acts "
        "on decisions for a limited set of flexible workloads. This mode is "
        "opt-in, defaults to dry-run, requires a signed policy to go live, "
        "supports an immediate kill switch, and logs every action for audit. "
        "Latency-hard workloads are excluded."))

    b.append(H1("Phases"))
    b.append(P(
        "The pilot is organized as a sequence of phases gated by evidence, not by "
        "a fixed calendar. Each phase is reversible and produces an artifact the "
        "customer can review independently before agreeing to proceed."))
    b.append(Sp(2))
    b.append(Tbl([
        ["Phase", "Activity", "Gate to next phase"],
        ["Onboarding", "Ingest trace; confirm regions and SLA constraints",
         "Trace validates; regions mapped"],
        ["Offline replay", "Produce historical-replay projection",
         "Projection reviewed with customer"],
        ["Shadow validation", "Record live decisions; realize against settlement",
         "Realized savings meet threshold"],
        ["Controlled execution", "Act on flexible workloads under policy",
         "Operational sign-off"],
    ], [1.4 * inch, 2.9 * inch, 2.0 * inch], bold_first_col=True))

    b.append(H1("Success metrics"))
    b.append(P(
        "A pilot is evaluated on evidence meaningful to infrastructure and finance "
        "reviewers: realized cost reduction versus the current-price-only "
        "baseline, measured on the customer's own workload in shadow mode; "
        "agreement between predicted and realized savings, which indicates how "
        "dependable the projection is; zero SLA violations or deadline misses "
        "introduced by optimized placement; and bounded downside events, which "
        "the safety gate is designed to contain. The historical benchmark sets "
        "expectations; the shadow-mode result is the figure a pilot is judged on."))

    b.append(H1("Security and data handling"))
    b.append(P(
        "The pilot's minimum footprint is a workload trace and read-only "
        "market-data access. Aurelius does not take custody of workloads, model "
        "weights, or data; in replay and shadow modes it does not connect to the "
        "customer's scheduler at all. Market-data credentials are read-only, no "
        "secrets are stored in the system, and controlled execution requires "
        "explicit, signed authorization. For controlled execution, the scheduler "
        "integration requires only the permission to submit and label jobs in the "
        "regions and queues explicitly in scope — the narrowest role that lets "
        "the agreed workloads be placed."))
    b.append(Note(
        "A formal compliance program (SOC 2) is on the roadmap and not yet in "
        "place. A pilot can be conducted entirely in read-only modes, which avoids "
        "granting any execution privilege while the compliance posture matures."))

    b.append(H1("Known limitations entering a pilot"))
    b.extend(Bullets([
        "Validated savings cover U.S. markets (CAISO, PJM, ERCOT); European markets require a connection that is implemented but not yet validated.",
        "Carbon-aware optimization currently has marginal-emissions coverage for CAISO only on the available data plan.",
        "Tier 2 and Tier 3 have been exercised against synthetic fixtures; live validation depends on customer-supplied queue and telemetry data.",
        "Controlled-execution adapters are unit-tested but not yet validated against live production infrastructure; first controlled execution should be scoped narrowly and reviewed jointly.",
    ]))

    # Appendix — visually separated and kept together (no orphaned tail page)
    b.append(KeepTogether([
        Divider(),
        H1("Appendix A — Workload trace format"),
        P("The workload trace is a simple tabular export. Four fields are "
          "sufficient to begin; additional fields refine the optimization and can "
          "be supplied incrementally."),
        Sp(2),
        Tbl([
            ["Field", "Required", "Description"],
            ["Job identifier", "Yes", "Unique reference for the job"],
            ["Workload type", "Yes", "Training, inference, batch, maintenance, etc."],
            ["Submission time", "Yes", "When the job became eligible to run"],
            ["Duration", "Yes", "Estimated runtime"],
            ["GPU count", "Optional", "Per-class default applied if omitted"],
            ["Deadline / max delay", "Optional", "Scheduling flexibility window"],
            ["Allowed / forbidden regions", "Optional", "Placement constraints"],
            ["SLA class", "Optional", "Best-effort, standard, or guaranteed"],
        ], [2.1 * inch, 1.0 * inch, 3.1 * inch], bold_first_col=True),
        Note("Engineering-level setup, command-line usage, and reproduction "
             "commands are maintained separately in the developer documentation "
             "and are omitted here by design."),
    ]))
    return b


# ================================================================= MAIN ======
def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    build("01_Aurelius_Executive_Summary.pdf",
          "Executive Summary",
          "Aurelius",
          "AI infrastructure orchestration for GPU cost optimization",
          "Executive Summary",
          doc_executive())

    build("02_Aurelius_Technical_Overview.pdf",
          "Technical Overview",
          "Technical Overview",
          "Architecture, control boundaries, and failure behavior",
          "Technical Overview",
          doc_technical())

    build("03_Aurelius_Benchmark_Appendix.pdf",
          "Benchmark Appendix",
          "Benchmark Appendix",
          "Validation methodology, baselines, and results",
          "Benchmark Appendix",
          doc_benchmark())

    build("04_Aurelius_Pilot_Deployment_Guide.pdf",
          "Pilot Deployment Guide",
          "Pilot Deployment Guide",
          "Prerequisites, deployment modes, and success metrics",
          "Pilot Deployment Guide",
          doc_pilot())


if __name__ == "__main__":
    main()
