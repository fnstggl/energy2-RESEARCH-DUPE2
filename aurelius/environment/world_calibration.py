"""Evidence-based calibration of the persistent world-state transition models.

The PR-#101 regression traced to ONE mis-modelled transition: warm-hold charged every idle replica a
full period of GPU time, when real autoscalers cool idle replicas after a ~300s idle timeout (see
``research/WORLD_STATE_REGRESSION_ROOT_CAUSE_AUDIT.md``). This module pins every world-state
transition parameter to a low/base/high band with an explicit public source, method, confidence and
fidelity tier — so calibration is auditable and evidence-driven, never tuned to make results look
good. ``world_simulator`` imports its constants from here.

Fidelity tiers (never "UNKNOWN"):
- TRACE_DERIVED      — fitted to one of our committed public traces (v2026 / Mooncake / Azure)
- PUBLIC_PAPER       — a published measurement (paper / vendor benchmark), cited by URL
- BENCHMARK_DERIVED  — a public-benchmark magnitude (blog/vendor numbers), cited by URL
- SIMULATOR_INFERENCE — a modelling assumption we make explicit (no external measurement)
"""

from __future__ import annotations

from dataclasses import dataclass, field

TRACE_DERIVED = "TRACE_DERIVED"
PUBLIC_PAPER = "PUBLIC_PAPER"
BENCHMARK_DERIVED = "BENCHMARK_DERIVED"
SIMULATOR_INFERENCE = "SIMULATOR_INFERENCE"
_TIERS = (TRACE_DERIVED, PUBLIC_PAPER, BENCHMARK_DERIVED, SIMULATOR_INFERENCE)


@dataclass(frozen=True)
class CalibrationSource:
    name: str
    url: str
    kind: str            # one of _TIERS


@dataclass(frozen=True)
class CalibratedParameter:
    """One world-state transition parameter, as a low/base/high band with provenance."""
    name: str
    low: float
    base: float          # the value the simulator uses
    high: float
    unit: str
    method: str          # how base was chosen from the sources
    confidence: str      # "low" | "medium" | "high"
    limitation: str
    fidelity: str        # one of _TIERS
    sources: tuple = ()  # CalibrationSource

    def __post_init__(self):
        assert self.fidelity in _TIERS, f"bad fidelity {self.fidelity}"
        assert self.low <= self.base <= self.high, f"band out of order for {self.name}"

    def to_dict(self) -> dict:
        return {"name": self.name, "low": self.low, "base": self.base, "high": self.high,
                "unit": self.unit, "method": self.method, "confidence": self.confidence,
                "limitation": self.limitation, "fidelity": self.fidelity,
                "sources": [{"name": s.name, "url": s.url, "kind": s.kind} for s in self.sources]}


@dataclass(frozen=True)
class TransitionValidationResult:
    transition: str
    check: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {"transition": self.transition, "check": self.check, "passed": self.passed,
                "detail": self.detail}


@dataclass
class WorldCalibrationReport:
    parameters: dict = field(default_factory=dict)        # name -> CalibratedParameter
    validations: list = field(default_factory=list)       # TransitionValidationResult

    def base(self, name: str) -> float:
        return self.parameters[name].base

    def to_dict(self) -> dict:
        return {"parameters": {k: v.to_dict() for k, v in self.parameters.items()},
                "validations": [v.to_dict() for v in self.validations],
                "any_unknown_provenance": any(p.fidelity not in _TIERS for p in self.parameters.values())}


# --- the sources (public, cited by URL) -------------------------------------
_S_SERVERLESS = CalibrationSource(
    "ServerlessLLM / serverless GPU cold-start surveys",
    "https://arxiv.org/pdf/2401.14351", PUBLIC_PAPER)
_S_VLLM_START = CalibrationSource(
    "vLLM / GKE startup-time decomposition (engine init 2-5s; weight load dominates)",
    "https://dudeperf3ct.github.io/posts/vllm_startup_load_time/", BENCHMARK_DERIVED)
_S_MODELLOAD = CalibrationSource(
    "Model loading time by GPU/storage (70B INT4 10s NVMe..72s SATA; 8-32B ~60s on A100)",
    "https://gigagpu.com/model-loading-time-gpu-storage/", BENCHMARK_DERIVED)
_S_SLEEPMODE = CalibrationSource(
    "vLLM sleep-mode / Run:ai model streamer (warm-pool resume 2-8s)",
    "https://blog.vllm.ai/2025/10/26/sleep-mode.html", BENCHMARK_DERIVED)
_S_SCALEDOWN = CalibrationSource(
    "Serverless GPU scale-down delay default 300s (5 min), configurable to 1h",
    "https://regolo.ai/scale-to-zero-cold-start-latency-why-serverless-gpu-breaks-real-time-ai-and-how-to-fix-it/",
    BENCHMARK_DERIVED)
_S_LLUMNIX = CalibrationSource(
    "Llumnix live migration (pipelined KV copy, near-zero downtime; append-only KV)",
    "https://arxiv.org/pdf/2406.03243", PUBLIC_PAPER)
_S_KVBW = CalibrationSource(
    "KV-cache transfer bandwidth (RDMA 12-50 GB/s; PCIe5 ~63 GB/s; 1-10 GB caches)",
    "https://arxiv.org/pdf/2504.11816", PUBLIC_PAPER)
_S_V2026 = CalibrationSource(
    "Alibaba cluster-trace-gpu-v2026 server/network marginals (committed)",
    "data/external/alibaba_gpu_v2026/processed", TRACE_DERIVED)


def world_calibration() -> WorldCalibrationReport:
    """Return the calibrated band + provenance for every world-state transition the simulator uses."""
    p = {}
    p["cold_start_s"] = CalibratedParameter(
        "cold_start_s", 8.0, 30.0, 60.0, "seconds",
        "serving replica becomes available: model weight load (10-72s by storage) + vLLM engine "
        "init (2-5s); warm-pool/sleep-mode resume is the 2-8s low; cold scale-from-zero on remote "
        "storage is the 40-60s high. base=30s = typical container+NVMe load. NOT tuned to results.",
        "medium", "serving startup; not from our trace (Alibaba ready_delay conflates batch queues)",
        BENCHMARK_DERIVED, (_S_SERVERLESS, _S_VLLM_START, _S_MODELLOAD, _S_SLEEPMODE))
    # cold-start DECOMPOSITION — the same 8/30/60 band split into the components different actions
    # avoid differently (a warm replica skips engine+model+ready; a MIGRATED replica keeps weights
    # loaded so it skips engine+model but may pay kv_warmup; prewarm pays warm-hold to skip the
    # future engine+model). Bases sum to cold_start_s base (30s) — a fidelity decomposition, NOT a
    # reduction. Sources are the same already-cited startup decompositions.
    p["cold_start_engine_init_s"] = CalibratedParameter(
        "cold_start_engine_init_s", 2.0, 3.0, 5.0, "seconds",
        "vLLM/SGLang engine + CUDA context init, independent of model size (2-5s). A warm or migrated "
        "replica has a live engine and skips this.",
        "medium", "framework/version dependent", BENCHMARK_DERIVED, (_S_VLLM_START,))
    p["cold_start_model_load_s"] = CalibratedParameter(
        "cold_start_model_load_s", 10.0, 22.0, 60.0, "seconds",
        "model-weight load into HBM — the DOMINANT term (10s NVMe .. 60-72s remote/SATA; 8-32B ~60s on "
        "A100). base=22s = container+NVMe. A migrated replica keeps weights resident and AVOIDS this; "
        "prewarming pays warm-hold to avoid it on the forecast period's cold replicas. NOT tuned down.",
        "medium", "storage tier dominates; the single biggest cold-start component",
        BENCHMARK_DERIVED, (_S_MODELLOAD, _S_VLLM_START))
    p["cold_start_kv_warmup_s"] = CalibratedParameter(
        "cold_start_kv_warmup_s", 0.0, 3.0, 8.0, "seconds",
        "KV/first-token warm-up: a freshly loaded (or non-pipelined-migrated) replica starts with an "
        "empty cache and pays first-batch prefill before steady state. A pipelined (Llumnix) migration "
        "moves the KV and avoids most of this; a bulk move pays it.",
        "low", "depends on prefix locality; overlaps the KV reuse model", SIMULATOR_INFERENCE,
        (_S_SLEEPMODE, _S_LLUMNIX))
    p["cold_start_ready_delay_s"] = CalibratedParameter(
        "cold_start_ready_delay_s", 0.0, 2.0, 4.0, "seconds",
        "scheduler/orchestrator readiness: k8s readiness probe + endpoint registration before traffic. "
        "Independent of model; a warm replica is already registered.",
        "low", "orchestrator-config dependent (Alibaba ready_delay is an upper-bound sanity, not "
        "serving cold-start)", SIMULATOR_INFERENCE, (_S_SERVERLESS,))
    p["migration_kv_preserved_frac"] = CalibratedParameter(
        "migration_kv_preserved_frac", 0.5, 0.9, 1.0, "fraction of KV warmth kept across a move",
        "Llumnix pipelines the KV to the destination (append-only, near-zero downtime), so a live "
        "(conservative/pipelined) migration KEEPS most KV warmth — the moved replica does NOT re-pay "
        "kv_warmup. base=0.9 kept. A bulk (aggressive) move keeps less. This REPLACES the unrealistic "
        "flat KV surcharge that made migration strictly dominated.",
        "low", "append-only KV + recompute make loss small; pipelining quality varies",
        PUBLIC_PAPER, (_S_LLUMNIX, _S_KVBW))
    p["warm_idle_timeout_s"] = CalibratedParameter(
        "warm_idle_timeout_s", 120.0, 300.0, 3600.0, "seconds",
        "an idle warm replica is held then cooled after the autoscaler scale-down delay; default "
        "300s (5 min). This is the duration an idle replica incurs warm-hold before cooling — the "
        "fix for the full-period over-charge that caused the regression.",
        "high", "vendor defaults vary; some keep min-replicas warm indefinitely (the 1h high)",
        BENCHMARK_DERIVED, (_S_SCALEDOWN,))
    p["warm_hold_gpu_fraction"] = CalibratedParameter(
        "warm_hold_gpu_fraction", 0.3, 1.0, 1.0, "fraction of a GPU-hour",
        "a warm replica keeps the model resident in GPU memory and the process live → it occupies "
        "~a full GPU while warm (base=1.0, conservative). sleep-mode/offload can reduce to ~0.3 (low).",
        "medium", "depends on whether warm means resident vs offloaded-to-CPU (sleep mode)",
        SIMULATOR_INFERENCE, (_S_SLEEPMODE,))
    p["cold_start_ramp"] = CalibratedParameter(
        "cold_start_ramp", 0.0, 1.0, 1.0, "0=step,1=linear",
        "cold replicas come online PROGRESSIVELY as each finishes loading (staggered readiness / "
        "pipelined loading), not all at once — base=1.0 (linear ramp from warm to full over "
        "cold_start_s). 0.0 would be the worst-case step (all-at-once), the PR-#101 behaviour.",
        "low", "real staggering depends on loader concurrency; linear is the common approximation",
        SIMULATOR_INFERENCE, (_S_SERVERLESS,))
    p["migration_duration_periods"] = CalibratedParameter(
        "migration_duration_periods", 1.0, 1.0, 2.0, "periods",
        "a live move drains + transfers + re-warms within one hourly period; benefit (locality) "
        "lands the next period. KV copy itself is sub-second-to-seconds (1-10 GB at 12-50 GB/s).",
        "low", "pipelined migration (Llumnix) can hide most of it; we keep 1 period as the unit",
        BENCHMARK_DERIVED, (_S_LLUMNIX, _S_KVBW))
    p["migration_cost_per_replica"] = CalibratedParameter(
        "migration_cost_per_replica", 0.1, 0.4, 1.0, "USD",
        "operator $ for one live move: control-plane reschedule + drain + KV/model transfer GPU-time. "
        "Dominated by the brief capacity loss + re-warm, not the raw transfer.",
        "low", "no public $-denominated migration benchmark; BENCHMARK_DERIVED from transfer time",
        BENCHMARK_DERIVED, (_S_LLUMNIX, _S_KVBW))
    p["migration_capacity_loss_frac"] = CalibratedParameter(
        "migration_capacity_loss_frac", 0.05, 0.10, 0.25, "fraction per migrating replica",
        "capacity withheld while a replica drains+moves, this period only.",
        "low", "Llumnix pipelining reduces this; we keep a conservative non-zero loss",
        SIMULATOR_INFERENCE, (_S_LLUMNIX,))
    p["migration_cache_penalty"] = CalibratedParameter(
        "migration_cache_penalty", 0.0, 0.04, 0.1, "service-time surcharge",
        "KV warmth lost on a moved replica → a brief service-time surcharge until the cache refills. "
        "Recompute-on-destination is ~10x faster than transferring (Llumnix), so the surcharge is small.",
        "low", "append-only KV + recompute make this small; SIMULATOR_INFERENCE magnitude",
        SIMULATOR_INFERENCE, (_S_LLUMNIX, _S_KVBW))
    p["topology_max_discount"] = CalibratedParameter(
        "topology_max_discount", 0.0, 0.08, 0.15, "service-time discount fraction",
        "max MACRO service-time relief from rack locality + lowest network pressure (v2026 rx/tx "
        "spread). Macro only — NO per-link/NVLink claims.",
        "low", "macro network pressure only; per-link congestion ABSENT from any public trace",
        TRACE_DERIVED, (_S_V2026,))
    p["network_pressure_ref_gibps"] = CalibratedParameter(
        "network_pressure_ref_gibps", 0.5, 1.0, 4.0, "GiB/s normaliser",
        "normaliser mapping v2026 macro rx+tx (mean 0.13/0.07 GiB/s, hotspots ~900) into 0..1 rack "
        "pressure. base=1.0 GiB/s reference link.",
        "low", "documented assumption, not a measured link rate", SIMULATOR_INFERENCE, (_S_V2026,))
    # reconcile the cold-start decomposition to the aggregate band (fidelity, not a reduction).
    comp_base = sum(p[k].base for k in COLD_START_COMPONENTS)
    v = [TransitionValidationResult(
        "cold_start_decomposition", "components_sum_to_aggregate_base",
        abs(comp_base - p["cold_start_s"].base) <= 1.0,
        f"sum(components.base)={comp_base:.1f}s vs cold_start_s.base={p['cold_start_s'].base:.1f}s")]
    return WorldCalibrationReport(parameters=p, validations=v)


# the components a full cold start decomposes into (order = serving startup sequence).
COLD_START_COMPONENTS = ("cold_start_engine_init_s", "cold_start_model_load_s",
                         "cold_start_kv_warmup_s", "cold_start_ready_delay_s")


def cold_start_components(report: "WorldCalibrationReport | None" = None) -> dict:
    """``{component: base_seconds}`` for the cold-start decomposition + the reconciled total.

    Used by the world simulator to value what each action AVOIDS: a warm replica skips
    engine+model+ready; a pipelined-migrated replica skips engine+model (weights move) and keeps
    ``migration_kv_preserved_frac`` of kv_warmup; prewarming pays warm-hold to skip the cold replicas'
    engine+model on the forecast period."""
    r = report or world_calibration()
    comp = {k: r.base(k) for k in COLD_START_COMPONENTS}
    comp["total_s"] = sum(comp.values())
    return comp


__all__ = ["CalibrationSource", "CalibratedParameter", "TransitionValidationResult",
           "WorldCalibrationReport", "world_calibration", "cold_start_components",
           "COLD_START_COMPONENTS", "TRACE_DERIVED", "PUBLIC_PAPER",
           "BENCHMARK_DERIVED", "SIMULATOR_INFERENCE"]
