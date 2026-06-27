"""FleetPlane — v2026-native (built from first principles, not the cluster engine).

Reads and calibrates the hourly fleet state **directly from the
cluster-trace-gpu-v2026 schema** (`pod_hourly` + `server_hourly` +
`network_hourly`) and a regional electricity series. Every field is computed from
the real distribution (TRACE_DERIVED), not a static heuristic table — which is
why this replaces (does not wrap) `simulation/cluster/engine.py` (heuristic-first,
hourly M/M/1, disconnected from v2026; see the KEEP/ADAPT/REPLACE/DELETE audit).

Explicit adapters load real (or sample) trace slices; nothing is row-joined to
the serving plane — the planes share *state variables*, not rows. Granularity is
hourly pod aggregates (that is what v2026 is); the per-second serving happens on
the serving plane, synchronized inside each fleet hour by the environment.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

from ..datasets.calibration import alibaba_v2026_serving_class_mix
from .ingestion import v2026_artifacts
from .schemas import TRACE_DERIVED, CalibratedParam, FleetState

# Repo-root-relative default sample slices (substitute the full trace when present).
_FIX = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests", "fixtures")
SAMPLE_POD_HOURLY = os.path.join(_FIX, "alibaba_gpu_v2026", "pod_hourly_sample.csv")
SAMPLE_SERVER_HOURLY = os.path.join(_FIX, "alibaba_gpu_v2026", "server_hourly_sample.csv")
SAMPLE_NETWORK_HOURLY = os.path.join(_FIX, "alibaba_gpu_v2026", "network_hourly_sample.csv")
SAMPLE_ELECTRICITY = os.path.join(_FIX, "electricity", "caiso_hourly_sample.csv")

_INFERENCE = frozenset({"online_inference", "offline_inference"})


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _norm_util(x: float) -> float:
    """SM util may be reported 0..100 or 0..1; normalize to 0..1."""
    return x / 100.0 if x > 1.5 else x


# ---------------------------------------------------------------------------
# Explicit trace adapters (load real or sample slices; never row-join)
# ---------------------------------------------------------------------------

def load_pod_hourly(path: str = SAMPLE_POD_HOURLY) -> list:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_server_hourly(path: str = SAMPLE_SERVER_HOURLY) -> list:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_network_hourly(path: str = SAMPLE_NETWORK_HOURLY) -> list:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_electricity(path: str = SAMPLE_ELECTRICITY) -> dict:
    """Return ``{hour: price_per_kwh}`` for one region's diurnal series."""
    out: dict = {}
    region = "unknown"
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            region = r.get("region", region)
            out[int(_f(r.get("hour")))] = _f(r.get("price_per_kwh"))
    return {"region": region, "by_hour": out}


def full_trace_marginals(processed_dir: str | None = None) -> dict | None:
    """Global fleet marginals from the FULL_TRACE_EXACT v2026 calibration artifacts.

    These are the real population distributions (6.5 B pod rows etc.) the
    environment calibrates its fleet DISTRIBUTIONS to — util/memory means, priority
    and GPU-type mixes, queue/ready delay, GPU request, network rx/tx. Returns
    ``None`` if the pod artifact is absent (env falls back to the sample). Topology
    and scale (GPU counts, racks) stay sample-derived; only distributions anchor."""
    pdir = processed_dir or v2026_artifacts.PROCESSED_DIR
    pod = v2026_artifacts.load_table("pod_hourly", pdir)
    if pod is None:
        return None
    srv = v2026_artifacts.load_table("server_hourly", pdir) or {"artifacts": {}, "label": "n/a"}
    net = v2026_artifacts.load_table("network_hourly", pdir) or {"artifacts": {}, "label": "n/a"}
    pa, sa, na = pod["artifacts"], srv["artifacts"], net["artifacts"]
    jt = (pa.get("job_type_public") or {}).get("fractions", {})
    online = jt.get("online_inference", 0.0)
    offline = jt.get("offline_inference", 0.0)
    be = offline / (online + offline) if (online + offline) > 0 else 0.0
    return {
        "util_mean_pct": pa["gpu_sm_util"]["mean"],                      # 0..100
        "mem_util_mean": (pa.get("gpu_mem_util") or {}).get("mean", 0.0),  # 0..1
        "priority_mix": (pa.get("priority_class") or {}).get("fractions", {}),
        "job_type_mix": jt,
        "best_effort_fraction": round(be, 4),
        "queue_delay_mean_s": (pa.get("schedule_delay_s") or {}).get("mean", 0.0),
        "ready_delay_mean_s": (pa.get("ready_delay_s") or {}).get("mean", 0.0),
        "gpu_request_mean": (pa.get("gpu_request") or {}).get("mean", 0.0),
        "gpu_type_mix": (sa.get("gpu_type") or {}).get("fractions", {}),
        "rx_mean": (na.get("rx_gibps") or {}).get("mean", 0.0),
        "tx_mean": (na.get("tx_gibps") or {}).get("mean", 0.0),
        "pod_label": pod.get("label", "n/a"),
        "server_label": srv.get("label", "n/a"),
        "network_label": net.get("label", "n/a"),
    }


# ---------------------------------------------------------------------------
# The v2026-native fleet plane
# ---------------------------------------------------------------------------

class V2026FleetPlane:
    """Produces a :class:`FleetState` per hour, all fields TRACE_DERIVED from v2026.

    ``net_ref_gibps`` normalizes macro rx+tx into a 0..1 pressure (documented
    assumption, not from the trace). The class-mix best-effort fraction reuses the
    validated ``datasets.calibration`` hook (which passed the quality test).
    """

    def __init__(
        self,
        *,
        pod_path: str = SAMPLE_POD_HOURLY,
        server_path: str = SAMPLE_SERVER_HOURLY,
        network_path: str = SAMPLE_NETWORK_HOURLY,
        electricity_path: str = SAMPLE_ELECTRICITY,
        net_ref_gibps: float = 50.0,
        trace_version: str = "v2026-sample",
        processed_dir: str | None = None,
        anchor_full_trace: bool = True,
    ) -> None:
        self.pods = load_pod_hourly(pod_path)
        self.servers = load_server_hourly(server_path)
        self.network = load_network_hourly(network_path)
        elec = load_electricity(electricity_path)
        self.region = elec["region"]
        self.price_by_hour = elec["by_hour"]
        self.net_ref_gibps = net_ref_gibps
        self.trace_version = trace_version
        self._class_mix = alibaba_v2026_serving_class_mix(pod_path)
        # When the FULL_TRACE_EXACT artifacts are available, anchor the fleet
        # DISTRIBUTIONS (util/mem/mixes/delays/network) to the real 6.5 B-row
        # marginals; topology/scale stay sample-derived. Opt-out via processed_dir
        # left None or anchor_full_trace=False (keeps the legacy sample behaviour).
        self.full_trace = (full_trace_marginals(processed_dir)
                           if (processed_dir is not None and anchor_full_trace) else None)
        if self.full_trace is not None:
            self.trace_version = "v2026-full_trace_exact"
        self._mean_price = (sum(self.price_by_hour.values()) / len(self.price_by_hour)
                            if self.price_by_hour else 0.06)

        self._pods_by_hour = self._group(self.pods)
        self._servers_by_hour = self._group(self.servers)
        self._net_by_hour = self._group(self.network)

    @staticmethod
    def _group(rows: list) -> dict:
        g: dict = defaultdict(list)
        for r in rows:
            g[int(_f(r.get("hour")))].append(r)
        return dict(g)

    def hours(self) -> list:
        return sorted(self._pods_by_hour)

    def _tier(self, name, value, table_column, method, limitations="") -> CalibratedParam:
        return CalibratedParam(
            name=name, value=value, source_dataset="alibaba_gpu_v2026",
            table_column=table_column, fitting_method=method,
            train_holdout_split="hour-partitioned (sample)",
            trace_version=self.trace_version, tier=TRACE_DERIVED,
            limitations=limitations, safe_for_headline=False)

    def params_at(self, hour: int) -> list:
        """The CalibratedParam provenance records behind ``state_at(hour)``."""
        fs = self.state_at(hour)
        return [
            self._tier("util_target", fs.util_target, "pod_hourly.avg_gpu_sm_util",
                       "mean (inference pods)"),
            self._tier("mem_pressure", fs.mem_pressure, "pod_hourly.avg_memory_util", "mean"),
            self._tier("priority_mix", fs.priority_mix, "pod_hourly.priority_class", "fraction"),
            self._tier("best_effort_fraction", fs.best_effort_fraction,
                       "pod_hourly.job_type_public", "offline/(online+offline)"),
            self._tier("queue_delay_s", fs.queue_delay_s, "pod_hourly.schedule_delay_sec", "mean"),
            self._tier("gpu_type_inventory", fs.gpu_type_inventory,
                       "server_hourly.gpu_spec_public/gpu_count", "sum by type"),
            self._tier("rack_locality", fs.rack_locality, "server_hourly.asw_id", "gpu_count by asw",
                       "rack tier only; intra-node fabric not in v2026"),
            self._tier("net_pressure", fs.net_pressure, "network_hourly.rx/tx_gibps_avg",
                       f"mean(rx+tx)/{self.net_ref_gibps} ref", "macro only; no per-link congestion"),
            self._tier("fragmentation", fs.fragmentation, "pod_hourly.gpu_request/server gpu_count",
                       "unallocated fraction proxy", "true stranding needs placement detail"),
            CalibratedParam(
                "energy_price_per_kwh", fs.energy_price_per_kwh, f"iso_{self.region.lower()}",
                "electricity.price_per_kwh", "hour-of-day lookup", "n/a", self.trace_version,
                TRACE_DERIVED, "regional marginal price; PUE/depreciation in CostModel", False),
        ]

    def full_trace_params(self) -> list:
        """Provenance for the fleet DISTRIBUTIONS anchored to the FULL_TRACE_EXACT
        artifacts (empty if not anchored). Tier is TRACE_DERIVED (fitted from a real
        public trace — here the full population) → headline-safe; the FULL_TRACE_EXACT
        source and the SAMPLE_FIXTURE temporal caveat are recorded in the method/note."""
        ft = self.full_trace
        if ft is None:
            return []

        def _p(name, value, col, method):
            return CalibratedParam(
                name=name, value=value, source_dataset="alibaba_gpu_v2026",
                table_column=col, fitting_method=method,
                train_holdout_split="full population (all 4440 partitions; no holdout split)",
                trace_version=self.trace_version, tier=TRACE_DERIVED,
                limitations="FULL_TRACE_EXACT global marginal; per-hour temporal shape "
                            "is SAMPLE_FIXTURE (v2026 has no per-hour breakdown)",
                safe_for_headline=True)

        return [
            _p("fleet_gpu_utilization", round(ft["util_mean_pct"] / 100.0, 4),
               "pod_hourly.avg_gpu_sm_util", "full-trace mean (FULL_TRACE_EXACT)"),
            _p("fleet_gpu_memory_util", round(ft["mem_util_mean"], 4),
               "pod_hourly.avg_memory_util", "full-trace mean (FULL_TRACE_EXACT)"),
            _p("fleet_priority_mix", ft["priority_mix"],
               "pod_hourly.priority_class", "full-trace fractions (FULL_TRACE_EXACT)"),
            _p("fleet_gpu_type_mix", ft["gpu_type_mix"],
               "server_hourly.gpu_spec_public", "full-trace fractions (FULL_TRACE_EXACT)"),
            _p("fleet_queue_delay_s", round(ft["queue_delay_mean_s"], 2),
               "pod_hourly.schedule_delay_sec", "full-trace mean (FULL_TRACE_EXACT)"),
            _p("fleet_gpu_request", round(ft["gpu_request_mean"], 4),
               "pod_hourly.gpu_request", "full-trace mean (FULL_TRACE_EXACT)"),
            _p("fleet_network_rx_tx", round(ft["rx_mean"] + ft["tx_mean"], 4),
               "network_hourly.rx/tx_gibps_avg", "full-trace mean (FULL_TRACE_EXACT)"),
        ]

    def state_at(self, hour: int) -> FleetState:
        pods = self._pods_by_hour.get(hour, [])
        servers = self._servers_by_hour.get(hour, [])
        nets = self._net_by_hour.get(hour, [])

        # --- server inventory + rack topology (server_hourly) ---
        inv: dict = defaultdict(int)
        rack: dict = defaultdict(int)
        for s in servers:
            gc = int(_f(s.get("gpu_count")))
            inv[s.get("gpu_spec_public", "unknown")] += gc
            rack[s.get("asw_id", "unknown")] += gc
        total_gpus = sum(inv.values())
        mix = {k: round(v / total_gpus, 4) for k, v in inv.items()} if total_gpus else {}

        # --- inference-pod utilization / memory / priority / queue delay ---
        inf = [p for p in pods if (p.get("job_type_public") or "") in _INFERENCE]
        util = [_norm_util(_f(p.get("avg_gpu_sm_util"))) for p in inf]
        mem = [_f(p.get("avg_memory_util")) for p in inf]
        sched = [_f(p.get("schedule_delay_sec")) for p in inf]
        ready = [_f(p.get("ready_delay_sec")) for p in inf]
        util_target = sum(util) / len(util) if util else 0.0
        mem_pressure = sum(mem) / len(mem) if mem else 0.0

        util_by_class: dict = {}
        for cls in ("HP", "LP", "Other"):
            vals = [_norm_util(_f(p.get("avg_gpu_sm_util"))) for p in inf
                    if p.get("priority_class") == cls]
            if vals:
                util_by_class[cls] = round(sum(vals) / len(vals), 4)

        prio_counts: dict = defaultdict(int)
        for p in inf:
            prio_counts[p.get("priority_class", "Other")] += 1
        n_inf = sum(prio_counts.values())
        priority_mix = ({k: round(v / n_inf, 4) for k, v in prio_counts.items()}
                        if n_inf else {})

        # --- packing / fragmentation (gpu_request vs server gpu_count) ---
        req = sum(_f(p.get("gpu_request")) for p in pods)
        fragmentation = max(0.0, min(1.0, 1.0 - req / total_gpus)) if total_gpus else 0.0
        capacity_envelope = max(1, int(round(total_gpus * (1.0 - 0.5 * fragmentation))))

        # --- macro network pressure (network_hourly) ---
        if nets:
            press = [(_f(n.get("rx_gibps_avg")) + _f(n.get("tx_gibps_avg"))) / self.net_ref_gibps
                     for n in nets]
            net_pressure = max(0.0, min(1.0, sum(press) / len(press)))
        else:
            net_pressure = 0.0

        be = self._class_mix.best_effort_fraction_by_count
        price = self.price_by_hour.get(hour % 24, self.price_by_hour.get(0, 0.06))
        queue_delay_s = sum(sched) / len(sched) if sched else 0.0
        ready_delay_s = sum(ready) / len(ready) if ready else 0.0

        fidelity = {k: TRACE_DERIVED for k in (
            "util_target", "mem_pressure", "priority_mix", "best_effort_fraction",
            "queue_delay_s", "gpu_type_inventory", "rack_locality", "net_pressure",
            "fragmentation", "capacity_envelope", "energy_price_per_kwh")}

        ft = self.full_trace
        if ft is not None:
            # Anchor DISTRIBUTIONS to the full-trace marginals. v2026 is globally
            # aggregated (no per-hour breakdown), so the honest representation is a
            # CONSTANT marginal across hours — the only legitimately-hourly cost
            # signal, the regional electricity price, still varies (real ISO series).
            # Topology/scale (GPU counts, racks, capacity) stay sample-derived.
            util_target = max(0.0, min(1.0, ft["util_mean_pct"] / 100.0))
            mem_pressure = max(0.0, min(1.0, ft["mem_util_mean"]))
            priority_mix = dict(ft["priority_mix"]) or priority_mix
            be = ft["best_effort_fraction"] or be
            queue_delay_s = ft["queue_delay_mean_s"]
            ready_delay_s = ft["ready_delay_mean_s"]
            net_pressure = max(0.0, min(1.0, (ft["rx_mean"] + ft["tx_mean"]) / self.net_ref_gibps))
            if ft["gpu_type_mix"]:
                mix = dict(ft["gpu_type_mix"])
            fidelity.update({k: "FULL_TRACE_EXACT" for k in (
                "util_target", "mem_pressure", "priority_mix", "best_effort_fraction",
                "queue_delay_s", "net_pressure", "gpu_type_mix")})
            fidelity.update({k: "SAMPLE_FIXTURE" for k in (
                "total_gpus", "gpu_type_inventory", "rack_locality", "capacity_envelope")})

        return FleetState(
            hour=hour, total_gpus=total_gpus, gpu_type_inventory=dict(inv),
            gpu_type_mix=mix, util_target=util_target, util_by_class=util_by_class,
            mem_pressure=mem_pressure, priority_mix=priority_mix,
            best_effort_fraction=be, queue_delay_s=queue_delay_s, ready_delay_s=ready_delay_s,
            rack_locality=dict(rack), net_pressure=net_pressure,
            capacity_envelope=capacity_envelope, fragmentation=fragmentation,
            energy_price_per_kwh=price, region=self.region, fidelity=fidelity)


__all__ = [
    "V2026FleetPlane", "load_pod_hourly", "load_server_hourly",
    "load_network_hourly", "load_electricity",
    "SAMPLE_POD_HOURLY", "SAMPLE_SERVER_HOURLY", "SAMPLE_NETWORK_HOURLY", "SAMPLE_ELECTRICITY",
]
