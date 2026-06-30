"""Heterogeneous GPU-type assignment — causal fixture model (Batch-1 Phase 3).

Routes workload classes to GPU *classes* (H100 / A100 / L40S / A10 / …) by their published FLOPs / memory
bandwidth / HBM capacity / power / cost. The economic lever Aurelius has that a single-pool serving
scheduler lacks: send latency-sensitive work to fast GPUs, batch/flexible work to cheap GPUs, and
memory-heavy work to high-HBM GPUs — minimizing $ at equal SLA.

**NOT_APPLICABLE to the production benchmark.** The canonical reward path costs an entire period at ONE
dominant GPU type (``gpu_type`` is constant per server; there is no per-workload assignment mechanism in the
cluster replay), so a heterogeneous-assignment ACTION cannot change — and cannot fake — the production
headline. This module is therefore a **controlled-fixture** causal model: you give it an explicit fleet mix
and a workload-class mix, and it scores each assignment policy through real per-GPU-type roofline timing +
per-type operator cost. On a HOMOGENEOUS fleet (one GPU type) every policy collapses to the same assignment
→ zero benefit (the no-fake-gain guarantee). It becomes a CONNECTED action once the fleet/cost path exposes
per-replica GPU-type assignment.

Fidelity: GPU FLOPs/bandwidth/HBM/TDP are PUBLIC_SPEC; per-type CapEx/lease/power are INFERRED public-list
(``cost_model``); the routing outcome is SIMULATOR_INFERENCE. ``diagnostic_oracle_assignment`` is a
NON-deployable brute-force upper bound (labelled). No reward bonus — gp/$ flows through latency (roofline) +
cost (per-type operator economics) + SLA (service time vs class target).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .cost_model import CostModel
from .kv_cache import gpu_mem_for
from .roofline import GPU_SPECS, ServingConfig, Workload, serving_point

GPU_ASSIGNMENT_OPTIONS = (
    "homogeneous_default", "fastest_for_latency_sensitive", "cheapest_for_batch",
    "memory_heavy_to_high_hbm", "balanced_heterogeneous", "diagnostic_oracle_assignment")
NON_DEPLOYABLE_POLICIES = frozenset({"diagnostic_oracle_assignment"})


@dataclass
class WorkloadClass:
    """One workload class to place. ``sla_s`` is its per-request completion target (None = batch/no SLA)."""
    name: str                      # "latency_sensitive" | "batch" | "memory_heavy" | …
    prompt_tokens: int
    decode_tokens: int
    requests_per_s: float
    sla_s: float | None = None
    kind: str = "latency"          # "latency" | "batch" | "memory_heavy"

    def workload(self) -> Workload:
        return Workload(prompt_tokens=self.prompt_tokens, decode_tokens=self.decode_tokens,
                        context_len=self.prompt_tokens + self.decode_tokens // 2)


@dataclass
class GPUType:
    name: str
    count: int

    @property
    def peak_flops(self) -> float:
        return GPU_SPECS.get(self.name, GPU_SPECS["A100"])["peak_flops"]

    @property
    def mem_bw(self) -> float:
        return GPU_SPECS.get(self.name, GPU_SPECS["A100"])["mem_bw"]

    @property
    def hbm_gib(self) -> float:
        return gpu_mem_for(self.name)


@dataclass
class AssignmentResult:
    policy: str
    deployable: bool
    selected_gpu_type_mix: dict
    workload_to_gpu_mapping: dict
    latency_by_gpu_type: dict
    cost_by_gpu_type: dict
    hbm_pressure_by_gpu_type: dict
    sla_by_class: dict
    total_sla_safe_goodput: float
    total_cost: float
    gp_per_dollar: float
    sla_violation_rate: float

    def to_dict(self) -> dict:
        return {
            "policy": self.policy, "deployable": self.deployable,
            "selected_gpu_type_mix": self.selected_gpu_type_mix,
            "workload_to_gpu_mapping": self.workload_to_gpu_mapping,
            "latency_by_gpu_type": {k: round(v, 5) for k, v in self.latency_by_gpu_type.items()},
            "cost_by_gpu_type": {k: round(v, 5) for k, v in self.cost_by_gpu_type.items()},
            "hbm_pressure_by_gpu_type": {k: round(v, 4) for k, v in self.hbm_pressure_by_gpu_type.items()},
            "sla_by_class": self.sla_by_class,
            "total_sla_safe_goodput": round(self.total_sla_safe_goodput, 3),
            "total_cost": round(self.total_cost, 5),
            "gp_per_dollar": round(self.gp_per_dollar, 2),
            "sla_violation_rate": round(self.sla_violation_rate, 5)}


def _completion_s(gpu: str, wc: WorkloadClass) -> dict:
    return serving_point(wc.workload(), ServingConfig(gpu=gpu, batch_size=16))


def _assign(policy: str, classes: list, fleet: list, cost_model: CostModel) -> dict:
    """Workload-class → GPU-type mapping under a policy. Deterministic. On a homogeneous fleet every policy
    returns the same map (the single type), so no policy can fake a benefit."""
    types = [g.name for g in fleet]
    if len(set(types)) <= 1:                                   # homogeneous → no choice exists
        only = types[0] if types else "A100"
        return {wc.name: only for wc in classes}
    # "fastest" = lowest serving latency on a reference latency workload (not raw bandwidth — prefill compute
    # matters too); "cheapest" = lowest $/GPU-hour depreciation; "highest_hbm" = most KV headroom.
    _ref = Workload(prompt_tokens=1024, decode_tokens=128, context_len=1088)
    by_latency = sorted(types, key=lambda t: serving_point(_ref, ServingConfig(gpu=t, batch_size=16))["completion_s"])
    by_cost = sorted(types, key=lambda t: cost_model._econ(t).depreciation_per_gpu_hour())
    by_hbm = sorted(types, key=lambda t: -gpu_mem_for(t))
    fastest, cheapest, highest_hbm = by_latency[0], by_cost[0], by_hbm[0]
    dominant = max(fleet, key=lambda g: g.count).name
    out = {}
    for wc in classes:
        if policy == "homogeneous_default":
            out[wc.name] = dominant
        elif policy == "fastest_for_latency_sensitive":
            out[wc.name] = fastest if wc.kind == "latency" else cheapest
        elif policy == "cheapest_for_batch":
            out[wc.name] = cheapest if wc.kind == "batch" else fastest
        elif policy == "memory_heavy_to_high_hbm":
            out[wc.name] = highest_hbm if wc.kind == "memory_heavy" else (fastest if wc.kind == "latency" else cheapest)
        elif policy == "balanced_heterogeneous":
            out[wc.name] = (fastest if wc.kind == "latency" else
                            highest_hbm if wc.kind == "memory_heavy" else cheapest)
        else:
            out[wc.name] = dominant
    return out


def _score_mapping(mapping: dict, classes: list, fleet: list, cost_model: CostModel, *,
                   period_s: float, energy_price: float) -> dict:
    """Score a class→GPU mapping: per-class latency / SLA, per-type cost / HBM pressure, total gp/$."""
    by_name = {wc.name: wc for wc in classes}
    fleet_by_type = {g.name: g for g in fleet}
    lat_by_type, cost_by_type, hbm_by_type, sla_by_class = {}, {}, {}, {}
    total_good = total_cost = total_req = total_viol = 0.0
    for cname, gtype in mapping.items():
        wc = by_name[cname]
        pt = _completion_s(gtype, wc)
        compl = pt["completion_s"]
        n = wc.requests_per_s * period_s
        gpu_h = pt["gpu_seconds"] * n / 3600.0
        cb = cost_model.operator_cost(gpu_hours=gpu_h, gpu_type=gtype, energy_price_per_kwh=energy_price,
                                      utilization=0.8, scenario="owned")
        cost_by_type[gtype] = cost_by_type.get(gtype, 0.0) + cb.total_operator_cost
        lat_by_type[gtype] = max(lat_by_type.get(gtype, 0.0), compl)
        # HBM pressure: this class's KV residency vs the GPU's HBM (memory-heavy classes press high).
        kv_gib = (wc.prompt_tokens + wc.decode_tokens) * pt.get("kv_bytes_per_token", 131072.0) / (1024 ** 3)
        offered_seqs = max(1.0, wc.requests_per_s * compl)
        hbm_press = min(1.0, kv_gib * offered_seqs / max(1.0, gpu_mem_for(gtype) - 16.0))
        hbm_by_type[gtype] = max(hbm_by_type.get(gtype, 0.0), hbm_press)
        sla_ok = wc.sla_s is None or compl <= wc.sla_s
        sla_by_class[cname] = {"gpu": gtype, "completion_s": round(compl, 5),
                               "sla_s": wc.sla_s, "sla_ok": bool(sla_ok),
                               "hbm_pressure": round(hbm_press, 4)}
        good_tokens = wc.decode_tokens * n
        if sla_ok:
            total_good += good_tokens
        else:
            total_viol += n
        total_req += n
        total_cost += cb.total_operator_cost
        _ = fleet_by_type  # fleet capacity is a fixture input; counts inform the mix only
    gp = total_good / max(total_cost, 1e-9)
    mix = {}
    for gtype in mapping.values():
        mix[gtype] = mix.get(gtype, 0) + 1
    return {"selected_gpu_type_mix": mix, "workload_to_gpu_mapping": dict(mapping),
            "latency_by_gpu_type": lat_by_type, "cost_by_gpu_type": cost_by_type,
            "hbm_pressure_by_gpu_type": hbm_by_type, "sla_by_class": sla_by_class,
            "total_sla_safe_goodput": total_good, "total_cost": total_cost,
            "gp_per_dollar": gp, "sla_violation_rate": total_viol / max(total_req, 1.0)}


def evaluate_assignment(policy: str, classes: list, fleet: list, *, cost_model: CostModel | None = None,
                        period_s: float = 60.0, energy_price: float = 0.06) -> AssignmentResult:
    """Score one assignment ``policy`` on a fixture (workload ``classes`` × ``fleet`` GPU mix). The oracle
    policy brute-forces the best class→type mapping by gp/$ (NON-deployable upper bound)."""
    cm = cost_model or CostModel()
    if policy == "diagnostic_oracle_assignment":
        types = list(dict.fromkeys(g.name for g in fleet))
        best, best_gp = None, -1.0
        for combo in product(types, repeat=len(classes)):
            mapping = {classes[i].name: combo[i] for i in range(len(classes))}
            sc = _score_mapping(mapping, classes, fleet, cm, period_s=period_s, energy_price=energy_price)
            if sc["gp_per_dollar"] > best_gp:
                best, best_gp = sc, sc["gp_per_dollar"]
        sc = best
    else:
        mapping = _assign(policy, classes, fleet, cm)
        sc = _score_mapping(mapping, classes, fleet, cm, period_s=period_s, energy_price=energy_price)
    return AssignmentResult(
        policy=policy, deployable=policy not in NON_DEPLOYABLE_POLICIES, **sc)


def compare_assignment_policies(classes: list, fleet: list, *, cost_model: CostModel | None = None,
                                period_s: float = 60.0, energy_price: float = 0.06,
                                policies=GPU_ASSIGNMENT_OPTIONS) -> dict:
    """``{policy: AssignmentResult.to_dict()}`` + the best DEPLOYABLE policy by gp/$ and the homogeneous
    baseline. On a homogeneous fleet all deployable policies tie (no fake heterogeneous benefit)."""
    res = {p: evaluate_assignment(p, classes, fleet, cost_model=cost_model, period_s=period_s,
                                  energy_price=energy_price) for p in policies}
    deployable = {p: r for p, r in res.items() if r.deployable}
    best = max(deployable, key=lambda p: deployable[p].gp_per_dollar)
    homogeneous = len({g.name for g in fleet}) <= 1
    return {"results": {p: r.to_dict() for p, r in res.items()},
            "best_deployable_policy": best,
            "homogeneous_fleet": homogeneous,
            "baseline_gp_per_dollar": res["homogeneous_default"].gp_per_dollar,
            "best_gp_per_dollar": deployable[best].gp_per_dollar}


__all__ = ["GPU_ASSIGNMENT_OPTIONS", "NON_DEPLOYABLE_POLICIES", "WorkloadClass", "GPUType",
           "AssignmentResult", "evaluate_assignment", "compare_assignment_policies"]
