"""Canonical production-like dataset assembly + the signal-coverage audit.

Two pieces, both deliberately honest about the gap between "production-like" and
"production":

* :mod:`aurelius.datasets.signal_matrix` — the machine-readable AUDIT of every
  signal the joint optimizer needs, which public dataset supplies it, and at what
  fidelity tier (measured / proxy / synthetic / simulator-only / absent).
* :mod:`aurelius.datasets.canonical` — the ASSEMBLER for the slice we can build
  today: a real interactive spine (Azure LLM 2024) + a documented best-effort
  batch overlay, the multi-class structure the unified replay engine needs to
  test whether combining serving levers compounds.

Companion prose design + risk analysis: ``research/CANONICAL_PRODUCTION_DATASET_DESIGN.md``.
"""

from .calibration import (
    ClassMix,
    alibaba_class_mix,
    alibaba_v2026_serving_class_mix,
    default_alibaba_class_mix,
)
from .canonical import (
    CanonicalManifest,
    assemble_calibrated,
    augment_with_best_effort,
    to_jobs,
)
from .signal_matrix import (
    CANONICAL_SIGNAL_MATRIX,
    CanonicalSignal,
    coverage_by_lever,
    coverage_by_tier,
    realizable_today,
    simulator_or_absent,
)

__all__ = [
    "CANONICAL_SIGNAL_MATRIX", "CanonicalSignal", "coverage_by_tier",
    "coverage_by_lever", "realizable_today", "simulator_or_absent",
    "to_jobs", "augment_with_best_effort", "assemble_calibrated",
    "CanonicalManifest", "ClassMix", "alibaba_class_mix",
    "default_alibaba_class_mix", "alibaba_v2026_serving_class_mix",
]
