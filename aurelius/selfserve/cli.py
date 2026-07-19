"""CLI for Aurelius Replay: ``python -m aurelius.selfserve <command>``.

Commands:

- ``inspect FILE``  — mapping plan + readiness only; no replay. ``--json``.
- ``run FILE``      — full pipeline; writes report.html + fingerprint.json.
- ``demo``          — generate a messy sample log, run the full pipeline on
                      it, and print the report path (zero-input first run).
- ``serve``         — local web UI on 127.0.0.1 (requires fastapi+uvicorn,
                      already project dependencies).

Exit codes: 0 = ran (ready/degraded), 2 = replay refused (insufficient),
1 = file unreadable / tool error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import TOOL_NAME, __version__
from .demo import write_demo_jobs_log, write_demo_serving_log
from .fields import WORKLOAD_CLASSES
from .fingerprint import write_fingerprint
from .pipeline import inspect_file, run_pipeline
from .report import write_report
from .sniff import TableReadError
from .validate import TIER_INSUFFICIENT

_DEFAULT_OUT = Path("aurelius_replay_runs")


def _parse_overrides(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--map expects canonical=source, got {pair!r}")
        k, v = pair.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _cmd_inspect(args: argparse.Namespace) -> int:
    try:
        _table, plan = inspect_file(
            args.file, workload_class=args.workload_class,
            overrides=_parse_overrides(args.map),
        )
    except TableReadError as exc:
        print(f"cannot read file: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
        return 0
    print(f"{TOOL_NAME} v{__version__} — mapping plan for {args.file}")
    print(f"workload class: {plan.workload_class} "
          f"(confidence {plan.class_confidence:.2f})")
    print(f"  {plan.class_evidence}")
    for m in plan.mappings:
        if m.source is None:
            continue
        print(f"  {m.canonical:<16} <- {m.source!r:<20} [{m.matched_by}] "
              f"{m.transform}")
    absent = [m.canonical for m in plan.mappings if m.source is None]
    if absent:
        print(f"  not present: {', '.join(absent)}")
    if plan.unmapped_source_columns:
        print(f"  ignored columns: "
              f"{', '.join(plan.unmapped_source_columns)}")
    for w in plan.warnings:
        print(f"  WARNING: {w}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    artifacts = run_pipeline(
        args.file,
        workload_class=args.workload_class,
        overrides=_parse_overrides(args.map),
        fleet_gpus=args.fleet_gpus,
        gpus_per_node=args.gpus_per_node,
        gpu_type=args.gpu_type,
        tick_seconds=args.tick_seconds,
        max_rows=args.max_rows,
    )
    if not artifacts.ok:
        print(f"cannot read file: {artifacts.error}", file=sys.stderr)
        return 1

    out_dir = Path(args.out) / artifacts.run_id
    report_path = write_report(artifacts, out_dir / "report.html")
    fp_path = write_fingerprint(artifacts.fingerprint,
                                out_dir / "schema_fingerprint.json")
    (out_dir / "summary.json").write_text(
        json.dumps(artifacts.to_dict(), indent=2, default=str))

    r = artifacts.readiness
    print(f"{TOOL_NAME} v{__version__}")
    print(f"  class:     {r.workload_class}")
    print(f"  readiness: {r.tier}  "
          f"({r.rows_usable:,}/{r.rows_read:,} rows usable)")
    if artifacts.replay_summary:
        outcome = artifacts.replay_summary.get("outcome", {})
        print(f"  replay:    constraint_aware vs "
              f"{artifacts.replay_summary.get('headline_baseline')}: "
              f"{outcome.get('constraint_aware_vs_headline')} "
              f"({outcome.get('margin_pct'):+.2f}% goodput/$)"
              " — directional only, not production savings")
    else:
        print(f"  replay:    SKIPPED — {artifacts.replay_skipped_reason}")
    print(f"  report:    {report_path}")
    print(f"  fingerprint (shareable, no log rows): {fp_path}")
    if args.json:
        print(json.dumps(artifacts.to_dict(), indent=2, default=str))
    return 2 if (r.tier == TIER_INSUFFICIENT) else 0


def _cmd_demo(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.workload_class == "serving":
        sample = write_demo_serving_log(out_dir / "demo_serving_gateway.csv")
    else:
        sample = write_demo_jobs_log(out_dir / "demo_slurm_sacct.csv")
    print(f"demo log (synthetic, messy on purpose): {sample}")
    ns = argparse.Namespace(
        file=str(sample), workload_class=None, map=[], out=str(out_dir),
        fleet_gpus=args.fleet_gpus, gpus_per_node=8, gpu_type=None,
        tick_seconds=60.0, max_rows=2_000_000, json=False,
    )
    return _cmd_run(ns)


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .webapp import create_app
    except ImportError as exc:
        print("serve requires fastapi + uvicorn (project deps): "
              f"{exc}", file=sys.stderr)
        return 1
    app = create_app(run_dir=Path(args.out))
    print(f"{TOOL_NAME} v{__version__} — http://{args.host}:{args.port} "
          "(local only; nothing is transmitted)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m aurelius.selfserve",
        description=f"{TOOL_NAME} — read-only local replay of your "
                    "scheduler/serving logs. Nothing leaves this machine.",
    )
    p.add_argument("--version", action="version",
                   version=f"{TOOL_NAME} {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def _common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--class", dest="workload_class",
                        choices=list(WORKLOAD_CLASSES), default=None,
                        help="force the workload class instead of detecting")
        sp.add_argument("--map", action="append", metavar="CANONICAL=SOURCE",
                        help="explicit column mapping (repeatable)")

    sp = sub.add_parser("inspect", help="mapping plan + readiness, no replay")
    sp.add_argument("file")
    _common(sp)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=_cmd_inspect)

    sp = sub.add_parser("run", help="full pipeline → report.html")
    sp.add_argument("file")
    _common(sp)
    sp.add_argument("--out", default=str(_DEFAULT_OUT))
    sp.add_argument("--fleet-gpus", type=int, default=None,
                    help="your real fleet size in GPUs (gpu_jobs class)")
    sp.add_argument("--gpus-per-node", type=int, default=8)
    sp.add_argument("--gpu-type", default=None)
    sp.add_argument("--tick-seconds", type=float, default=60.0)
    sp.add_argument("--max-rows", type=int, default=2_000_000)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=_cmd_run)

    sp = sub.add_parser("demo", help="generate a sample log and run on it")
    sp.add_argument("--class", dest="workload_class",
                    choices=["serving", "gpu_jobs"], default="gpu_jobs")
    sp.add_argument("--out", default=str(_DEFAULT_OUT / "demo"))
    sp.add_argument("--fleet-gpus", type=int, default=None)
    sp.set_defaults(fn=_cmd_demo)

    sp = sub.add_parser("serve", help="local web UI (127.0.0.1)")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8712)
    sp.add_argument("--out", default=str(_DEFAULT_OUT))
    sp.set_defaults(fn=_cmd_serve)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
