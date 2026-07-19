"""Deterministic demo log generators — messy on purpose.

The demo exists so a prospect (or the two of us) can see the whole flow in
under a minute with zero real data. Both samples deliberately use
vendor-flavored column names, mixed encodings and a few corrupt rows, so the
demo also demonstrates the importer doing its job — mapping, unit inference,
and the parse-error ledger — not just the replay.

Synthetic data, clearly labelled. Never used for any benchmark claim.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

_DAY_S = 86_400


def write_demo_serving_log(out_path: str | Path, *, seed: int = 20260701,
                           hours: float = 3.0) -> Path:
    """Gateway-style LLM request log: epoch-ms ts, vendor column names.

    Load is tuned into the multi-replica regime (tens of RPS with strong
    diurnal swing + bursts) so provisioning policies actually differentiate;
    a trace light enough to fit one replica ties every policy by construction.
    """
    rng = random.Random(seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0_ms = 1_772_000_000_000  # fixed epoch base (2026), deterministic
    models = [("chat-70b", 0.55), ("chat-8b", 0.30), ("code-34b", 0.15)]
    rows = []
    t = 0.0
    horizon_s = hours * 3600
    sessions = [f"s-{rng.randrange(16**8):08x}" for _ in range(400)]
    burst_until = -1.0
    while t < horizon_s:
        cycle_frac = (t % (horizon_s / 2)) / (horizon_s / 2)
        diurnal = 1.0 + 0.85 * math.sin(2 * math.pi * (cycle_frac - 0.3))
        if t > burst_until and rng.random() < 0.0015:
            burst_until = t + rng.uniform(45, 150)   # sustained burst window
        burst = 5.0 if t <= burst_until else 1.0
        rate = max(2.0, 11.0 * diurnal) * burst      # requests / second
        t += rng.expovariate(rate)
        r = rng.random()
        acc = 0.0
        model = models[-1][0]
        for name, share in models:
            acc += share
            if r <= acc:
                model = name
                break
        prompt = max(8, int(rng.lognormvariate(5.6, 0.9)))
        output = max(4, int(rng.lognormvariate(5.1, 1.0)))
        status = 200 if rng.random() > 0.012 else rng.choice([500, 503, 429])
        session = rng.choice(sessions) if rng.random() < 0.62 else ""
        rows.append({
            "ts": int(t0_ms + t * 1000),
            "model_name": model,
            "input_tok": prompt,
            "gen_tok": output,
            "latency_ms": int(80 + output * rng.uniform(8, 14)),
            "http_status": status,
            "session": session,
            "region": rng.choice(["us-east-1", "us-west-2"]),
            "api_ver": "v2",
        })
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for i, row in enumerate(rows):
            if i == 1234:                      # one corrupt row on purpose
                fh.write("garbage,line,that,should,be,counted,not,crash\n")
                continue
            w.writerow(row)
    return out


def write_demo_jobs_log(out_path: str | Path, *, seed: int = 20260702,
                        days: float = 14.0) -> Path:
    """Slurm-``sacct``-flavored GPU job log: TRES strings, [D-]HH:MM:SS."""
    rng = random.Random(seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    base = 1_772_000_000  # epoch seconds, deterministic
    users = [f"user{n:02d}" for n in range(14)]
    partitions = ["training", "batch", "interactive"]
    rows = []
    t = 0.0
    horizon = days * _DAY_S
    jid = 41_000
    while t < horizon:
        day_frac = (t % _DAY_S) / _DAY_S
        diurnal = 1.0 + 0.9 * math.sin(2 * math.pi * (day_frac - 0.25))
        rate_per_hour = max(0.6, 4.0 * diurnal)
        t += rng.expovariate(rate_per_hour / 3600.0)
        jid += 1
        gpus = rng.choice([1, 1, 1, 2, 2, 4, 4, 8, 8, 8])
        dur_s = int(min(3 * _DAY_S, max(300, rng.lognormvariate(8.6, 1.3))))
        wait_s = int(max(0, rng.lognormvariate(6.5, 1.6)))
        submit = base + int(t)
        start = submit + wait_s
        end = start + dur_s
        state = rng.choices(
            ["COMPLETED", "FAILED", "CANCELLED by 1042", "TIMEOUT"],
            weights=[86, 7, 5, 2])[0]
        gpu_type = rng.choices(["h100", "a100"], weights=[70, 30])[0]
        d, rem = divmod(dur_s, _DAY_S)
        h, rem = divmod(rem, 3600)
        m, s = divmod(rem, 60)
        elapsed = (f"{d}-{h:02d}:{m:02d}:{s:02d}" if d
                   else f"{h:02d}:{m:02d}:{s:02d}")
        started = "" if state.startswith("CANCELLED") and rng.random() < 0.5 \
            else _iso(start)
        rows.append({
            "JobID": jid,
            "User": rng.choice(users),
            "Partition": rng.choice(partitions),
            "Submit": _iso(submit),
            "Start": started,
            "End": _iso(end) if started else "",
            "Elapsed": elapsed if started else "00:00:00",
            "State": state,
            "AllocTRES": (f"billing={gpus * 8},cpu={gpus * 16},"
                          f"gres/gpu:{gpu_type}={gpus},mem={gpus * 128}G"),
            "ExitCode": "0:0" if state == "COMPLETED" else "1:0",
        })
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return out


def _iso(epoch_s: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S")
