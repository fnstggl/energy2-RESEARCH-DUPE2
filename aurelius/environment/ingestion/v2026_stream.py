"""v2026 incremental FULL_TRACE_EXACT ingestion — stream the OSS ZIPs by partition.

Proves (and implements) that the 351 GB Alibaba v2026 dataset can be calibrated
*incrementally* with **bounded disk** and **FULL_TRACE_EXACT** quality — no full
extraction, no sampling. The mechanism (verified against the live OSS bucket):

  1. The archives are real ZIPs on Aliyun OSS which **supports HTTP Range**.
  2. Reading only the ZIP central directory (a tail range) lists every parquet
     partition (``day=…/hour=…/part-000.parquet``) — pod_hourly has 4,440, each
     **STORED** (uncompressed in the ZIP), median ~81 MB.
  3. Each partition is range-fetched to a bounded work dir, read row-group by
     row-group with pyarrow, folded into **mergeable exact aggregators**, then
     deleted. Peak disk ≈ one partition (~112 MB), never 351 GB.
  4. A manifest checkpoints after every partition → **resumable / restart-safe**.

Exactness (per the build spec): count / sum / sumsq / min / max / fixed-bin
histograms / category counters are processed over **every row exactly once** and
are mathematically equivalent to conventional full processing (cross-partition
float reduction order is the only difference) → ``FULL_TRACE_EXACT``. Percentiles
derived from histogram bins are ``FULL_TRACE_APPROX`` with documented bins. Nothing
is sampled; a partial run (e.g. interrupted) is labeled ``SUBSET_TRACE`` honestly.

Requires ``pyarrow`` (an optional ingestion-time dep; the stdlib FleetPlane only
consumes the JSON artifacts this writes). Network egress for a full pod_hourly pass
transfers 351 GB once (time-bound, resumable) — NOT a disk or feasibility limit.
"""

from __future__ import annotations

import io
import json
import math
import os
import subprocess
import zipfile
from dataclasses import dataclass, field

from ..data_tier import (
    FULL_TRACE_APPROX,
    FULL_TRACE_EXACT,
    SUBSET_TRACE,
)

OSS_BASE = "https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-gpu-v2026/data/"
ARCHIVES = {
    "pod_hourly": OSS_BASE + "asi_opensource_pod_hourly.zip",
    "server_hourly": OSS_BASE + "asi_opensource_server_hourly.zip",
    "network_hourly": OSS_BASE + "asi_opensource_network_hourly.zip",
    "job_execution_summary": OSS_BASE + "asi_opensource_job_execution_summary.zip",
}


# ---------------------------------------------------------------------------
# HTTP-range seekable file (curl-backed: proxy + CA bundle already work)
# ---------------------------------------------------------------------------

class HttpRangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP Range via curl, for zipfile/pyarrow."""

    def __init__(self, url: str, size: int, *, timeout: int = 120, retries: int = 4):
        self.url, self._size, self._pos = url, size, 0
        self.timeout, self.retries = timeout, retries

    def seekable(self): return True
    def readable(self): return True

    def seek(self, off, whence=0):
        self._pos = off if whence == 0 else (self._pos + off if whence == 1 else self._size + off)
        return self._pos

    def tell(self): return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self._size - self._pos
        if n == 0 or self._pos >= self._size:
            return b""
        end = min(self._pos + n, self._size) - 1
        want = end - self._pos + 1
        for attempt in range(self.retries):
            out = subprocess.run(
                ["curl", "-sS", "--max-time", str(self.timeout), "-r",
                 f"{self._pos}-{end}", self.url], capture_output=True).stdout
            if len(out) == want:
                self._pos += len(out)
                return out
        # short read after retries → return what we got (caller/zipfile will error loudly)
        self._pos += len(out)
        return out


def head_size(url: str, *, timeout: int = 30) -> int:
    r = subprocess.run(["curl", "-sS", "--max-time", str(timeout), "-I", url],
                       capture_output=True, text=True).stdout
    for line in r.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":")[1].strip())
    raise RuntimeError(f"no Content-Length for {url}")


# ---------------------------------------------------------------------------
# Mergeable EXACT aggregators (serializable for checkpoints)
# ---------------------------------------------------------------------------

@dataclass
class ExactStats:
    """Exact count/sum/sumsq/min/max → exact mean & variance (order-independent)."""
    n: int = 0
    s: float = 0.0
    ss: float = 0.0
    mn: float = math.inf
    mx: float = -math.inf

    def update(self, xs) -> None:
        for x in xs:
            if x is None:
                continue
            x = float(x)
            self.n += 1
            self.s += x
            self.ss += x * x
            if x < self.mn:
                self.mn = x
            if x > self.mx:
                self.mx = x

    def merge(self, o: "ExactStats") -> None:
        self.n += o.n
        self.s += o.s
        self.ss += o.ss
        self.mn = min(self.mn, o.mn)
        self.mx = max(self.mx, o.mx)

    def to_dict(self) -> dict:
        mean = self.s / self.n if self.n else 0.0
        var = (self.ss / self.n - mean * mean) if self.n else 0.0
        return {"label": FULL_TRACE_EXACT, "n": self.n, "mean": mean,
                "variance": max(0.0, var), "min": (self.mn if self.n else 0.0),
                "max": (self.mx if self.n else 0.0)}

    @classmethod
    def from_state(cls, d: dict) -> "ExactStats":
        return cls(d["n"], d["s"], d["ss"], d["mn"], d["mx"])

    def state(self) -> dict:
        return {"n": self.n, "s": self.s, "ss": self.ss, "mn": self.mn, "mx": self.mx}


@dataclass
class ExactHistogram:
    """Fixed-bin histogram over every row (exact bin counts) → APPROX percentiles."""
    lo: float
    hi: float
    nbins: int = 50
    counts: list = field(default_factory=list)
    below: int = 0
    above: int = 0

    def __post_init__(self):
        if not self.counts:
            self.counts = [0] * self.nbins

    def update(self, xs) -> None:
        w = (self.hi - self.lo) / self.nbins
        for x in xs:
            if x is None:
                continue
            x = float(x)
            if x < self.lo:
                self.below += 1
            elif x >= self.hi:
                self.above += 1
            else:
                self.counts[min(self.nbins - 1, int((x - self.lo) / w))] += 1

    def merge(self, o: "ExactHistogram") -> None:
        self.counts = [a + b for a, b in zip(self.counts, o.counts)]
        self.below += o.below
        self.above += o.above

    def percentile(self, q: float) -> float:
        total = self.below + sum(self.counts) + self.above
        if total == 0:
            return 0.0
        target = q * total
        cum = self.below
        w = (self.hi - self.lo) / self.nbins
        for i, c in enumerate(self.counts):
            if cum + c >= target:
                return self.lo + (i + 0.5) * w
            cum += c
        return self.hi

    def to_dict(self) -> dict:
        return {"label": FULL_TRACE_APPROX, "method": "fixed-bin histogram",
                "bins": self.nbins, "range": [self.lo, self.hi],
                "p50": self.percentile(0.50), "p95": self.percentile(0.95),
                "p99": self.percentile(0.99),
                "below_range": self.below, "above_range": self.above}

    def state(self) -> dict:
        return {"lo": self.lo, "hi": self.hi, "nbins": self.nbins,
                "counts": self.counts, "below": self.below, "above": self.above}

    @classmethod
    def from_state(cls, d: dict) -> "ExactHistogram":
        return cls(d["lo"], d["hi"], d["nbins"], list(d["counts"]), d["below"], d["above"])


@dataclass
class ExactCounter:
    """Exact category counts (priority mix, job/model type, GPU type, asw locality)."""
    counts: dict = field(default_factory=dict)

    def update(self, xs) -> None:
        for x in xs:
            k = str(x)
            self.counts[k] = self.counts.get(k, 0) + 1

    def merge(self, o: "ExactCounter") -> None:
        for k, v in o.counts.items():
            self.counts[k] = self.counts.get(k, 0) + v

    def to_dict(self) -> dict:
        total = sum(self.counts.values()) or 1
        return {"label": FULL_TRACE_EXACT, "counts": dict(self.counts),
                "fractions": {k: v / total for k, v in self.counts.items()}}

    def state(self) -> dict:
        return {"counts": self.counts}

    @classmethod
    def from_state(cls, d: dict) -> "ExactCounter":
        return cls(dict(d["counts"]))


# ---------------------------------------------------------------------------
# Partition streaming (range-extract one parquet at a time, bounded disk)
# ---------------------------------------------------------------------------

def list_partitions(archive_url: str) -> tuple:
    """Return ``(zipfile, [parquet ZipInfo, ...])`` reading ONLY the central dir."""
    zf = zipfile.ZipFile(HttpRangeFile(archive_url, head_size(archive_url)))
    parts = [i for i in zf.infolist() if i.filename.endswith(".parquet")]
    return zf, parts


def _atomic_write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


@dataclass
class StreamResult:
    archive: str
    n_partitions_total: int
    n_partitions_done: int
    artifacts: dict                    # name -> aggregator.to_dict()
    label: str                         # FULL_TRACE_EXACT/APPROX vs SUBSET_TRACE
    bytes_streamed: int

    def to_dict(self) -> dict:
        return {
            "archive": self.archive, "n_partitions_total": self.n_partitions_total,
            "n_partitions_done": self.n_partitions_done, "artifacts": self.artifacts,
            "label": self.label, "bytes_streamed": self.bytes_streamed,
            "complete": self.n_partitions_done == self.n_partitions_total,
        }


def stream_archive(
    archive_url: str, build_aggs, fold_table, *,
    work_dir: str, manifest_path: str, max_partitions: int | None = None,
) -> StreamResult:
    """Stream every partition of a remote OSS archive once (bounded disk, resumable).

    ``build_aggs() -> dict[str, agg]`` constructs fresh aggregators;
    ``fold_table(aggs, pyarrow_table)`` updates them from one partition.
    """
    zf, parts = list_partitions(archive_url)
    return _stream_parts(archive_url, zf, parts, build_aggs, fold_table,
                         work_dir=work_dir, manifest_path=manifest_path,
                         max_partitions=max_partitions)


def stream_local_zip(
    zip_path: str, build_aggs, fold_table, *,
    work_dir: str, manifest_path: str, max_partitions: int | None = None,
) -> StreamResult:
    """Same streaming logic over a LOCAL zip of parquet partitions (for tests /
    already-downloaded archives) — identical exactness + resume semantics."""
    zf = zipfile.ZipFile(zip_path)
    parts = [i for i in zf.infolist() if i.filename.endswith(".parquet")]
    return _stream_parts(zip_path, zf, parts, build_aggs, fold_table,
                         work_dir=work_dir, manifest_path=manifest_path,
                         max_partitions=max_partitions)


# Archives at/under this many bytes are downloaded WHOLE (one bandwidth-bound
# curl) then streamed locally — far faster than thousands of per-partition range
# round-trips. Still bounded disk (must fit the work dir). pod_hourly (351 GB) is
# above this → always range-streamed.
PREFETCH_MAX_BYTES = 10 * 1024**3   # 10 GB


def stream_archive_prefetched(
    archive_url: str, build_aggs, fold_table, *,
    work_dir: str, manifest_path: str, max_partitions: int | None = None,
) -> StreamResult:
    """Download the whole archive once (bounded by ``PREFETCH_MAX_BYTES``), then
    stream it locally. Resumable: if a completed manifest exists the download is
    skipped. The downloaded zip is removed after processing."""
    os.makedirs(work_dir, exist_ok=True)
    size = head_size(archive_url)
    if size > PREFETCH_MAX_BYTES:
        raise ValueError(f"{archive_url} is {size:,} B > prefetch cap "
                         f"{PREFETCH_MAX_BYTES:,}; use stream_archive (range)")
    local_zip = os.path.join(work_dir, "archive.zip")
    if not (os.path.exists(local_zip) and os.path.getsize(local_zip) == size):
        rc = subprocess.run(["curl", "-sS", "--fail", "--max-time", "1800",
                             "-o", local_zip, archive_url]).returncode
        if rc != 0 or os.path.getsize(local_zip) != size:
            raise RuntimeError(f"whole-archive download failed for {archive_url}")
    try:
        return stream_local_zip(local_zip, build_aggs, fold_table, work_dir=work_dir,
                                manifest_path=manifest_path, max_partitions=max_partitions)
    finally:
        if os.path.exists(local_zip):
            os.remove(local_zip)


def _stream_parts(
    src_id: str, zf, parts, build_aggs, fold_table, *,
    work_dir: str, manifest_path: str, max_partitions: int | None = None,
) -> StreamResult:
    """Shared core: fold each parquet partition into mergeable exact aggregators,
    one bounded temp file at a time, checkpointing the manifest after each."""
    import pyarrow.parquet as pq  # optional ingestion-time dependency

    os.makedirs(work_dir, exist_ok=True)
    total = len(parts)

    aggs = build_aggs()
    done: set = set()
    bytes_streamed = 0
    if os.path.exists(manifest_path):
        m = json.load(open(manifest_path))
        done = set(m.get("processed", []))
        bytes_streamed = m.get("bytes_streamed", 0)
        for name, agg in aggs.items():
            if name in m.get("state", {}):
                agg.__dict__.update(type(agg).from_state(m["state"][name]).__dict__)

    todo = [p for p in parts if p.filename not in done]
    if max_partitions is not None:
        todo = todo[:max_partitions]

    def _checkpoint():
        _atomic_write_json(manifest_path, {
            "archive": src_id, "processed": sorted(done),
            "bytes_streamed": bytes_streamed, "n_partitions_total": total,
            "state": {k: v.state() for k, v in aggs.items()}})

    # Checkpoint every N partitions (not every one) — for tables with large counter
    # state (e.g. server_hourly asw_id: thousands of keys) writing the full state
    # each partition is O(n^2). A crash loses <= N partitions; resume re-streams them.
    checkpoint_every = 50
    local = os.path.join(work_dir, "part.parquet")
    for i, p in enumerate(todo):
        # Extract + read with retry — transient proxy/range corruption (a wrong-bytes
        # 200, a truncated member) is retried (re-fetched), not fatal; the run is also
        # checkpointed so a hard failure resumes from here next run.
        for attempt in range(5):
            try:
                # Copy the member in bounded chunks: each src.read(chunk) is a
                # separate small range request, so a slow proxy can't blow the
                # per-read timeout (the failure mode of one giant 80 MB read).
                with zf.open(p) as src, open(local, "wb") as dst:
                    while True:
                        buf = src.read(8 * 1024 * 1024)
                        if not buf:
                            break
                        dst.write(buf)
                if os.path.getsize(local) != p.file_size:   # byte-size verification
                    raise RuntimeError(f"size {os.path.getsize(local)} != {p.file_size}")
                pf = pq.ParquetFile(local)
                aggs_part = build_aggs()                     # fold into a scratch first
                for rg in range(pf.num_row_groups):          # row-group iter: bounded memory
                    fold_table(aggs_part, pf.read_row_group(rg))
                break
            except (zipfile.BadZipFile, RuntimeError, OSError) as e:
                if attempt == 4:
                    raise RuntimeError(f"partition {p.filename} failed after retries: {e}")
                continue
            finally:
                if os.path.exists(local):
                    os.remove(local)                         # cleanup → bounded disk
        for k, v in aggs_part.items():                       # commit only on success
            aggs[k].merge(v)
        bytes_streamed += p.file_size
        done.add(p.filename)
        if (i + 1) % checkpoint_every == 0:            # periodic failure-safe checkpoint
            _checkpoint()

    _checkpoint()                                      # always checkpoint final state
    n_done = len(done)
    complete = n_done == total
    label = FULL_TRACE_EXACT if complete else SUBSET_TRACE
    return StreamResult(
        archive=src_id, n_partitions_total=total, n_partitions_done=n_done,
        artifacts={k: v.to_dict() for k, v in aggs.items()},
        label=label, bytes_streamed=bytes_streamed)


def _fold_local_parquet(path: str, build_aggs, fold_table) -> dict:
    """Fold one local parquet file (row-group by row-group, bounded memory) into a
    fresh aggregator set and return its serialized state. Pure/standalone (no
    globals, no network) so it is unit-testable and identical to serial folding."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    ap = build_aggs()
    for rg in range(pf.num_row_groups):
        fold_table(ap, pf.read_row_group(rg))
    return {k: v.state() for k, v in ap.items()}


# Per-process state for the multiprocessing pool: each worker opens its OWN
# HttpRangeFile/ZipFile once (central dir read) and reuses it for every task.
_MP: dict = {}


def _mp_init(archive_url, size, build_aggs, fold_table, work_dir, read_chunk, timeout):
    _MP.update(
        zf=zipfile.ZipFile(HttpRangeFile(archive_url, size, timeout=timeout)),
        byname=None, build=build_aggs, fold=fold_table, work=work_dir,
        chunk=read_chunk)
    _MP["byname"] = {i.filename: i for i in _MP["zf"].infolist()}


def _mp_download_fold(pname: str):
    """Worker task: download ONE whole partition (one bandwidth-bound curl) and
    fold it with the identical exact aggregators → (pname, state, file_size).
    Download (I/O) + fold (CPU) run together across processes, so oversubscribing
    workers overlaps the two and breaks the single-thread fold ceiling."""
    g = _MP
    p = g["byname"][pname]
    safe = pname.replace("/", "_").replace("=", "")
    dest = os.path.join(g["work"], f"mp_{safe}")
    last = None
    for attempt in range(5):
        try:
            with g["zf"].open(p) as src, open(dest, "wb") as d:
                while True:
                    buf = src.read(g["chunk"])
                    if not buf:
                        break
                    d.write(buf)
            if os.path.getsize(dest) != p.file_size:
                raise RuntimeError(f"size {os.path.getsize(dest)} != {p.file_size}")
            state = _fold_local_parquet(dest, g["build"], g["fold"])
            os.remove(dest)
            return pname, state, p.file_size
        except (zipfile.BadZipFile, RuntimeError, OSError) as e:
            last = e
            if os.path.exists(dest):
                os.remove(dest)
    raise RuntimeError(f"partition {pname} failed after retries: {last}")


def stream_archive_parallel(
    archive_url: str, build_aggs, fold_table, *,
    work_dir: str, manifest_path: str, workers: int = 8,
    checkpoint_every: int = 20, max_partitions: int | None = None,
    read_chunk: int = 128 * 1024 * 1024, timeout: int = 300,
) -> StreamResult:
    """Range-stream an archive with a POOL OF WORKER PROCESSES.

    pod_hourly (351 GB) has TWO bottlenecks: per-partition download (fixed best by
    fetching each whole partition in one curl, not many small chunks) and folding,
    which is CPU-bound Python — the GIL serialises threads at ~1 partition/12 s. So
    each worker is a *process* that downloads one whole partition and folds it with
    the SAME exact aggregators (bit-for-bit identical to serial folding); the main
    process merges the returned per-partition state (order-independent →
    FULL_TRACE_EXACT) and checkpoints. Oversubscribe ``workers`` past the core
    count so I/O downloads overlap CPU folds. Bounded disk (≈ workers partitions in
    flight); resumable from the manifest.
    """
    import multiprocessing as mp

    os.makedirs(work_dir, exist_ok=True)
    size = head_size(archive_url)
    main_zf = zipfile.ZipFile(HttpRangeFile(archive_url, size))
    parts = [i for i in main_zf.infolist() if i.filename.endswith(".parquet")]
    total = len(parts)

    aggs = build_aggs()
    done: set = set()
    bytes_streamed = 0
    if os.path.exists(manifest_path):
        m = json.load(open(manifest_path))
        done = set(m.get("processed", []))
        bytes_streamed = m.get("bytes_streamed", 0)
        for name, agg in aggs.items():
            if name in m.get("state", {}):
                agg.__dict__.update(type(agg).from_state(m["state"][name]).__dict__)

    todo = [p.filename for p in parts if p.filename not in done]
    if max_partitions is not None:
        todo = todo[:max_partitions]

    def _checkpoint():
        _atomic_write_json(manifest_path, {
            "archive": archive_url, "processed": sorted(done),
            "bytes_streamed": bytes_streamed, "n_partitions_total": total,
            "state": {k: v.state() for k, v in aggs.items()}})

    processed = 0
    if todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            workers, initializer=_mp_init,
            initargs=(archive_url, size, build_aggs, fold_table,
                      work_dir, read_chunk, timeout),
        ) as pool:
            for pname, state, fsize in pool.imap_unordered(_mp_download_fold, todo):
                for k, st in state.items():            # merge exact per-partition state
                    aggs[k].merge(type(aggs[k]).from_state(st))
                done.add(pname)
                bytes_streamed += fsize
                processed += 1
                if processed % checkpoint_every == 0:
                    _checkpoint()

    _checkpoint()
    n_done = len(done)
    label = FULL_TRACE_EXACT if n_done == total else SUBSET_TRACE
    return StreamResult(
        archive=archive_url, n_partitions_total=total, n_partitions_done=n_done,
        artifacts={k: v.to_dict() for k, v in aggs.items()},
        label=label, bytes_streamed=bytes_streamed)


__all__ = [
    "OSS_BASE", "ARCHIVES", "HttpRangeFile", "head_size", "list_partitions",
    "ExactStats", "ExactHistogram", "ExactCounter", "StreamResult",
    "stream_archive", "stream_local_zip", "stream_archive_prefetched",
    "stream_archive_parallel", "PREFETCH_MAX_BYTES",
]
