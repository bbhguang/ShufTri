"""Shared, paper-faithful plumbing for the ShufTri experiment programs.

The functions in this module deliberately separate three things which were
implemented by the experiment programs:

* generation of the *directed* PSI-CA schedule specified by the paper;
* execution of that schedule by the native Ristretto255 backend; and
* the public, reproducible randomness used by the DP experiment.

The native backend samples its cryptographic scalars and the Round-2 private
permutation from libsodium's CSPRNG.  Those secret coins are never derived
from, or written to, the experiment seed ledger.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


POINT_BYTES = 32
SHUFTRI_SHUFFLER_PAYLOAD_BYTES = 8
SHUFTRI_PLUS_SHUFFLER_PAYLOAD_BYTES = 16
DEFAULT_EPSILON = 1.0
DEFAULT_DELTA = 1e-8


@dataclass(frozen=True)
class GraphData:
    """A validated simple, undirected graph."""

    source_path: str
    source_sha256: str
    declared_nodes: int | None
    node_ids: tuple[int, ...]
    adjacency: Mapping[int, frozenset[int]]
    edge_count: int
    removed_self_loops: int
    duplicate_edge_rows: int

    @property
    def n(self) -> int:
        return len(self.node_ids)

    @property
    def directed_incidence_count(self) -> int:
        return sum(len(self.adjacency[u]) for u in self.node_ids)


@dataclass(frozen=True)
class DirectedSession:
    """One paper-level PSI call; direction is semantically significant."""

    session_id: int
    initiator: int
    responder: int
    workload: str


@dataclass(frozen=True)
class BudgetCalibration:
    target_epsilon: float
    delta: float
    theta_sh: float
    epsilon_raw: float
    sampling_probability: float
    epsilon_effective: float
    amplified_epsilon: float


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_int_pair(row: Sequence[str]) -> tuple[int, int] | None:
    if len(row) < 2:
        return None
    try:
        return int(row[0].strip()), int(row[1].strip())
    except ValueError:
        return None


def _is_recognized_edge_header(row: Sequence[str]) -> bool:
    if len(row) < 2:
        return False
    left, right = row[0].strip().lower(), row[1].strip().lower()
    return (left, right) in {
        ("node", "node"),
        ("u", "v"),
        ("source", "target"),
        ("src", "dst"),
        ("from", "to"),
    }


def read_graph(path: os.PathLike[str] | str) -> GraphData:
    """Read either the paper dataset format or an ordinary two-column CSV.

    The paper-format file begins with ``#nodes``, the declared node count on the next
    line, and a CSV header on the third line.  Ordinary ``u,v`` fixtures are
    accepted too.  Repeated/reverse-repeated edges are de-duplicated, while
    self-loops are counted and removed.  Node identifiers must be non-negative
    integers because the native backend uses a canonical uint64 encoding.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.reader(handle) if row and any(x.strip() for x in row)]
    if not rows:
        raise ValueError(f"empty graph file: {source}")

    declared_nodes: int | None = None
    start = 0
    if rows[0][0].strip().lower() == "#nodes":
        if len(rows) < 2:
            raise ValueError("#nodes marker is missing its node count")
        try:
            declared_nodes = int(rows[1][0].strip())
        except ValueError as exc:
            raise ValueError("invalid declared node count") from exc
        if declared_nodes < 0:
            raise ValueError("declared node count must be non-negative")
        start = 2

    edges: set[tuple[int, int]] = set()
    observed: set[int] = set()
    removed_self_loops = 0
    duplicate_edge_rows = 0
    data_rows = rows[start:]
    if data_rows and _is_recognized_edge_header(data_rows[0]):
        data_rows = data_rows[1:]
    for row_number, row in enumerate(data_rows, start=start + 1):
        pair = _parse_int_pair(row)
        if pair is None:
            raise ValueError(f"malformed edge row {row_number}: {row!r}")
        u, v = pair
        if u < 0 or v < 0:
            raise ValueError("node identifiers must be non-negative integers")
        observed.update((u, v))
        if u == v:
            removed_self_loops += 1
            continue
        edge = (u, v) if u < v else (v, u)
        if edge in edges:
            duplicate_edge_rows += 1
        edges.add(edge)

    if declared_nodes is not None:
        if observed and max(observed) >= declared_nodes:
            raise ValueError(
                f"node id {max(observed)} is outside declared range "
                f"[0,{declared_nodes - 1}]"
            )
        node_ids = tuple(range(declared_nodes))
    else:
        node_ids = tuple(sorted(observed))
    if not node_ids:
        raise ValueError("graph contains no declared or observed nodes")

    mutable: dict[int, set[int]] = {u: set() for u in node_ids}
    for u, v in edges:
        mutable[u].add(v)
        mutable[v].add(u)
    adjacency = {u: frozenset(mutable[u]) for u in node_ids}

    incidence_count = sum(len(adjacency[u]) for u in node_ids)
    if incidence_count != 2 * len(edges):
        raise AssertionError("undirected graph invariant sum(degree)=2m failed")
    for u in node_ids:
        if any(u not in adjacency[v] for v in adjacency[u]):
            raise AssertionError("input graph is not symmetric after parsing")

    return GraphData(
        source_path=str(source),
        source_sha256=sha256_file(source),
        declared_nodes=declared_nodes,
        node_ids=node_ids,
        adjacency=adjacency,
        edge_count=len(edges),
        removed_self_loops=removed_self_loops,
        duplicate_edge_rows=duplicate_edge_rows,
    )


def degree_statistics(graph: GraphData) -> dict[str, object]:
    values = np.asarray([len(graph.adjacency[u]) for u in graph.node_ids], dtype=float)
    maximum = int(values.max(initial=0))
    max_nodes = [u for u in graph.node_ids if len(graph.adjacency[u]) == maximum]
    return {
        "minimum": int(values.min()),
        "median": float(np.quantile(values, 0.50, method="linear")),
        "p90": float(np.quantile(values, 0.90, method="linear")),
        "p95": float(np.quantile(values, 0.95, method="linear")),
        "p99": float(np.quantile(values, 0.99, method="linear")),
        "maximum": maximum,
        "maximum_degree_nodes": max_nodes,
        "mean": float(values.mean()),
    }


def count_undirected_triangles(graph: GraphData) -> int:
    """Count each triangle exactly once in the original undirected graph."""

    total = 0
    adjacency = graph.adjacency
    for u in graph.node_ids:
        for v in adjacency[u]:
            if v <= u:
                continue
            total += sum(1 for w in adjacency[u].intersection(adjacency[v]) if w > v)
    return total


def theta_shuffle(n: int, delta: float) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0,1)")
    return math.log(n / (16.0 * math.log(2.0 / delta)))


def shuffle_amplification(n: int, epsilon_local: float, delta: float) -> float:
    """The paper's Eq. (1), evaluated with stable elementary functions."""

    if epsilon_local < 0.0:
        raise ValueError("epsilon_local must be non-negative")
    exponent = math.exp(epsilon_local)
    ratio = math.tanh(epsilon_local / 2.0)
    inner = ratio * (
        8.0 * math.sqrt(exponent * math.log(4.0 / delta)) / math.sqrt(n)
        + 8.0 * exponent / n
    )
    return math.log1p(inner)


def invert_shuffle_amplification(
    n: int,
    target_epsilon: float,
    delta: float,
    *,
    iterations: int = 240,
) -> float:
    """Numerically invert Eq. (1) without silently clamping at ``theta_sh``.

    Algorithm 2 explicitly applies ``min(inverse, theta_sh)`` whereas
    Algorithm 3 needs the *unclamped* inverse in order for its sampling branch
    to be reachable.  This function therefore returns the raw inverse; callers
    apply the algorithm-specific rule.
    """

    if target_epsilon < 0.0:
        raise ValueError("target epsilon must be non-negative")
    if target_epsilon == 0.0:
        return 0.0
    lo = 0.0
    hi = max(1.0, theta_shuffle(n, delta), target_epsilon)
    while shuffle_amplification(n, hi, delta) < target_epsilon:
        hi *= 2.0
        if hi > 64.0:
            raise ArithmeticError("failed to bracket shuffle-amplification inverse")
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if shuffle_amplification(n, mid, delta) < target_epsilon:
            lo = mid
        else:
            hi = mid
    result = (lo + hi) / 2.0
    reconstructed = shuffle_amplification(n, result, delta)
    if not math.isclose(reconstructed, target_epsilon, rel_tol=2e-12, abs_tol=2e-12):
        raise ArithmeticError("shuffle-amplification inversion did not converge")
    return result


def calibrate_shuftri_budget(n: int, epsilon: float, delta: float) -> BudgetCalibration:
    raw = invert_shuffle_amplification(n, epsilon, delta)
    threshold = theta_shuffle(n, delta)
    if threshold <= 0.0:
        raise ValueError(
            "shuffle-amplification precondition is infeasible for this small "
            "n/delta because theta_sh <= 0; schedule helpers remain valid, "
            "but no privacy-calibrated full run is claimed"
        )
    effective = min(raw, threshold)
    return BudgetCalibration(
        target_epsilon=epsilon,
        delta=delta,
        theta_sh=threshold,
        epsilon_raw=raw,
        sampling_probability=1.0,
        epsilon_effective=effective,
        amplified_epsilon=shuffle_amplification(n, effective, delta),
    )


def calibrate_wedge_budget(n: int, epsilon: float, delta: float) -> BudgetCalibration:
    raw = invert_shuffle_amplification(n, epsilon, delta)
    threshold = theta_shuffle(n, delta)
    if threshold <= 0.0:
        raise ValueError(
            "shuffle-amplification precondition is infeasible for this small "
            "n/delta because theta_sh <= 0; schedule helpers remain valid, "
            "but no privacy-calibrated full run is claimed"
        )
    if raw <= threshold:
        probability = 1.0
    else:
        probability = math.expm1(threshold) / math.expm1(raw)
    probability = min(1.0, max(0.0, probability))
    effective = math.log1p(probability * math.expm1(raw))
    if effective > threshold + 2e-12:
        raise AssertionError("WedgeCalib produced epsilon_local above theta_sh")
    return BudgetCalibration(
        target_epsilon=epsilon,
        delta=delta,
        theta_sh=threshold,
        epsilon_raw=raw,
        sampling_probability=probability,
        epsilon_effective=effective,
        amplified_epsilon=shuffle_amplification(n, effective, delta),
    )


def floor_linear_quantile(values: Sequence[float] | np.ndarray, q: float = 0.90) -> int:
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0,1]")
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot take a quantile of an empty sequence")
    return math.floor(float(np.quantile(array, q, method="linear")))


def estimate_projection_bound(
    graph: GraphData,
    epsilon_tau_local: float,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray, dict[str, object]]:
    """Algorithm 2 with disclosed valid-range handling for noisy ``tau``."""

    if epsilon_tau_local <= 0.0:
        raise ValueError("epsilon_tau_local must be positive")
    degrees = np.asarray([len(graph.adjacency[u]) for u in graph.node_ids], dtype=float)
    noisy = degrees + rng.laplace(0.0, 1.0 / epsilon_tau_local, graph.n)
    raw_tau = floor_linear_quantile(noisy, 0.90)
    maximum_valid = max(1, graph.n - 1)
    tau = min(max(raw_tau, 1), maximum_valid)
    handling = {
        "raw_floor_q90": raw_tau,
        "released_tau": tau,
        "clamped": tau != raw_tau,
        "clamp_rule": "max(1,min(raw_tau,n-1)); paper does not specify edge cases",
        "quantile_method": "NumPy linear",
        "laplace_sensitivity": 1.0,
        "laplace_scale": 1.0 / epsilon_tau_local,
    }
    return tau, noisy, handling


def project_local_neighbors(
    graph: GraphData,
    tau: int,
    rng: np.random.Generator,
) -> dict[int, frozenset[int]]:
    """Independently project each local adjacency set, without mutualizing."""

    if tau < 1:
        raise ValueError("tau must be positive")
    projected: dict[int, frozenset[int]] = {}
    for u in graph.node_ids:
        ordered = np.asarray(sorted(graph.adjacency[u]), dtype=np.int64)
        if ordered.size <= tau:
            retained = ordered.tolist()
        else:
            retained = rng.choice(ordered, size=tau, replace=False).tolist()
        projected[u] = frozenset(int(v) for v in retained)
    return projected


def generate_shuftri_sessions(
    graph: GraphData,
    *,
    first_session_id: int = 0,
) -> list[DirectedSession]:
    sessions: list[DirectedSession] = []
    session_id = first_session_id
    for initiator in graph.node_ids:
        for responder in sorted(graph.adjacency[initiator]):
            sessions.append(DirectedSession(session_id, initiator, responder, "ShufTri"))
            session_id += 1
    if len(sessions) != 2 * graph.edge_count:
        raise AssertionError("ShufTri must schedule exactly 2m directed sessions")
    return sessions


def generate_shuftriplus_sessions(
    projected: Mapping[int, frozenset[int]],
    probability: float,
    rng: np.random.Generator,
    *,
    first_session_id: int = 0,
) -> list[DirectedSession]:
    """Sample every *directed* projected incidence independently."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0,1]")
    sessions: list[DirectedSession] = []
    session_id = first_session_id
    for initiator in sorted(projected):
        for responder in sorted(projected[initiator]):
            selected = bool(rng.random() < probability)
            if selected:
                sessions.append(
                    DirectedSession(session_id, initiator, responder, "ShufTriPlus")
                )
            session_id += 1
    return sessions


def theoretical_group_payload_bytes(
    session: DirectedSession,
    local_sets: Mapping[int, frozenset[int]],
) -> int:
    d_i = len(local_sets[session.initiator])
    d_j = len(local_sets[session.responder])
    return POINT_BYTES * (2 * d_i + d_j)


def derive_public_seeds(master_seed: int) -> dict[str, int]:
    """Derive only non-secret experiment seeds from a documented master seed."""

    labels = (
        "degree_noise",
        "degree_shuffle",
        "projection",
        "poisson_sampling",
        "wedge_noise",
        "wedge_shuffle",
        "backend_order",
    )
    sequence = np.random.SeedSequence(master_seed)
    children = sequence.spawn(len(labels))
    return {
        label: int(child.generate_state(1, dtype=np.uint64)[0])
        for label, child in zip(labels, children)
    }


def write_sets_csv(path: os.PathLike[str] | str, local_sets: Mapping[int, frozenset[int]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("owner", "item"))
        for owner in sorted(local_sets):
            for item in sorted(local_sets[owner]):
                writer.writerow((owner, item))


def write_sessions_csv(path: os.PathLike[str] | str, sessions: Sequence[DirectedSession]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("session_id", "initiator", "responder", "workload"))
        for item in sessions:
            writer.writerow((item.session_id, item.initiator, item.responder, item.workload))


def write_projection_csv(
    path: os.PathLike[str] | str,
    graph: GraphData,
    projected: Mapping[int, frozenset[int]],
    noisy_degrees: Sequence[float],
) -> None:
    destination = Path(path)
    rows = (
        {
            "node": u,
            "original_degree": len(graph.adjacency[u]),
            "noisy_degree": float(noisy_degree),
            "projected_degree": len(projected[u]),
            "projected_neighbors": ";".join(str(v) for v in sorted(projected[u])),
        }
        for u, noisy_degree in zip(graph.node_ids, noisy_degrees)
    )
    _write_rows(
        destination,
        ("node", "original_degree", "noisy_degree", "projected_degree", "projected_neighbors"),
        rows,
    )


def _write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


NATIVE_RAW_FIELDS = (
    "task_index", "schedule_position", "session_id", "workload", "rep",
    "initiator", "responder", "d_i", "d_j", "cardinality",
    "plaintext_cardinality", "correct", "status", "latency_ms",
    "scalar_rng_ms", "hash_to_group_ms", "initiator_blind_ms",
    "request_serialize_ms", "responder_parse_ms", "responder_compute_ms",
    "responder_shuffle_ms", "response_serialize_ms", "initiator_parse_ms",
    "initiator_finalize_ms", "matching_ms", "group_payload_bytes",
    "framing_overhead_bytes", "round1_serialized_bytes",
    "round2_serialized_bytes", "total_serialized_bytes", "allocation_bytes",
    "rss_before_bytes", "rss_after_bytes", "process_peak_rss_bytes",
)

NATIVE_SUMMARY_FIELDS = (
    "backend_version", "libsodium_version", "group", "hash_to_group",
    "frame_version", "threads", "repetitions", "warmup_per_session",
    "order_seed", "session_rows", "completed_calls", "wall_s",
    "throughput_calls_per_s", "latency_median_ms", "latency_p95_ms",
    "latency_max_ms", "baseline_rss_bytes", "peak_rss_bytes",
    "incremental_peak_rss_bytes", "network_transport",
)


def run_plaintext_oracle(
    local_sets: Mapping[int, frozenset[int]],
    sessions: Sequence[DirectedSession],
    raw_path: Path,
    summary_path: Path,
    *,
    repetitions: int,
    warmup: int,
    threads: int,
    order_seed: int,
) -> None:
    """Produce correctness results without pretending they are PSI measurements."""

    del warmup  # no timed warm-up exists for the non-cryptographic oracle
    rows: list[dict[str, object]] = []
    task_index = 0
    for repetition in range(repetitions):
        for position, session in enumerate(sessions):
            d_i = len(local_sets[session.initiator])
            d_j = len(local_sets[session.responder])
            cardinality = len(
                local_sets[session.initiator].intersection(local_sets[session.responder])
            )
            row = {field: "" for field in NATIVE_RAW_FIELDS}
            row.update(
                {
                    "task_index": task_index,
                    "schedule_position": position,
                    "session_id": session.session_id,
                    "workload": session.workload,
                    "rep": repetition,
                    "initiator": session.initiator,
                    "responder": session.responder,
                    "d_i": d_i,
                    "d_j": d_j,
                    "cardinality": cardinality,
                    "plaintext_cardinality": cardinality,
                    "correct": 1,
                    "status": 0,
                    # This is the paper's point-payload formula, not serialized bytes.
                    "group_payload_bytes": POINT_BYTES * (2 * d_i + d_j),
                }
            )
            rows.append(row)
            task_index += 1
    _write_rows(raw_path, NATIVE_RAW_FIELDS, rows)
    summary = {field: "" for field in NATIVE_SUMMARY_FIELDS}
    summary.update(
        {
            "backend_version": "plaintext-correctness-oracle-NOT-PSI",
            "group": "none",
            "hash_to_group": "none",
            "threads": threads,
            "repetitions": repetitions,
            "warmup_per_session": 0,
            "order_seed": order_seed,
            "session_rows": len(sessions),
            "completed_calls": len(rows),
            "network_transport": "none",
        }
    )
    _write_rows(summary_path, NATIVE_SUMMARY_FIELDS, (summary,))
    json_dump(
        raw_path.parent / "backend_execution.json",
        {
            "backend": "plaintext-correctness-oracle-NOT-PSI",
            "command": None,
            "exit_code": 0,
            "stdout_log": None,
            "stderr_log": None,
            "warning": "No cryptographic latency or serialized-byte measurement exists.",
        },
    )


def run_native_backend(
    executable: os.PathLike[str] | str,
    sets_path: Path,
    sessions_path: Path,
    raw_path: Path,
    summary_path: Path,
    *,
    threads: int,
    repetitions: int,
    warmup: int,
    order_seed: int,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    binary = Path(executable).expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(
            f"native PSI backend not found: {binary}; build it with `make -C src`"
        )
    command = [
        str(binary), "batch", str(sets_path), str(sessions_path), str(raw_path),
        str(summary_path), str(threads), str(repetitions), str(warmup),
        str(order_seed),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    json_dump(
        raw_path.parent / "backend_execution.json",
        {
            "backend": "native-ristretto255",
            "command": command,
            "exit_code": completed.returncode,
            "wall_seconds_observed_by_python": elapsed,
            "stdout_log": stdout_path.name,
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_log": stderr_path.name,
            "stderr_sha256": sha256_file(stderr_path),
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"native PSI backend failed with exit code {completed.returncode}; "
            f"see {stderr_path}"
        )


def read_csv_rows(path: os.PathLike[str] | str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def session_cardinalities(raw_rows: Sequence[Mapping[str, str]]) -> dict[int, int]:
    grouped: dict[int, set[int]] = defaultdict(set)
    for row in raw_rows:
        if int(row["status"]) != 0 or int(row["correct"]) != 1:
            raise RuntimeError(f"PSI correctness/status failure in session {row['session_id']}")
        grouped[int(row["session_id"])].add(int(row["cardinality"]))
    result: dict[int, int] = {}
    for session_id, values in grouped.items():
        if len(values) != 1:
            raise RuntimeError(f"non-deterministic PSI cardinality in session {session_id}")
        result[session_id] = next(iter(values))
    return result


def _numeric_values(rows: Sequence[Mapping[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field, "")
        if value is None or value == "":
            continue
        values.append(float(value))
    return values


def write_metric_summary(
    path: Path,
    raw_rows: Sequence[Mapping[str, str]],
    *,
    backend: str,
) -> None:
    specifications = (
        ("latency_ms", "ms"),
        ("group_payload_bytes", "bytes"),
        ("framing_overhead_bytes", "bytes"),
        ("round1_serialized_bytes", "bytes"),
        ("round2_serialized_bytes", "bytes"),
        ("total_serialized_bytes", "bytes"),
        ("allocation_bytes", "bytes"),
        ("process_peak_rss_bytes", "bytes"),
    )
    rows: list[dict[str, object]] = []
    for metric, unit in specifications:
        values = _numeric_values(raw_rows, metric)
        if not values:
            continue
        array = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "unit": unit,
                "count": len(values),
                "median": float(np.quantile(array, 0.50, method="linear")),
                "p95": float(np.quantile(array, 0.95, method="linear")),
                "max": float(array.max()),
                "evidence": (
                    "measured by native Ristretto255 backend"
                    if backend == "ristretto255"
                    else "analytical/plaintext only; not a cryptographic measurement"
                ),
            }
        )
    _write_rows(
        path,
        ("metric", "unit", "count", "median", "p95", "max", "evidence"),
        rows,
    )


def aggregate_session_values(
    sessions: Sequence[DirectedSession],
    raw_rows: Sequence[Mapping[str, str]],
    node_ids: Sequence[int],
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """Return per-initiator wedge, serialized, and point-payload totals.

    Repetitions are benchmark repeats of the same logical schedule.  The
    logical algorithm therefore consumes one cardinality and one message-size
    value per session, not a sum over repetitions.
    """

    cards = session_cardinalities(raw_rows)
    by_session: dict[int, list[Mapping[str, str]]] = defaultdict(list)
    for row in raw_rows:
        by_session[int(row["session_id"])].append(row)

    wedge = {u: 0 for u in node_ids}
    serialized = {u: 0 for u in node_ids}
    payload = {u: 0 for u in node_ids}
    for session in sessions:
        if session.session_id not in cards:
            raise RuntimeError(f"backend omitted session {session.session_id}")
        wedge[session.initiator] += cards[session.session_id]
        rows = by_session[session.session_id]
        payload_values = {int(row["group_payload_bytes"]) for row in rows}
        if len(payload_values) != 1:
            raise RuntimeError("group payload changed across repetitions")
        payload[session.initiator] += next(iter(payload_values))
        serialized_values = {
            int(row["total_serialized_bytes"])
            for row in rows
            if row.get("total_serialized_bytes", "") != ""
        }
        if len(serialized_values) > 1:
            raise RuntimeError("serialized size changed across repetitions")
        if serialized_values:
            serialized[session.initiator] += next(iter(serialized_values))
    return wedge, serialized, payload


def aggregate_role_metrics(
    sessions: Sequence[DirectedSession],
    raw_rows: Sequence[Mapping[str, str]],
    node_ids: Sequence[int],
) -> dict[str, dict[int, float]]:
    """Aggregate logical-call timing and physical outbound role metrics.

    The paper attributes the entire two-message session cost to its initiator.
    For deployment analysis we additionally report physical outbound traffic:
    Round 1 is sent by the initiator and Round 2 by the responder.  A median
    across benchmark repetitions represents one logical call.
    """

    by_session: dict[int, list[Mapping[str, str]]] = defaultdict(list)
    for row in raw_rows:
        by_session[int(row["session_id"])].append(row)

    fields = (
        "sessions_as_responder",
        "total_session_participations",
        "latency_ms_attributed_to_initiator",
        "round1_outbound_bytes_as_initiator",
        "round2_outbound_bytes_as_responder",
        "physical_role_outbound_bytes",
    )
    result = {field: {u: 0.0 for u in node_ids} for field in fields}
    for session in sessions:
        rows = by_session.get(session.session_id, [])
        if not rows:
            raise RuntimeError(f"backend omitted session {session.session_id}")
        result["sessions_as_responder"][session.responder] += 1
        result["total_session_participations"][session.initiator] += 1
        result["total_session_participations"][session.responder] += 1

        latencies = _numeric_values(rows, "latency_ms")
        if latencies:
            result["latency_ms_attributed_to_initiator"][session.initiator] += float(
                np.quantile(np.asarray(latencies), 0.50, method="linear")
            )
        request = _numeric_values(rows, "round1_serialized_bytes")
        response = _numeric_values(rows, "round2_serialized_bytes")
        if request:
            request_bytes = float(np.quantile(np.asarray(request), 0.50, method="linear"))
            result["round1_outbound_bytes_as_initiator"][session.initiator] += request_bytes
            result["physical_role_outbound_bytes"][session.initiator] += request_bytes
        if response:
            response_bytes = float(np.quantile(np.asarray(response), 0.50, method="linear"))
            result["round2_outbound_bytes_as_responder"][session.responder] += response_bytes
            result["physical_role_outbound_bytes"][session.responder] += response_bytes
    return result


def write_per_user_csv(
    path: Path,
    *,
    node_ids: Sequence[int],
    local_sets: Mapping[int, frozenset[int]],
    candidate_counts: Mapping[int, int],
    executed_counts: Mapping[int, int],
    wedge: Mapping[int, int],
    noisy_wedge: Mapping[int, float],
    serialized: Mapping[int, int],
    group_payload: Mapping[int, int],
    shuffler_payload_bytes: int,
    role_metrics: Mapping[str, Mapping[int, float]],
) -> None:
    fields = (
        "node", "local_set_size", "candidate_directed_sessions",
        "executed_directed_sessions", "wedge_sum", "noisy_wedge_sum",
        "psi_group_payload_bytes_attributed_to_initiator",
        "psi_serialized_bytes_attributed_to_initiator",
        "psi_latency_ms_attributed_to_initiator",
        "psi_sessions_as_responder", "psi_total_session_participations",
        "round1_outbound_bytes_as_initiator",
        "round2_outbound_bytes_as_responder", "physical_role_outbound_bytes",
        "shuffler_numeric_payload_bytes",
    )
    rows = (
        {
            "node": u,
            "local_set_size": len(local_sets[u]),
            "candidate_directed_sessions": candidate_counts[u],
            "executed_directed_sessions": executed_counts[u],
            "wedge_sum": wedge[u],
            "noisy_wedge_sum": noisy_wedge[u],
            "psi_group_payload_bytes_attributed_to_initiator": group_payload[u],
            "psi_serialized_bytes_attributed_to_initiator": serialized[u],
            "psi_latency_ms_attributed_to_initiator": role_metrics[
                "latency_ms_attributed_to_initiator"
            ][u],
            "psi_sessions_as_responder": int(role_metrics["sessions_as_responder"][u]),
            "psi_total_session_participations": int(
                role_metrics["total_session_participations"][u]
            ),
            "round1_outbound_bytes_as_initiator": role_metrics[
                "round1_outbound_bytes_as_initiator"
            ][u],
            "round2_outbound_bytes_as_responder": role_metrics[
                "round2_outbound_bytes_as_responder"
            ][u],
            "physical_role_outbound_bytes": role_metrics["physical_role_outbound_bytes"][u],
            "shuffler_numeric_payload_bytes": shuffler_payload_bytes,
        }
        for u in node_ids
    )
    _write_rows(path, fields, rows)


def write_key_value_summary(path: Path, rows: Iterable[tuple[str, object, str, str]]) -> None:
    _write_rows(
        path,
        ("metric", "value", "unit", "evidence"),
        (
            {"metric": metric, "value": value, "unit": unit, "evidence": evidence}
            for metric, value, unit, evidence in rows
        ),
    )


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact_hashes(directory: Path, names: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            result[name] = sha256_file(candidate)
    return result


def runtime_environment() -> dict[str, object]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def process_peak_rss_bytes() -> int:
    """Return this Python process's observed peak RSS in bytes."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS/BSD reports bytes.
    if sys.platform.startswith("linux"):
        value *= 1024
    return value


def backend_summary_row(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    if len(rows) != 1:
        raise RuntimeError(f"expected one backend summary row in {path}")
    return rows[0]


def session_count_by_initiator(
    sessions: Sequence[DirectedSession], node_ids: Sequence[int]
) -> dict[int, int]:
    result = {u: 0 for u in node_ids}
    for session in sessions:
        result[session.initiator] += 1
    return result


def make_output_directory(path: os.PathLike[str] | str) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def run_backend(
    *,
    backend: str,
    executable: Path,
    local_sets: Mapping[int, frozenset[int]],
    sessions: Sequence[DirectedSession],
    output_directory: Path,
    threads: int,
    repetitions: int,
    warmup: int,
    order_seed: int,
) -> tuple[Path, Path]:
    sets_path = output_directory / "sets.csv"
    sessions_path = output_directory / "directed_sessions.csv"
    raw_path = output_directory / "psi_raw.csv"
    summary_path = output_directory / "psi_backend_summary.csv"
    write_sets_csv(sets_path, local_sets)
    write_sessions_csv(sessions_path, sessions)

    if backend == "ristretto255":
        run_native_backend(
            executable,
            sets_path,
            sessions_path,
            raw_path,
            summary_path,
            threads=threads,
            repetitions=repetitions,
            warmup=warmup,
            order_seed=order_seed,
            stdout_path=output_directory / "backend_stdout.log",
            stderr_path=output_directory / "backend_stderr.log",
        )
    elif backend == "plaintext":
        run_plaintext_oracle(
            local_sets,
            sessions,
            raw_path,
            summary_path,
            repetitions=repetitions,
            warmup=warmup,
            threads=threads,
            order_seed=order_seed,
        )
    else:
        raise ValueError(f"unknown backend: {backend}")
    return raw_path, summary_path


def prepare_only_manifest(
    *,
    algorithm: str,
    graph: GraphData,
    output_directory: Path,
    parameters: Mapping[str, object],
    seeds: Mapping[str, int],
    local_sets: Mapping[int, frozenset[int]],
    sessions: Sequence[DirectedSession],
) -> None:
    manifest = {
        "schema_version": 1,
        "status": "schedule_prepared_backend_not_executed",
        "algorithm": algorithm,
        "input": {
            "path": graph.source_path,
            "sha256": graph.source_sha256,
            "nodes": graph.n,
            "edges": graph.edge_count,
        },
        "parameters": dict(parameters),
        "seeds": {
            "public_experiment_seeds": dict(seeds),
            "cryptographic_coins": "not generated in prepare-only mode",
        },
        "schedule": {
            "local_set_elements": sum(len(value) for value in local_sets.values()),
            "directed_sessions": len(sessions),
        },
        "environment": runtime_environment(),
        "artifacts_sha256": artifact_hashes(
            output_directory, ("sets.csv", "directed_sessions.csv")
        ),
    }
    json_dump(output_directory / "manifest.json", manifest)


def dataclass_dict(value: object) -> dict[str, object]:
    return asdict(value)  # narrow wrapper keeps callers readable
