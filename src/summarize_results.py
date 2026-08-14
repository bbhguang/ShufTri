#!/usr/bin/env python3
"""Derive reproducible summaries from preserved PSI experiment CSV files.

The script never substitutes a latency model for a missing measurement.  It
uses NumPy-compatible linear interpolation for percentiles, but intentionally
depends only on the Python standard library so the archived results can be
reduced on a clean machine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REQUIRED_RUNS_PER_EXPERIMENT = 20


RAW_COLUMNS = {
    "workload",
    "initiator",
    "latency_ms",
    "total_serialized_bytes",
    "status",
    "correct",
}
BACKEND_SUMMARY_COLUMNS = {
    "threads",
    "completed_calls",
    "wall_s",
    "throughput_calls_per_s",
    "peak_rss_bytes",
}
INDEX_COLUMNS = {
    "category",
    "algorithm",
    "run_id",
    "threads",
    "seed",
    "raw_csv",
    "backend_summary_csv",
    "result_summary_csv",
    "per_user_csv",
    "status",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build measured/exact/derived CSV tables from secure PSI raw "
            "results; missing experiment families produce header-only tables."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
        help="experiment results root (default: PACKAGE/generated_results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="derived-table directory (default: RESULTS_DIR/derived)",
    )
    parser.add_argument(
        "--run-index",
        type=Path,
        default=None,
        help="optional run_index.csv written by run_experiments.py",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "exit nonzero on correctness/accounting failures or incomplete "
            f"{REQUIRED_RUNS_PER_EXPERIMENT}-run experiment cells"
        ),
    )
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_fields(path: Path) -> set[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            return set(next(reader, []))
    except (OSError, UnicodeDecodeError, csv.Error):
        return set()


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def as_float(value: object, default: float = math.nan) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite(values: Iterable[object]) -> list[float]:
    result: list[float] = []
    for value in values:
        number = as_float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def quantile_linear(values: Iterable[object], q: float) -> float | str:
    ordered = sorted(finite(values))
    if not ordered:
        return ""
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def maximum(values: Iterable[object]) -> float | str:
    numbers = finite(values)
    return max(numbers) if numbers else ""


def minimum(values: Iterable[object]) -> float | str:
    numbers = finite(values)
    return min(numbers) if numbers else ""


def total(values: Iterable[object]) -> float:
    return sum(finite(values))


def render_number(value: object, suffix: str = "") -> str:
    number = as_float(value)
    return f"{number:.6g}{suffix}" if math.isfinite(number) else "not available"


def relpath(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def infer_category(path: Path) -> str:
    text = "/".join(path.parts).lower()
    if "micro" in text:
        return "micro"
    if "concurrency" in text or re.search(r"(?:^|[/_-])c(?:1|4|8)(?:[/_-]|$)", text):
        return "concurrency"
    if "e2e" in text or "end_to_end" in text or "full" in text:
        return "e2e"
    return "unknown"


def infer_algorithm(path: Path) -> str:
    text = "/".join(path.parts).lower()
    if "shuftri+" in text or "shuftri_plus" in text or "shuftriplus" in text:
        return "ShufTri+"
    if "shuftri" in text:
        return "ShufTri"
    return ""


def infer_run_id(path: Path) -> str:
    match = re.search(r"run[_-]?(\d+)", "/".join(path.parts).lower())
    return f"run_{int(match.group(1)):02d}" if match else path.parent.name


def resolve_index_path(value: str, results_dir: Path, index_path: Path) -> Path | None:
    value = value.strip()
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    from_results = results_dir / candidate
    if from_results.exists():
        return from_results
    return index_path.parent / candidate


def load_index(results_dir: Path, index_path: Path) -> list[dict[str, object]]:
    if not index_path.exists():
        return []
    fields = csv_fields(index_path)
    if not INDEX_COLUMNS.issubset(fields):
        missing = ", ".join(sorted(INDEX_COLUMNS - fields))
        raise ValueError(f"{index_path}: run index is missing columns: {missing}")
    entries: list[dict[str, object]] = []
    for row in read_csv(index_path):
        entry: dict[str, object] = dict(row)
        for key in ("raw_csv", "backend_summary_csv", "result_summary_csv", "per_user_csv"):
            entry[key] = resolve_index_path(row.get(key, ""), results_dir, index_path)
        entries.append(entry)
    return entries


def discover_entries(results_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    """Fallback discovery used when an interrupted run did not update its index."""
    entries: list[dict[str, object]] = []
    for path in sorted(results_dir.rglob("*.csv")):
        if output_dir == path or output_dir in path.parents or path.name == "run_index.csv":
            continue
        fields = csv_fields(path)
        kind = ""
        if RAW_COLUMNS.issubset(fields):
            kind = "raw_csv"
        elif BACKEND_SUMMARY_COLUMNS.issubset(fields):
            kind = "backend_summary_csv"
        elif path.name == "result_summary.csv":
            kind = "result_summary_csv"
        elif path.name == "per_user.csv":
            kind = "per_user_csv"
        if not kind:
            continue
        key = (infer_category(path), infer_algorithm(path), infer_run_id(path), path.parent)
        match = next((entry for entry in entries if entry["_key"] == key), None)
        if match is None:
            match = {
                "_key": key,
                "category": key[0],
                "algorithm": key[1],
                "run_id": key[2],
                "threads": "",
                "seed": "",
                "status": "discovered",
                "raw_csv": None,
                "backend_summary_csv": None,
                "result_summary_csv": None,
                "per_user_csv": None,
            }
            entries.append(match)
        match[kind] = path
    for entry in entries:
        entry.pop("_key", None)
    return entries


def valid_raw_rows(path: Path, audits: list[dict[str, object]]) -> tuple[list[dict[str, str]], int]:
    rows = read_csv(path)
    valid: list[dict[str, str]] = []
    invalid = 0
    byte_mismatches = 0
    for row in rows:
        correct = is_true(row.get("correct"))
        status_ok = as_int(row.get("status"), -1) == 0
        if correct and status_ok:
            valid.append(row)
        else:
            invalid += 1
        di = as_int(row.get("d_i"), -1)
        dj = as_int(row.get("d_j"), -1)
        serialized = as_int(row.get("total_serialized_bytes"), -1)
        if di >= 0 and dj >= 0 and serialized >= 0:
            expected = (2 * di + dj) * 32 + 64
            if serialized != expected:
                byte_mismatches += 1
    audits.append(
        {
            "check": "backend_raw_correctness",
            "artifact": str(path),
            "observed": len(rows) - invalid,
            "expected": len(rows),
            "pass": invalid == 0,
            "detail": f"invalid status/cardinality rows={invalid}",
        }
    )
    audits.append(
        {
            "check": "serialized_byte_identity",
            "artifact": str(path),
            "observed": len(rows) - byte_mismatches,
            "expected": len(rows),
            "pass": byte_mismatches == 0,
            "detail": "expected total=(2*d_i+d_j)*32 + two 32-byte frame headers",
        }
    )
    return valid, len(rows)


def backend_row(entry: Mapping[str, object]) -> dict[str, str] | None:
    path = entry.get("backend_summary_csv")
    if not isinstance(path, Path) or not path.exists():
        return None
    rows = read_csv(path)
    return rows[0] if rows else None


def stat_triplet(rows: Sequence[Mapping[str, object]], column: str, prefix: str) -> dict[str, object]:
    return {
        f"{prefix}_median": quantile_linear((row.get(column, "") for row in rows), 0.50),
        f"{prefix}_p95": quantile_linear((row.get(column, "") for row in rows), 0.95),
        f"{prefix}_max": maximum(row.get(column, "") for row in rows),
    }


def summarize_micro(
    entries: Sequence[Mapping[str, object]],
    results_dir: Path,
    output_dir: Path,
    audits: list[dict[str, object]],
) -> list[dict[str, object]]:
    raw_rows: list[dict[str, str]] = []
    observed_rows = 0
    source_paths: list[Path] = []
    summaries: list[dict[str, str]] = []
    for entry in entries:
        if entry.get("category") != "micro":
            continue
        raw_path = entry.get("raw_csv")
        if isinstance(raw_path, Path) and raw_path.exists():
            valid, observed = valid_raw_rows(raw_path, audits)
            raw_rows.extend(valid)
            observed_rows += observed
            source_paths.append(raw_path)
        summary = backend_row(entry)
        if summary:
            summaries.append(summary)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        grouped[row.get("workload", "Unlabelled")].append(row)
    output_rows: list[dict[str, object]] = []
    order = {"common": 0, "medium": 1, "high": 2, "hub-tail": 3, "hub_tail": 3}
    for workload in sorted(grouped, key=lambda name: (order.get(name.lower(), 50), name)):
        group = grouped[workload]
        row: dict[str, object] = {
            "scope": "workload",
            "workload": workload,
            "observed_calls": len(group),
            "successful_calls": len(group),
            "batch_wall_s": "",
            "throughput_calls_per_s": "",
            "peak_rss_bytes": maximum(item.get("process_peak_rss_bytes", "") for item in group),
            "throughput_basis": "not separable from mixed-batch wall timer",
            "evidence": "derived from measured per-call rows; bytes are exact frame accounting",
        }
        row.update(stat_triplet(group, "latency_ms", "latency_ms"))
        row.update(stat_triplet(group, "total_serialized_bytes", "serialized_bytes"))
        row.update(stat_triplet(group, "allocation_bytes", "allocation_bytes"))
        output_rows.append(row)

    if raw_rows:
        completed = sum(as_int(row.get("completed_calls")) for row in summaries)
        wall = total(row.get("wall_s") for row in summaries)
        row = {
            "scope": "overall",
            "workload": "Overall",
            "observed_calls": observed_rows,
            "successful_calls": len(raw_rows),
            "batch_wall_s": wall if summaries else "",
            "throughput_calls_per_s": completed / wall if summaries and wall > 0 else "",
            "peak_rss_bytes": maximum(row.get("peak_rss_bytes", "") for row in summaries),
            "throughput_basis": "sum(completed_calls)/sum(measured backend wall_s)",
            "evidence": "derived from measured per-call rows and measured batch wall timer",
        }
        row.update(stat_triplet(raw_rows, "latency_ms", "latency_ms"))
        row.update(stat_triplet(raw_rows, "total_serialized_bytes", "serialized_bytes"))
        row.update(stat_triplet(raw_rows, "allocation_bytes", "allocation_bytes"))
        output_rows.append(row)

    fields = [
        "scope", "workload", "observed_calls", "successful_calls",
        "latency_ms_median", "latency_ms_p95", "latency_ms_max",
        "serialized_bytes_median", "serialized_bytes_p95", "serialized_bytes_max",
        "allocation_bytes_median", "allocation_bytes_p95", "allocation_bytes_max",
        "batch_wall_s", "throughput_calls_per_s", "peak_rss_bytes",
        "throughput_basis", "evidence",
    ]
    write_csv(output_dir / "psi_micro_summary.csv", fields, output_rows)

    phase_definitions: list[tuple[str, tuple[str, ...], str]] = [
        ("scalar_rng", ("scalar_rng_ms",), "both parties' fresh scalar generation"),
        ("hash_to_group", ("hash_to_group_ms",), "both input sets"),
        ("initiator_blind", ("initiator_blind_ms",), "Ristretto scalar multiplication"),
        ("request_serialize", ("request_serialize_ms",), "application frame construction"),
        ("responder_parse", ("responder_parse_ms",), "frame and point validation"),
        ("responder_compute", ("responder_compute_ms",), "reblind request and evaluate responder set"),
        ("responder_shuffle", ("responder_shuffle_ms",), "two CSPRNG Fisher-Yates permutations"),
        ("response_serialize", ("response_serialize_ms",), "application frame construction"),
        ("initiator_parse", ("initiator_parse_ms",), "frame and point validation"),
        ("initiator_finalize", ("initiator_finalize_ms",), "final Ristretto scalar multiplication"),
        ("token_matching", ("matching_ms",), "sort and exact token matching"),
        (
            "serialization_and_parsing_total",
            ("request_serialize_ms", "responder_parse_ms", "response_serialize_ms", "initiator_parse_ms"),
            "derived sum of the four measured framing/parse phase timers",
        ),
        ("complete_session", ("latency_ms",), "complete timed local session including cleanup"),
    ]
    phase_rows: list[dict[str, object]] = []
    for phase, columns, detail in phase_definitions:
        values = [sum(as_float(row.get(column), 0.0) for column in columns) for row in raw_rows]
        if not values:
            continue
        phase_rows.append(
            {
                "phase": phase,
                "measured_calls": len(values),
                "latency_ms_median": quantile_linear(values, 0.50),
                "latency_ms_p95": quantile_linear(values, 0.95),
                "latency_ms_max": max(values),
                "detail": detail,
                "evidence": (
                    "derived sum of measured phase timers"
                    if len(columns) > 1 else "measured monotonic-clock phase timer"
                ),
            }
        )
    write_csv(
        output_dir / "psi_phase_summary.csv",
        ["phase", "measured_calls", "latency_ms_median", "latency_ms_p95",
         "latency_ms_max", "detail", "evidence"],
        phase_rows,
    )
    return output_rows


def summarize_concurrency(
    entries: Sequence[Mapping[str, object]],
    output_dir: Path,
    audits: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_rows: list[dict[str, object]] = []
    pooled_raw: dict[int, list[dict[str, str]]] = defaultdict(list)
    for entry in entries:
        if entry.get("category") != "concurrency":
            continue
        summary = backend_row(entry)
        if not summary:
            continue
        threads = as_int(summary.get("threads") or entry.get("threads"), 0)
        row: dict[str, object] = {
            "run_id": entry.get("run_id", ""),
            "threads": threads,
            "seed": entry.get("seed", ""),
            "completed_calls": summary.get("completed_calls", ""),
            "wall_s": summary.get("wall_s", ""),
            "throughput_calls_per_s": summary.get("throughput_calls_per_s", ""),
            "latency_median_ms": summary.get("latency_median_ms", ""),
            "latency_p95_ms": summary.get("latency_p95_ms", ""),
            "latency_max_ms": summary.get("latency_max_ms", ""),
            "baseline_rss_bytes": summary.get("baseline_rss_bytes", ""),
            "peak_rss_bytes": summary.get("peak_rss_bytes", ""),
            "incremental_peak_rss_bytes": summary.get("incremental_peak_rss_bytes", ""),
            "evidence": "measured fresh-process backend run",
        }
        run_rows.append(row)
        raw_path = entry.get("raw_csv")
        if isinstance(raw_path, Path) and raw_path.exists():
            valid, _ = valid_raw_rows(raw_path, audits)
            pooled_raw[threads].extend(valid)

    run_rows.sort(key=lambda row: (as_int(row["threads"]), str(row["run_id"])))
    run_fields = [
        "run_id", "threads", "seed", "completed_calls", "wall_s",
        "throughput_calls_per_s", "latency_median_ms", "latency_p95_ms",
        "latency_max_ms", "baseline_rss_bytes", "peak_rss_bytes",
        "incremental_peak_rss_bytes", "evidence",
    ]
    write_csv(output_dir / "concurrency_runs.csv", run_fields, run_rows)

    aggregate_rows: list[dict[str, object]] = []
    by_threads: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        by_threads[as_int(row["threads"])].append(row)
    for threads in sorted(by_threads):
        runs = by_threads[threads]
        calls = sum(as_int(row["completed_calls"]) for row in runs)
        wall = total(row["wall_s"] for row in runs)
        raw = pooled_raw.get(threads, [])
        item: dict[str, object] = {
            "threads": threads,
            "fresh_runs": len(runs),
            "required_runs": REQUIRED_RUNS_PER_EXPERIMENT,
            "run_requirement_met": len(runs) == REQUIRED_RUNS_PER_EXPERIMENT,
            "completed_calls_total": calls,
            "wall_s_total": wall,
            "aggregate_throughput_calls_per_s": calls / wall if wall > 0 else "",
            "per_run_throughput_median": quantile_linear((row["throughput_calls_per_s"] for row in runs), 0.50),
            "per_run_throughput_p95": quantile_linear((row["throughput_calls_per_s"] for row in runs), 0.95),
            "per_run_throughput_max": maximum(row["throughput_calls_per_s"] for row in runs),
            "pooled_latency_median_ms": quantile_linear((row.get("latency_ms", "") for row in raw), 0.50),
            "pooled_latency_p95_ms": quantile_linear((row.get("latency_ms", "") for row in raw), 0.95),
            "pooled_latency_max_ms": maximum(row.get("latency_ms", "") for row in raw),
            "peak_rss_median_bytes": quantile_linear((row["peak_rss_bytes"] for row in runs), 0.50),
            "peak_rss_p95_bytes": quantile_linear((row["peak_rss_bytes"] for row in runs), 0.95),
            "peak_rss_max_bytes": maximum(row["peak_rss_bytes"] for row in runs),
            "incremental_peak_rss_max_bytes": maximum(row["incremental_peak_rss_bytes"] for row in runs),
            "evidence": "derived across measured fresh-process runs",
        }
        aggregate_rows.append(item)
        audits.append(
            {
                "check": "concurrency_required_fresh_runs",
                "artifact": f"threads={threads}",
                "observed": len(runs),
                "expected": REQUIRED_RUNS_PER_EXPERIMENT,
                "pass": len(runs) == REQUIRED_RUNS_PER_EXPERIMENT,
                "detail": (
                    f"protocol requires {REQUIRED_RUNS_PER_EXPERIMENT} fresh "
                    "processes for each concurrency level"
                ),
            }
        )
    aggregate_fields = [
        "threads", "fresh_runs", "required_runs", "run_requirement_met",
        "completed_calls_total",
        "wall_s_total", "aggregate_throughput_calls_per_s",
        "per_run_throughput_median", "per_run_throughput_p95", "per_run_throughput_max",
        "pooled_latency_median_ms", "pooled_latency_p95_ms", "pooled_latency_max_ms",
        "peak_rss_median_bytes", "peak_rss_p95_bytes", "peak_rss_max_bytes",
        "incremental_peak_rss_max_bytes", "evidence",
    ]
    write_csv(output_dir / "concurrency_summary.csv", aggregate_fields, aggregate_rows)
    return run_rows, aggregate_rows


def pick(row: Mapping[str, object], *names: str) -> object:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def read_result_summary(path: Path) -> dict[str, object]:
    """Accept the algorithm CLIs' metric,value table and older one-row tables."""
    rows = read_csv(path)
    if not rows:
        return {}
    if {"metric", "value"}.issubset(rows[0]):
        return {
            row["metric"]: row.get("value", "")
            for row in rows
            if row.get("metric", "")
        }
    return dict(rows[0])


def read_manifest_parameters(run_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}, {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {}
    parameters = document.get("parameters", {})
    return (
        parameters if isinstance(parameters, dict) else {},
        document if isinstance(document, dict) else {},
    )


def summarize_e2e(
    entries: Sequence[Mapping[str, object]],
    output_dir: Path,
    audits: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_rows: list[dict[str, object]] = []
    per_user_run_rows: list[dict[str, object]] = []
    pooled_users: dict[str, list[dict[str, float]]] = defaultdict(list)
    for entry in entries:
        if entry.get("category") != "e2e":
            continue
        result_path = entry.get("result_summary_csv")
        result: dict[str, object] = {}
        if isinstance(result_path, Path) and result_path.exists():
            result = read_result_summary(result_path)
        backend = backend_row(entry) or {}
        run_dir = result_path.parent if isinstance(result_path, Path) else Path(".")
        parameters, manifest = read_manifest_parameters(run_dir)
        algorithm = str(
            pick(result, "algorithm")
            or entry.get("algorithm")
            or manifest.get("algorithm", "")
            or infer_algorithm(Path(str(result_path or "")))
        )
        run_id = str(entry.get("run_id", ""))
        noncrypto_pre = as_float(pick(result, "noncryptographic_pre_backend_wall"))
        noncrypto_post = as_float(pick(result, "noncryptographic_post_backend_wall"))
        noncrypto_sum: object = ""
        if math.isfinite(noncrypto_pre) and math.isfinite(noncrypto_post):
            noncrypto_sum = noncrypto_pre + noncrypto_post
        node_count = as_int(pick(result, "nodes"), 0)
        shuffler_per_user = as_float(pick(result, "shuffler_numeric_payload_per_user"), 0.0)
        shuffler_total: object = node_count * shuffler_per_user if node_count > 0 else ""
        psi_serialized = as_float(pick(result, "total_application_serialized_bytes"))
        algorithm_total: object = ""
        if math.isfinite(psi_serialized) and shuffler_total != "":
            algorithm_total = psi_serialized + float(shuffler_total)
        item = {
            "algorithm": algorithm,
            "run_id": run_id,
            "seed": pick(result, "seed", "public_seed") or entry.get("seed", ""),
            "threads": pick(result, "threads") or backend.get("threads", entry.get("threads", "")),
            "sessions": pick(
                result, "sessions", "session_count", "psi_sessions", "completed_calls",
                "directed_psi_sessions", "executed_directed_psi_sessions",
            ) or backend.get("completed_calls", ""),
            "tau": pick(result, "tau", "projection_tau") or parameters.get("tau", ""),
            "sampling_probability": pick(result, "sampling_probability", "p") or parameters.get("sampling_probability", ""),
            "crypto_wall_s": pick(
                result, "crypto_wall_s", "psi_wall_s", "backend_wall_s", "backend_wall_time"
            ) or backend.get("wall_s", ""),
            "non_crypto_wall_s": pick(result, "non_crypto_wall_s", "noncrypto_wall_s") or noncrypto_sum,
            "end_to_end_wall_s": pick(
                result, "end_to_end_wall_s", "e2e_wall_s", "total_wall_s",
                "full_pipeline_wall_through_reporting",
            ),
            "throughput_sessions_per_s": pick(
                result, "throughput_sessions_per_s", "psi_throughput_calls_per_s", "backend_throughput"
            ) or backend.get("throughput_calls_per_s", ""),
            "peak_rss_bytes": pick(result, "peak_rss_bytes", "full_pipeline_peak_rss") or backend.get("peak_rss_bytes", ""),
            "backend_peak_rss_bytes": pick(result, "backend_process_peak_rss") or backend.get("peak_rss_bytes", ""),
            "full_pipeline_peak_rss_bytes": pick(result, "full_pipeline_peak_rss") or backend.get("peak_rss_bytes", ""),
            "total_group_payload_bytes": pick(result, "total_group_payload_bytes"),
            "total_application_serialized_bytes": pick(result, "total_application_serialized_bytes"),
            "shuffler_numeric_payload_total_bytes": shuffler_total,
            "psi_plus_shuffler_numeric_payload_bytes": algorithm_total,
            "relative_error": pick(result, "relative_error"),
            "epsilon": pick(result, "epsilon") or parameters.get("epsilon_total", parameters.get("epsilon_wedge", "")),
            "delta": pick(result, "delta") or parameters.get("delta_total", parameters.get("delta", "")),
            "evidence": "measured direct full-schedule run" if result else "backend measurement present; frontend timing missing",
        }
        run_rows.append(item)

        users: list[dict[str, float]] = []
        per_user_path = entry.get("per_user_csv")
        if isinstance(per_user_path, Path) and per_user_path.exists():
            for row in read_csv(per_user_path):
                users.append(
                    {
                        "sessions": as_float(row.get("executed_directed_sessions"), 0.0),
                        "responder_sessions": as_float(row.get("psi_sessions_as_responder"), 0.0),
                        "participations": as_float(row.get("psi_total_session_participations"), 0.0),
                        "latency": as_float(row.get("psi_latency_ms_attributed_to_initiator"), 0.0),
                        "bytes": as_float(row.get("psi_serialized_bytes_attributed_to_initiator"), 0.0),
                        "physical_bytes": as_float(row.get("physical_role_outbound_bytes"), 0.0),
                    }
                )
        else:
            raw_path = entry.get("raw_csv")
            if isinstance(raw_path, Path) and raw_path.exists():
                valid, _ = valid_raw_rows(raw_path, audits)
                by_user: dict[str, dict[str, float]] = defaultdict(
                    lambda: {
                        "sessions": 0.0, "responder_sessions": 0.0,
                        "participations": 0.0, "latency": 0.0,
                        "bytes": 0.0, "physical_bytes": 0.0,
                    }
                )
                for row in valid:
                    initiator = row.get("initiator", "")
                    responder = row.get("responder", "")
                    by_user[initiator]["sessions"] += 1
                    by_user[responder]["responder_sessions"] += 1
                    by_user[initiator]["participations"] += 1
                    by_user[responder]["participations"] += 1
                    by_user[initiator]["latency"] += as_float(row.get("latency_ms"), 0.0)
                    by_user[initiator]["bytes"] += as_float(row.get("total_serialized_bytes"), 0.0)
                    by_user[initiator]["physical_bytes"] += as_float(row.get("round1_serialized_bytes"), 0.0)
                    by_user[responder]["physical_bytes"] += as_float(row.get("round2_serialized_bytes"), 0.0)
                users = list(by_user.values())
        if users:
            pooled_users[algorithm].extend(users)
            per_user_run_rows.append(
                {
                    "algorithm": algorithm,
                    "run_id": run_id,
                    "users_with_initiated_sessions": len(users),
                    "initiated_sessions_median": quantile_linear((user["sessions"] for user in users), 0.50),
                    "initiated_sessions_p95": quantile_linear((user["sessions"] for user in users), 0.95),
                    "initiated_sessions_max": maximum(user["sessions"] for user in users),
                    "responder_sessions_median": quantile_linear((user["responder_sessions"] for user in users), 0.50),
                    "responder_sessions_p95": quantile_linear((user["responder_sessions"] for user in users), 0.95),
                    "responder_sessions_max": maximum(user["responder_sessions"] for user in users),
                    "total_participations_median": quantile_linear((user["participations"] for user in users), 0.50),
                    "total_participations_p95": quantile_linear((user["participations"] for user in users), 0.95),
                    "total_participations_max": maximum(user["participations"] for user in users),
                    "total_psi_latency_ms_median": quantile_linear((user["latency"] for user in users), 0.50),
                    "total_psi_latency_ms_p95": quantile_linear((user["latency"] for user in users), 0.95),
                    "total_psi_latency_ms_max": maximum(user["latency"] for user in users),
                    "total_serialized_bytes_median": quantile_linear((user["bytes"] for user in users), 0.50),
                    "total_serialized_bytes_p95": quantile_linear((user["bytes"] for user in users), 0.95),
                    "total_serialized_bytes_max": maximum(user["bytes"] for user in users),
                    "physical_role_outbound_bytes_median": quantile_linear((user["physical_bytes"] for user in users), 0.50),
                    "physical_role_outbound_bytes_p95": quantile_linear((user["physical_bytes"] for user in users), 0.95),
                    "physical_role_outbound_bytes_max": maximum(user["physical_bytes"] for user in users),
                    "evidence": (
                        "derived for every declared user by the algorithm driver"
                        if isinstance(per_user_path, Path) and per_user_path.exists()
                        else "derived for observed initiators from measured backend session rows"
                    ),
                }
            )

    run_rows.sort(key=lambda row: (str(row["algorithm"]), str(row["run_id"])))
    run_fields = [
        "algorithm", "run_id", "seed", "threads", "sessions", "tau",
        "sampling_probability", "crypto_wall_s", "non_crypto_wall_s",
        "end_to_end_wall_s", "throughput_sessions_per_s", "peak_rss_bytes",
        "backend_peak_rss_bytes", "full_pipeline_peak_rss_bytes",
        "total_group_payload_bytes", "total_application_serialized_bytes",
        "shuffler_numeric_payload_total_bytes", "psi_plus_shuffler_numeric_payload_bytes",
        "relative_error",
        "epsilon", "delta", "evidence",
    ]
    write_csv(output_dir / "e2e_runs.csv", run_fields, run_rows)

    aggregate_rows: list[dict[str, object]] = []
    by_algorithm: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        by_algorithm[str(row["algorithm"])].append(row)
    for algorithm in sorted(by_algorithm):
        rows = by_algorithm[algorithm]
        relative_errors = finite(row["relative_error"] for row in rows)
        aggregate_rows.append(
            {
                "algorithm": algorithm,
                "measured_runs": len(rows),
                "required_runs": REQUIRED_RUNS_PER_EXPERIMENT,
                "run_requirement_met": len(rows) == REQUIRED_RUNS_PER_EXPERIMENT,
                "sessions_median": quantile_linear((row["sessions"] for row in rows), 0.50),
                "crypto_wall_s_median": quantile_linear((row["crypto_wall_s"] for row in rows), 0.50),
                "crypto_wall_s_p95": quantile_linear((row["crypto_wall_s"] for row in rows), 0.95),
                "crypto_wall_s_max": maximum(row["crypto_wall_s"] for row in rows),
                "end_to_end_wall_s_median": quantile_linear((row["end_to_end_wall_s"] for row in rows), 0.50),
                "end_to_end_wall_s_p95": quantile_linear((row["end_to_end_wall_s"] for row in rows), 0.95),
                "end_to_end_wall_s_max": maximum(row["end_to_end_wall_s"] for row in rows),
                "throughput_median_sessions_per_s": quantile_linear((row["throughput_sessions_per_s"] for row in rows), 0.50),
                "throughput_p95_sessions_per_s": quantile_linear((row["throughput_sessions_per_s"] for row in rows), 0.95),
                "throughput_max_sessions_per_s": maximum(row["throughput_sessions_per_s"] for row in rows),
                "backend_peak_rss_median_bytes": quantile_linear((row["backend_peak_rss_bytes"] for row in rows), 0.50),
                "backend_peak_rss_p95_bytes": quantile_linear((row["backend_peak_rss_bytes"] for row in rows), 0.95),
                "backend_peak_rss_max_bytes": maximum(row["backend_peak_rss_bytes"] for row in rows),
                "full_pipeline_peak_rss_median_bytes": quantile_linear((row["full_pipeline_peak_rss_bytes"] for row in rows), 0.50),
                "full_pipeline_peak_rss_p95_bytes": quantile_linear((row["full_pipeline_peak_rss_bytes"] for row in rows), 0.95),
                "full_pipeline_peak_rss_max_bytes": maximum(row["full_pipeline_peak_rss_bytes"] for row in rows),
                "total_group_payload_bytes_median": quantile_linear((row["total_group_payload_bytes"] for row in rows), 0.50),
                "total_group_payload_bytes_p95": quantile_linear((row["total_group_payload_bytes"] for row in rows), 0.95),
                "total_group_payload_bytes_max": maximum(row["total_group_payload_bytes"] for row in rows),
                "total_application_serialized_bytes_median": quantile_linear((row["total_application_serialized_bytes"] for row in rows), 0.50),
                "total_application_serialized_bytes_p95": quantile_linear((row["total_application_serialized_bytes"] for row in rows), 0.95),
                "total_application_serialized_bytes_max": maximum(row["total_application_serialized_bytes"] for row in rows),
                "shuffler_numeric_payload_total_bytes_median": quantile_linear((row["shuffler_numeric_payload_total_bytes"] for row in rows), 0.50),
                "shuffler_numeric_payload_total_bytes_p95": quantile_linear((row["shuffler_numeric_payload_total_bytes"] for row in rows), 0.95),
                "shuffler_numeric_payload_total_bytes_max": maximum(row["shuffler_numeric_payload_total_bytes"] for row in rows),
                "psi_plus_shuffler_numeric_payload_bytes_median": quantile_linear((row["psi_plus_shuffler_numeric_payload_bytes"] for row in rows), 0.50),
                "psi_plus_shuffler_numeric_payload_bytes_p95": quantile_linear((row["psi_plus_shuffler_numeric_payload_bytes"] for row in rows), 0.95),
                "psi_plus_shuffler_numeric_payload_bytes_max": maximum(row["psi_plus_shuffler_numeric_payload_bytes"] for row in rows),
                "mean_relative_error": (
                    sum(relative_errors) / len(relative_errors) if relative_errors else ""
                ),
                "relative_error_median": quantile_linear((row["relative_error"] for row in rows), 0.50),
                "relative_error_p95": quantile_linear((row["relative_error"] for row in rows), 0.95),
                "relative_error_max": maximum(row["relative_error"] for row in rows),
                "evidence": "derived across measured direct full-schedule runs",
            }
        )
        audits.append(
            {
                "check": "e2e_required_complete_runs",
                "artifact": f"algorithm={algorithm}",
                "observed": len(rows),
                "expected": REQUIRED_RUNS_PER_EXPERIMENT,
                "pass": len(rows) == REQUIRED_RUNS_PER_EXPERIMENT,
                "detail": (
                    f"protocol requires {REQUIRED_RUNS_PER_EXPERIMENT} complete "
                    "runs for each algorithm"
                ),
            }
        )
    aggregate_fields = [
        "algorithm", "measured_runs", "required_runs", "run_requirement_met",
        "sessions_median", "crypto_wall_s_median",
        "crypto_wall_s_p95", "crypto_wall_s_max", "end_to_end_wall_s_median",
        "end_to_end_wall_s_p95", "end_to_end_wall_s_max",
        "throughput_median_sessions_per_s", "throughput_p95_sessions_per_s",
        "throughput_max_sessions_per_s", "backend_peak_rss_median_bytes",
        "backend_peak_rss_p95_bytes", "backend_peak_rss_max_bytes",
        "full_pipeline_peak_rss_median_bytes", "full_pipeline_peak_rss_p95_bytes",
        "full_pipeline_peak_rss_max_bytes", "total_group_payload_bytes_median",
        "total_group_payload_bytes_p95", "total_group_payload_bytes_max",
        "total_application_serialized_bytes_median",
        "total_application_serialized_bytes_p95",
        "total_application_serialized_bytes_max",
        "shuffler_numeric_payload_total_bytes_median",
        "shuffler_numeric_payload_total_bytes_p95",
        "shuffler_numeric_payload_total_bytes_max",
        "psi_plus_shuffler_numeric_payload_bytes_median",
        "psi_plus_shuffler_numeric_payload_bytes_p95",
        "psi_plus_shuffler_numeric_payload_bytes_max", "mean_relative_error",
        "relative_error_median",
        "relative_error_p95", "relative_error_max", "evidence",
    ]
    write_csv(output_dir / "e2e_summary.csv", aggregate_fields, aggregate_rows)

    per_user_fields = [
        "algorithm", "run_id", "users_with_initiated_sessions",
        "initiated_sessions_median", "initiated_sessions_p95", "initiated_sessions_max",
        "responder_sessions_median", "responder_sessions_p95", "responder_sessions_max",
        "total_participations_median", "total_participations_p95", "total_participations_max",
        "total_psi_latency_ms_median", "total_psi_latency_ms_p95", "total_psi_latency_ms_max",
        "total_serialized_bytes_median", "total_serialized_bytes_p95",
        "total_serialized_bytes_max", "physical_role_outbound_bytes_median",
        "physical_role_outbound_bytes_p95", "physical_role_outbound_bytes_max", "evidence",
    ]
    write_csv(output_dir / "per_user_psi_summary.csv", per_user_fields, per_user_run_rows)

    pooled_rows: list[dict[str, object]] = []
    for algorithm in sorted(pooled_users):
        users = pooled_users[algorithm]
        pooled_rows.append(
            {
                "algorithm": algorithm,
                "measured_runs": len(by_algorithm.get(algorithm, [])),
                "user_run_observations": len(users),
                "initiated_sessions_median": quantile_linear((user["sessions"] for user in users), 0.50),
                "initiated_sessions_p95": quantile_linear((user["sessions"] for user in users), 0.95),
                "initiated_sessions_max": maximum(user["sessions"] for user in users),
                "responder_sessions_median": quantile_linear((user["responder_sessions"] for user in users), 0.50),
                "responder_sessions_p95": quantile_linear((user["responder_sessions"] for user in users), 0.95),
                "responder_sessions_max": maximum(user["responder_sessions"] for user in users),
                "total_participations_median": quantile_linear((user["participations"] for user in users), 0.50),
                "total_participations_p95": quantile_linear((user["participations"] for user in users), 0.95),
                "total_participations_max": maximum(user["participations"] for user in users),
                "initiator_attributed_latency_ms_median": quantile_linear((user["latency"] for user in users), 0.50),
                "initiator_attributed_latency_ms_p95": quantile_linear((user["latency"] for user in users), 0.95),
                "initiator_attributed_latency_ms_max": maximum(user["latency"] for user in users),
                "initiator_attributed_serialized_bytes_median": quantile_linear((user["bytes"] for user in users), 0.50),
                "initiator_attributed_serialized_bytes_p95": quantile_linear((user["bytes"] for user in users), 0.95),
                "initiator_attributed_serialized_bytes_max": maximum(user["bytes"] for user in users),
                "physical_role_outbound_bytes_median": quantile_linear((user["physical_bytes"] for user in users), 0.50),
                "physical_role_outbound_bytes_p95": quantile_linear((user["physical_bytes"] for user in users), 0.95),
                "physical_role_outbound_bytes_max": maximum(user["physical_bytes"] for user in users),
                "evidence": "pooled user-run observations derived from measured full-schedule PSI rows",
            }
        )
    pooled_fields = [
        "algorithm", "measured_runs", "user_run_observations",
        "initiated_sessions_median", "initiated_sessions_p95", "initiated_sessions_max",
        "responder_sessions_median", "responder_sessions_p95", "responder_sessions_max",
        "total_participations_median", "total_participations_p95", "total_participations_max",
        "initiator_attributed_latency_ms_median", "initiator_attributed_latency_ms_p95",
        "initiator_attributed_latency_ms_max", "initiator_attributed_serialized_bytes_median",
        "initiator_attributed_serialized_bytes_p95", "initiator_attributed_serialized_bytes_max",
        "physical_role_outbound_bytes_median", "physical_role_outbound_bytes_p95",
        "physical_role_outbound_bytes_max", "evidence",
    ]
    write_csv(output_dir / "per_user_psi_overall_summary.csv", pooled_fields, pooled_rows)
    return run_rows, aggregate_rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or (results_dir / "derived")).resolve()
    index_path = (args.run_index or (results_dir / "run_index.csv")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_index(results_dir, index_path)
    discovered = discover_entries(results_dir, output_dir)
    known_paths = {
        str(value.resolve())
        for entry in entries
        for key in ("raw_csv", "backend_summary_csv", "result_summary_csv", "per_user_csv")
        if isinstance((value := entry.get(key)), Path)
    }
    for entry in discovered:
        paths = [entry.get(key) for key in ("raw_csv", "backend_summary_csv", "result_summary_csv", "per_user_csv")]
        if not any(isinstance(path, Path) and str(path.resolve()) in known_paths for path in paths):
            entries.append(entry)

    audits: list[dict[str, object]] = []
    micro = summarize_micro(entries, results_dir, output_dir, audits)
    _, concurrency = summarize_concurrency(entries, output_dir, audits)
    _, e2e = summarize_e2e(entries, output_dir, audits)

    audit_fields = ["check", "artifact", "observed", "expected", "pass", "detail"]
    write_csv(output_dir / "consistency_audit.csv", audit_fields, audits)
    present = {
        "micro": any(row.get("scope") == "overall" for row in micro),
        "concurrency": bool(concurrency),
        "e2e": bool(e2e),
    }
    manifest = {
        "results_dir": str(results_dir),
        "run_index": str(index_path) if index_path.exists() else None,
        "entries_considered": len(entries),
        "output_dir": str(output_dir),
        "percentile_method": "linear interpolation at q*(n-1), NumPy-compatible",
        "available": present,
        "audit_failures": sum(not is_true(row["pass"]) for row in audits),
    }
    (output_dir / "summary_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.strict and manifest["audit_failures"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
