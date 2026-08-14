#!/usr/bin/env python3
"""Paper-faithful ShufTri with a real, directed PSI-CA schedule.

For every ``j in N(k)`` this program creates the distinct directed session
``k -> j``; consequently a simple undirected graph has exactly ``2m``
sessions.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from paper_common import (
    DEFAULT_DELTA,
    DEFAULT_EPSILON,
    SHUFTRI_SHUFFLER_PAYLOAD_BYTES,
    aggregate_role_metrics,
    aggregate_session_values,
    artifact_hashes,
    backend_summary_row,
    calibrate_shuftri_budget,
    count_undirected_triangles,
    dataclass_dict,
    degree_statistics,
    derive_public_seeds,
    generate_shuftri_sessions,
    json_dump,
    make_output_directory,
    prepare_only_manifest,
    process_peak_rss_bytes,
    read_csv_rows,
    read_graph,
    run_backend,
    runtime_environment,
    session_count_by_initiator,
    write_key_value_summary,
    write_metric_summary,
    write_per_user_csv,
    write_sessions_csv,
    write_sets_csv,
)


def build_paper_schedule(graph):
    """Public helper used by tests and reproduction tooling."""

    return graph.adjacency, generate_shuftri_sessions(graph)


def _optional_float(row: Mapping[str, str], field: str) -> float | None:
    value = row.get(field, "")
    return None if value in (None, "") else float(value)


def run_experiment(
    csv_path: str,
    *,
    output_dir: str,
    epsilon: float = DEFAULT_EPSILON,
    delta: float = DEFAULT_DELTA,
    seed: int = 42,
    backend: str = "ristretto255",
    psi_binary: str | None = None,
    threads: int = 1,
    repetitions: int = 1,
    warmup: int = 0,
    prepare_only: bool = False,
    sensitivity_degree_bound: int | None = None,
) -> dict[str, object]:
    """Run one auditable ShufTri experiment and return its manifest."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0,1)")
    if threads < 1 or repetitions < 1 or warmup < 0:
        raise ValueError("threads/repetitions must be positive and warmup non-negative")

    pipeline_started = time.perf_counter()
    graph = read_graph(csv_path)
    graph_loaded_at = time.perf_counter()
    local_sets, sessions = build_paper_schedule(graph)
    destination = make_output_directory(output_dir)
    public_seeds = derive_public_seeds(seed)
    seeds_document = {
        "master_public_experiment_seed": seed,
        "derived_public_experiment_seeds": public_seeds,
        "scope": (
            "DP noise, shuffler permutation, and benchmark task order only; "
            "these seeds are not cryptographic secrets"
        ),
        "cryptographic_randomness": (
            "fresh per-session libsodium CSPRNG scalars and private Round-2 "
            "permutations; deliberately neither seeded nor logged"
        ),
    }
    json_dump(destination / "seeds.json", seeds_document)
    write_sets_csv(destination / "sets.csv", local_sets)
    write_sessions_csv(destination / "directed_sessions.csv", sessions)

    budget = calibrate_shuftri_budget(graph.n, epsilon, delta)
    if budget.epsilon_effective <= 0.0:
        raise ValueError("shuffle calibration produced a non-positive local budget")

    if sensitivity_degree_bound is None:
        sensitivity = 2.0 * graph.n
        sensitivity_mode = "literal_Algorithm_1_2n"
        sensitivity_assumption = "none"
    else:
        if sensitivity_degree_bound < 1:
            raise ValueError("sensitivity_degree_bound must be positive")
        sensitivity = 2.0 * sensitivity_degree_bound
        sensitivity_mode = "explicit_public_degree_bound_override_NOT_literal_Algorithm_1"
        sensitivity_assumption = (
            "all admissible inputs satisfy the public degree bound; this is an "
            "experimental restricted-domain override"
        )
    noise_scale = sensitivity / budget.epsilon_effective

    parameters: dict[str, object] = {
        "epsilon_wedge": epsilon,
        "delta": delta,
        "epsilon_wedge_local": budget.epsilon_effective,
        "theta_sh": budget.theta_sh,
        "sensitivity": sensitivity,
        "sensitivity_mode": sensitivity_mode,
        "sensitivity_assumption": sensitivity_assumption,
        "laplace_scale": noise_scale,
        "directed_session_rule": "one k->j session for every j in N(k); exactly 2m",
        "shuffler_numeric_payload_bytes_per_user": SHUFTRI_SHUFFLER_PAYLOAD_BYTES,
        "backend": backend,
        "threads": threads,
        "repetitions": repetitions,
        "warmup_per_session": warmup,
    }

    if prepare_only:
        prepare_only_manifest(
            algorithm="ShufTri",
            graph=graph,
            output_directory=destination,
            parameters=parameters,
            seeds=public_seeds,
            local_sets=local_sets,
            sessions=sessions,
        )
        return {"status": "schedule_prepared_backend_not_executed", **parameters}

    executable = (
        Path(psi_binary).expanduser()
        if psi_binary is not None
        else Path(__file__).resolve().with_name("psi_backend")
    )
    backend_started = time.perf_counter()
    raw_path, backend_summary_path = run_backend(
        backend=backend,
        executable=executable,
        local_sets=local_sets,
        sessions=sessions,
        output_directory=destination,
        threads=threads,
        repetitions=repetitions,
        warmup=warmup,
        order_seed=public_seeds["backend_order"],
    )
    backend_finished = time.perf_counter()
    raw_rows = read_csv_rows(raw_path)
    wedge, serialized, group_payload = aggregate_session_values(
        sessions, raw_rows, graph.node_ids
    )
    if sum(wedge.values()) != 6 * count_undirected_triangles(graph):
        raise AssertionError("ShufTri wedge identity sum(W)=6T failed")
    aggregation_finished = time.perf_counter()

    noise_rng = np.random.default_rng(public_seeds["wedge_noise"])
    noisy_values = np.asarray([wedge[u] for u in graph.node_ids], dtype=float)
    noisy_values += noise_rng.laplace(0.0, noise_scale, graph.n)
    noisy_wedge = {u: float(value) for u, value in zip(graph.node_ids, noisy_values)}

    # The random permutation is executed explicitly even though aggregation is
    # order invariant, so the reproduction record matches Algorithm 1.
    shuffled = noisy_values.copy()
    np.random.default_rng(public_seeds["wedge_shuffle"]).shuffle(shuffled)
    estimate = float(shuffled.sum()) / 6.0
    true_triangles = count_undirected_triangles(graph)
    relative_error = (
        abs(estimate - true_triangles) / true_triangles
        if true_triangles > 0
        else math.nan
    )

    candidates = {u: len(local_sets[u]) for u in graph.node_ids}
    executed = session_count_by_initiator(sessions, graph.node_ids)
    role_metrics = aggregate_role_metrics(sessions, raw_rows, graph.node_ids)
    write_per_user_csv(
        destination / "per_user.csv",
        node_ids=graph.node_ids,
        local_sets=local_sets,
        candidate_counts=candidates,
        executed_counts=executed,
        wedge=wedge,
        noisy_wedge=noisy_wedge,
        serialized=serialized,
        group_payload=group_payload,
        shuffler_payload_bytes=SHUFTRI_SHUFFLER_PAYLOAD_BYTES,
        role_metrics=role_metrics,
    )
    write_metric_summary(destination / "psi_metric_summary.csv", raw_rows, backend=backend)

    native_summary = backend_summary_row(backend_summary_path)
    backend_wall = _optional_float(native_summary, "wall_s")
    backend_throughput = _optional_float(native_summary, "throughput_calls_per_s")
    backend_peak_rss = _optional_float(native_summary, "peak_rss_bytes")
    reporting_finished = time.perf_counter()
    pipeline_through_aggregation = aggregation_finished - pipeline_started
    pipeline_through_reporting = reporting_finished - pipeline_started
    noncrypto_pre_seconds = backend_started - pipeline_started
    noncrypto_post_seconds = reporting_finished - backend_finished
    python_peak_rss = process_peak_rss_bytes()
    overall_peak_rss = max(python_peak_rss, int(backend_peak_rss or 0))
    measured_label = (
        "measured by native Ristretto255 backend"
        if backend == "ristretto255"
        else "plaintext correctness oracle; not cryptographic performance"
    )
    write_key_value_summary(
        destination / "result_summary.csv",
        (
            ("nodes", graph.n, "nodes", "parsed input"),
            ("undirected_edges", graph.edge_count, "edges", "parsed input"),
            ("true_triangles", true_triangles, "triangles", "exact original graph"),
            ("directed_psi_sessions", len(sessions), "calls", "paper schedule; exactly 2m"),
            ("unnoised_wedge_sum", sum(wedge.values()), "wedges", measured_label),
            ("unnoised_triangle_estimate", sum(wedge.values()) / 6.0, "triangles", measured_label),
            ("noisy_triangle_estimate", estimate, "triangles", "public seeded DP noise"),
            ("relative_error", relative_error, "ratio", "against original T(G)"),
            ("laplace_sensitivity", sensitivity, "wedge units", sensitivity_mode),
            ("laplace_scale", noise_scale, "wedge units", "sensitivity/epsilon_local"),
            ("total_group_payload_bytes", sum(group_payload.values()), "bytes", measured_label),
            (
                "total_application_serialized_bytes",
                sum(serialized.values()) if backend == "ristretto255" else "",
                "bytes",
                measured_label,
            ),
            ("shuffler_numeric_payload_per_user", 8, "bytes", "one binary64; framing excluded"),
            ("backend_wall_time", backend_wall if backend_wall is not None else "", "s", measured_label),
            (
                "backend_throughput",
                backend_throughput if backend_throughput is not None else "",
                "calls/s",
                measured_label,
            ),
            (
                "full_pipeline_wall_through_aggregation",
                pipeline_through_aggregation,
                "s",
                "measured wall clock: graph load, schedule, backend, and aggregation",
            ),
            (
                "full_pipeline_wall_through_reporting",
                pipeline_through_reporting,
                "s",
                "measured wall clock through result-table generation",
            ),
            ("noncryptographic_pre_backend_wall", noncrypto_pre_seconds, "s", "measured"),
            ("noncryptographic_post_backend_wall", noncrypto_post_seconds, "s", "measured"),
            ("python_process_peak_rss", python_peak_rss, "bytes", "OS resource observation"),
            ("backend_process_peak_rss", backend_peak_rss or "", "bytes", measured_label),
            ("full_pipeline_peak_rss", overall_peak_rss, "bytes", "max of observed Python/backend peaks"),
        ),
    )

    artifact_names = (
        "sets.csv", "directed_sessions.csv", "psi_raw.csv",
        "psi_backend_summary.csv", "psi_metric_summary.csv", "per_user.csv",
        "result_summary.csv", "seeds.json", "backend_stdout.log",
        "backend_stderr.log", "backend_execution.json",
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "algorithm": "ShufTri",
        "implementation_semantics": "paper Algorithm 1; directed; no edge deduplication",
        "input": {
            "path": graph.source_path,
            "sha256": graph.source_sha256,
            "nodes": graph.n,
            "edges": graph.edge_count,
            "removed_self_loops": graph.removed_self_loops,
            "duplicate_edge_rows": graph.duplicate_edge_rows,
            "degree_statistics": degree_statistics(graph),
        },
        "parameters": parameters,
        "budget_calibration": dataclass_dict(budget),
        "schedule": {
            "candidate_directed_sessions": graph.directed_incidence_count,
            "executed_directed_sessions": len(sessions),
            "assertion_2m": len(sessions) == 2 * graph.edge_count,
        },
        "protocol": {
            "backend": backend,
            "group": "Ristretto255 (prime-order group in the Curve25519 family)",
            "security_model": "semi-honest",
            "network_transport": "none; local application-layer serialization",
            "cryptographic_randomness": seeds_document["cryptographic_randomness"],
            "plaintext_backend_warning": (
                None
                if backend == "ristretto255"
                else "NON-CRYPTOGRAPHIC correctness oracle; no latency or serialized-byte claim"
            ),
        },
        "results": {
            "true_triangles": true_triangles,
            "unnoised_wedge_sum": sum(wedge.values()),
            "noisy_triangle_estimate": estimate,
            "relative_error": relative_error,
            "backend_summary": native_summary,
            "timing": {
                "graph_load_seconds": graph_loaded_at - pipeline_started,
                "noncryptographic_pre_backend_seconds": noncrypto_pre_seconds,
                "backend_invocation_seconds": backend_finished - backend_started,
                "noncryptographic_post_backend_seconds": noncrypto_post_seconds,
                "full_pipeline_through_aggregation_seconds": pipeline_through_aggregation,
                "full_pipeline_through_reporting_seconds": pipeline_through_reporting,
            },
            "memory": {
                "python_peak_rss_bytes": python_peak_rss,
                "backend_peak_rss_bytes": backend_peak_rss,
                "full_pipeline_peak_rss_bytes": overall_peak_rss,
            },
        },
        "backend_execution": __import__("json").loads(
            (destination / "backend_execution.json").read_text(encoding="utf-8")
        ),
        "seeds": seeds_document,
        "environment": runtime_environment(),
        "artifacts_sha256": artifact_hashes(destination, artifact_names),
    }
    json_dump(destination / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-faithful ShufTri: schedule every ordered incidence and run "
            "a Ristretto255 PSI-CA backend."
        )
    )
    parser.add_argument("csv_path", help="paper-format or two-column edge CSV")
    parser.add_argument("--output-dir", default="results/ShufTri", help="artifact directory")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--seed", type=int, default=42, help="public experiment master seed")
    parser.add_argument("--backend", choices=("ristretto255", "plaintext"), default="ristretto255")
    parser.add_argument("--psi-binary", default=None, help="native psi_backend path")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0, help="warm-ups per directed session")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write sets/schedule/manifest but do not execute either backend",
    )
    parser.add_argument(
        "--sensitivity-degree-bound",
        type=int,
        default=None,
        metavar="D",
        help=(
            "explicit restricted-domain override sensitivity=2D; omitted means "
            "literal Algorithm 1 sensitivity=2n"
        ),
    )
    parser.add_argument(
        "--D",
        type=int,
        default=None,
        help="deprecated alias for --sensitivity-degree-bound (never used silently)",
    )
    args = parser.parse_args(argv)
    if args.D is not None:
        if args.sensitivity_degree_bound is not None and args.D != args.sensitivity_degree_bound:
            parser.error("--D and --sensitivity-degree-bound disagree")
        args.sensitivity_degree_bound = args.D
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_experiment(
        args.csv_path,
        output_dir=args.output_dir,
        epsilon=args.epsilon,
        delta=args.delta,
        seed=args.seed,
        backend=args.backend,
        psi_binary=args.psi_binary,
        threads=args.threads,
        repetitions=args.repetitions,
        warmup=args.warmup,
        prepare_only=args.prepare_only,
        sensitivity_degree_bound=args.sensitivity_degree_bound,
    )
    print(
        f"[DONE] ShufTri status={manifest['status']} "
        f"artifacts={Path(args.output_dir).expanduser().resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
