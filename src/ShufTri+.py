#!/usr/bin/env python3
"""Paper-faithful ShufTri+ with independent local projection and PSI-CA.

The projected object is intentionally *not* mutualized: ``j in N'(k)`` does
not imply ``k in N'(j)``.  Every projected directed incidence receives its
own Bernoulli draw, and every selected incidence retains its original
initiator/responder orientation when passed to the native backend.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from paper_common import (
    DEFAULT_DELTA,
    DEFAULT_EPSILON,
    SHUFTRI_PLUS_SHUFFLER_PAYLOAD_BYTES,
    aggregate_role_metrics,
    aggregate_session_values,
    artifact_hashes,
    backend_summary_row,
    calibrate_shuftri_budget,
    calibrate_wedge_budget,
    count_undirected_triangles,
    dataclass_dict,
    degree_statistics,
    derive_public_seeds,
    estimate_projection_bound,
    generate_shuftriplus_sessions,
    json_dump,
    make_output_directory,
    prepare_only_manifest,
    process_peak_rss_bytes,
    project_local_neighbors,
    read_csv_rows,
    read_graph,
    run_backend,
    runtime_environment,
    session_count_by_initiator,
    write_key_value_summary,
    write_metric_summary,
    write_per_user_csv,
    write_projection_csv,
    write_sessions_csv,
    write_sets_csv,
)


def build_paper_projection_and_schedule(
    graph,
    *,
    tau: int,
    probability: float,
    projection_seed: int,
    sampling_seed: int,
):
    """Public deterministic helper for tests and reproduction tooling."""

    projected = project_local_neighbors(
        graph, tau, np.random.default_rng(projection_seed)
    )
    sessions = generate_shuftriplus_sessions(
        projected, probability, np.random.default_rng(sampling_seed)
    )
    return projected, sessions


def _optional_float(row: Mapping[str, str], field: str) -> float | None:
    value = row.get(field, "")
    return None if value in (None, "") else float(value)


def run_experiment(
    csv_path: str,
    *,
    output_dir: str,
    epsilon: float = DEFAULT_EPSILON,
    delta: float = DEFAULT_DELTA,
    epsilon_tau_fraction: float = 0.1,
    seed: int = 42,
    backend: str = "ristretto255",
    psi_binary: str | None = None,
    threads: int = 1,
    repetitions: int = 1,
    warmup: int = 0,
    prepare_only: bool = False,
) -> dict[str, object]:
    """Run one auditable paper-mode ShufTri+ experiment."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0,1)")
    if not 0.0 < epsilon_tau_fraction < 1.0:
        raise ValueError("epsilon_tau_fraction must be in (0,1)")
    if threads < 1 or repetitions < 1 or warmup < 0:
        raise ValueError("threads/repetitions must be positive and warmup non-negative")

    pipeline_started = time.perf_counter()
    graph = read_graph(csv_path)
    graph_loaded_at = time.perf_counter()
    destination = make_output_directory(output_dir)
    public_seeds = derive_public_seeds(seed)

    epsilon_tau = epsilon * epsilon_tau_fraction
    epsilon_wedge = epsilon - epsilon_tau
    # The paper specifies the epsilon split; this implementation records an
    # equal numerical delta split for reproducibility.
    delta_tau = delta / 2.0
    delta_wedge = delta / 2.0
    degree_budget = calibrate_shuftri_budget(graph.n, epsilon_tau, delta_tau)
    wedge_budget = calibrate_wedge_budget(graph.n, epsilon_wedge, delta_wedge)
    if degree_budget.epsilon_effective <= 0.0 or wedge_budget.epsilon_raw <= 0.0:
        raise ValueError("budget calibration produced a non-positive local budget")

    degree_rng = np.random.default_rng(public_seeds["degree_noise"])
    tau, noisy_degrees, tau_handling = estimate_projection_bound(
        graph, degree_budget.epsilon_effective, degree_rng
    )
    # Execute the degree-report shuffle explicitly; Q_0.9 is permutation
    # invariant, but the operation is part of Algorithm 2.
    shuffled_degrees = noisy_degrees.copy()
    np.random.default_rng(public_seeds["degree_shuffle"]).shuffle(shuffled_degrees)
    # The same floor(linear Q90) result was already computed above.  This
    # assertion guards accidental dependence on user order.
    shuffled_tau = math.floor(float(np.quantile(shuffled_degrees, 0.90, method="linear")))
    if shuffled_tau != tau_handling["raw_floor_q90"]:
        raise AssertionError("degree shuffling changed the order-invariant Q90")

    projected, sessions = build_paper_projection_and_schedule(
        graph,
        tau=tau,
        probability=wedge_budget.sampling_probability,
        projection_seed=public_seeds["projection"],
        sampling_seed=public_seeds["poisson_sampling"],
    )
    write_sets_csv(destination / "sets.csv", projected)
    write_sessions_csv(destination / "directed_sessions.csv", sessions)
    write_projection_csv(destination / "projection.csv", graph, projected, noisy_degrees)

    candidate_count = sum(len(projected[u]) for u in graph.node_ids)
    asymmetric_count = sum(
        1
        for u in graph.node_ids
        for v in projected[u]
        if u not in projected[v]
    )
    noise_scale = 2.0 * tau / wedge_budget.epsilon_raw
    parameters: dict[str, object] = {
        "epsilon_total": epsilon,
        "delta_total": delta,
        "epsilon_tau": epsilon_tau,
        "epsilon_wedge": epsilon_wedge,
        "delta_tau": delta_tau,
        "delta_wedge": delta_wedge,
        "delta_split_convention": "equal split; paper experiment does not specify numeric split",
        "epsilon_tau_local": degree_budget.epsilon_effective,
        "epsilon_wedge_0": wedge_budget.epsilon_raw,
        "epsilon_wedge_local": wedge_budget.epsilon_effective,
        "sampling_probability": wedge_budget.sampling_probability,
        "tau": tau,
        "tau_release": tau_handling,
        "projection": "independent uniform subset without replacement per user; not mutualized",
        "sampling": "independent Bernoulli(p) per projected directed incidence",
        "laplace_sensitivity": 2.0 * tau,
        "laplace_scale": noise_scale,
        "shuffler_numeric_payload_bytes_per_user": SHUFTRI_PLUS_SHUFFLER_PAYLOAD_BYTES,
        "backend": backend,
        "threads": threads,
        "repetitions": repetitions,
        "warmup_per_session": warmup,
    }
    seeds_document = {
        "master_public_experiment_seed": seed,
        "derived_public_experiment_seeds": public_seeds,
        "scope": (
            "degree/wedge DP noise, shuffle permutations, local projection, "
            "Poisson sampling, and benchmark task order only"
        ),
        "cryptographic_randomness": (
            "fresh per-session libsodium CSPRNG scalars and private Round-2 "
            "permutations; deliberately neither seeded nor logged"
        ),
    }
    json_dump(destination / "seeds.json", seeds_document)

    if prepare_only:
        prepare_only_manifest(
            algorithm="ShufTriPlus",
            graph=graph,
            output_directory=destination,
            parameters=parameters,
            seeds=public_seeds,
            local_sets=projected,
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
        local_sets=projected,
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
    aggregation_finished = time.perf_counter()

    noise_rng = np.random.default_rng(public_seeds["wedge_noise"])
    noisy_values = np.asarray([wedge[u] for u in graph.node_ids], dtype=float)
    noisy_values += noise_rng.laplace(0.0, noise_scale, graph.n)
    noisy_wedge = {u: float(value) for u, value in zip(graph.node_ids, noisy_values)}
    shuffled = noisy_values.copy()
    np.random.default_rng(public_seeds["wedge_shuffle"]).shuffle(shuffled)
    probability = wedge_budget.sampling_probability
    if probability <= 0.0:
        raise AssertionError("sampling probability must be positive")
    estimate = float(shuffled.sum()) / (6.0 * probability)
    unnoised_estimate = sum(wedge.values()) / (6.0 * probability)
    true_triangles = count_undirected_triangles(graph)
    relative_error = (
        abs(estimate - true_triangles) / true_triangles
        if true_triangles > 0
        else math.nan
    )

    candidates = {u: len(projected[u]) for u in graph.node_ids}
    executed = session_count_by_initiator(sessions, graph.node_ids)
    role_metrics = aggregate_role_metrics(sessions, raw_rows, graph.node_ids)
    write_per_user_csv(
        destination / "per_user.csv",
        node_ids=graph.node_ids,
        local_sets=projected,
        candidate_counts=candidates,
        executed_counts=executed,
        wedge=wedge,
        noisy_wedge=noisy_wedge,
        serialized=serialized,
        group_payload=group_payload,
        shuffler_payload_bytes=SHUFTRI_PLUS_SHUFFLER_PAYLOAD_BYTES,
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
            ("tau", tau, "neighbors", "private noisy-degree linear Q90 then floor"),
            ("projected_directed_incidences", candidate_count, "incidences", "measured preprocessing"),
            ("asymmetric_projected_incidences", asymmetric_count, "incidences", "measured preprocessing"),
            ("sampling_probability", probability, "probability", "Algorithm 3 calibration"),
            ("executed_directed_psi_sessions", len(sessions), "calls", "realized Bernoulli schedule"),
            ("unnoised_wedge_sum", sum(wedge.values()), "wedges", measured_label),
            ("unnoised_triangle_estimate", unnoised_estimate, "triangles", measured_label),
            ("noisy_triangle_estimate", estimate, "triangles", "public seeded DP noise"),
            ("relative_error", relative_error, "ratio", "against original T(G)"),
            ("laplace_sensitivity", 2 * tau, "wedge units", "literal Algorithm 4"),
            ("laplace_scale", noise_scale, "wedge units", "2tau/epsilon_wedge_0"),
            ("total_group_payload_bytes", sum(group_payload.values()), "bytes", measured_label),
            (
                "total_application_serialized_bytes",
                sum(serialized.values()) if backend == "ristretto255" else "",
                "bytes",
                measured_label,
            ),
            ("shuffler_numeric_payload_per_user", 16, "bytes", "two binary64 values; framing excluded"),
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
                "measured wall clock: graph load, private projection, backend, aggregation",
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
        "sets.csv", "directed_sessions.csv", "projection.csv", "psi_raw.csv",
        "psi_backend_summary.csv", "psi_metric_summary.csv", "per_user.csv",
        "result_summary.csv", "seeds.json", "backend_stdout.log",
        "backend_stderr.log", "backend_execution.json",
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "algorithm": "ShufTriPlus",
        "implementation_semantics": (
            "paper Algorithms 2-4; independent asymmetric local projection; "
            "directed Bernoulli sampling; no deduplication"
        ),
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
        "budget_calibration": {
            "degree": dataclass_dict(degree_budget),
            "wedge": dataclass_dict(wedge_budget),
        },
        "schedule": {
            "projected_directed_incidences": candidate_count,
            "asymmetric_projected_incidences": asymmetric_count,
            "executed_directed_sessions": len(sessions),
            "conditional_expected_sessions": probability * candidate_count,
            "initiator_responder_orientation_preserved": True,
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
            "unnoised_triangle_estimate": unnoised_estimate,
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
        "privacy_statement": {
            "paper_edge_dp_upper_bound_epsilon": 2.0 * epsilon,
            "paper_edge_dp_upper_bound_delta": 2.0 * math.exp(2.0 * epsilon) * delta,
            "note": "algorithm inputs epsilon,delta are distinct from Theorem 7 edge-DP bound",
        },
        "backend_execution": json.loads(
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
            "Paper-faithful ShufTri+: private linear-Q90 projection, independent "
            "local uniform projection, directed Bernoulli sampling, and native PSI-CA."
        )
    )
    parser.add_argument("csv_path", help="paper-format or two-column edge CSV")
    parser.add_argument("--output-dir", default="results/ShufTriPlus", help="artifact directory")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument(
        "--epsilon-tau-fraction",
        type=float,
        default=0.1,
        help="degree-phase fraction; paper experiment uses 0.1",
    )
    parser.add_argument("--seed", type=int, default=42, help="public experiment master seed")
    parser.add_argument("--backend", choices=("ristretto255", "plaintext"), default="ristretto255")
    parser.add_argument("--psi-binary", default=None, help="native psi_backend path")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0, help="warm-ups per directed session")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write projection/sets/schedule/manifest but do not execute a backend",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_experiment(
        args.csv_path,
        output_dir=args.output_dir,
        epsilon=args.epsilon,
        delta=args.delta,
        epsilon_tau_fraction=args.epsilon_tau_fraction,
        seed=args.seed,
        backend=args.backend,
        psi_binary=args.psi_binary,
        threads=args.threads,
        repetitions=args.repetitions,
        warmup=args.warmup,
        prepare_only=args.prepare_only,
    )
    print(
        f"[DONE] ShufTriPlus status={manifest['status']} "
        f"artifacts={Path(args.output_dir).expanduser().resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
