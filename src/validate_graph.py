#!/usr/bin/env python3
"""Validate an input graph against its packaged dataset profile."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from paper_common import degree_statistics, read_graph


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("edges", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    graph = read_graph(args.edges)
    degrees = degree_statistics(graph)
    observed = {
        "sha256": sha256(args.edges),
        "nodes": graph.n,
        "edges": graph.edge_count,
        "max_degree": int(degrees["maximum"]),
    }
    expected = {
        "sha256": profile["sha256"],
        "nodes": int(profile["nodes"]),
        "edges": int(profile["edges"]),
        "max_degree": int(profile["max_degree"]),
    }
    if observed != expected:
        raise ValueError(f"dataset profile mismatch: expected={expected}, observed={observed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "graph_validation.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        writer.writerows(
            (
                ("dataset", profile["dataset"]),
                ("input_sha256", observed["sha256"]),
                ("nodes", observed["nodes"]),
                ("undirected_edges", observed["edges"]),
                ("directed_incidences", 2 * observed["edges"]),
                ("degree_min", degrees["minimum"]),
                ("degree_median", degrees["median"]),
                ("degree_p90", degrees["p90"]),
                ("degree_p95", degrees["p95"]),
                ("degree_p99", degrees["p99"]),
                ("degree_max", degrees["maximum"]),
                ("degree_mean", degrees["mean"]),
            )
        )
    print(json.dumps(observed, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
