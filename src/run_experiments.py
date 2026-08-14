#!/usr/bin/env python3
"""Staged reproduction driver for the ShufTri secure-PSI experiments.

No stage estimates a cryptographic runtime.  The expensive ``e2e`` stage calls
the native backend for every generated directed session.  Use ``--dry-run`` to
inspect commands without starting benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
INDEX_FIELDS = [
    "category", "algorithm", "run_id", "threads", "seed", "raw_csv",
    "backend_summary_csv", "result_summary_csv", "per_user_csv", "status",
]
MICRO_SEED = 20260811
EXPERIMENT_RUN_COUNT = 20
FULL_SEEDS = tuple(range(42, 42 + EXPERIMENT_RUN_COUNT))


class CommandFailure(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and run the Enron or Facebook PSI-CA experiments."
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=("smoke", "prepare", "micro", "concurrency", "e2e", "summarize", "all"),
        help="repeatable stage selector (default: smoke); 'all' includes full E2E",
    )
    parser.add_argument("--dataset", required=True, choices=("enron", "facebook"))
    parser.add_argument("--edges", type=Path, default=None, help="override the packaged edge table")
    parser.add_argument("--results-dir", type=Path, default=None, help="output root")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for algorithm CLIs")
    parser.add_argument("--psi-binary", type=Path, default=SRC / "psi_backend", help="native PSI backend")
    parser.add_argument("--full-threads", type=int, default=8, help="worker count for direct E2E runs")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing them")
    parser.add_argument("--skip-build", action="store_true", help="do not compile the backend before run stages")
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="skip completed run directories instead of refusing to mix new and existing evidence",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile_linear(values: Sequence[int], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def load_graph(path: Path) -> tuple[int, list[tuple[int, int]], dict[int, tuple[int, ...]]]:
    """Read the paper's #nodes/count/header format or a two-column CSV."""
    if not path.exists():
        raise FileNotFoundError(f"missing edge table: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    declared_nodes: int | None = None
    edges: set[tuple[int, int]] = set()
    index = 0
    while index < len(lines):
        text = lines[index].strip()
        index += 1
        if not text:
            continue
        if text.lower() == "#nodes":
            if index >= len(lines):
                raise ValueError("#nodes is not followed by a count")
            declared_nodes = int(lines[index].strip())
            index += 1
            continue
        if text.startswith("#") or text.lower() in {"node,node", "source,target", "src,dst", "u,v"}:
            continue
        pieces = [piece.strip() for piece in text.split(",")]
        if len(pieces) != 2:
            raise ValueError(f"malformed edge row {index}: {text!r}")
        try:
            u, v = (int(pieces[0]), int(pieces[1]))
        except ValueError:
            if not edges:
                continue
            raise ValueError(f"non-integer edge row {index}: {text!r}")
        if u == v:
            raise ValueError(f"self-loop at row {index}: {u}")
        if u < 0 or v < 0:
            raise ValueError(f"negative node identifier at row {index}")
        edge = (u, v) if u < v else (v, u)
        if edge in edges:
            raise ValueError(f"duplicate undirected edge at row {index}: {edge}")
        edges.add(edge)
    if not edges:
        raise ValueError("edge table contains no edges")
    observed_nodes = max(max(edge) for edge in edges) + 1
    node_count = declared_nodes if declared_nodes is not None else observed_nodes
    if observed_nodes > node_count:
        raise ValueError(f"node id requires {observed_nodes} nodes but #nodes declares {node_count}")
    neighbors: dict[int, set[int]] = {node: set() for node in range(node_count)}
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
    frozen = {node: tuple(sorted(items)) for node, items in neighbors.items()}
    return node_count, sorted(edges), frozen


def prepare_micro_inputs(edges_path: Path, results_dir: Path) -> tuple[Path, Path, dict[str, object]]:
    node_count, edges, neighbors = load_graph(edges_path)
    degrees = [len(neighbors[node]) for node in range(node_count)]
    thresholds = {
        "q50": quantile_linear(degrees, 0.50),
        "q90": quantile_linear(degrees, 0.90),
        "q99": quantile_linear(degrees, 0.99),
    }
    strata: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for edge in edges:
        maximum_degree = max(degrees[edge[0]], degrees[edge[1]])
        if maximum_degree <= thresholds["q50"]:
            workload = "Common"
        elif maximum_degree <= thresholds["q90"]:
            workload = "Medium"
        elif maximum_degree <= thresholds["q99"]:
            workload = "High"
        else:
            workload = "Hub-tail"
        strata[workload].append(edge)
    rng = random.Random(MICRO_SEED)
    selected: dict[str, list[tuple[int, int]]] = {}
    for workload in ("Common", "Medium", "High", "Hub-tail"):
        candidates = strata[workload]
        if len(candidates) < 100:
            raise ValueError(f"{workload} contains only {len(candidates)} edges; 100 required")
        selected[workload] = sorted(rng.sample(candidates, 100))

    input_dir = results_dir / "micro" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    sets_path = input_dir / "sets.csv"
    sessions_path = input_dir / "directed_sessions.csv"
    sampled_edges_path = input_dir / "sampled_edges.csv"
    with sets_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("owner", "item"))
        for owner in range(node_count):
            for item in neighbors[owner]:
                writer.writerow((owner, item))
    with sessions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("session_id", "initiator", "responder", "workload"))
        session_id = 0
        for workload in ("Common", "Medium", "High", "Hub-tail"):
            for u, v in selected[workload]:
                writer.writerow((session_id, u, v, workload))
                session_id += 1
                writer.writerow((session_id, v, u, workload))
                session_id += 1
    with sampled_edges_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("workload", "u", "v", "degree_u", "degree_v", "max_endpoint_degree"))
        for workload in ("Common", "Medium", "High", "Hub-tail"):
            for u, v in selected[workload]:
                writer.writerow((workload, u, v, degrees[u], degrees[v], max(degrees[u], degrees[v])))
    manifest = {
        "created_utc": utc_now(),
        "edge_file": str(edges_path.resolve()),
        "edge_sha256": sha256(edges_path),
        "node_count": node_count,
        "edge_count": len(edges),
        "max_degree": max(degrees),
        "degree_quantiles_linear": thresholds,
        "public_sampling_seed": MICRO_SEED,
        "sampled_undirected_edges_per_workload": 100,
        "directed_sessions": 800,
        "workload_candidate_counts": {name: len(strata[name]) for name in ("Common", "Medium", "High", "Hub-tail")},
        "sets_sha256": sha256(sets_path),
        "sessions_sha256": sha256(sessions_path),
        "sampled_edges_sha256": sha256(sampled_edges_path),
    }
    (input_dir / "workload_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sets_path, sessions_path, manifest


class Recorder:
    def __init__(self, results_dir: Path, dry_run: bool) -> None:
        self.results_dir = results_dir
        self.logs_dir = results_dir / "logs"
        self.dry_run = dry_run
        self.counter = 0
        self.command_log = self.logs_dir / "commands.jsonl"
        if not dry_run:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            if self.command_log.exists():
                for line in self.command_log.read_text(encoding="utf-8").splitlines():
                    try:
                        self.counter = max(self.counter, int(json.loads(line).get("sequence", 0)))
                    except (ValueError, json.JSONDecodeError, AttributeError):
                        continue

    def run(self, label: str, command: Sequence[object], cwd: Path = ROOT) -> None:
        self.counter += 1
        if not self.dry_run:
            while any(self.logs_dir.glob(f"{self.counter:03d}_*")):
                self.counter += 1
        safe_label = re_safe(label)
        prefix = self.logs_dir / f"{self.counter:03d}_{safe_label}"
        argv = [str(item) for item in command]
        record = {
            "sequence": self.counter,
            "label": label,
            "started_utc": utc_now(),
            "cwd": str(cwd.resolve()),
            "argv": argv,
            "dry_run": self.dry_run,
        }
        if self.dry_run:
            print("DRY RUN:", " ".join(argv))
            return
        else:
            started = time.monotonic()
            completed = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)
            (prefix.with_suffix(".stdout.log")).write_text(completed.stdout, encoding="utf-8")
            (prefix.with_suffix(".stderr.log")).write_text(completed.stderr, encoding="utf-8")
            (prefix.with_suffix(".exit_code")).write_text(f"{completed.returncode}\n", encoding="utf-8")
            record.update(
                {
                    "exit_code": completed.returncode,
                    "elapsed_s": time.monotonic() - started,
                    "finished_utc": utc_now(),
                    "stdout_log": str(prefix.with_suffix(".stdout.log")),
                    "stderr_log": str(prefix.with_suffix(".stderr.log")),
                    "exit_code_file": str(prefix.with_suffix(".exit_code")),
                }
            )
        with self.command_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if not self.dry_run and record["exit_code"] != 0:
            raise CommandFailure(f"{label} failed; see {prefix}.stderr.log")


def re_safe(label: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in label)


def ensure_fresh_run_dir(path: Path, allow_existing: bool, dry_run: bool) -> bool:
    complete = path / ".complete"
    if complete.exists() and allow_existing:
        print(f"SKIP completed run: {path}")
        return False
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty run directory {path}; choose a new --results-dir "
            "or pass --allow-existing to skip runs bearing .complete"
        )
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)
    return True


def relative_or_absolute(path: Path, results_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(results_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def register_run(results_dir: Path, row: dict[str, object], dry_run: bool) -> None:
    if dry_run:
        return
    index_path = results_dir / "run_index.csv"
    exists = index_path.exists()
    with index_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in INDEX_FIELDS})


def mark_complete(run_dir: Path, dry_run: bool) -> None:
    if not dry_run:
        (run_dir / ".complete").write_text(utc_now() + "\n", encoding="utf-8")


def capture(command: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (result.stdout or result.stderr).strip()
    return text if result.returncode == 0 and text else None


def record_environment(
    results_dir: Path, edges: Path, backend: Path, python_executable: str, dry_run: bool
) -> None:
    if dry_run:
        return
    sources = [
        SRC / "psi_backend.c", SRC / "psi_backend.h", SRC / "paper_common.py",
        SRC / "ShufTri.py", SRC / "ShufTri+.py", Path(__file__), SRC / "summarize_results.py",
        SRC / "validate_graph.py", SRC / "Makefile",
    ]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources if path.exists()}
    if backend.exists():
        hashes[str(backend.resolve())] = sha256(backend)
    sodium_header = Path("/opt/anaconda3/include/sodium/version.h")
    sodium_version = None
    if sodium_header.exists():
        hashes[str(sodium_header)] = sha256(sodium_header)
        for line in sodium_header.read_text(encoding="utf-8").splitlines():
            if line.startswith("#define SODIUM_VERSION_STRING"):
                sodium_version = line.split(maxsplit=2)[-1].strip('"')
                break
    hardware: dict[str, object] = {}
    if platform.system() == "Darwin":
        profiler = capture(["system_profiler", "SPHardwareDataType", "-json"])
        if profiler:
            try:
                entries = json.loads(profiler).get("SPHardwareDataType", [])
                if entries and isinstance(entries[0], dict):
                    # Deliberately exclude serial number, UUID, and UDID.
                    for key in (
                        "chip_type", "machine_name", "machine_model",
                        "number_processors", "physical_memory",
                    ):
                        if key in entries[0]:
                            hardware[key] = entries[0][key]
            except (json.JSONDecodeError, AttributeError):
                pass
    environment = {
        "recorded_utc": utc_now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hardware_overview_nonidentifying": hardware,
        "python": sys.version,
        "python_executable": sys.executable,
        "requested_python_executable": python_executable,
        "requested_python_version": capture([python_executable, "--version"]),
        "numpy_version": capture([python_executable, "-c", "import numpy; print(numpy.__version__)"]),
        "compiler_version": capture([os.environ.get("CC", "cc"), "--version"]),
        "libsodium_version_from_build_header": sodium_version,
        "edge_file": str(edges.resolve()),
        "edge_sha256": sha256(edges) if edges.exists() else None,
        "source_and_binary_sha256": hashes,
        "secret_randomness": "libsodium OS CSPRNG; not seeded or logged",
        "public_seed_domains": "edge selection, task order, projection, Bernoulli sampling, and DP noise only",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_backend_batch(
    recorder: Recorder,
    backend: Path,
    sets_path: Path,
    sessions_path: Path,
    run_dir: Path,
    threads: int,
    repetitions: int,
    warmup: int,
    order_seed: int,
    label: str,
) -> None:
    recorder.run(
        label,
        [
            backend, "batch", sets_path, sessions_path,
            run_dir / "psi_raw.csv", run_dir / "psi_backend_summary.csv",
            threads, repetitions, warmup, order_seed,
        ],
    )


def register_backend(
    results_dir: Path,
    run_dir: Path,
    category: str,
    run_id: str,
    threads: int,
    seed: int,
    dry_run: bool,
) -> None:
    register_run(
        results_dir,
        {
            "category": category,
            "algorithm": "PSI-CA",
            "run_id": run_id,
            "threads": threads,
            "seed": seed,
            "raw_csv": relative_or_absolute(run_dir / "psi_raw.csv", results_dir),
            "backend_summary_csv": relative_or_absolute(run_dir / "psi_backend_summary.csv", results_dir),
            "result_summary_csv": "",
            "per_user_csv": "",
            "status": "complete",
        },
        dry_run,
    )


def run_algorithm(
    recorder: Recorder,
    python: str,
    script: Path,
    edges: Path,
    backend: Path,
    run_dir: Path,
    algorithm: str,
    threads: int,
    seed: int,
    plaintext: bool,
    prepare_only: bool,
    max_degree: int,
) -> None:
    command: list[object] = [
        python, script, edges, "--output-dir", run_dir,
        "--backend", "plaintext" if plaintext else "ristretto255",
        "--threads", threads, "--repetitions", 1, "--warmup", 0,
        "--seed", seed, "--epsilon", 1, "--delta", "1e-8",
    ]
    if not plaintext:
        command.extend(("--psi-binary", backend))
    if algorithm == "ShufTri":
        command.extend(("--sensitivity-degree-bound", max_degree))
    else:
        command.extend(("--epsilon-tau-fraction", 0.1))
    if prepare_only:
        command.append("--prepare-only")
    recorder.run(f"{algorithm}_{run_dir.name}", command)


def register_algorithm(
    results_dir: Path,
    run_dir: Path,
    category: str,
    algorithm: str,
    run_id: str,
    threads: int,
    seed: int,
    dry_run: bool,
) -> None:
    register_run(
        results_dir,
        {
            "category": category,
            "algorithm": algorithm,
            "run_id": run_id,
            "threads": threads,
            "seed": seed,
            "raw_csv": relative_or_absolute(run_dir / "psi_raw.csv", results_dir),
            "backend_summary_csv": relative_or_absolute(run_dir / "psi_backend_summary.csv", results_dir),
            "result_summary_csv": relative_or_absolute(run_dir / "result_summary.csv", results_dir),
            "per_user_csv": relative_or_absolute(run_dir / "per_user.csv", results_dir),
            "status": "complete",
        },
        dry_run,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stages = args.stage or ["smoke"]
    if "all" in stages:
        stages = [
            "smoke", "prepare", "micro", "concurrency", "e2e", "summarize",
        ]
    stages = list(dict.fromkeys(stages))
    results_dir = (args.results_dir or (ROOT / "generated_results" / args.dataset)).resolve()
    edges = (args.edges or (ROOT / "data" / args.dataset / "edges.csv")).resolve()
    profile = (ROOT / "data" / args.dataset / "profile.json").resolve()
    backend = args.psi_binary.resolve()
    recorder = Recorder(results_dir, args.dry_run)

    needs_data = any(
        stage in stages
        for stage in ("prepare", "micro", "concurrency", "e2e")
    )
    if needs_data and not edges.exists():
        raise FileNotFoundError(
            f"{edges} does not exist; restore the packaged file or pass --edges."
        )
    if needs_data and not profile.exists():
        raise FileNotFoundError(f"missing dataset profile: {profile}")

    if any(
        stage in stages
        for stage in ("smoke", "prepare", "micro", "concurrency", "e2e")
    ):
        if not args.skip_build:
            recorder.run("build_backend", ["make", "-C", SRC, "all"])
        recorder.run("backend_selftest", [backend, "selftest"])
        record_environment(results_dir, edges, backend, args.python, args.dry_run)

    sets_path: Path | None = None
    sessions_path: Path | None = None
    graph_manifest: dict[str, object] | None = None
    if any(
        stage in stages
        for stage in ("prepare", "micro", "concurrency", "e2e")
    ):
        sets_path, sessions_path, graph_manifest = prepare_micro_inputs(edges, results_dir)
        recorder.run(
            "validate_graph",
            [args.python, SRC / "validate_graph.py", edges, "--profile", profile,
             "--output-dir", results_dir / "validation"],
        )

    if "prepare" in stages:
        max_degree = int(graph_manifest["max_degree"])
        for algorithm, filename in (("ShufTri", "ShufTri.py"), ("ShufTri+", "ShufTri+.py")):
            run_dir = results_dir / "prepared" / algorithm.replace("+", "_plus") / "seed_42"
            if ensure_fresh_run_dir(run_dir, args.allow_existing, args.dry_run):
                run_algorithm(
                    recorder, args.python, SRC / filename, edges, backend, run_dir,
                    algorithm, 1, 42, False, True, max_degree,
                )
                mark_complete(run_dir, args.dry_run)

    if "micro" in stages:
        assert sets_path is not None and sessions_path is not None
        run_dir = results_dir / "micro" / "run_01"
        if ensure_fresh_run_dir(run_dir, args.allow_existing, args.dry_run):
            run_backend_batch(
                recorder, backend, sets_path, sessions_path, run_dir,
                threads=1, repetitions=10, warmup=1, order_seed=MICRO_SEED,
                label="micro_run_01",
            )
            register_backend(results_dir, run_dir, "micro", "run_01", 1, MICRO_SEED, args.dry_run)
            mark_complete(run_dir, args.dry_run)

    if "concurrency" in stages:
        assert sets_path is not None and sessions_path is not None
        for threads in (1, 4, 8):
            for run_number in range(1, EXPERIMENT_RUN_COUNT + 1):
                order_seed = MICRO_SEED + threads * 1000 + run_number
                run_id = f"run_{run_number:02d}"
                run_dir = results_dir / "concurrency" / f"c{threads}" / run_id
                if not ensure_fresh_run_dir(run_dir, args.allow_existing, args.dry_run):
                    continue
                run_backend_batch(
                    recorder, backend, sets_path, sessions_path, run_dir,
                    threads=threads, repetitions=1, warmup=1, order_seed=order_seed,
                    label=f"concurrency_c{threads}_{run_id}",
                )
                register_backend(results_dir, run_dir, "concurrency", run_id, threads, order_seed, args.dry_run)
                mark_complete(run_dir, args.dry_run)

    if "e2e" in stages:
        assert graph_manifest is not None
        max_degree = int(graph_manifest["max_degree"])
        for algorithm, filename in (("ShufTri", "ShufTri.py"), ("ShufTri+", "ShufTri+.py")):
            slug = algorithm.replace("+", "_plus")
            for run_number, seed in enumerate(FULL_SEEDS, start=1):
                run_id = f"run_{run_number:02d}"
                run_dir = results_dir / "e2e" / slug / run_id
                if not ensure_fresh_run_dir(run_dir, args.allow_existing, args.dry_run):
                    continue
                run_algorithm(
                    recorder, args.python, SRC / filename, edges, backend, run_dir,
                    algorithm, args.full_threads, seed, False, False, max_degree,
                )
                register_algorithm(results_dir, run_dir, "e2e", algorithm, run_id, args.full_threads, seed, args.dry_run)
                mark_complete(run_dir, args.dry_run)

    if "summarize" in stages:
        recorder.run(
            "summarize_results",
            [args.python, SRC / "summarize_results.py", "--results-dir", results_dir],
        )

    if not args.dry_run:
        print(f"Completed stages: {', '.join(stages)}")
        print(f"Evidence root: {results_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CommandFailure, FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
