# ShufTri PSI experiment artifact

This package contains the minimum code and data needed to run the ShufTri and
ShufTri+ experiments on Enron and Facebook with the Ristretto255 PSI-CA
backend. Published aggregate results and the 8,000-call PSI microbenchmark raw
CSV for each dataset are under `results/`.

## Requirements

- Python 3.10 or newer and NumPy 2.1.3;
- a C11 compiler and POSIX threads;
- libsodium with Ristretto255 support.

The Makefile defaults to `/opt/anaconda3` for libsodium. Override
`SODIUM_PREFIX` or `SODIUM_LIB` when necessary.

## Build and check

```bash
python3 -m pip install -r requirements.txt
make -C src psi_backend
src/psi_backend selftest
```

## Run

The default stage is a backend smoke test:

```bash
./reproduce.sh --dataset enron
./reproduce.sh --dataset facebook
```

Run the complete experiment matrix for either dataset:

```bash
./reproduce.sh --dataset enron --stage all
./reproduce.sh --dataset facebook --stage all
```

The complete matrix performs one preparation run, an 8,000-call PSI
microbenchmark, twenty fresh runs at each concurrency level (1, 4, and 8),
and twenty full runs of each algorithm. It is intentionally expensive. Stages
can be run separately:

```bash
./reproduce.sh --dataset enron --stage prepare --stage micro
./reproduce.sh --dataset enron --stage concurrency --allow-existing
./reproduce.sh --dataset enron --stage e2e --allow-existing
./reproduce.sh --dataset enron --stage summarize --allow-existing
```

Generated files are written to `generated_results/DATASET` unless
`--results-dir` is supplied. A completed run is never overwritten;
`--allow-existing` only skips directories marked complete.

`EXPERIMENT_PROTOCOL.md` records the fixed parameters and seeds.
`SECURITY.md` records the PSI implementation and measurement boundary.
