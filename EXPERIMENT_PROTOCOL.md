# Experiment protocol

- Inputs: `epsilon=1`, `delta=1e-8`.
- ShufTri uses the public dataset maximum degree for the experimental
  sensitivity bound.
- ShufTri+ uses the 0.1/0.9 epsilon split, equal delta split, linear noisy
  90th percentile followed by floor, independent uniform per-user projection,
  and independent sampling of every projected directed incidence.
- Public master seeds for the 20 complete runs: 42 through 61.
- PSI microbenchmark seed: 20260811.
- Microbenchmark: 100 real edges from each of Common, Medium, High, and
  Hub-tail degree strata; both directions; one warm-up and ten retained
  repetitions (8,000 measured calls).
- Concurrency: the same 800 directed sessions, twenty fresh runs at each of 1,
  4, and 8 threads; one warm-up and one retained repetition per run.
- Complete runtime: twenty runs per algorithm at eight threads. Every scheduled
  directed session executes the native PSI backend.
- Triangle-counting accuracy: arithmetic mean over the twenty complete runs.
- PSI latency and runtime statistics: median, maximum, and linear P95 at
  `q*(n-1)`.

The public seeds control workload selection, task order, projection, sampling,
and DP noise. Cryptographic scalars and private shuffles use fresh OS-CSPRNG
randomness and are not seeded or logged.
