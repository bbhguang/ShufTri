import argparse
import math
import multiprocessing as mp
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path

BYTES_PER_KB = 1024.0




_TRIAL_CTX = None

def read_graph(csv_path: str):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    n = int(lines[1].strip())
    adj = defaultdict(set)
    for raw in lines[3:]:
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(',')
        if len(parts) < 2:
            continue
        try:
            u, v = int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            continue
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    for i in range(n):
        if i not in adj:
            adj[i] = set()
    return n, {k: v for k, v in adj.items()}




def _feldman_threshold(n, delta):
    val = math.log(n / (16.0 * math.log(2.0 / delta)))
    return max(val, 1e-6)


def shuffle_amplify(eps_L, n, delta):
    eps_L = max(eps_L, 1e-15)
    e = math.exp(eps_L)
    inner = ((e - 1.0) / (e + 1.0)) * (
        8.0 * math.sqrt(e * math.log(4.0 / delta)) / math.sqrt(n)
        + 8.0 * e / n
    )
    return math.log(max(1.0 + inner, 1e-15))


def compute_p(eps_base, n, delta):
    threshold = _feldman_threshold(n, delta)
    if eps_base <= threshold:
        return 1.0
    p = (math.exp(threshold) - 1.0) / (math.exp(eps_base) - 1.0)
    return float(np.clip(p, 1e-12, 1.0))


def invert_shuffle(eps_dp_target, n, delta):
    upper = _feldman_threshold(n, delta)
    if shuffle_amplify(upper, n, delta) <= eps_dp_target:
        return upper
    lo, hi = 1e-15, upper
    for _ in range(300):
        mid = (lo + hi) / 2.0
        if shuffle_amplify(mid, n, delta) < eps_dp_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0




def estimate_D(adj, n, eps_D_base, rng):

    true_degs = np.array([len(adj[u]) for u in range(n)], dtype=float)
    noisy = true_degs + rng.laplace(0.0, 1.0 / eps_D_base, n)
    D = int(round(float(np.percentile(noisy, 90))))
    return max(D, 1)


def project_graph(adj, n, D):


    tentative = {}
    for u in range(n):
        tentative[u] = set(sorted(adj[u])[:D])

    N_proj = {}
    for u in range(n):
        N_proj[u] = {v for v in tentative[u] if u in tentative[v]}
    return N_proj


def count_true_triangles(N_proj, n):


    T = 0
    for u in range(n):
        nbrs_u = N_proj[u]
        for v in nbrs_u:
            if v > u:

                T += len(nbrs_u & {w for w in N_proj[v] if w > v})
    return T


def precompute_wedges(N_proj, n):


    wedge = {}
    for u in range(n):
        for v in N_proj[u]:
            if v > u:
                wedge[(u, v)] = len(N_proj[u] & N_proj[v])
    return wedge


def parse_eps_values(text):
    vals = []
    seen = set()
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
        value = round(float(token), 10)
        if value < 0:
            raise ValueError(f"epsilon must be non-negative: {token}")
        if value in seen:
            continue
        seen.add(value)
        vals.append(value)
    if not vals:
        raise ValueError("epsilon list must not be empty.")
    return vals


def _trial_mp_context():
    if sys.platform.startswith('linux'):
        return mp.get_context('fork')
    return mp.get_context('spawn')


def _init_trial_worker(n, N_proj, wedge, T_true, p, D, eps_base):
    global _TRIAL_CTX
    _TRIAL_CTX = {
        'n': n,
        'N_proj': N_proj,
        'wedge': wedge,
        'T_true': T_true,
        'p': p,
        'D': D,
        'eps_base': eps_base,
    }


def _run_trial_worker(seed):
    rng = np.random.default_rng(seed)
    return run_trial(
        _TRIAL_CTX['n'],
        _TRIAL_CTX['N_proj'],
        _TRIAL_CTX['wedge'],
        _TRIAL_CTX['T_true'],
        _TRIAL_CTX['p'],
        _TRIAL_CTX['D'],
        _TRIAL_CTX['eps_base'],
        rng,
    )


def run_trials(n, N_proj, wedge, T_true, p, D, eps_base,
               num_trials, num_workers, seed_base):
    seeds = [seed_base + i for i in range(num_trials)]
    max_workers = max(1, min(num_workers, num_trials))

    def _run_serial():
        results = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            results.append(run_trial(n, N_proj, wedge, T_true, p, D, eps_base, rng))
        return results

    if max_workers == 1:
        return _run_serial()

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=_trial_mp_context(),
            initializer=_init_trial_worker,
            initargs=(n, N_proj, wedge, T_true, p, D, eps_base),
        ) as executor:
            return list(executor.map(_run_trial_worker, seeds))
    except (OSError, PermissionError) as exc:
        print(f"[WARN] Parallel trial initialization failed; using serial execution: {exc}")
        return _run_serial()


def relative_error_cpp_style(est, true_val, node_num):
    denom = max(true_val, 0.001 * node_num)
    return abs(est - true_val) / denom




def run_trial(n, N_proj, wedge, T_true, p, D, eps_base, rng):


    S = [None] * n
    for u in range(n):
        nbrs = list(N_proj[u])
        if nbrs:
            mask = rng.random(len(nbrs)) < p
            S[u] = {v for v, m in zip(nbrs, mask) if m}
        else:
            S[u] = set()


    W = np.zeros(n, dtype=float)
    for u in range(n):
        for v in S[u]:
            key = (u, v) if u < v else (v, u)
            W[u] += wedge[key]


    sensitivity = 2.0 * (D - 1)
    if sensitivity > 0.0:
        b = sensitivity / eps_base
        W_tilde = W + rng.laplace(0.0, b, n)
    else:
        W_tilde = W.copy()


    T_hat = float(W_tilde.sum()) / (6.0 * p)


    if D <= 1:
        var = 0.0
    else:
        var = (
            (1.0 - p) * (D - 1) * T_true / (6.0 * p)
            + 2.0 * n * (D - 1) ** 2 / (9.0 * p ** 2 * eps_base ** 2)
        )


    # C_PSI(initiator→responder) = (2×d_init + d_resp) × 32


    sampled_edges = set()
    for u in range(n):
        for v in S[u]:
            sampled_edges.add((min(u, v), max(u, v)))

    C_PSI_total = 0
    for (u, v) in sampled_edges:
        du, dv = len(N_proj[u]), len(N_proj[v])

        if du <= dv:
            d_init, d_resp = du, dv
        else:
            d_init, d_resp = dv, du
        C_PSI_total += (2 * d_init + d_resp) * 32

    C_PSI_avg   = C_PSI_total / n
    C_shuffle   = 8
    C_total_avg = C_PSI_avg + C_shuffle

    return T_hat, var, C_PSI_avg, C_shuffle, C_total_avg




def run_experiment(csv_path: str, eps_base_list=None, num_trials: int = 20,
                   delta: float = 1e-8, num_workers: int = 1,
                   seed: int = 42, output_file=None):
    if eps_base_list is None:
        eps_base_list = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    print(f"[LOAD] {csv_path}")
    n, adj = read_graph(csv_path)
    total_edges = sum(len(v) for v in adj.values()) // 2
    print(f"  nodes n={n}, edges |E|={total_edges}\n")

    rows = []

    for eps_idx, eps_base in enumerate(eps_base_list):
        print(f"▶ ε_base={eps_base:.2f}", end="  ", flush=True)


        p         = compute_p(eps_base, n, delta)
        eps_L     = math.log(1.0 + p * (math.exp(eps_base) - 1.0))
        eps_dp    = shuffle_amplify(eps_L, n, delta)
        eps_total = eps_dp / 0.9
        eps_D     = eps_total * 0.1
        eps_D_base = invert_shuffle(eps_D, n, delta)


        rng_d  = np.random.default_rng(seed + eps_idx * 1000003 + 17)
        D      = estimate_D(adj, n, eps_D_base, rng_d)


        N_proj = project_graph(adj, n, D)


        proj_edges = sum(len(nbrs) for nbrs in N_proj.values()) // 2


        T_true = count_true_triangles(N_proj, n)

        wedge  = precompute_wedges(N_proj, n)

        print(f"D={D}  T_true={T_true}  projected edges={proj_edges}  "
              f"p={p:.4f}  ε_L={eps_L:.4f}  ε_dp={eps_dp:.4f}  ε_D={eps_D:.4f}")

        trial_results = run_trials(
            n=n,
            N_proj=N_proj,
            wedge=wedge,
            T_true=T_true,
            p=p,
            D=D,
            eps_base=eps_base,
            num_trials=num_trials,
            num_workers=num_workers,
            seed_base=seed + eps_idx * 1000003 + 1000,
        )
        t_hat_list = [item[0] for item in trial_results]
        vars_list = [item[1] for item in trial_results]
        c_psi_list = [item[2] for item in trial_results]
        c_shuf_list = [item[3] for item in trial_results]
        c_tot_list = [item[4] for item in trial_results]
        mre_list = [
            relative_error_cpp_style(t_hat, T_true, n)
            for t_hat in t_hat_list
        ]
        mre = float(np.mean(mre_list))

        rows.append({
            'epsilon_base': eps_base,
            'MRE': mre,
            'VAR': float(np.mean(vars_list)),
            'C_PSI(KB/user)': float(np.mean(c_psi_list)) / BYTES_PER_KB,
            'C_shuf(KB/user)': float(np.mean(c_shuf_list)) / BYTES_PER_KB,
            'C_tot(KB/user)': float(np.mean(c_tot_list)) / BYTES_PER_KB,
            'epsilon_L': eps_L,
            'epsilon_dp': eps_dp,
            'epsilon_total': eps_total,
            'epsilon_D': eps_D,
            'p': p,
            'D': D,
            'T_true': T_true,
            'num_trials': num_trials,
            'num_workers': min(max(1, num_workers), num_trials),
        })

        print(f"  MRE={mre:.6e}")

    df = pd.DataFrame(rows)

    fmt = lambda x: f'{x:.4f}'
    sep = "=" * 160
    print(f"\n{sep}")
    print(f"RESULTS TABLE ({num_trials}-trial means, 4 decimal places)")
    print(sep)
    print(df.to_string(index=False, float_format=fmt))
    print(sep)

    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[DONE] Results written to: {output_path}")

    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="AmpPSIShuf with configurable epsilon values, trials, "
                    "parallel workers, and CSV output."
    )
    parser.add_argument(
        'csv_path',
        help="Path to the input edges.csv file.",
    )
    parser.add_argument(
        '--eps-values',
        default='0.1,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0',
        help="Comma-separated epsilon_base values.",
    )
    parser.add_argument(
        '--num-trials',
        type=int,
        default=20,
        help="Trials per epsilon_base value; default: 20.",
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help="Parallel worker processes per epsilon value; default: 1.",
    )
    parser.add_argument(
        '--delta',
        type=float,
        default=1e-8,
        help="Delta used for shuffle amplification; default: 1e-8.",
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Base random seed; default: 42.",
    )
    parser.add_argument(
        '--output-file',
        default=None,
        help="Optional output CSV path.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_experiment(
        csv_path=args.csv_path,
        eps_base_list=parse_eps_values(args.eps_values),
        num_trials=args.num_trials,
        delta=args.delta,
        num_workers=args.num_workers,
        seed=args.seed,
        output_file=args.output_file,
    )
