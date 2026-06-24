"""
sim_study.py — Monte Carlo evaluation of FAMF-SEM.

Design
------
k = 24 items, q = 2 traits (12 items each), n = 300 per replication.
Inter-trait correlation psi in {0, .15, .40}; psi = 0 gives Type I error,
.15 power, .40 bias/RMSE under a strong association.
CMV conditions:
  none        no method factor;
  contrast    acquiescence-style, design-driven differential CMV
              (method loadings mixed-sign, driven mostly by reverse keying
              plus a serial-position effect);
  misaligned  same method loadings, but calibration receives permuted
              metadata (Z uninformative);
  common      equal positive loadings (0.35) on all items — boundary case
              in which an equal-loading CLF is correctly specified and the
              method variance is largely absorbed by the baseline fit.
Models: Baseline (traits only); CLF (equal loadings, free variance);
FAMF (fixed calibrated pattern, free variance).
Outcomes: bias and RMSE of the inter-trait correlation; rejection rate of
H0: cov(F1,F2)=0 by likelihood-ratio test (1 df, alpha = .05); convergence.

Usage:  python3 sim_study.py <start> <count> [reps] [outfile]
        (chunked & resumable; appends to outfile)
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool

warnings.filterwarnings("ignore")

from famf import (encode_metadata, calibrate, trait_model_desc,
                  famf_model_desc, clf_model_desc)

K, NHALF, N = 24, 12, 300
ITEMS = [f"y{i+1}" for i in range(K)]
SCALES = {"F1": ITEMS[:NHALF], "F2": ITEMS[NHALF:]}
RMS_W = 0.35
LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
CRIT = 3.841459  # chi2(1), alpha = .05


def make_metadata(rng):
    rev = np.array([1 if i % 2 == 1 else 0 for i in range(K)])
    page = np.repeat([1, 2, 3], K // 3)
    order = np.arange(1, K + 1)
    length = rng.integers(5, 15, K)
    return pd.DataFrame({"Reversed": rev, "Page": page,
                         "Order": order, "Length": length})


def true_w(meta, cond, rng):
    if cond == "none":
        return np.zeros(K)
    if cond == "common":
        return np.full(K, RMS_W)
    Z, _ = encode_metadata(meta)
    gamma = np.array([0.70, 0.0, 0.12, -0.06])
    w = Z @ gamma + rng.normal(0, 0.06, K)
    return w * (RMS_W / np.sqrt(np.mean(w ** 2)))


def gen_data(psi, w, rng):
    lam = rng.uniform(0.55, 0.85, K)
    F = rng.multivariate_normal([0, 0], [[1, psi], [psi, 1]], N)
    M = rng.normal(0, 1, N)
    theta = 1 - lam ** 2
    Y = np.empty((N, K))
    for i in range(K):
        f = F[:, 0] if i < NHALF else F[:, 1]
        Y[:, i] = lam[i] * f + w[i] * M + rng.normal(0, np.sqrt(theta[i]), N)
    return pd.DataFrame(Y, columns=ITEMS)


_CACHE = {}


def _get_model(desc, cache_key=None):
    """Constant model descriptions are parsed once per worker process and
    re-fit with fresh data; FAMF models (per-rep numeric loadings) are
    always rebuilt."""
    from semopy import Model
    if cache_key is None:
        return Model(desc)
    if cache_key not in _CACHE:
        _CACHE[cache_key] = Model(desc)
    return _CACHE[cache_key]


def fit_pair(desc, df, cache_key=None):
    """Fit full and cov-restricted model; return corr, LR, model, ok."""
    try:
        m = _get_model(desc, cache_key)
        r = m.fit(df)
        psi = m.mx_psi
        # F1,F2 are the first two latents in mx_psi for these models
        v1, v2, cv = psi[0, 0], psi[1, 1], psi[0, 1]
        if v1 <= 0 or v2 <= 0 or not np.isfinite(r.fun):
            return None
        corr = cv / np.sqrt(v1 * v2)
        m0 = _get_model(desc + "\nF1 ~~ 0*F2",
                        None if cache_key is None else cache_key + "_r")
        r0 = m0.fit(df)
        lr = max(0.0, (len(df) - 1) * (r0.fun - r.fun))
        return corr, lr, m, True
    except Exception:
        return None


def one_rep(args):
    rep, psi, cond = args
    rng = np.random.default_rng(10_000 * rep + hash((psi, cond)) % 9973)
    meta = make_metadata(rng)
    w_star = true_w(meta, cond, rng)
    df = gen_data(psi, w_star, rng)
    rows = []

    rb = fit_pair(trait_model_desc(SCALES), df, cache_key="base")
    if rb:
        corr, lr, base, _ = rb
        rows.append((rep, psi, cond, "Baseline", corr, lr, True))
    else:
        rows.append((rep, psi, cond, "Baseline", np.nan, np.nan, False))
        base = None

    rc = fit_pair(clf_model_desc(SCALES, ITEMS), df, cache_key="clf")
    rows.append((rep, psi, cond, "CLF") +
                ((rc[0], rc[1], True) if rc else (np.nan, np.nan, False)))

    if base is not None:
        meta_cal = meta.copy()
        if cond == "misaligned":
            perm = np.random.default_rng(rep).permutation(K)
            meta_cal = meta.iloc[perm].reset_index(drop=True)
        try:
            sigma = base.calc_sigma()[0]
            cal = calibrate(df, sigma, meta_cal, signal="eigen",
                            lambdas=LAMBDAS)
            rf = fit_pair(famf_model_desc(SCALES, ITEMS, cal["w"]), df)
        except Exception:
            rf = None
        rows.append((rep, psi, cond, "FAMF") +
                    ((rf[0], rf[1], True) if rf else (np.nan, np.nan, False)))
    else:
        rows.append((rep, psi, cond, "FAMF", np.nan, np.nan, False))
    return rows


def job_list(reps):
    cells = [(psi, cond)
             for psi in (0.0, 0.15, 0.40)
             for cond in ("none", "contrast", "misaligned", "common")]
    return [(rep, psi, cond) for psi, cond in cells for rep in range(reps)]


def main():
    start = int(sys.argv[1]); count = int(sys.argv[2])
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    outfile = sys.argv[4] if len(sys.argv) > 4 else "sim_results.csv"
    jobs = job_list(reps)[start:start + count]
    if not jobs:
        print("no jobs in range"); return
    t0 = time.time()
    with Pool(4) as pool:
        chunks = pool.map(one_rep, jobs, chunksize=10)
    rows = [r for ch in chunks for r in ch]
    res = pd.DataFrame(rows, columns=["rep", "psi", "cond", "model",
                                      "corr", "lr", "conv"])
    res.to_csv(outfile, mode="a", header=not os.path.exists(outfile),
               index=False)
    print(f"jobs {start}..{start+len(jobs)-1} done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
