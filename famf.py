"""
famf.py — Metadata-only Feature-Augmented Method Factor (FAMF-SEM).

Reference implementation accompanying:
  Yaslioglu, M. "A Metadata-Only Feature-Augmented Method Factor for Ex-Post
  Correction and Attribution of Common Method Variance in Self-Report Health
  Instruments."

Pipeline
--------
1. Fit a baseline trait-only CFA; obtain the standardized residual
   correlation matrix R_res = R_observed - R_implied.
2. Signal: leading eigenvector v1 of R_res (signed; scaled by sqrt of the
   leading eigenvalue). Under a single method factor with loadings w,
   R_res ~ w w', so v1*sqrt(eig1) estimates w up to sign.
   (A magnitude-only fallback, mean |residual|, is also provided.)
3. Ridge-regress the signal on an intercept plus encoded item metadata
   (binary features effects-coded, numeric features standardized); the
   penalty applies to the feature coefficients only, and lambda is chosen
   by Generalized Cross-Validation (GCV; Golub, Heath & Wahba, 1979).
4. Normalize the fitted weights to sum(w^2)=k (a convention only) and fix
   them as the method-factor loading PATTERN in the SEM; the method-factor
   variance phi is FREELY estimated (phi>=0), so the data set the overall
   strength while the pattern stays fixed by design.
   With an uninformative Z the prediction collapses to the intercept,
   weights become equal, and FAMF reduces to the equal-loading CLF.

Requires: numpy, pandas, semopy (pip install semopy).
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------
def encode_metadata(meta: pd.DataFrame, binary=("Reversed", "Polarity")):
    """Effects-code binary features (center), z-score numeric features.
    Returns (Z, colnames)."""
    cols, names = [], []
    for c in meta.columns:
        x = meta[c].astype(float).to_numpy()
        if c in binary:
            cols.append(x - x.mean())
        else:
            s = x.std(ddof=1)
            cols.append((x - x.mean()) / s if s > 1e-12 else x * 0.0)
        names.append(c)
    return np.column_stack(cols), names


# ----------------------------------------------------------------------
# Residual signal
# ----------------------------------------------------------------------
def residual_corr(data: pd.DataFrame, sigma_implied: np.ndarray) -> np.ndarray:
    """Standardized residual correlation matrix (observed - implied)."""
    R_obs = np.corrcoef(data.to_numpy(), rowvar=False)
    d = np.sqrt(np.diag(sigma_implied))
    R_imp = sigma_implied / np.outer(d, d)
    return R_obs - R_imp


def eigen_signal(R_res: np.ndarray) -> np.ndarray:
    """Signed signal: leading eigenvector scaled by sqrt(leading eigenvalue).
    The diagonal of R_res is ~0 and does not affect eigenvectors up to a
    shift; we symmetrize for numerical safety."""
    S = (R_res + R_res.T) / 2.0
    vals, vecs = np.linalg.eigh(S)
    i = int(np.argmax(vals))
    lam1 = max(vals[i], 0.0)
    v = vecs[:, i] * np.sqrt(lam1)
    # orientation convention: majority-positive
    if np.sum(v) < 0:
        v = -v
    return v


def absmean_signal(R_res: np.ndarray) -> np.ndarray:
    """Magnitude-only fallback: mean absolute off-diagonal residual."""
    k = R_res.shape[0]
    A = np.abs(R_res).copy()
    np.fill_diagonal(A, 0.0)
    return A.sum(axis=1) / (k - 1)


# ----------------------------------------------------------------------
# Ridge with intercept + GCV
# ----------------------------------------------------------------------
def ridge_gcv(Z: np.ndarray, m: np.ndarray,
              lambdas=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)):
    """Ridge regression of m on [1, Z]; intercept unpenalized.
    Lambda chosen by GCV(l) = (1/k)||(I-H)m||^2 / (1 - tr(H)/k)^2.
    Returns dict(gamma0, gamma, w_raw, lambda, gcv_path, r2)."""
    k, p = Z.shape
    X = np.column_stack([np.ones(k), Z])
    D = np.eye(p + 1)
    D[0, 0] = 0.0                      # do not penalize the intercept
    best, path = None, []
    XtX = X.T @ X
    Xtm = X.T @ m
    for lam in lambdas:
        beta = np.linalg.solve(XtX + lam * D, Xtm)
        H = X @ np.linalg.solve(XtX + lam * D, X.T)
        resid = m - X @ beta
        trH = np.trace(H)
        denom = max(1e-9, (1.0 - trH / k)) ** 2
        gcv = (resid @ resid) / k / denom
        path.append((lam, gcv))
        if best is None or gcv < best[1]:
            best = (lam, gcv, beta)
    lam, _, beta = best
    w_raw = X @ beta
    ssm = np.sum((m - m.mean()) ** 2)
    r2 = 1.0 - np.sum((m - w_raw) ** 2) / ssm if ssm > 1e-12 else 0.0
    return dict(gamma0=beta[0], gamma=beta[1:], w_raw=w_raw,
                lam=lam, gcv_path=path, r2=r2)


def normalize_pattern(w_raw: np.ndarray) -> np.ndarray:
    """Scale (NO centering) so that sum(w^2)=k. Convention only: the
    method variance phi is freely estimated in the SEM."""
    k = len(w_raw)
    ss = float(np.sum(w_raw ** 2))
    if ss < 1e-12:
        return np.ones(k)              # degenerate -> CLF pattern
    return w_raw * np.sqrt(k / ss)


def calibrate(data: pd.DataFrame, sigma_implied: np.ndarray,
              meta: pd.DataFrame, signal="eigen", lambdas=None):
    """Full calibration. Returns dict with weights and diagnostics."""
    R_res = residual_corr(data, sigma_implied)
    m = eigen_signal(R_res) if signal == "eigen" else absmean_signal(R_res)
    Z, names = encode_metadata(meta)
    kw = {} if lambdas is None else {"lambdas": lambdas}
    fit = ridge_gcv(Z, m, **kw)
    w = normalize_pattern(fit["w_raw"])
    return dict(w=w, m=m, R_res=R_res, Z=Z, feature_names=names, **fit)


# ----------------------------------------------------------------------
# semopy model builders
# ----------------------------------------------------------------------
def trait_model_desc(scales: dict) -> str:
    """scales: {'F1': ['y1',...], 'F2': [...]} -> baseline CFA string."""
    lines = [f"{f} =~ " + " + ".join(items) for f, items in scales.items()]
    fs = list(scales)
    for a in range(len(fs)):
        for b in range(a + 1, len(fs)):
            lines.append(f"{fs[a]} ~~ {fs[b]}")
    return "\n".join(lines)


def famf_model_desc(scales: dict, items: list, w: np.ndarray) -> str:
    """Trait CFA + method factor M with FIXED loading pattern w and FREE
    variance phi, orthogonal to all traits."""
    lines = [trait_model_desc(scales)]
    lines.append("M =~ " + " + ".join(f"{w[i]:.5f}*{it}"
                                      for i, it in enumerate(items)))
    lines.append("M ~~ M")             # phi free
    for f in scales:
        lines.append(f"M ~~ 0*{f}")
    return "\n".join(lines)


def clf_model_desc(scales: dict, items: list) -> str:
    """Equal-loading common latent factor: loadings fixed to 1, variance
    free, orthogonal to traits."""
    w = np.ones(len(items))
    return famf_model_desc(scales, items, w)
