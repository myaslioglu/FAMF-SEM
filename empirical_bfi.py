"""
empirical_bfi.py — FAMF-SEM empirical illustration on the SAPA bfi data
(25 self-report items, 5 factors; Revelle, psych R package; distributed
via pydataset). Items measure Agreeableness, Conscientiousness,
Extraversion, Emotional Stability (reflected Neuroticism), and Openness;
the Emotional Stability block (anger, irritability, mood, feeling blue,
panic) is directly relevant to mental-health screening.

Usage: python3 empirical_bfi.py
"""
import warnings, json
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from famf import calibrate, famf_model_desc, clf_model_desc, trait_model_desc
from semopy import Model
from semopy.stats import calc_stats
from pydataset import data

ITEMS = [f"{s}{i}" for s in "ACENO" for i in range(1, 6)]
SCALES = {s: [f"{s}{i}" for i in range(1, 6)] for s in "ACENO"}
KEY = {  # administered wording direction (+1/-1), from bfi.dictionary
 "A1":-1,"A2":1,"A3":1,"A4":1,"A5":1, "C1":1,"C2":1,"C3":1,"C4":-1,"C5":-1,
 "E1":-1,"E2":-1,"E3":1,"E4":1,"E5":1, "N1":-1,"N2":-1,"N3":-1,"N4":-1,"N5":-1,
 "O1":1,"O2":-1,"O3":1,"O4":1,"O5":-1}


def factor_corrs(model, factors):
    ins = model.inspect()
    V = {f: float(ins[(ins.lval == f) & (ins.op == "~~")
                      & (ins.rval == f)].Estimate.iloc[0]) for f in factors}
    C = {}
    for a in range(len(factors)):
        for b in range(a + 1, len(factors)):
            f, g = factors[a], factors[b]
            row = ins[(ins.lval == f) & (ins.op == "~~") & (ins.rval == g)]
            if row.empty:
                row = ins[(ins.lval == g) & (ins.op == "~~") & (ins.rval == f)]
            C[(f, g)] = float(row.Estimate.iloc[0]) / np.sqrt(V[f] * V[g])
    return C


def main():
    bfi = data("bfi")
    d = data("bfi.dictionary")
    df = bfi[ITEMS].dropna().astype(float)
    n = len(df)

    # standard scoring: reflect reverse-keyed items (6-point scale)
    for it in ITEMS:
        if KEY[it] < 0:
            df[it] = 7 - df[it]

    # metadata: keying direction; item length = words in the actual item text
    texts = d.loc[ITEMS, "Item"]
    meta = pd.DataFrame({
        "Polarity": [KEY[i] for i in ITEMS],
        "Length":   [len(str(t).split()) for t in texts]}, index=ITEMS)
    print(f"n = {n} complete cases, k = {len(ITEMS)} items")
    print(meta.T)

    # 1) baseline 5-factor CFA
    base = Model(trait_model_desc(SCALES))
    base.fit(df)
    sigma = base.calc_sigma()[0]
    st_b = calc_stats(base)

    # 2) calibration
    cal = calibrate(df, sigma, meta, signal="eigen")
    print(f"\nGCV lambda = {cal['lam']},  R2(m, m_hat) = {cal['r2']:.3f}")
    print("gamma0 (intercept) = %.4f" % cal["gamma0"])
    for nm, g in zip(cal["feature_names"], cal["gamma"]):
        print(f"gamma[{nm}] = {g:.4f}")
    w = cal["w"]
    print("\nweights w_i (pattern, sum w^2 = k):")
    print(pd.Series(np.round(w, 3), index=ITEMS).to_string())
    # correlation between weight and keying
    print("corr(w, Polarity) = %.3f"
          % np.corrcoef(w, meta.Polarity.astype(float))[0, 1])

    # 3) FAMF refit (fixed pattern, free phi)
    famf = Model(famf_model_desc(SCALES, ITEMS, w))
    famf.fit(df)
    ins = famf.inspect()
    phi = float(ins[(ins.lval == "M") & (ins.op == "~~")
                    & (ins.rval == "M")].Estimate.iloc[0])
    st_f = calc_stats(famf)

    # 4) CLF comparison
    clf = Model(clf_model_desc(SCALES, ITEMS))
    clf.fit(df)
    insc = clf.inspect()
    phic = float(insc[(insc.lval == "M") & (insc.op == "~~")
                      & (insc.rval == "M")].Estimate.iloc[0])
    st_c = calc_stats(clf)

    # method variance share per item (FAMF): w_i^2 phi / total var
    sv = famf.calc_sigma()[0]
    share = np.array([w[i] ** 2 * phi / sv[i, i] for i in range(len(ITEMS))])
    print(f"\nphi (FAMF method variance) = {phi:.4f}")
    print(f"mean method-variance share  = {share.mean()*100:.1f}% "
          f"(max {share.max()*100:.1f}%)")
    print(f"phi (CLF) = {phic:.4f}")

    def fitrow(st):
        return (float(st.chi2.iloc[0]), int(st.DoF.iloc[0]),
                float(st.CFI.iloc[0]), float(st.RMSEA.iloc[0]))
    for nm, st in [("Baseline", st_b), ("CLF", st_c), ("FAMF", st_f)]:
        c2, dof, cfi, rm = fitrow(st)
        print(f"{nm:9s} chi2={c2:9.1f} df={dof} CFI={cfi:.3f} RMSEA={rm:.3f}")

    fb = factor_corrs(base, list(SCALES))
    ff = factor_corrs(famf, list(SCALES))
    fc = factor_corrs(clf, list(SCALES))
    print("\nfactor correlations: baseline -> FAMF (delta) | CLF")
    for kpair in fb:
        print(f"{kpair[0]}-{kpair[1]}: {fb[kpair]:+.3f} -> {ff[kpair]:+.3f} "
              f"({ff[kpair]-fb[kpair]:+.3f}) | {fc[kpair]:+.3f}")

    json.dump({
        "n": n, "lam": cal["lam"], "r2": cal["r2"],
        "gamma0": cal["gamma0"],
        "gamma": dict(zip(cal["feature_names"], map(float, cal["gamma"]))),
        "w": dict(zip(ITEMS, map(float, w))),
        "phi": phi, "phi_clf": phic,
        "share_mean": float(share.mean()), "share_max": float(share.max()),
        "fit": {"base": fitrow(st_b), "clf": fitrow(st_c), "famf": fitrow(st_f)},
        "corr_base": {f"{a}-{b}": v for (a, b), v in fb.items()},
        "corr_famf": {f"{a}-{b}": v for (a, b), v in ff.items()},
        "corr_clf":  {f"{a}-{b}": v for (a, b), v in fc.items()},
    }, open("empirical_results.json", "w"), indent=1)
    print("\nsaved -> empirical_results.json")


if __name__ == "__main__":
    main()
