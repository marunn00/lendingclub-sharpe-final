"""102 글래스박스: M1a EBM(interpret), M1b WoE 스코어카드(optbinning + 로지스틱). 점수 = PD."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; M = importlib.import_module("102_common"); B, K, log = M.B, M.K, M.log
import numpy as np, pandas as pd
CATC = lambda X: [c for c in X.columns if c in B.NOMINAL or c in B.ORD]

def m_ebm(tr, te, Xtr, Xte):
    from interpret.glassbox import ExplainableBoostingClassifier as EBM
    cc = [c for c in CATC(Xtr) if c in B.NOMINAL]                                   # 명목형만 범주, sub_grade 는 서수 숫자
    f = lambda Z: Z.assign(**{c: Z[c].map(lambda v: "NA" if v != v else str(int(v))) for c in cc})
    ft = ["nominal" if c in cc else "continuous" for c in Xtr.columns]
    m = EBM(feature_types=ft, interactions=10, max_bins=128, learning_rate=0.05, outer_bags=2, inner_bags=0, max_rounds=1500, early_stopping_rounds=40, n_jobs=10, random_state=0)
    m.fit(f(Xtr), M.y_all[tr]); p = m.predict_proba(f(Xte))[:, 1]
    import pickle; pickle.dump(m, open(M.W2 / "ebm_last.pkl", "wb"))
    return {"M1a": (p, p)}

def m_woe(tr, te, Xtr, Xte):
    from optbinning import BinningProcess
    from sklearn.linear_model import LogisticRegression
    cc = CATC(Xtr); names = list(Xtr.columns); ytr = M.y_all[tr]
    bp = BinningProcess(names, categorical_variables=cc, max_n_prebins=40, min_prebin_size=0.01, min_bin_size=0.02, max_n_bins=8, n_jobs=1, verbose=False)
    bp.fit(Xtr.to_numpy(np.float64), ytr)
    Wtr, Wte = bp.transform(Xtr.to_numpy(np.float64), metric="woe", metric_missing="empirical"), bp.transform(Xte.to_numpy(np.float64), metric="woe", metric_missing="empirical")
    lr = LogisticRegression(C=0.5, max_iter=500).fit(Wtr, ytr); p = lr.predict_proba(Wte)[:, 1]
    pd.DataFrame({"var": names, "coef": lr.coef_[0]}).to_csv(M.W2 / "woe_coef_last.csv", index=False, encoding="utf-8-sig")
    bp.summary().to_csv(M.W2 / "woe_binning_last.csv", index=False, encoding="utf-8-sig")
    return {"M1b": (p, p)}
