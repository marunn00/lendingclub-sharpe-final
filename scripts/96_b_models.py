"""96_b: 시간분할 외표본 모형. LightGBM(부도 분류·NPV 회귀, 고정 시드) + 로지스틱 기준선, 특성집합 5종.
분할: primary = 학습 발행≤2014(확정) / 시험 2015–16 (36m); strict = 학습 발행≤2014 & 확정월<2015-01; 60m = 학습 2012 / 시험 2013–14.
저장: _work/scores_<split>.parquet, _work/importance.csv, _work/model_metrics.csv"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; C = importlib.import_module("96_common")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
log = C.log
L = pd.read_parquet(C.WORK / "loans.parquet"); F = pd.read_parquet(C.WORK / "feats.parquet")
raw = [c for c in F if c.startswith("raw__")]; nv = [c for c in F if c.startswith("nv_")]
ACC = [c for c in nv if c.startswith(("nv_c04", "nv_c05", "nv_c07", "nv_is_owner"))]
FLAG = ["nv_new_acc_share24","nv_acc24_per_hist","nv_nonrevol_bal","nv_owner_x_hpi_x_mort","nv_stateur_x_burden","nv_zc_logrevbal","nv_lag_dr_cum","nv_lag_ra_spread","nv_lagdr_stale_mo"]
FS = {"raw141": raw, "raw+채택": raw + ACC, "raw+채택+FLAG": raw + ACC + [c for c in FLAG if c in F], "raw+nv전체": raw + nv, "nv만": nv}
FS = {k: [c for c in dict.fromkeys(v) if c in F] for k, v in FS.items()}
log({k: len(v) for k, v in FS.items()})
y = L.default.to_numpy(); v = (L.NPV_short / L.A).to_numpy()
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
         random_state=C.SEED, deterministic=True, force_row_wise=True, n_jobs=8, verbose=-1)
SPLITS = {
 "primary36": (lambda: (L.term == 36) & (L.issue_year <= 2014), lambda: (L.term == 36) & L.issue_year.between(2015, 2016), list(FS)),
 "strict36": (lambda: (L.term == 36) & (L.issue_year <= 2014) & (L.res_month < 120), lambda: (L.term == 36) & L.issue_year.between(2015, 2016), list(FS)),
 "m60": (lambda: (L.term == 60) & (L.issue_year == 2012), lambda: (L.term == 60) & L.issue_year.between(2013, 2014), ["raw141", "raw+nv전체"]),
}
imp, mets = [], []
log("60m strict(학습 확정월<2013-01) 가능 표본:", int(((L.term == 60) & (L.issue_year == 2012) & (L.res_month < 96)).sum()))
for sp, (trf, tef, fsets) in SPLITS.items():
    tr, te = np.where(trf())[0], np.where(tef())[0]
    log(sp, "train", len(tr), "test", len(te))
    S = pd.DataFrame({"row": L.row.to_numpy()[te]})
    for fsn in fsets:
        cols = FS[fsn]; Xtr, Xte = F.loc[tr, cols], F.loc[te, cols]
        m = lgb.LGBMClassifier(objective="binary", **P).fit(Xtr, y[tr]); p = m.predict_proba(Xte)[:, 1]
        r = lgb.LGBMRegressor(objective="huber", alpha=0.15, **P).fit(Xtr, v[tr]); q = r.predict(Xte)
        S[f"clf|{fsn}"] = p; S[f"reg|{fsn}"] = q
        mets.append(dict(split=sp, fs=fsn, model="lgb_clf", auc=roc_auc_score(y[te], p), n_feat=len(cols), n_train=len(tr), n_test=len(te)))
        mets.append(dict(split=sp, fs=fsn, model="lgb_reg", auc=roc_auc_score(y[te], -q), n_feat=len(cols), n_train=len(tr), n_test=len(te)))
        if fsn in ("raw141", "raw+nv전체"):
            rs = np.random.RandomState(C.SEED).choice(len(te), 20000, replace=False)
            for nm, mod in (("clf", m), ("reg", r)):
                sh = mod.booster_.predict(Xte.iloc[rs], pred_contrib=True)[:, :-1]
                g = mod.booster_.feature_importance("gain")
                imp.append(pd.DataFrame(dict(split=sp, fs=fsn, model=nm, feature=cols, mean_abs_shap=np.abs(sh).mean(0), gain_share=g / g.sum())))
        if fsn in ("raw141", "raw+nv전체"):
            keep = [c for c in cols if not c.startswith(("raw__addr_state", "raw__purpose", "raw__home_ownership", "raw__verification", "raw__initial", "raw__application", "raw__emp_length", "raw__grade"))]
            Z = pd.concat([Xtr[keep], Xte[keep]]); med = Z.median(); Z = Z.fillna(med).fillna(0)
            ind = pd.concat([Xtr[keep], Xte[keep]]).isna().astype(float)[[c for c in keep if F[c].isna().any()]].add_suffix("_na")
            Z = pd.concat([Z, ind], axis=1); sc = StandardScaler().fit(Z.iloc[:len(tr)])
            lr = LogisticRegression(C=0.1, max_iter=300, random_state=C.SEED).fit(sc.transform(Z.iloc[:len(tr)]), y[tr])
            lp = lr.predict_proba(sc.transform(Z.iloc[len(tr):]))[:, 1]
            S[f"logit|{fsn}"] = lp
            mets.append(dict(split=sp, fs=fsn, model="logit_clf", auc=roc_auc_score(y[te], lp), n_feat=Z.shape[1], n_train=len(tr), n_test=len(te)))
        log(sp, fsn, [(x["model"], round(x["auc"], 4)) for x in mets[-3:]])
    S.to_parquet(C.WORK / f"scores_{sp}.parquet", index=False)
pd.concat(imp).to_csv(C.WORK / "importance.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(mets).to_csv(C.WORK / "model_metrics.csv", index=False, encoding="utf-8-sig")
log("done")
