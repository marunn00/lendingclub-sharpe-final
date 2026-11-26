"""140 시대 열 제거 검정: 점수에서 발급연도·발급 시점 금리 열을 빼면 월 간 배분과 Sharpe가 어떻게 달라지는가.
학습 OOF 5겹, 같은 단일 회귀(XGB, z=r−rf)를 열 전체/시대 열 제외 두 번 적합해 같은 방식으로 비교한다."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, numpy as np, pandas as pd
M = importlib.import_module("102_common"); MD = importlib.import_module("102_models"); R = importlib.import_module("121_rulers"); ROOT = M.K.C.ROOT
OUT = ROOT / "산출물" / "목적검수"; ERA = ["issue_year", "tsy10_at_issue", "tsy_matched_at_issue", "spread_fedfunds"]
d = R.frame(ROOT / "lending_club_2020_train.csv").set_index("id").loc[M.L.id.astype(str)].reset_index(); z = (d.r1 - d.rf).to_numpy(); zt = np.clip(z, -3, 0.5)
S3 = np.load(M.W2 / "oof_V3_t1.npz")["score"]; keep = ~np.isnan(S3); sc = {"v3 기준(참고)": S3.copy(), "전체 열": np.full(M.n, np.nan), "시대 열 제외": np.full(M.n, np.nan)}
for k in range(5):
    tr, te, Xtr, Xte = M.fold_data(k, M.IX, "full"); era = [c for c in Xtr.columns if c in ERA]; assert len(era) >= 1, list(Xtr.columns)[:20]
    for nm, cols in (("전체 열", list(Xtr.columns)), ("시대 열 제외", [c for c in Xtr.columns if c not in ERA])):
        p, _ = MD._xgb_reg(Xtr[cols].to_numpy(), zt[tr], np.ones(len(tr)), Xte[cols].to_numpy()); sc[nm][te] = -p
    print("fold", k, "era cols", era, flush=True)
rows = []
from scipy.stats import spearmanr
for lab, sub in (("전체(미확정 포함)", np.ones(len(d), bool)), ("만기 확정 표본", d.ref.to_numpy())):
    kk = keep & sub; dd = d[kk].reset_index(drop=True); rf, r = dd.rf.to_numpy(), dd.r1.to_numpy(); SRf = lambda x: (x.mean() - rf.mean()) / x.std(ddof=1)
    for nm, s in sc.items():
        s = s[kk]; a = s < np.quantile(s, 0.26); pm = pd.Series(s).groupby(dd.ym.to_numpy()).rank(method="first", pct=True).to_numpy() <= 0.26
        q_m = pd.Series(a).groupby(dd.ym.to_numpy()).mean(); mt = pd.Series(r - rf).groupby(dd.ym.to_numpy()).mean()
        rows.append(dict(표본=lab, 점수=nm, Sharpe_전체한줄=float(SRf(np.where(a, r, rf))), Sharpe_월안선별만=float(SRf(np.where(pm, r, rf))), 월간배분_기여=float(SRf(np.where(a, r, rf)) - SRf(np.where(pm, r, rf))), 승인율_월간_표준편차=float(q_m.std()), 스피어만_승인율_월평균초과수익=float(spearmanr(q_m, mt.reindex(q_m.index))[0])))
o = pd.DataFrame(rows); o.to_csv(OUT / "E8_시대열_제거.csv", index=False, encoding="utf-8-sig"); print(o.round(4).to_string())
