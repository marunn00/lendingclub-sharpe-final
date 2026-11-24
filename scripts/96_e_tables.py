"""96_e: 시장 기준선·빈티지·등급, 다리표, D–G/A–C, 2016 등급, 60m vs 36m, 순손실·빈티지 곡선, 스트레스 격자, 보정·ROC·리프트·AUC CI.
모든 입력은 train_새테이블.parquet 에서 만든 _work/loans.parquet·scores_*.parquet 와 같은 표의 total_rec_prncp·recoveries 열."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; C = importlib.import_module("96_common")
import numpy as np, pandas as pd, polars as pl
from sklearn.metrics import roc_auc_score, roc_curve
log = C.log; W = C.WORK
L = pd.read_parquet(W / "loans.parquet")
t = pl.read_parquet(C.TABLE, columns=["loan_status", "issue_d", "term", "funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee"]).filter(pl.col("issue_d").is_not_null() & pl.col("loan_status").is_in(["Fully Paid", "Charged Off", "Default"]))
t = t.with_columns(pl.col("term").str.strip_chars().str.slice(0, 2).cast(pl.Int32).alias("tm"), pl.col("issue_d").str.slice(4, 4).cast(pl.Int32).alias("iy"))
t = t.filter(((pl.col("tm") == 36) & pl.col("iy").is_between(2012, 2016)) | ((pl.col("tm") == 60) & pl.col("iy").is_between(2012, 2014))).to_pandas()
assert len(t) == len(L) and np.allclose(pd.to_numeric(t.funded_amnt).to_numpy(), L.A.to_numpy())
for c in ["total_rec_prncp", "recoveries", "collection_recovery_fee"]: L[c] = pd.to_numeric(t[c]).fillna(0).to_numpy()
L["net_loss"] = L.A - L.total_rec_prncp - L.recoveries + L.collection_recovery_fee   # 순손실(원금 손실 − 순회수), 사후 확정값
L["AS"] = L.A * L.S; L["AT"] = L.A * L.rT
# ---------- 1. 시장 기준선: 집합 지표
def agg(g, H):
    A = g.A.sum(); safe = (g.AS.sum() / A) ** (12 / H) - 1
    return dict(n=len(g), A=A, default_rate=g.default.mean(), default_rate_amt=(g.A * g.default).sum() / A, E_npv=g.NPV_short.sum() / A,
        R_reinv=(g.TW_short.sum() / A) ** (12 / H) - 1 - safe, R_noreinv=(g.cash_total.sum() / A) ** (12 / H) - 1 - safe, safe_ann=safe,
        R_noreinv_locked_tsy=(g.cash_total.sum() / A) ** (12 / H) - 1 - g.AT.sum() / A, E_npv_tsy=g.NPV_tsy.sum() / A, lifetime_net_loss=g.net_loss.sum() / A,
        ret_total_pymnt_ann=(0.99 * g.total_pymnt.sum() / A) ** (12 / H) - 1 - g.AT.sum() / A)
rows = []
for (tm, y), g in L.groupby(["term", "issue_year"]): rows.append(dict(term=tm, year=y, grade="전체", **agg(g, tm)))
for (tm, y, gr), g in L.groupby(["term", "issue_year", "grade"]): rows.append(dict(term=tm, year=y, grade=gr, **agg(g, tm)))
for (tm, y, gg), g in L.assign(gg=np.where(L.grade.isin(list("ABC")), "A-C", "D-G")).groupby(["term", "issue_year", "gg"]): rows.append(dict(term=tm, year=y, grade=gg, **agg(g, tm)))
MK = pd.DataFrame(rows); MK.to_csv(W / "market.csv", index=False, encoding="utf-8-sig"); log("market")
# 월 블록 부트스트랩 CI (연도별 전체 R_reinv, 블록 3개월, 2000회)
rng = np.random.RandomState(C.SEED); ci = []
for (tm, y), g in L.groupby(["term", "issue_year"]):
    mo = g.groupby("issue_month")[["A", "TW_short", "AS", "NPV_short", "cash_total"]].sum(); M = len(mo); nb = -(-M // 3)
    st = rng.randint(0, M, (2000, nb)); idx = ((st[:, :, None] + np.arange(3)[None, None, :]) % M).reshape(2000, -1)[:, :M]
    a = mo.to_numpy()[idx].sum(1); H = tm
    R = (a[:, 1] / a[:, 0]) ** (12 / H) - 1 - ((a[:, 2] / a[:, 0]) ** (12 / H) - 1); E = a[:, 3] / a[:, 0]; Rn = (a[:, 4] / a[:, 0]) ** (12 / H) - 1 - ((a[:, 2] / a[:, 0]) ** (12 / H) - 1)
    ci.append(dict(term=tm, year=y, R_reinv_lo=np.percentile(R, 2.5), R_reinv_hi=np.percentile(R, 97.5), E_npv_lo=np.percentile(E, 2.5), E_npv_hi=np.percentile(E, 97.5), R_noreinv_lo=np.percentile(Rn, 2.5), R_noreinv_hi=np.percentile(Rn, 97.5)))
pd.DataFrame(ci).to_csv(W / "market_ci.csv", index=False, encoding="utf-8-sig")
# ---------- 2. 다리표 (36m 2014–2016)
br = []
for y in (2014, 2015, 2016):
    g = L[(L.term == 36) & (L.issue_year == y)]; a = agg(g, 36); A = g.A.sum()
    br.append(dict(year=y, old_style_0p99TP_locked_tsy=a["ret_total_pymnt_ann"], same_assump_cash_locked_tsy=a["R_noreinv_locked_tsy"], R_noreinv=a["R_noreinv"], R_reinv=a["R_reinv"],
                   safe_locked_tsy=g.AT.sum() / A, safe_realized_ffr=a["safe_ann"]))
pd.DataFrame(br).to_csv(W / "bridge.csv", index=False, encoding="utf-8-sig")
# ---------- 3. 60m vs 36m 짝지음 (등급×발행연도 2012–14, 36m 자본 가중), 월 블록 부트스트랩
sub = L[L.issue_year <= 2014].copy(); sub["cell"] = sub.grade + sub.issue_year.astype(str)
mo = sorted(sub.issue_month.unique()); M = len(mo)
cells = sorted(sub.cell.unique())
def tens(col, tm):
    x = sub[sub.term == tm].pivot_table(index="issue_month", columns="cell", values=col, aggfunc="sum").reindex(index=mo, columns=cells).fillna(0.0); return x.to_numpy()
T = {tm: {c: tens(c, tm) for c in ["A", "TW_short", "AS", "NPV_short"]} for tm in (36, 60)}
def pair_stat(cnt):
    out = []
    for tm in (36, 60):
        A = cnt @ T[tm]["A"]; out.append((((cnt @ T[tm]["TW_short"]) / A) ** (12 / tm) - 1 - (((cnt @ T[tm]["AS"]) / A) ** (12 / tm) - 1), (cnt @ T[tm]["NPV_short"]) / A, A))
    ok = (out[0][2] > 0) & (out[1][2] > 0); w = out[0][2] * ok
    d0 = np.where(ok, out[1][0] - out[0][0], 0.0); d1 = np.where(ok, out[1][1] - out[0][1], 0.0)
    return (d0 * w).sum(1) / w.sum(1), (d1 * w).sum(1) / w.sum(1)
nb = -(-M // 6); st = rng.randint(0, M, (2000, nb)); idx = ((st[:, :, None] + np.arange(6)[None, None, :]) % M).reshape(2000, -1)[:, :M]
cnt = np.zeros((2000, M)); np.add.at(cnt, (np.repeat(np.arange(2000), M), idx.ravel()), 1)
p0 = pair_stat(np.ones((1, M))); pb = pair_stat(cnt)
pd.DataFrame([dict(metric="R_reinv 연환산 차(60m−36m) %p", value=p0[0][0] * 100, lo=np.percentile(pb[0], 2.5) * 100, hi=np.percentile(pb[0], 97.5) * 100, n_cells=len(cells)),
              dict(metric="원금당 NPV 차(60m−36m) %p (기간 상이, 참고)", value=p0[1][0] * 100, lo=np.percentile(pb[1], 2.5) * 100, hi=np.percentile(pb[1], 97.5) * 100, n_cells=len(cells))]).to_csv(W / "pair60.csv", index=False, encoding="utf-8-sig")
log("pair60")
# ---------- 4. 빈티지×나이 곡선
V = np.load(W / "vintage_age.npz"); vr = []
for tm in (36, 60):
    for y in range(2012, 2017):
        if f"{tm}_{y}_A" not in V: continue
        A = float(V[f"{tm}_{y}_A"]); net, wo = V[f"{tm}_{y}_net"], V[f"{tm}_{y}_wo"]
        cn, cw = np.cumsum(net) / A, np.cumsum(wo) / A
        for a in range(len(net)): vr.append(dict(term=tm, year=y, age=a, cum_net_cash_per_A=cn[a], cum_writeoff_per_A=cw[a], writeoff_in_month=wo[a] / A))
pd.DataFrame(vr).to_csv(W / "vintage_curves.csv", index=False, encoding="utf-8-sig")
# ---------- 5. 스트레스 격자 (36m 2012–16, 전체/연도/등급)
def stress(g, label):
    A = g.A.sum(); d = g.default == 1; Ad = g.A[d].sum(); pi = Ad / A
    Vnd = g.NPV_short[~d].sum() / g.A[~d].sum(); Vd = g.NPV_short[d].sum() / Ad; ell = g.net_loss[d].sum() / Ad
    out = []
    for m in (1, 1.25, 1.5, 2, 3):
        for k in (1, 1.1, 1.25):
            p = min(m * pi, 1); out.append(dict(group=label, pd_mult=m, lgd_mult=k, excess_pct=((1 - p) * Vnd + p * (Vd - (k - 1) * ell)) * 100))
    be = Vnd / (pi * (Vnd - Vd)) if Vnd > 0 else np.nan
    return out, dict(group=label, base_excess_pct=(Vnd * (1 - pi) + Vd * pi) * 100, pi=pi, V_nondefault=Vnd, V_default=Vd, loss_per_default_dollar=ell, breakeven_pd_mult=be)
g36 = L[L.term == 36]; srows, brows = [], []
for lab, g in [("전체 2012–16", g36)] + [(f"{y}", g36[g36.issue_year == y]) for y in range(2012, 2017)] + [(f"등급{gr}", g36[g36.grade == gr]) for gr in "ABCDEFG"]:
    a, b = stress(g, lab); srows += a; brows.append(b)
pd.DataFrame(srows).to_csv(W / "stress_grid.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(brows).to_csv(W / "stress_breakeven.csv", index=False, encoding="utf-8-sig"); log("stress")
# ---------- 6. 보정·ROC·리프트·AUC CI (primary36 시험)
ALL = {k: [] for k in ("cal", "roc", "lift", "aucs")}
for SPN in ("strict36", "primary36"):
    S = pd.read_parquet(W / f"scores_{SPN}.parquet").set_index("row"); D = L.loc[S.index]; y = D.default.to_numpy()
    mon = D.issue_month.to_numpy(); ums = np.sort(np.unique(mon)); pos = {m: np.where(mon == m)[0] for m in ums}
    cal, roc, lift, aucs = [], [], [], []
    for nm in ["clf|raw141", "clf|raw+채택", "clf|raw+채택+FLAG", "clf|raw+nv전체", "clf|nv만", "logit|raw141", "logit|raw+nv전체"]:
        p = S[nm].to_numpy(); q = pd.qcut(p, 10, labels=False, duplicates="drop")
        for k in range(10):
            s = q == k; cal.append(dict(model=nm, decile=k + 1, mean_pred=p[s].mean(), obs_rate=y[s].mean(), n=int(s.sum())))
            lift.append(dict(model=nm, decile=k + 1, lift=y[s].mean() / y.mean(), capture=y[s].sum() / y.sum()))
        f, t_, _ = roc_curve(y, p); sel = np.linspace(0, len(f) - 1, 300).astype(int)
        for i in sel: roc.append(dict(model=nm, fpr=f[i], tpr=t_[i]))
        r2 = np.random.RandomState(C.SEED); bs = []
        for _ in range(100):
            st = r2.randint(0, len(ums), -(-len(ums) // 6)); mm = ((st[:, None] + np.arange(6)[None, :]) % len(ums)).ravel()[:len(ums)]
            ix = np.concatenate([pos[ums[i]] for i in mm]); bs.append(roc_auc_score(y[ix], p[ix]))
        lg = np.log(p.clip(1e-6, 1 - 1e-6) / (1 - p.clip(1e-6, 1 - 1e-6)))
        from sklearn.linear_model import LogisticRegression
        cs = LogisticRegression(C=1e6, max_iter=200).fit(lg[:, None], y)
        aucs.append(dict(model=nm, auc=roc_auc_score(y, p), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), brier=float(((p - y) ** 2).mean()), base_rate=y.mean(), mean_pred=p.mean(), cal_slope=float(cs.coef_[0, 0]), cal_intercept=float(cs.intercept_[0]), top_decile_lift=lift[-1]["lift"]))
    for nm in ["reg|raw141", "reg|raw+nv전체"]:
        p = S[nm].to_numpy(); aucs.append(dict(model=nm + "(−예측NPV를 부도점수로)", auc=roc_auc_score(y, -p), lo=np.nan, hi=np.nan, brier=np.nan, base_rate=y.mean(), mean_pred=np.nan, cal_slope=np.nan, cal_intercept=np.nan, top_decile_lift=np.nan))
    for k, v in (("cal", cal), ("roc", roc), ("lift", lift), ("aucs", aucs)):
        ALL[k] += [dict(split=SPN, **r) for r in v]
for k, fn in (("cal", "calibration"), ("roc", "roc"), ("lift", "lift"), ("aucs", "auc")): pd.DataFrame(ALL[k]).to_csv(W / f"{fn}.csv", index=False, encoding="utf-8-sig")
log("done")
