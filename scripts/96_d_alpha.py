"""96_d: 같은 셀(발행월×세부등급) 안 상위 30% 알파 — 합산 방식, 투입자본당·전체자본당, 월 이동블록(6개월) 부트스트랩 2000회(seed 0), 특성집합 쌍 차이.
R(S)=(ΣTW_S/ΣA_S)^(12/H)−1 은 안전금리 차감 전. 알파_투입 = (ΣTW_sel/ΣA_sel)^(12/H) − (Σ_c A_sel,c·TW_c/A_c / ΣA_sel)^(12/H); 알파_전체 = 알파_투입 × ΣA_sel/ΣA_all.
(안전금리 항은 같은 셀 무작위와 선택이 같은 S 가중이라 상쇄.) 선택 = 셀 안 우선순위 정렬 후 금액 누적 ≤ 30%."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; C = importlib.import_module("96_common")
import numpy as np, pandas as pd
log = C.log
L = pd.read_parquet(C.WORK / "loans.parquet").set_index("row")
B, BL, FR = 2000, 6, 0.30
def cell_aggs(d, pr, H):
    key = d.issue_month.astype(str) + "|" + d.sub_grade
    df = pd.DataFrame(dict(m=d.issue_month.to_numpy(), c=key.to_numpy(), A=d.A.to_numpy(), TW=d.TW_short.to_numpy(), pr=pr))
    df = df.sort_values(["c", "pr"], kind="stable")
    g = df.groupby("c", sort=False)
    df["cumA"] = g.A.cumsum(); df["totA"] = g.A.transform("sum"); df["totTW"] = g.TW.transform("sum")
    sel = df.cumA <= FR * df.totA
    df["selA"] = np.where(sel, df.A, 0.0); df["selTW"] = np.where(sel, df.TW, 0.0); df["randTW"] = df.selA * df.totTW / df.totA
    return df.groupby("m")[["selA", "selTW", "randTW", "A"]].sum().sort_index()
def alpha_from(cnt, ag, H):
    sA, sT, rT, A = (cnt @ ag[c].to_numpy() for c in ["selA", "selTW", "randTW", "A"])
    inv = (sT / sA) ** (12 / H) - (rT / sA) ** (12 / H)
    return inv, inv * sA / A
def run(d, S, names, H, rng_seed=0):
    rng = np.random.RandomState(rng_seed); aggs = {}
    for nm in names:
        pr = -(d.NPV_short / d.A).to_numpy() if nm == "오라클" else (S[nm].to_numpy() if nm.split("|")[0] in ("clf", "logit") else -S[nm].to_numpy())
        aggs[nm] = cell_aggs(d, pr, H)
    M = len(next(iter(aggs.values()))); nb = -(-M // BL)
    st = rng.randint(0, M, (B, nb)); idx = ((st[:, :, None] + np.arange(BL)[None, None, :]) % M).reshape(B, -1)[:, :M]
    cnt = np.zeros((B, M)); np.add.at(cnt, (np.repeat(np.arange(B), M), idx.ravel()), 1)
    one = np.ones((1, M)); res = {}
    for nm, ag in aggs.items():
        res[nm] = (alpha_from(one, ag, H), alpha_from(cnt, ag, H), float(ag.selA.sum() / ag.A.sum()))
    return res, M
rows = []
SP = {"strict36": (36, {"헤드라인2015-16": (2015, 2016), "2015": (2015, 2015), "2016": (2016, 2016)}), "primary36": (36, {"헤드라인2015-16": (2015, 2016)}), "m60": (60, {"헤드라인2013-14": (2013, 2014)})}
pairs = [("reg|raw+nv전체", "reg|raw141"), ("clf|raw+nv전체", "clf|raw141"), ("reg|raw+채택", "reg|raw141"), ("reg|raw+채택+FLAG", "reg|raw141"), ("reg|raw141", "clf|raw141"), ("reg|raw+nv전체", "clf|raw+nv전체"), ("reg|nv만", "reg|raw141")]
for sp, (H, subs) in SP.items():
    S = pd.read_parquet(C.WORK / f"scores_{sp}.parquet").set_index("row"); D = L.loc[S.index]
    names = [c for c in S.columns if not c.startswith("logit") or True] + ["오라클"]
    for sn, (y0, y1) in subs.items():
        m = D.issue_year.between(y0, y1).to_numpy()
        res, M = run(D[m], S[m], names, H)
        for nm, ((i0, t0), (ib, tb), share) in res.items():
            rows.append(dict(split=sp, subset=sn, model=nm, months=M, sel_share=share, alpha_invested_pp=i0[0] * 100, inv_lo=np.percentile(ib, 2.5) * 100, inv_hi=np.percentile(ib, 97.5) * 100,
                             alpha_total_pp=t0[0] * 100, tot_lo=np.percentile(tb, 2.5) * 100, tot_hi=np.percentile(tb, 97.5) * 100, kind="level"))
        for a, b in pairs:
            if a in res and b in res:
                di, dt_ = res[a][1][0] - res[b][1][0], res[a][1][1] - res[b][1][1]
                rows.append(dict(split=sp, subset=sn, model=f"{a} − {b}", months=M, sel_share=np.nan, alpha_invested_pp=(res[a][0][0][0] - res[b][0][0][0]) * 100, inv_lo=np.percentile(di, 2.5) * 100, inv_hi=np.percentile(di, 97.5) * 100,
                             alpha_total_pp=(res[a][0][1][0] - res[b][0][1][0]) * 100, tot_lo=np.percentile(dt_, 2.5) * 100, tot_hi=np.percentile(dt_, 97.5) * 100, kind="pair_diff"))
        log(sp, sn)
pd.DataFrame(rows).to_csv(C.WORK / "alpha.csv", index=False, encoding="utf-8-sig")
log("done")
