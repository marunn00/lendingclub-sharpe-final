"""96_c: 승인율 곡선·프런티어·전략 비교(전량/등급컷/부도모형/NPV모형/로지스틱/오라클)·허들. 월 빈티지(발행월) 시계열 기준.
승인 = 발행월 안에서 우선순위 점수로 정렬해 금액 누적비율 ≤ a 인 대출(월별 같은 승인율). 거절 자본은 실현 FEDFUNDS(안전).
월 초과수익 e_m(a) = R_total: ((ΣTW_sel+Σ(A_rej)·S_m)/ΣA)^(12/H) − S_m^(12/H). Sharpe = mean(e_m)/sd(e_m). 블록 부트스트랩: 순환 이동블록 6개월, 2000회, seed 0.
구방식(22_frontier) Sharpe: 대출별 연환산 초과(0.99·total_pymnt, 국채)의 횡단 sd, 시험 전체 풀 정렬."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; C = importlib.import_module("96_common")
import numpy as np, pandas as pd
log = C.log
L = pd.read_parquet(C.WORK / "loans.parquet").set_index("row")
AG = np.round(np.arange(0.01, 1.0001, 0.01), 2); NA = len(AG)
B, BL = 2000, 6
SUBSETS = {"헤드라인2015-16": (2015, 2016), "2015": (2015, 2015), "2016": (2016, 2016)}
def priority(S, name, rng, d):
    kind, fs = name.split("|") if "|" in name else (name, None)
    if name == "전량(무작위)": return rng.rand(len(d))
    if name == "등급컷": return d.sub_grade.astype("category").cat.codes.to_numpy() + 0.001 * rng.rand(len(d))
    if name == "오라클(실현NPV)": return -(d.NPV_short / d.A).to_numpy()
    return S[name].to_numpy() if kind in ("clf", "logit") else -S[name].to_numpy()
def month_aggs(d, pr, H, cell=False):
    """cell=False: 발행월 안 순위(금액 누적 ≤ a). cell=True: 발행월×세부등급 셀 안 순위(구 정의)."""
    ms = np.sort(d.issue_month.unique()); M = len(ms); mpos = {m: i for i, m in enumerate(ms)}
    selA, selTW, selNPV = (np.zeros((M, NA)) for _ in range(3)); Am, Sm = np.zeros(M), np.zeros(M)
    mon = d.issue_month.to_numpy(); key = pd.Series(mon).to_numpy() if not cell else (pd.Series(mon).astype(str) + "|" + d.sub_grade.reset_index(drop=True)).to_numpy()
    Aall, TWall, NPVall, Sall = d.A.to_numpy(), d.TW_short.to_numpy(), d.NPV_short.to_numpy(), d.S.to_numpy()
    order = np.argsort(key, kind="stable"); ks = key[order]; cuts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        g = order[c0:c1]; i = mpos[mon[g[0]]]; o = g[np.argsort(pr[g], kind="stable")]
        cA = np.concatenate([[0], np.cumsum(Aall[o])]); tot = cA[-1]
        cT = np.concatenate([[0], np.cumsum(TWall[o])]); cN = np.concatenate([[0], np.cumsum(NPVall[o])])
        k = np.searchsorted(cA[1:] / tot, AG + 1e-12, side="right")
        selA[i] += cA[k]; selTW[i] += cT[k]; selNPV[i] += cN[k]; Am[i] += tot; Sm[i] = Sall[o[0]]
    return dict(ms=ms, selA=selA, selTW=selTW, selNPV=selNPV, Am=Am, Sm=Sm, H=H)
def sel_mask(d, pr, cell, a=0.30):
    mon = d.issue_month.to_numpy(); key = mon if not cell else (pd.Series(mon).astype(str) + "|" + d.sub_grade.reset_index(drop=True)).to_numpy()
    df = pd.DataFrame(dict(k=key, A=d.A.to_numpy(), pr=pr, i=np.arange(len(d)))).sort_values(["k", "pr"], kind="stable")
    g = df.groupby("k", sort=False); m = (g.A.cumsum() <= a * g.A.transform("sum")).to_numpy()
    out = np.zeros(len(d), bool); out[df.i.to_numpy()[m]] = True; return out
def series(ag):
    num = ag["selTW"] + (ag["Am"][:, None] - ag["selA"]) * ag["Sm"][:, None]
    return (num / ag["Am"][:, None]) ** (12 / ag["H"]) - ag["Sm"][:, None] ** (12 / ag["H"])   # e_m(a)
def pooled(ag, cnt):
    H = ag["H"]; W = cnt @ ag["Am"]
    num = cnt @ (ag["selTW"] + (ag["Am"][:, None] - ag["selA"]) * ag["Sm"][:, None])
    safe = (cnt @ (ag["Am"] * ag["Sm"]) / W) ** (12 / H)
    return (num / W[:, None] if num.ndim > 1 else num / W) ** (12 / H) - (safe[:, None] if num.ndim > 1 else safe), (cnt @ ag["selNPV"]) / W[:, None]
def boot_idx(M, rng):
    nb = -(-M // BL); st = rng.randint(0, M, (B, nb)); idx = ((st[:, :, None] + np.arange(BL)[None, None, :]) % M).reshape(B, -1)[:, :M]; return idx
def analyse(ag, rng):
    e = series(ag); M = e.shape[0]; idx = boot_idx(M, rng)
    cnt = np.zeros((B, M)); np.add.at(cnt, (np.repeat(np.arange(B), M), idx.ravel()), 1)
    e_b = e[idx]; sr_b = e_b.mean(1) / e_b.std(1, ddof=1)
    Rb, Eb = pooled(ag, cnt); R0, E0 = pooled(ag, np.ones((1, M))); R0, E0 = R0[0], E0[0]
    sr = e.mean(0) / e.std(0, ddof=1)
    q = lambda x: np.percentile(x, [2.5, 97.5], axis=0)
    cur = pd.DataFrame(dict(a=AG, excess_pooled=R0, E_npv_total=E0, e_mean=e.mean(0), e_sd=e.std(0, ddof=1), sharpe=sr,
        excess_lo=q(Rb)[0], excess_hi=q(Rb)[1], sharpe_lo=q(sr_b)[0], sharpe_hi=q(sr_b)[1], enpv_lo=q(Eb)[0], enpv_hi=q(Eb)[1]))
    am_sr = AG[np.nanargmax(sr_b, 1)]; am_ex = AG[np.argmax(Rb, 1)]
    return cur, dict(argmax_sharpe=AG[np.nanargmax(sr)], argmax_sharpe_ci=np.percentile(am_sr, [2.5, 97.5]).tolist(), argmax_excess=AG[np.argmax(R0)],
                     argmax_excess_ci=np.percentile(am_ex, [2.5, 97.5]).tolist(), months=M), (cnt, e, ag)
def oldstyle(d, pr):
    o = np.argsort(pr, kind="stable"); A = d.A.to_numpy()[o]
    r = ((0.99 * d.total_pymnt.to_numpy()[o] / A).clip(1e-9) ** (12 / d.term.to_numpy()[o]) - 1); t = d.rT.to_numpy()[o]
    cA, cAr, cAr2 = (np.concatenate([[0], np.cumsum(x)]) for x in (A, A * r, A * r * r)); W = cA[-1]
    tA, tAt, tAt2 = (np.concatenate([[0], np.cumsum(x)]) for x in (A, A * t, A * t * t))
    k = np.searchsorted(cA[1:] / W, AG + 1e-12, side="right")
    Rp = (cAr[k] + (tAt[-1] - tAt[k])) / W; rbar = tAt[-1] / W
    var = ((cAr2[k] - 2 * Rp * cAr[k] + Rp ** 2 * cA[k]) + ((tAt2[-1] - tAt2[k]) - 2 * Rp * (tAt[-1] - tAt[k]) + Rp ** 2 * (W - cA[k]))) / W
    return Rp - rbar, np.sqrt(var)
SPL = {"strict36": 36, "primary36": 36, "m60": 60}
curves, summ, hur, comp = [], [], [], []
for sp, H in SPL.items():
    S = pd.read_parquet(C.WORK / f"scores_{sp}.parquet").set_index("row"); D = L.loc[S.index]
    names = ["전량(무작위)", "등급컷", "오라클(실현NPV)"] + list(S.columns)
    subs = SUBSETS if sp == "strict36" else {"헤드라인": (int(D.issue_year.min()), int(D.issue_year.max()))} if sp == "m60" else {"헤드라인2015-16": (2015, 2016)}
    for sn, (y0, y1) in subs.items():
        sel = D.issue_year.between(y0, y1).to_numpy(); d = D[sel].reset_index(drop=True); Ss = S[sel].reset_index(drop=True)
        jobs = [(nm, "월") for nm in names] + [(nm, "셀") for nm in names if nm not in ("전량(무작위)", "등급컷")]
        for nm, unit in jobs:
            cell = unit == "셀"; rng = np.random.RandomState(C.SEED); pr = priority(Ss, nm, rng, d)
            ag = month_aggs(d, pr, H, cell); cur, sm, (cnt, e, _) = analyse(ag, np.random.RandomState(C.SEED))
            ox, osd = oldstyle(d, pr); cur["old_excess"] = ox; cur["old_sd"] = osd; cur["old_SR"] = ox / osd
            cur.insert(0, "split", sp); cur.insert(1, "subset", sn); cur.insert(2, "strategy", nm); cur.insert(3, "rank_unit", unit); curves.append(cur)
            r30, r53 = cur.iloc[29], cur.iloc[52]
            Rb, _ = pooled(ag, cnt); d3053 = Rb[:, 29] - Rb[:, 52]; e_b = e[boot_idx(e.shape[0], np.random.RandomState(C.SEED))]
            sr_b = e_b.mean(1) / e_b.std(1, ddof=1)
            summ.append(dict(split=sp, subset=sn, strategy=nm, rank_unit=unit, H=H, months=sm["months"], argmax_sharpe=sm["argmax_sharpe"], argmax_sharpe_lo=sm["argmax_sharpe_ci"][0], argmax_sharpe_hi=sm["argmax_sharpe_ci"][1],
                argmax_excess=sm["argmax_excess"], argmax_excess_lo=sm["argmax_excess_ci"][0], argmax_excess_hi=sm["argmax_excess_ci"][1],
                R30=r30.excess_pooled, R53=r53.excess_pooled, R100=cur.iloc[99].excess_pooled, diff30_53=r30.excess_pooled - r53.excess_pooled, diff30_53_lo=np.percentile(d3053, 2.5), diff30_53_hi=np.percentile(d3053, 97.5),
                SR30=r30.sharpe, SR53=r53.sharpe, SR30_lo=r30.sharpe_lo, SR30_hi=r30.sharpe_hi, SR53_lo=r53.sharpe_lo, SR53_hi=r53.sharpe_hi, SR30_minus_SR53_lo=np.percentile(sr_b[:, 29] - sr_b[:, 52], 2.5), SR30_minus_SR53_hi=np.percentile(sr_b[:, 29] - sr_b[:, 52], 97.5),
                SRmax=cur.sharpe.max(), old_argmax_SR=AG[cur.old_SR.values.argmax()], old_SR_max=cur.old_SR.max(), old_excess_at_SRmax=cur.old_excess.values[cur.old_SR.values.argmax()], old_argmax_excess=AG[cur.old_excess.values.argmax()]))
            for h in (0, 0.5, 1, 1.5, 2, 3, 4):
                c = (1 + h / 100) ** (H / 12) - 1
                val = (ag["selNPV"] - ag["selA"] * c).sum(0) / ag["Am"].sum()
                hur.append(dict(split=sp, subset=sn, strategy=nm, rank_unit=unit, h_pp=h, best_a=AG[np.argmax(val)], best_profit_pct=val.max() * 100, profit_at_100_pct=val[-1] * 100))
            if sn.startswith("헤드라인") and nm.split("|")[0] in ("clf", "reg") and nm.split("|")[1] in ("raw141", "raw+nv전체"):
                m = sel_mask(d, pr, cell); x = d[m]
                comp.append(dict(split=sp, strategy=nm, rank_unit=unit, sel_share_amt=x.A.sum() / d.A.sum(), int_rate_mean_pct=np.average(x.int_rate, weights=x.A), default_rate=x.default.mean(),
                                 share_CG_amt=x.A[x.grade.isin(list("CDEFG"))].sum() / x.A.sum(), share_AB_amt=x.A[x.grade.isin(list("AB"))].sum() / x.A.sum(), monthly_excess_sd_pp=cur.e_sd.iloc[29] * 100, monthly_excess_mean_pp=cur.e_mean.iloc[29] * 100))
        log(sp, sn, "done")
pd.concat(curves).to_csv(C.WORK / "curves.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(summ).to_csv(C.WORK / "strategy_summary.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(hur).to_csv(C.WORK / "hurdle.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(comp).to_csv(C.WORK / "selection_composition.csv", index=False, encoding="utf-8-sig")
log("done")
