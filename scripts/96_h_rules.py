"""96_h: 규칙 기준선 vs 모형(raw141) — strict36, 시험 2015–16(36m), 포트폴리오 수준 비교.
규칙: 전량, 등급컷(A–B/A–C/A–D), 금리 낮은 순/높은 순(발행월 안 순위), 세부등급만(낮은 위험 순). 모형: clf|raw141, reg|raw141(발행월 안 순위).
승인율 일치 비교: 30%, 53%, 규칙 자체 승인율(등급컷·전량: 시험 기간 금액 비중을 1% 격자로 반올림해 모형을 같은 승인율로 맞춤).
월 초과수익 e_m = ((ΣTW_sel+Σ(A−A_sel)·S_m)/ΣA)^(12/H) − S_m^(12/H) (거절 자본 = FEDFUNDS). Sharpe = mean/sd(월 빈티지 e_m), 연환산 아님.
6개월 순환 블록 부트스트랩 2000회 seed 0, 모형과 규칙은 같은 재표집 인덱스(쌍 차이)."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; C = importlib.import_module("96_common")
import numpy as np, pandas as pd
src = open(__import__("pathlib").Path(__file__).parent / "96_c_strategies.py").read()
exec(src[src.index("L = pd.read_parquet"):src.index("SPL = ")])   # 엔진 함수 재사용(월 집계·부트스트랩 인덱스)
H = 36
S = pd.read_parquet(C.WORK / "scores_strict36.parquet").set_index("row"); D = L.loc[S.index]
sel = D.issue_year.between(2015, 2016).to_numpy(); d = D[sel].reset_index(drop=True); Ss = S[sel].reset_index(drop=True)
ms = np.sort(d.issue_month.unique()); M = len(ms); mpos = {m: i for i, m in enumerate(ms)}
mi = d.issue_month.map(mpos).to_numpy()
Am = np.bincount(mi, d.A.to_numpy()); Sm = np.array([d.S[mi == i].iloc[0] for i in range(M)])
idx = boot_idx(M, np.random.RandomState(C.SEED)); cnt = np.zeros((B, M)); np.add.at(cnt, (np.repeat(np.arange(B), M), idx.ravel()), 1)
def stats(selA, selTW):
    e = ((selTW + (Am - selA) * Sm) / Am) ** (12 / H) - Sm ** (12 / H)
    def R(c):
        W = c @ Am; num = c @ (selTW + (Am - selA) * Sm); return (num / W) ** (12 / H) - (c @ (Am * Sm) / W) ** (12 / H)
    eb = e[idx]; return R(np.ones(M)), R(cnt), e.mean() / e.std(ddof=1), eb.mean(1) / eb.std(1, ddof=1), float(selA.sum() / Am.sum())
rng = np.random.RandomState(C.SEED); noise = 1e-4 * rng.rand(len(d))
PR = {"규칙|금리 낮은 순": d.int_rate.to_numpy() + noise, "규칙|금리 높은 순": -d.int_rate.to_numpy() + noise,
      "규칙|세부등급만(낮은 위험 순)": d.sub_grade.astype("category").cat.codes.to_numpy() + noise,
      "모형|부도확률(raw141)": Ss["clf|raw141"].to_numpy(), "모형|NPV(raw141)": -Ss["reg|raw141"].to_numpy()}
AGG = {k: month_aggs(d, p, H) for k, p in PR.items()}
res = {}   # (name, matched rate) -> stats
for k, ag in AGG.items():
    for a in (0.30, 0.53):
        j = int(round(a * 100)) - 1; res[(k, a)] = stats(ag["selA"][:, j], ag["selTW"][:, j])
GR = {"규칙|등급컷 A–B": list("AB"), "규칙|등급컷 A–C": list("ABC"), "규칙|등급컷 A–D": list("ABCD"), "규칙|전량(A–G)": list("ABCDEFG")}
own = {}
for k, gs in GR.items():
    m = d.grade.isin(gs).to_numpy(); selA = np.bincount(mi[m], d.A.to_numpy()[m], minlength=M); selTW = np.bincount(mi[m], d.TW_short.to_numpy()[m], minlength=M)
    st = stats(selA, selTW); share = st[4]; own[k] = share; res[(k, "own")] = st
    j = int(round(share * 100)) - 1
    for mk in ("모형|부도확률(raw141)", "모형|NPV(raw141)"): res[(mk, k)] = stats(AGG[mk]["selA"][:, j], AGG[mk]["selTW"][:, j])
q = lambda x: np.percentile(x, [2.5, 97.5])
rows = []
def add(name, a, st, kind):
    rows.append(dict(전략=name, 구분=kind, 비교_승인율=a, 실제_승인율_금액비중=st[4], 초과수익_pp=st[0] * 100, 초과수익_lo_pp=q(st[1])[0] * 100, 초과수익_hi_pp=q(st[1])[1] * 100, Sharpe=st[2], Sharpe_lo=q(st[3])[0], Sharpe_hi=q(st[3])[1]))
for (k, a), st in res.items():
    if a in (0.30, 0.53): lab = a
    elif a == "own": lab = f"자체 {own[k]*100:.0f}%"
    else: lab = f"{a.split('|')[1]} 승인율 일치({own[a]*100:.0f}%)"
    add(k, lab, st, "모형" if k.startswith("모형") else "규칙")
T = pd.DataFrame(rows)
# 쌍 차이: 모형 − 최선 규칙(점추정 초과수익 최대)
pairs = []
def pair(label, a_lab, mk, rk, stm, str_):
    dR = (stm[1] - str_[1]) * 100; dS = stm[3] - str_[3]
    pairs.append(dict(비교=label, 승인율=a_lab, 모형=mk, 최선_규칙=rk, 차_초과수익_pp=(stm[0] - str_[0]) * 100, lo_pp=q(dR)[0], hi_pp=q(dR)[1], 차_Sharpe=stm[2] - str_[2], Sharpe_lo=q(dS)[0], Sharpe_hi=q(dS)[1]))
for a in (0.30, 0.53):
    rules = [k for k in PR if k.startswith("규칙")]; bR = max(rules, key=lambda k: res[(k, a)][0]); bS = max(rules, key=lambda k: res[(k, a)][2])
    for mk in ("모형|부도확률(raw141)", "모형|NPV(raw141)"):
        pair("초과수익 기준 최선 규칙", f"{int(a*100)}%", mk, bR, res[(mk, a)], res[(bR, a)])
        pair("Sharpe 기준 최선 규칙", f"{int(a*100)}%", mk, bS, res[(mk, a)], res[(bS, a)])
for k in GR:
    for mk in ("모형|부도확률(raw141)", "모형|NPV(raw141)"): pair("규칙 자체 승인율", f"{k.split('|')[1]} ({own[k]*100:.0f}%)", mk, k, res[(mk, k)], res[(k, "own")])
P = pd.DataFrame(pairs)
T.to_csv(C.WORK / "rules_summary.csv", index=False, encoding="utf-8-sig"); P.to_csv(C.WORK / "rules_pairs.csv", index=False, encoding="utf-8-sig")
O = C.WORK.parent; T.to_csv(O / "표_규칙기준선.csv", index=False, encoding="utf-8-sig"); P.to_csv(O / "표_규칙기준선_쌍차이.csv", index=False, encoding="utf-8-sig")
# 곡선(그림용): 월 안 순위 규칙/모형의 승인율별 Sharpe·초과수익
cur = []
for k, ag in AGG.items():
    c, _, _ = analyse(ag, np.random.RandomState(C.SEED)); c.insert(0, "strategy", k); cur.append(c)
pd.concat(cur).to_csv(C.WORK / "rules_curves.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(T.round(3).to_string()); print(P.round(3).to_string()); C.log("done")
