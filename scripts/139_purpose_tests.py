"""139 목적·정합성 검정: Sharpe가 어디서 나오는가(월 안 선별 vs 월 간 배분), 월 간 배분과 월 평균 초과수익이 거시 변수로 설명·예측되는가.
입력(읽기만): 학습 OOF 점수(대안모델/_work/oof_V3_t1.npz), 시험 점수(최종보고서/test_scores_v3.parquet), 시간변수_거시.csv.
출력: 산출물/목적검수/*.csv, 요약.json. 라쏘·릿지는 Sharpe 추정이 아니라 거시 변수 정보량 검정 도구로 쓴다."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, json, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
M = importlib.import_module("102_common"); R = importlib.import_module("121_rulers"); ROOT = M.K.C.ROOT
MODE = sys.argv[1]; OUT = ROOT / "산출물" / "목적검수" / MODE; OUT.mkdir(parents=True, exist_ok=True); rng = np.random.RandomState(0); Q = 0.26

dtr = R.frame(ROOT / "lending_club_2020_train.csv").set_index("id").loc[M.L.id.astype(str)].reset_index()
S3 = np.load(M.W2 / "oof_V3_t1.npz")["score"]; keep = ~np.isnan(S3); dtr = dtr[keep].reset_index(drop=True); dtr["score"] = S3[keep]; dtr["src"] = "학습OOF"
dte = R.frame(ROOT / "lending_club_2020_test.csv").merge(pd.read_parquet(ROOT / "산출물/최종보고서/test_scores_v3.parquet")[["id", "score_V3"]], on="id", how="left", validate="1:1")
dte = dte.rename(columns={"score_V3": "score"}); dte["src"] = "시험"
D = pd.concat([dtr, dte], ignore_index=True); D = D[D.ref].reset_index(drop=True) if MODE == "matured" else D; D["z"] = D.r1 - D.rf; D["T"] = D["T"].astype(int)
D["gi"] = D.sub_grade.map({g: i for i, g in enumerate(sorted(D.sub_grade.unique()))}); D["m60"] = (D["T"] == 60).astype(float)
print("모집단", len(D), D.groupby("src").size().to_dict())

# ---------- E1 Sharpe 분해 ----------
def SR(x, rf): return (x.mean() - rf.mean()) / x.std(ddof=1)
rf, r = D.rf.to_numpy(), D.r1.to_numpy(); cut = np.quantile(D.score, Q); ap = (D.score < cut).to_numpy()
rk_in_month = D.groupby("ym").score.rank(method="first", pct=True).to_numpy(); ap_m = rk_in_month <= Q
cell = ap_m.copy()
sr_all = SR(r, rf); sr_pool = SR(np.where(ap, r, rf), rf)
sr_within = SR(np.where(ap_m, r, rf), rf)
rs = []
for s in range(20):
    a = np.random.RandomState(s).rand(len(D)) < Q; rs.append(SR(np.where(a, r, rf), rf))
sr_rand = float(np.mean(rs))
dec = pd.DataFrame([["전부 승인", sr_all, 1.0], ["무작위 26% 승인(20회 평균)", sr_rand, Q], ["월마다 26% 고정, 월 안 점수 순위(월 안 선별만)", sr_within, Q], ["전체 한 줄 기준선, 승인 26%(최종 규칙)", sr_pool, ap.mean()]], columns=["전략", "Sharpe", "승인율"])
dec.to_csv(OUT / "E1_Sharpe_분해.csv", index=False, encoding="utf-8-sig")

# ---------- 월 표 ----------
g = D.assign(ap=ap, apm=ap_m).groupby("ym")
mt = pd.DataFrame(dict(n=g.size(), zbar=g.z.mean(), rbar=g.r1.mean(), pd=g.dflt.mean(), q=g.ap.mean(), q_in=g.apm.mean(), gi=g.gi.mean(), m60=g.m60.mean(), amt=g.funded_amnt.mean(), rf=g.rf.mean(), sc=g.score.mean()))
za = D[ap].groupby("ym").z.mean(); mt["z_ap"] = za; mt["sel_prem"] = mt.z_ap - mt.zbar
mt = mt[mt.n >= 200]; mt["date"] = pd.to_datetime(mt.index.astype(str), format="%Y%m"); mt.to_csv(OUT / "월표.csv", encoding="utf-8-sig")
per = pd.cut(mt.index, [200600, 201112, 201612, 202012], labels=["2007–11", "2012–16", "2017–20"])
pr = mt.groupby(per, observed=True).apply(lambda t: pd.Series(dict(월수=len(t), 대출수=int(t.n.sum()), 승인율=float(np.average(t.q, weights=t.n)), 평균_초과수익_전체=float(np.average(t.zbar, weights=t.n)), 평균_초과수익_승인=float(np.nansum(t.z_ap * t.q * t.n) / np.sum(t.q * t.n)), 평균_부도율=float(np.average(t.pd, weights=t.n)), 평균_등급지수=float(np.average(t.gi, weights=t.n)), 평균_rf=float(np.average(t.rf, weights=t.n)))))
pr.to_csv(OUT / "E2_시기별_승인과_수준.csv", encoding="utf-8-sig")
c_q_z = float(np.corrcoef(mt.q, mt.zbar)[0, 1]); from scipy.stats import spearmanr; sp_q_z = float(spearmanr(mt.q, mt.zbar)[0]); sp_q_rf = float(spearmanr(mt.q, mt.rf)[0]); sp_q_gi = float(spearmanr(mt.q, mt.gi)[0])
# 분산 분해: 대출 수익 x의 월 간 분산 비중
x = np.where(ap, r, rf); between = pd.Series(x).groupby(D.ym.to_numpy()).transform("mean"); var_between = float(between.var(ddof=1) / pd.Series(x).var(ddof=1))

# ---------- 거시 입력 ----------
mc = pd.read_csv(ROOT / "산출물/시간변수_테이블/시간변수_거시.csv", encoding="utf-8-sig"); mc["ym"] = pd.to_datetime(mc.month).dt.year * 100 + pd.to_datetime(mc.month).dt.month; mc = mc.set_index("ym")
cols = ["UNRATE", "FEDFUNDS", "DGS3", "SLOPE_10y3y", "BAA10Y", "VIXCLS", "NFCI", "STLFSI4", "CPI_yoy", "INDPRO_yoy", "PAYEMS_yoy", "HPI_yoy", "UMCSENT"]
mc = mc[cols].ffill(); mcl = mc.shift(1)                      # 공개지연 1개월: 발급월 값 대신 직전 월 값
ym_ok = [y for y in mt.index if y in mcl.index and mcl.loc[y].notna().all()]; T = mt.loc[ym_ok]; Xm = mcl.loc[ym_ok]
ur = mc.UNRATE; idx = list(mc.index); life = pd.Series({y: ur.get(idx[idx.index(y) + 18], np.nan) - ur.loc[y] for y in T.index if idx.index(y) + 18 < len(idx)}).dropna()

# ---------- E3 사후 변수(대출 기간 중 실업률 변화)와 사전 거시의 설명력 ----------
def r2(y, X): X1 = np.c_[np.ones(len(y)), X]; b = np.linalg.lstsq(X1, y, rcond=None)[0]; e = y - X1 @ b; return 1 - e @ e / ((y - y.mean()) @ (y - y.mean()))
ys = T.zbar.loc[life.index].to_numpy(); e3 = [["대출 기간 중 실업률 변화(사후)", len(life), r2(ys, life.to_numpy())]]
for nm_, c_ in (("발급 시점 실업률 수준(사전)", ["UNRATE"]), ("발급 시점 금리 3개(사전)", ["FEDFUNDS", "DGS3", "SLOPE_10y3y"])):
    e3.append([nm_, len(life), r2(ys, Xm.loc[life.index, c_].to_numpy())])
e3.append(["발급 시점 13개 거시(사전, 표본 안)", len(life), r2(ys, Xm.loc[life.index].to_numpy())])
pd.DataFrame(e3, columns=["설명 변수", "월 수", "R2(표본 안)"]).to_csv(OUT / "E3_사후_대_사전.csv", index=False, encoding="utf-8-sig")

# ---------- E4 앞으로 가며 예측: 내부 구성 vs 거시 ----------
Cc = T[["gi", "m60", "amt"]].to_numpy(); Mm = Xm.to_numpy(); Cm = np.c_[Cc, Mm]; L0 = 36
def fwd(y, X, kind):
    pr_ = []
    for t in range(L0, len(y)):
        tr = slice(0, t)
        if X is None: pr_.append(y[tr].mean()); continue
        sc = StandardScaler().fit(X[tr]); A, B = sc.transform(X[tr]), sc.transform(X[t:t + 1])
        mod = RidgeCV(alphas=np.logspace(-2, 3, 30)) if kind == "ridge" else LassoCV(cv=TimeSeriesSplit(4), max_iter=20000, alphas=40)
        pr_.append(mod.fit(A, y[tr] - y[tr].mean()).predict(B)[0] + y[tr].mean())
    return np.array(pr_)
def blk_ci(d, bl=6, B=1000):
    n = len(d); st = np.arange(0, n, bl); out = []
    for _ in range(B):
        ii = np.concatenate([np.arange(s, min(s + bl, n)) for s in rng.choice(st, len(st))])[:n]; out.append(d[ii].mean())
    return np.percentile(out, [2.5, 97.5])
rows = []
for tgt, lab in (("zbar", "월 평균 초과수익"), ("rbar", "월 평균 총수익"), ("q", "승인율(공통 점수선)")):
    y = T[tgt].to_numpy(); yy = y[L0:]; base = fwd(y, None, "ridge"); sb = (yy - base) ** 2
    for nm_, X in (("내부 구성 3개(등급지수·60개월 비중·금액)", Cc), ("거시 13개(1개월 지연)", Mm), ("구성 + 거시", Cm)):
        for kind in ("ridge", "lasso"):
            p = fwd(y, X, kind); se = (yy - p) ** 2; d_ = sb - se
            rows.append(dict(대상=lab, 입력=nm_, 방법=kind, 평가월수=len(yy), OOS_R2=float(1 - se.sum() / sb.sum()), 손실차_평균=float(d_.mean()), 구간하한=float(blk_ci(d_)[0]), 구간상한=float(blk_ci(d_)[1])))
e4 = pd.DataFrame(rows); e4.to_csv(OUT / "E4_순차예측.csv", index=False, encoding="utf-8-sig")

# ---------- E7 월별 승인 규모를 거시 예측으로 조절하면 얼마나 이득인가(완전예지 상한과 비교) ----------
zb = T.zbar.to_numpy(); pz = fwd(zb, Mm, "ridge"); evm = T.index[L0:]; thr = np.array([np.median(zb[:t]) for t in range(L0, len(T))])
msk = D.ym.isin(evm).to_numpy(); Dm = D[msk].reset_index(drop=True); rkm = rk_in_month[msk]; apm_f = ap[msk]
def sums(a): x = np.where(a, Dm.r1.to_numpy(), Dm.rf.to_numpy()); return pd.DataFrame(dict(n=1, s=x, s2=x * x, f=Dm.rf.to_numpy())).groupby(Dm.ym.to_numpy()).sum()
def sr_from(S): n = S.n.sum(); m = S.s.sum() / n; v = (S.s2.sum() - n * m * m) / (n - 1); return (m - S.f.sum() / n) / np.sqrt(v)
strat = {"고정 26%(월 안 순위)": rkm <= Q, "거시 예측으로 조절(상위 1.5배·하위 0.5배)": rkm <= (pd.Series(np.where(pz > thr, 1.5, 0.5) * Q, index=evm).reindex(Dm.ym).to_numpy()), "완전예지로 조절(상한)": rkm <= (pd.Series(np.where(zb[L0:] > thr, 1.5, 0.5) * Q, index=evm).reindex(Dm.ym).to_numpy()), "전체 한 줄 기준선(최종 규칙)": apm_f}
SS = {k: sums(v) for k, v in strat.items()}; base = SS["고정 26%(월 안 순위)"]; mon = base.index; B_ = 1000; bl = 6; st = np.arange(0, len(mon), bl); idxs = []
for _ in range(B_): idxs.append(np.concatenate([np.arange(s, min(s + bl, len(mon))) for s in rng.choice(st, len(st))])[:len(mon)])
rows = []
for k, S in SS.items():
    bs = np.array([sr_from(S.iloc[i]) - sr_from(base.iloc[i]) for i in idxs]); rows.append(dict(전략=k, Sharpe=float(sr_from(S)), 고정대비_차=float(sr_from(S) - sr_from(base)), 구간하한=float(np.percentile(bs, 2.5)), 구간상한=float(np.percentile(bs, 97.5)), 승인율=float(np.mean([0]) if False else (np.average(strat[k]))), 평가월수=len(mon)))
pd.DataFrame(rows).to_csv(OUT / "E7_거시조절_이득.csv", index=False, encoding="utf-8-sig")

# ---------- E5 라쏘 선택 안정성: 월 블록 부트스트랩 ----------
def stab(y, X, B=200, bl=6):
    n = len(y); st = np.arange(0, n, bl); cnt = np.zeros(X.shape[1])
    for _ in range(B):
        ii = np.concatenate([np.arange(s, min(s + bl, n)) for s in rng.choice(st, len(st))])[:n]; sc = StandardScaler().fit(X[ii])
        m = LassoCV(cv=5, max_iter=20000, alphas=30).fit(sc.transform(X[ii]), y[ii]); cnt += (np.abs(m.coef_) > 1e-8)
    return cnt / B
s_z = stab(T.zbar.to_numpy(), Mm); s_q = stab(T.q.to_numpy(), Mm)
pd.DataFrame(dict(변수=cols, 선택비율_월평균초과수익=s_z, 선택비율_승인율=s_q)).to_csv(OUT / "E5_라쏘_선택안정성.csv", index=False, encoding="utf-8-sig")

# ---------- E6 월 안 선별의 안정성 ----------
sp = mt.sel_prem.dropna()
json.dump(dict(표본=len(D), 월수=len(mt), 월수_거시=len(T), 분산_월간비중=var_between, 상관_q_zbar=c_q_z, 스피어만_q_zbar=sp_q_z, 스피어만_q_rf=sp_q_rf, 스피어만_q_gi=sp_q_gi, 선별프리미엄_평균=float(sp.mean()), 선별프리미엄_양수월=float((sp > 0).mean()), 사후사전_월수=int(len(life)), 승인율_공통선=float(ap.mean())), open(OUT / "요약.json", "w"), ensure_ascii=False, indent=1)
print(dec.round(4).to_string(index=False)); print(pr.round(4).to_string()); print(pd.DataFrame(e3).round(3).to_string()); print(e4.round(4).to_string()); print(open(OUT / "요약.json").read())
print(pd.read_csv(OUT / "E5_라쏘_선택안정성.csv").round(2).to_string()); print(pd.read_csv(OUT / "E7_거시조절_이득.csv").round(4).to_string())
