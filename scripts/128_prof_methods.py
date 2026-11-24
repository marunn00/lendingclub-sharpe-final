"""128 교수님 강조 사항의 실증 구현(최종 보고서 Ⅵ장). 최종 모델(v3)·운영 지점(q* = 124 학습 OOF)은 바꾸지 않는 진단·보완 방안 분석.
기준(실행 전 고정)
 표본: 시험 결과 확정 대출 중 만기 기준 코호트(36개월 ≤ 2017-03, 60개월 ≤ 2015-03 발급; 우측 중도절단 영향 최소화). 초과수익 z = T1 대출 수익 r1 − rf(%p).
 A 듀레이션: 계약 가중평균만기(WAL)·맥컬리 듀레이션(쿠폰 할인) 대 실현 WAL(= 받은 이자 ÷ 금리 ÷ 원금, 잔액·년). 수익률 환산 3종 비교, 듀레이션 일관 수익률 y = (총수령/원금)^(1/실현 WAL) − 1로 T1을 다시 계산.
 B OLS(statsmodels, 발급월 군집 표준오차): M1 z~스프레드, M2 +세부등급 더미, M3 +부도확률, M4 +우리 예측 초과수익 μ̂. 예측 불편성(Mincer–Zarnowitz) z = α + β·μ̂, H0: α=0, β=1.
 C 10분위 롱−숏: 발급월×만기 안에서 점수 10분위(1 = 가장 좋음). 월별 D1 − D10 평균 초과수익, Newey–West t(시차 6). 비교: 세부등급·부도확률 순위.
 D 1·2차 모먼트: μ̂ 10분위별 예측 평균 대 실현 평균, σ̂ 10분위별 예측 표준편차 대 실현 표준편차.
 E 레짐 전환(statsmodels MarkovRegression, 2국면, 평균·분산 전환): 학습 파일 36개월 대출의 발급월 평균 z(월 300건 이상, ≤ 2017-03). 국면별로 시험 T1(최종 모형 대 전부 승인).
 F 림보(사람 심사) 3분류: 림보 = 승인 경계 ±2.5%p ∪ [승인 예정 ∩ (3모형 부도확률 불일치 상위 5% ∪ 희소 구간(학습의 등급×목적×만기 1,000건 미만))]. T1: 자동 규칙 / 림보 전부 거절 / 심사자 정확도 70%(난수 시드 0) / 완전 심사(상한).
 G 비정형(직함): 학습 빈도 30건 미만 직함 = 희소. 희소·일반·미기재별 AUC. 희소 직함의 키워드 직군(규칙 공개)별 잔차(실제 부도 − 예측 부도확률), n ≥ 1,000만.
출력: 산출물/최종보고서/표/교수_*.csv, 교수_요약.json"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, json, re, numpy as np, pandas as pd, polars as pl, statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
R = importlib.import_module("121_rulers"); K = R.K; CF = K.C; log = K.log          # 'C' 는 회귀식의 범주 함수 C() 와 겹치므로 쓰지 않는다
OUT = CF.ROOT / "산출물" / "최종보고서"; TB = OUT / "표"; TEST, TRAIN = CF.ROOT / "lending_club_2020_test.csv", CF.ROOT / "lending_club_2020_train.csv"
Q = float(pd.read_csv(TB / "v3_판정.csv").set_index("model").loc["V3", "q_star"]); SUMM = dict(q_star=Q)

def extra(csv):
    e = pl.read_csv(csv, columns=["id", "int_rate", "emp_title", "purpose", "grade"], infer_schema_length=0).to_pandas(); e["id"] = e.id.astype(str)
    e["rate"] = pd.to_numeric(e.int_rate.str.strip().str.replace("%", ""), errors="coerce") / 100; return e.drop(columns="int_rate")      # total_rec_int 은 R.frame 에 있음
d = R.frame(TEST).merge(extra(TEST), on="id", validate="1:1")
d = d.merge(pd.read_parquet(OUT / "test_scores_v3.parquet"), on="id", validate="1:1").merge(pd.read_parquet(CF.ROOT / "산출물/테스트평가/test_scores_v2.parquet")[["id", "pd_xgb", "pd_lgb", "pd_cat"]], on="id", validate="1:1")
d = d.merge(pd.read_parquet(CF.ROOT / "산출물/테스트평가/test_scores.parquet")[["id", "score_B0"]], on="id", validate="1:1"); assert len(d) == 744608 and d.rate.notna().all()
o_all, rk_all = R.order_rank(d.score_V3.to_numpy()); APP = R.sel(rk_all, Q)                        # 운영 규칙: 시험 확정 풀 상위 q*
d["z"] = (d.r1 - d.rf) * 100; d["mu"] = d.exp_gain_v3 * 100; d["sig"] = d.risk_v3 * 100; d["spread"] = (d.rate - d.rf) * 100
E = d[d.ref].reset_index(drop=True); log("만기 기준 코호트", len(E))

# ---- A 듀레이션
def contract(rate, T):
    i = rate / 12; k = np.arange(T); g = (1 + i) ** T; bal = (g - (1 + i) ** k) / (g - 1); wal = bal.sum() / 12
    t = np.arange(1, T + 1); v = (1 + i) ** (-t.astype(float)); return wal, (t * v).sum() / v.sum() / 12
u = d[["rate", "T"]].drop_duplicates(); cw = {(r, t): contract(r, t) for r, t in zip(u.rate, u["T"])}
d["wal_c"] = [cw[(r, t)][0] for r, t in zip(d.rate, d["T"])]; d["dur_c"] = [cw[(r, t)][1] for r, t in zip(d.rate, d["T"])]
d["wal_r"] = d.total_rec_int / (d.rate * d.funded_amnt); d["y_wal"] = (d.total_pymnt / d.funded_amnt) ** (1 / np.maximum(d.wal_r, 1 / 12)) - 1
L = pd.read_parquet(CF.ROOT / "산출물/테스트평가/loans_test.parquet")[["id", "ann", "A", "roe_ok"]]; L["id"] = L.id.astype(str); d = d.merge(L, on="id", validate="1:1")
d["r_hold"] = np.where(d.roe_ok, d.ann / d.A, np.nan); E = d[d.ref].reset_index(drop=True)
DUR = E.assign(상태=np.where(E.dflt, "부도", "완납")).groupby(["T", "상태"]).agg(n=("z", "size"), 계약만기_년=("T", lambda s: s.iloc[0] / 12), 계약WAL_년=("wal_c", "mean"), 계약듀레이션_년=("dur_c", "mean"),
      실현WAL_년=("wal_r", "mean"), 약정금리=("rate", "mean"), 수익_계약만기환산=("r1", "mean"), 수익_실보유기간환산=("r_hold", "mean"), 수익_듀레이션일관=("y_wal", "mean")).reset_index()
DUR.to_csv(TB / "교수_A_듀레이션.csv", index=False, encoding="utf-8-sig"); print(DUR.round(4).to_string())
dd = d.copy(); dd["r1"] = dd.y_wal; TA = []
for k, sc in (("V3", d.score_V3.to_numpy()), ("B0", d.score_B0.to_numpy()), ("GRADE", R.grade_score(d))):
    o = R.order_rank(sc)[0]; TA.append(dict(strategy=k, T1_계약만기=R.t1_curve(d, o, qs=np.array([Q]))[0][0], T1_듀레이션일관=R.t1_curve(dd, o, qs=np.array([Q]))[0][0],
                                            T1_듀레이션일관_최대=float(np.nanmax(R.t1_curve(dd, o)[0])), q_최대=float(R.QS[np.nanargmax(R.t1_curve(dd, o)[0])])))
TA.append(dict(strategy="ALL", T1_계약만기=R.t1_curve(d, o_all, qs=np.array([1.0]))[0][0], T1_듀레이션일관=R.t1_curve(dd, o_all, qs=np.array([1.0]))[0][0]))
TA = pd.DataFrame(TA); TA.to_csv(TB / "교수_A_듀레이션_T1.csv", index=False, encoding="utf-8-sig"); print(TA.round(4).to_string())

# ---- B OLS(발급월 군집)
E["p"] = E.pd_blend * 100; cl = dict(cov_type="cluster", cov_kwds={"groups": E.mi.to_numpy()})
FORM = {"M1 스프레드": "z ~ spread", "M2 +세부등급": "z ~ spread + C(sub_grade)", "M3 +부도확률": "z ~ spread + C(sub_grade) + p", "M4 +우리 예측": "z ~ spread + C(sub_grade) + p + mu"}
OLS = []
for nm, f in FORM.items():
    r = smf.ols(f, data=E).fit(**cl); rn = smf.ols(f, data=E).fit()
    row = dict(모형=nm, N=int(r.nobs), R2=r.rsquared, 조정R2=r.rsquared_adj, AIC=r.aic)
    for v in ("spread", "p", "mu"):
        if v in r.params: row.update({f"b_{v}": r.params[v], f"se_{v}": r.bse[v], f"t_{v}": r.tvalues[v], f"se_naive_{v}": rn.bse[v]})
    OLS.append(row)
mz = smf.ols("z ~ mu", data=E).fit(**cl); w = mz.wald_test("Intercept = 0, mu = 1", scalar=True)
OLS.append(dict(모형="MZ 불편성 z = α + β·μ̂", N=int(mz.nobs), R2=mz.rsquared, b_mu=mz.params["mu"], se_mu=mz.bse["mu"], t_mu=mz.tvalues["mu"], alpha=mz.params["Intercept"], se_alpha=mz.bse["Intercept"], wald_F=float(w.statistic), wald_p=float(w.pvalue)))
OLS = pd.DataFrame(OLS); OLS.to_csv(TB / "교수_B_OLS.csv", index=False, encoding="utf-8-sig"); print(OLS.round(4).to_string())

# ---- C 10분위 롱−숏(발급월×만기 안)
def deciles(score):
    g = pd.DataFrame(dict(s=score, m=E.mi, t=E["T"])); return np.clip(np.ceil(g.groupby(["m", "t"]).s.rank(method="first", pct=True).to_numpy() * 10), 1, 10).astype(int)
LS, DEC = [], []
for k, sc in (("최종 모형", E.score_V3.to_numpy()), ("부도확률 순위", E.score_B0.to_numpy()), ("세부등급 순위", R.grade_score(E))):
    E["dec"] = deciles(sc); DEC += [dict(순위=k, 분위=q, n=int(len(x)), 평균초과수익=x.z.mean(), 부도율=x.dflt.mean(), 평균스프레드=x.spread.mean()) for q, x in E.groupby("dec")]
    mm = E.groupby(["mi", "dec"]).z.mean().unstack(); ls = (mm[1] - mm[10]).dropna(); nw = sm.OLS(ls.to_numpy(), np.ones(len(ls))).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    LS.append(dict(순위=k, 개월=len(ls), 롱숏_평균=float(ls.mean()), NW_t=float(nw.tvalues[0]), 양수_월비율=float((ls > 0).mean()), 롱숏_Sharpe=float(ls.mean() / ls.std(ddof=1)),
                   D1_평균=float(mm[1].mean()), D10_평균=float(mm[10].mean()), 단조성_Spearman=float(pd.Series(mm.mean().to_numpy()).corr(pd.Series(np.arange(1, 11)), method="spearman"))))
    if k == "최종 모형": ls.rename("롱숏").to_frame().assign(ym=E.groupby("mi").ym.first().reindex(ls.index).to_numpy()).to_csv(TB / "교수_C_롱숏_월별.csv", encoding="utf-8-sig")
pd.DataFrame(DEC).to_csv(TB / "교수_C_10분위.csv", index=False, encoding="utf-8-sig"); LS = pd.DataFrame(LS); LS.to_csv(TB / "교수_C_롱숏.csv", index=False, encoding="utf-8-sig"); print(LS.round(3).to_string())

# ---- D 1·2차 모먼트 보정
MO = []
for v, lab in (("mu", "평균"), ("sig", "표준편차")):
    E["b"] = pd.qcut(E[v], 10, labels=False, duplicates="drop")
    for b, x in E.groupby("b"): MO.append(dict(모먼트=lab, 분위=int(b) + 1, n=len(x), 예측=x[v].mean(), 실현=(x.z.mean() if v == "mu" else x.z.std(ddof=1))))
MO = pd.DataFrame(MO); MO.to_csv(TB / "교수_D_모먼트.csv", index=False, encoding="utf-8-sig")
SUMM["모먼트_상관"] = {lab: float(np.corrcoef(g.예측, g.실현)[0, 1]) for lab, g in MO.groupby("모먼트")}; print(MO.round(3).to_string())

# ---- E 레짐 전환(학습 파일로 적합 → 시험 월 분류)
tr = R.frame(TRAIN); s36 = tr[(tr["T"] == 36) & (tr.ym <= 201703)].assign(z=lambda x: (x.r1 - x.rf) * 100)
g = s36.groupby("mi").agg(n=("z", "size"), z=("z", "mean"), ym=("ym", "first")); g = g[g.n >= 300].sort_index()
mr = sm.tsa.MarkovRegression(g.z.to_numpy(), k_regimes=2, trend="c", switching_variance=True).fit(search_reps=50, disp=False)
pp = dict(zip(mr.model.param_names, mr.params)); mu_r = [pp[f"const[{i}]"] for i in range(2)]; bad = int(np.argmin(mu_r)); P = mr.regime_transition
g["p_bad"] = mr.smoothed_marginal_probabilities[:, bad]; g.to_csv(TB / "교수_E_레짐_월별.csv", encoding="utf-8-sig")
REG = dict(months=len(g), first=int(g.ym.iloc[0]), last=int(g.ym.iloc[-1]), mean=mu_r, sigma2=[pp[f"sigma2[{i}]"] for i in range(2)], bad=bad,
           p_stay=[float(P[i, i, 0]) for i in range(2)], dur=[float(1 / (1 - P[i, i, 0])) for i in range(2)], llf=float(mr.llf), aic=float(mr.aic), bad_months=int((g.p_bad > 0.5).sum()))
lin = sm.OLS(g.z.to_numpy(), np.ones(len(g))).fit(); REG["aic_1국면"] = float(lin.aic)
badm = set(g.index[g.p_bad > 0.5]); goodm = set(g.index[g.p_bad <= 0.5]); RG = []
for nm, ms in (("악화 국면", badm), ("정상 국면", goodm)):
    sub = d.mi.isin(ms).to_numpy() & d.ref.to_numpy()
    for k, m in (("최종 모형", APP), ("전부 승인", np.ones(len(d), bool))):
        o = np.argsort(~m, kind="stable"); RG.append(dict(국면=nm, 전략=k, 대출=int(sub.sum()), T1=R.t1_curve(d, o, sub=sub, qs=np.array([m.mean()]))[0][0], 승인부도율=d.dflt[m & sub].mean(), 전체부도율=d.dflt[sub].mean()))
RG = pd.DataFrame(RG); RG.to_csv(TB / "교수_E_레짐_성과.csv", index=False, encoding="utf-8-sig"); SUMM["레짐"] = REG; print(REG); print(RG.round(4).to_string())

# ---- F 림보(사람 심사)
pdm = d[["pd_xgb", "pd_lgb", "pd_cat"]].to_numpy(); dis = pdm.std(1) / pdm.mean(1); L2 = dis >= np.quantile(dis, 0.95)
XTR = extra(TRAIN); trx = tr.merge(XTR[["id", "purpose", "grade"]], on="id"); cnt = trx.groupby(["grade", "purpose", "T"]).size()
cell = pd.MultiIndex.from_arrays([d.grade, d.purpose, d["T"]]); L3 = cnt.reindex(cell).fillna(0).to_numpy() < 1000
rq = rk_all / len(d); L1 = np.abs(rq - Q) <= 0.025; LIM = L1 | (APP & (L2 | L3))
def t1m(m): return R.t1_curve(d, np.argsort(~m, kind="stable"), qs=np.array([m.mean()]))[0][0]
good = ~d.dflt.to_numpy(); rs = np.random.RandomState(0); hum70 = np.where(rs.rand(len(d)) < 0.7, good, ~good)
FL = [dict(방식="자동 규칙(그대로)", T1=t1m(APP), 승인율=APP.mean()), dict(방식="림보 전부 거절", T1=t1m(APP & ~LIM), 승인율=(APP & ~LIM).mean()),
      dict(방식="림보 심사(정확도 70%)", T1=t1m((APP & ~LIM) | (LIM & hum70)), 승인율=((APP & ~LIM) | (LIM & hum70)).mean()),
      dict(방식="림보 완전 심사(상한)", T1=t1m((APP & ~LIM) | (LIM & good)), 승인율=((APP & ~LIM) | (LIM & good)).mean())]
FL = pd.DataFrame(FL); FL.to_csv(TB / "교수_F_림보.csv", index=False, encoding="utf-8-sig")
SUMM["림보"] = dict(비중=float(LIM.mean()), 경계=float(L1.mean()), 불일치_승인=float((APP & L2).mean()), 희소_승인=float((APP & L3).mean()),
                  부도율_림보승인=float(d.dflt[LIM & APP].mean()), 부도율_비림보승인=float(d.dflt[APP & ~LIM].mean()), z_sd_림보=float(d.z[LIM].std()), z_sd_비림보승인=float(d.z[APP & ~LIM].std()))
print(FL.round(4).to_string()); print(SUMM["림보"])

# ---- G 비정형(직함) 시범
norm = lambda s: s.fillna("").str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
fq = norm(XTR.emp_title).value_counts(); d["et"] = norm(d.emp_title); d["etn"] = d.et.map(fq).fillna(0)
d["직함군"] = np.where(d.et == "", "미기재", np.where(d.etn < 30, "희소(학습 30건 미만)", "일반"))
GG = [dict(직함군=k, n=len(x), 부도율=x.dflt.mean(), 예측부도확률=x.pd_blend.mean(), AUC=roc_auc_score(x.dflt, x.pd_blend)) for k, x in d.groupby("직함군")]
RULES = [("의료", r"nurse|\brn\b|lpn|cna|medical|physician|doctor|pharm|dental|therap|health|hospital"), ("교육", r"teacher|professor|educat|school|instructor|tutor"),
         ("운전·운송", r"driver|truck|delivery|transport|\bcdl\b|courier|logistic"), ("공공·군", r"police|sheriff|fire|army|navy|air force|military|usps|postal|government|federal|state of|county|city of"),
         ("기술·IT", r"engineer|developer|software|programmer|technician|\bit\b|analyst|scientist"), ("영업", r"sales|realtor|broker|agent|account exec"),
         ("관리직", r"manager|director|supervisor|\bvp\b|president|chief|executive|lead"), ("자영·소유", r"owner|self|founder|proprietor|partner"),
         ("서비스", r"server|cashier|bartender|cook|chef|retail|customer service|clerk|housekeep|waitress"), ("생산·건설", r"mechanic|electric|plumb|carpent|construct|welder|operator|laborer|maint|warehouse|forklift|install"),
         ("사무", r"admin|assistant|secretary|receptionist|office|clerical|bookkeep|accountant|accounting|payroll")]
def occ(s):
    out = np.full(len(s), "기타", dtype=object)
    for nm, pat in reversed(RULES): out[s.str.contains(pat, regex=True).to_numpy()] = nm
    return out
rare = d[d.직함군 == "희소(학습 30건 미만)"].copy(); rare["직군"] = occ(rare.et); rare["resid"] = rare.dflt.astype(float) - rare.pd_blend
OC = rare.groupby("직군").agg(n=("resid", "size"), 실제부도율=("dflt", "mean"), 예측부도확률=("pd_blend", "mean"), 잔차=("resid", "mean"), 잔차_se=("resid", lambda x: x.std(ddof=1) / np.sqrt(len(x)))).reset_index()
OC = OC[OC.n >= 1000].sort_values("잔차"); OC["t"] = OC.잔차 / OC.잔차_se
ct = {k: float(sm.OLS(x.resid.to_numpy(), np.ones(len(x))).fit(cov_type="cluster", cov_kwds={"groups": x.mi.to_numpy()}).tvalues[0]) for k, x in rare.groupby("직군")}
OC["t_군집"] = OC.직군.map(ct)                                                # 같은 달 대출의 공통 충격 반영(발급월 군집)
pd.DataFrame(GG).to_csv(TB / "교수_G_직함군.csv", index=False, encoding="utf-8-sig"); OC.to_csv(TB / "교수_G_희소직함_직군잔차.csv", index=False, encoding="utf-8-sig"); print(pd.DataFrame(GG).round(4).to_string()); print(OC.round(4).to_string())
SUMM["직함"] = dict(희소비중=float((d.직함군 == "희소(학습 30건 미만)").mean()), 미기재비중=float((d.직함군 == "미기재").mean()), 유의직군=int((OC.t_군집.abs() > 2.58).sum()), 유의직군_단순=int((OC.t.abs() > 2.58).sum()), 직군수=len(OC), 희소_평균잔차=float(rare.resid.mean()))
json.dump(SUMM, open(TB / "교수_요약.json", "w"), ensure_ascii=False, indent=1, default=float)
