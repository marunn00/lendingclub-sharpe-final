"""126 통계 품질 점검(최종 보고서 Ⅱ·Ⅵ·Ⅶ장): 표본 대표성, 우측 중도절단, 경쟁위험 누적부도, 부도확률 보정 검정. 모형 선택에는 쓰지 않는다(시험 결과 열은 채점·서술 통계에만 사용).
1) 대표성: 거절 신청(rejected_2007_to_2018Q4.csv) 대 승인 대출(학습+시험 파일 = 승인 대출 전체) 연도별 건수·플랫폼 승인율.
   신용점수 비교는 2013-11-05 이전 신청만(그 뒤 거절 자료 Risk_Score는 FICO가 아닌 Vantage 점수 — LendingClub RejectStats 사전).
2) 우측 중도절단: 시험 파일 발급연도별 결과 확정 비율, 확정 대출만의 부도율(단순) 대 경쟁위험 누적부도(36개월 시점).
3) 경쟁위험 누적부도(Aalen–Johansen, 월 단위): 사건 1 = 부도(마지막 납입 다음 달), 사건 2 = 완납(마지막 납입 달), 중도절단 = 상환 중·연체·유예(마지막 납입 달).
   시험 파일 전체 1,170,198건(상환 중 포함). 승인 = 최종 v3 점수 상위 q*(124 학습 OOF) — 시험 전체 신청 풀 기준.
4) 보정 검정(시험 결과 확정 대출): Cox 보정 회귀 logit P(부도) = a + b·logit(p̂) (a = 0, b = 1이면 보정 양호), Hosmer–Lemeshow 10분위 χ²(df 8), ECE.
출력: 산출물/최종보고서/표/대표성_연도별.csv, 중도절단_연도별.csv, 누적부도_CIF.csv, 보정검정.json"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, json, numpy as np, pandas as pd, polars as pl, statsmodels.api as sm
from scipy.stats import chi2
K = importlib.import_module("98_common"); C = K.C; log = K.log
OUT = C.ROOT / "산출물" / "최종보고서"; TB = OUT / "표"
TRAIN, TEST, REJ = C.ROOT / "lending_club_2020_train.csv", C.ROOT / "lending_club_2020_test.csv", C.ROOT / "rejected_2007_to_2018Q4.csv"

# ---- 1) 대표성
rj = (pl.scan_csv(REJ, infer_schema_length=0).select([pl.col("Application Date").alias("d"), pl.col("Risk_Score").cast(pl.Float64, strict=False).alias("rs"),
      pl.col("Debt-To-Income Ratio").str.replace("%", "").cast(pl.Float64, strict=False).alias("dti")])
      .with_columns(pl.col("d").str.slice(0, 4).cast(pl.Int32, strict=False).alias("yr"), (pl.col("d") < "2013-11-05").alias("fico_era")))
R1 = rj.group_by("yr").agg(pl.len().alias("거절_신청"), pl.col("dti").median().alias("거절_DTI_중앙")).collect().to_pandas()
R2 = rj.filter(pl.col("fico_era") & (pl.col("rs") > 0)).group_by("yr").agg(pl.col("rs").median().alias("거절_FICO_중앙"), (pl.col("rs") < 660).mean().alias("거절_FICO660미만")).collect().to_pandas()
acc = pl.concat([pl.scan_csv(f, infer_schema_length=0).select(["issue_d", "fico_range_low", "dti"]) for f in (TRAIN, TEST)]).filter(pl.col("issue_d").is_not_null())
acc = acc.with_columns(pl.col("issue_d").str.slice(-4).cast(pl.Int32).alias("yr"), pl.col("fico_range_low").cast(pl.Float64, strict=False).alias("fico"), pl.col("dti").cast(pl.Float64, strict=False).alias("dti"))
A1 = acc.group_by("yr").agg(pl.len().alias("승인_대출"), pl.col("fico").median().alias("승인_FICO_중앙"), (pl.col("fico") < 660).mean().alias("승인_FICO660미만"), pl.col("dti").median().alias("승인_DTI_중앙")).collect().to_pandas()
REP = A1.merge(R1, on="yr", how="outer").merge(R2, on="yr", how="left").sort_values("yr"); REP = REP[REP.yr.between(2007, 2020)]
REP["플랫폼_승인율"] = REP.승인_대출 / (REP.승인_대출 + REP.거절_신청); REP.loc[REP.yr > 2018, "플랫폼_승인율"] = np.nan
REP.to_csv(TB / "대표성_연도별.csv", index=False, encoding="utf-8-sig"); print(REP.round(3).to_string())
log("승인 대출 합계", int(REP.승인_대출.sum()), "거절 신청 합계(2007–2018)", int(REP.거절_신청.sum()))

# ---- 2)·3) 중도절단·경쟁위험 누적부도(시험 파일 전체)
t = pl.read_csv(TEST, columns=["id", "issue_d", "term", "loan_status", "last_pymnt_d"], infer_schema_length=0).filter(pl.col("issue_d").is_not_null()).to_pandas()
mi = C.mcode_str(t.issue_d); lp = C.mcode_str(t.last_pymnt_d); hasp = lp >= 0; m = np.where(hasp, np.maximum(lp - mi, 0), 0)
st = t.loan_status.fillna(""); ev = np.where(st.str.contains("Charged Off|Default"), 1, np.where(st.str.contains("Fully Paid"), 2, 0))
tt = np.where(ev == 1, m + 1, np.where(ev == 2, np.maximum(m, 1), m)).astype(int); T_ = t.term.str.strip().str[:2].astype(int).to_numpy(); yr = t.issue_d.str[-4:].astype(int).to_numpy()
S = pd.read_parquet(OUT / "test_scores_v3.parquet")[["id", "score_V3"]]; assert (S.id.to_numpy() == t.id.astype(str).to_numpy()).all()
q = float(pd.read_csv(TB / "v3_판정.csv").set_index("model").loc["V3", "q_star"]); o = np.argsort(S.score_V3.to_numpy(), kind="stable"); ap = np.zeros(len(t), bool); ap[o[:int(round(q * len(t)))]] = True
def cif(tm, e, horizon=60):
    """Aalen–Johansen: 부도 누적발생률 F1(m), m = 0..horizon."""
    W = max(horizon, int(tm.max())) + 2; ex = np.bincount(tm, minlength=W); risk = ex[::-1].cumsum()[::-1]
    d1, d2 = np.bincount(tm[e == 1], minlength=W), np.bincount(tm[e == 2], minlength=W); S_, F = 1.0, [0.0]
    for k in range(1, horizon + 1):
        if risk[k] == 0: F.append(F[-1]); continue
        F.append(F[-1] + S_ * d1[k] / risk[k]); S_ *= 1 - (d1[k] + d2[k]) / risk[k]
    return np.array(F)
rows = []
for term in (36, 60):
    for g, mk in (("전체 신청", np.ones(len(t), bool)), ("최종 모형 승인", ap), ("최종 모형 거절", ~ap)):
        s = mk & (T_ == term); F = cif(tt[s], ev[s], term)
        rows += [dict(만기=term, 집단=g, 개월=k, 누적부도=F[k], n=int(s.sum())) for k in range(term + 1)]
CIF = pd.DataFrame(rows); CIF.to_csv(TB / "누적부도_CIF.csv", index=False, encoding="utf-8-sig")
print(CIF[CIF.개월.isin([12, 24, 36, 60])].pivot_table(index=["만기", "개월"], columns="집단", values="누적부도").round(4).to_string())
cz = []
for y in range(2007, 2021):
    s = (yr == y) & (T_ == 36); res = s & (ev > 0)
    if s.sum() == 0: continue
    cz.append(dict(발급연도=y, 대출_36개월=int(s.sum()), 결과확정비율=res.sum() / s.sum(), 확정대출_부도율=(ev[res] == 1).mean(), 누적부도_36개월=cif(tt[s], ev[s], 36)[36], 승인율_최종모형=ap[s].mean()))
CZ = pd.DataFrame(cz); CZ.to_csv(TB / "중도절단_연도별.csv", index=False, encoding="utf-8-sig"); print(CZ.round(3).to_string())
wv = []                                                                     # 같은 발급연도 안 승인·거절 비교: 발급 시기 차이와 선별 효과를 분리
for y in (2015, 2016, 2017):
    for g, mk in (("승인", ap), ("거절", ~ap)):
        s = (yr == y) & (T_ == 36) & mk; wv.append(dict(발급연도=y, 집단=g, n=int(s.sum()), 누적부도_36개월=cif(tt[s], ev[s], 36)[36]))
WV = pd.DataFrame(wv); WV.to_csv(TB / "누적부도_발급연도내.csv", index=False, encoding="utf-8-sig"); print(WV.round(4).to_string())

# ---- 4) 보정 검정(시험 결과 확정 대출, 3모형 평균 부도확률)
V2 = pd.read_parquet(C.ROOT / "산출물/테스트평가/test_scores_v2.parquet")[["id", "pd_blend"]]; d = pd.DataFrame(dict(id=t.id.astype(str), ev=ev, st=st)).merge(V2, on="id")
d = d[d.st.isin(["Fully Paid", "Charged Off", "Default"])]                 # 다른 표와 같은 결과 확정 744,608건(정책 미충족 상태 제외)
y = (d.ev == 1).to_numpy().astype(float); p = np.clip(d.pd_blend.to_numpy(), 1e-6, 1 - 1e-6); lg = np.log(p / (1 - p))
fit = sm.Logit(y, sm.add_constant(lg)).fit(disp=0); a, b = fit.params; sa, sb = fit.bse
dec = pd.qcut(p, 10, labels=False); g = pd.DataFrame(dict(dec=dec, y=y, p=p)).groupby("dec").agg(n=("y", "size"), o=("y", "sum"), e=("p", "sum"))
hl = float((((g.o - g.e) ** 2) / (g.e * (1 - g.e / g.n))).sum()); ece = float((g.n * np.abs(g.o / g.n - g.e / g.n)).sum() / g.n.sum())
CAL = dict(n=int(len(y)), intercept=float(a), intercept_se=float(sa), slope=float(b), slope_se=float(sb), HL_chi2=hl, HL_df=8, HL_p=float(chi2.sf(hl, 8)), ECE=ece,
           mean_pred=float(p.mean()), mean_obs=float(y.mean()), max_abs_decile_gap=float(np.abs(g.o / g.n - g.e / g.n).max()))
json.dump(CAL, open(TB / "보정검정.json", "w"), ensure_ascii=False, indent=1); print(CAL)
