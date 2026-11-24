"""119 이수용 보고서(Test Data 최종보고서_이수용.pdf) 평가 방식으로 우리 모델 평가.
- 평가 대상: 만기가 지난 대출(발급월 기준: 36개월 ≤ 2017-03, 60개월 ≤ 2015-03 — 이수용 보고서 5장 건수 467,101건·60개월 62,522건과 맞는 경계), 상태로 거르지 않음(진행 중·연체도 실현 상환액 그대로)
- 대출별 x = 실현 총수익률 − rf × 잔액·년, 실현 총수익률 = (total_pymnt − funded_amnt)/funded_amnt, 잔액·년 = total_rec_int/(int_rate·funded_amnt)
  Sharpe = 승인 대출 x 의 평균 ÷ 표준편차(대출 단위, 동일가중). rf = 0/1/2/3% 고정, '국채' = 발급월·만기 맞춤(36→3년, 60→5년) 월평균
- 연수익률(잔액 기준) = Σ(total_pymnt − funded) ÷ Σ(total_rec_int/int_rate) (금액 가중)
- 95% 구간: 발급월 블록 부트스트랩 2,000회(발급월을 복원추출)
- 정책: 전부 승인, A~B등급만, A등급만, 부도확률 낮은 순 20%(우리 XGBoost PD), 우리 방식 승인 20/25/50/53%(발급월·만기 안 금액 기준, 시험 파일 전체에 적용한 규칙을 이 범위로 제한)
출력: 산출물/우수사례평가/이수용/표/*.csv, fig/*.png"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, numpy as np, pandas as pd, polars as pl
K = importlib.import_module("98_common"); C = K.C; log = K.log
OUT = C.ROOT / "산출물" / "우수사례평가" / "이수용"; TB = OUT / "표"; [p.mkdir(parents=True, exist_ok=True) for p in (TB, OUT / "fig")]
COLS = ["id", "issue_d", "term", "grade", "loan_status", "funded_amnt", "int_rate", "total_pymnt", "total_rec_int"]
d = pl.read_csv(C.ROOT / "lending_club_2020_test.csv", columns=COLS, infer_schema_length=0, null_values=["", "NA"]).filter(pl.col("issue_d").is_not_null()).to_pandas()
for c in ("funded_amnt", "total_pymnt", "total_rec_int"): d[c] = pd.to_numeric(d[c], errors="coerce")
d["rate"] = pd.to_numeric(d.int_rate.str.replace("%", ""), errors="coerce") / 100; d["T"] = d.term.str.strip().str[:2].astype(int); d["mi"] = C.mcode_str(d.issue_d)
S = pd.read_parquet(C.ROOT / "산출물/테스트평가/test_scores.parquet")[["id", "pd", "score_B0"]].merge(pd.read_parquet(C.ROOT / "산출물/테스트평가/test_scores_v2.parquet")[["id", "score_M6v2"]], on="id")
S = S.merge(pd.read_parquet(C.ROOT / "산출물/최종보고서/test_scores_v3.parquet")[["id", "score_V3"]], on="id", validate="1:1")      # '우리 방식' = 최종 v3, v2는 참고
d["id"] = d.id.astype(str); d = d.merge(S, on="id", validate="1:1"); assert len(d) == len(S)
# 정책 판정은 시험 파일 전체에서(우리 운영 규칙 그대로) 먼저 하고, 평가 범위로 자른다
def approve_monthly(score, a):
    o = np.lexsort((score, d["T"].to_numpy(), d.mi.to_numpy())); g = pd.DataFrame(dict(m=d.mi.to_numpy()[o], t=d["T"].to_numpy()[o], A=d.funded_amnt.to_numpy()[o]))
    cum = g.groupby(["m", "t"]).A.transform(lambda s: s.cumsum() / s.sum()).to_numpy(); ap = np.zeros(len(d), bool); ap[o] = cum <= a + 1e-12; return ap
m3, m5 = C.mcode_str(pd.Series(["Mar-2017"]))[0], C.mcode_str(pd.Series(["Mar-2015"]))[0]
E = ((d["T"] == 36) & (d.mi <= m3)) | ((d["T"] == 60) & (d.mi <= m5)); E = E.to_numpy()
POL = {"전부 승인": np.ones(len(d), bool), "A~B등급만": d.grade.isin(["A", "B"]).to_numpy(), "A등급만": (d.grade == "A").to_numpy()}
for a in (0.20, 0.25, 0.50, 0.53): POL[f"우리 방식 {int(a * 100)}%"] = approve_monthly(d.score_V3.to_numpy(), a)
POL["v2 20%"] = approve_monthly(d.score_M6v2.to_numpy(), 0.20)
POL["부도확률 낮은 순 20%"] = approve_monthly(d.score_B0.to_numpy(), 0.20)
RATES = (0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.53, 0.60, 0.80, 1.0); CUR = {a: approve_monthly(d.score_V3.to_numpy(), a)[E] for a in RATES}
d = d[E].reset_index(drop=True); POL = {k: v[E] for k, v in POL.items()}
unres = ~d.loan_status.isin(["Fully Paid", "Charged Off", "Default", "Does not meet the credit policy. Status:Fully Paid", "Does not meet the credit policy. Status:Charged Off"])
log("평가 대상", len(d), "36m", int((d["T"] == 36).sum()), "60m", int((d["T"] == 60).sum()), "미확정 비율 %.4f" % unres.mean())
# rf: 발급월 맞춤 국채(98_e 와 같은 월평균, 발급월)
def tb_month(name):
    t = pd.read_csv(C.MACRO / f"{name}.csv"); t.columns = ["dd", "v"]; t["v"] = pd.to_numeric(t.v, errors="coerce"); t = t.dropna()
    m = (t.dd.str[:4].astype(int) - 2005) * 12 + t.dd.str[5:7].astype(int) - 1; a = np.full(216, np.nan); g = t.groupby(m).v.mean(); g = g[(g.index >= 0) & (g.index < 216)]; a[g.index.to_numpy()] = g.to_numpy() / 100; return a
t3, t5 = tb_month("DGS3"), tb_month("DGS5"); d["tsy"] = np.where(d["T"] == 36, t3[d.mi], t5[d.mi])
d["R"] = (d.total_pymnt - d.funded_amnt) / d.funded_amnt; d["BY"] = d.total_rec_int / (d.rate * d.funded_amnt); d["dflt"] = d.loan_status.str.contains("Charged Off|Default")
RF = {"rf 0%": 0.0, "rf 1%": 0.01, "rf 2%": 0.02, "rf 3%": 0.03, "국채": None}
X = {k: (d.R - (d.tsy if v is None else v) * d.BY).to_numpy() for k, v in RF.items()}
def sharpe(x, m): v = x[m]; return v.mean() / v.std(ddof=1)
def ann_ret(m): return float((d.total_pymnt - d.funded_amnt)[m].sum() / (d.total_rec_int / d.rate)[m].sum())
rows = []
for p, m in POL.items():
    r = dict(정책=p, 승인율=m.mean(), 승인건수=int(m.sum()), 부도율=d.dflt[m].mean(), 연수익률=ann_ret(m))
    r.update({k: sharpe(X[k], m) for k in RF}); rows.append(r)
TAB = pd.DataFrame(rows); TAB.to_csv(TB / "정책표.csv", index=False, encoding="utf-8-sig"); print(TAB.round(4).to_string())
print("평균 국채 rf", round(d.tsy.mean(), 4), "36m", round(d.tsy[d["T"] == 36].mean(), 4), "60m", round(d.tsy[d["T"] == 60].mean(), 4))
# 발급월 블록 부트스트랩: 월별 (n, Σx, Σx²) 집계 → 복원추출한 월로 합산
months = np.sort(d.mi.unique()); mpos = np.searchsorted(months, d.mi.to_numpy()); nm = len(months)
agg = {(p, k): np.stack([np.bincount(mpos, w * 1.0, nm), np.bincount(mpos, w * X[k], nm), np.bincount(mpos, w * X[k] ** 2, nm)]) for p, m in POL.items() for w in [m.astype(float)] for k in RF}
idx = np.random.RandomState(0).randint(0, nm, (2000, nm)); cnt = np.stack([np.bincount(i, minlength=nm) for i in idx])
def boot(p, k): n, s1, s2 = (cnt @ v for v in agg[(p, k)]); mu = s1 / n; return mu / np.sqrt((s2 - n * mu * mu) / (n - 1))
CI = []
for k in RF:
    for p in ("우리 방식 20%", "우리 방식 25%", "우리 방식 50%", "우리 방식 53%"):
        b = boot(p, k); r = dict(rf=k, 정책=p, SR_lo=np.percentile(b, 2.5), SR_hi=np.percentile(b, 97.5))
        for base in ("전부 승인", "A등급만", "A~B등급만", "부도확률 낮은 순 20%", "v2 20%"):
            df_ = b - boot(base, k); r[f"−{base}"] = df_.mean(); r[f"−{base}_lo"], r[f"−{base}_hi"] = np.percentile(df_, [2.5, 97.5]); r[f"{base}보다_높은_비율"] = (df_ > 0).mean()
        CI.append(r)
CI = pd.DataFrame(CI); CI.to_csv(TB / "신뢰구간.csv", index=False, encoding="utf-8-sig"); print(CI.round(3).to_string())
# 만기별·발급연도별(국채 rf)
yr = d.issue_d.str[-4:].astype(int)
BY_ = pd.DataFrame([dict(구분=f"{t}개월", **{p: sharpe(X["국채"], POL[p] & (d["T"] == t).to_numpy()) for p in ("전부 승인", "A등급만", "우리 방식 25%", "우리 방식 50%")}) for t in (36, 60)]
                   + [dict(구분=str(y), **{p: sharpe(X["국채"], POL[p] & (yr == y).to_numpy()) for p in ("전부 승인", "A등급만", "우리 방식 25%", "우리 방식 50%")}) for y in range(2010, 2018)])
BY_.to_csv(TB / "만기_연도별.csv", index=False, encoding="utf-8-sig"); print(BY_.round(3).to_string())
# 승인율 곡선(우리 방식, 발급월 규칙의 목표 승인율; 실제 승인 비율은 평가 범위 기준)
CU = pd.DataFrame([dict(목표승인율=a, 실제승인율=m.mean(), 부도율=d.dflt[m].mean(), 연수익률=ann_ret(m), **{k: sharpe(X[k], m) for k in RF}) for a, m in CUR.items()])
CU.to_csv(TB / "승인율_곡선.csv", index=False, encoding="utf-8-sig"); print(CU.round(4).to_string())
