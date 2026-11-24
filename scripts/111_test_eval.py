"""111 시험 채점: 110 점수(학습 파일만으로 만든 것) × 시험 확정 대출의 실제 현금흐름(98_e 정의, loans_test.parquet).
지표·운영 지점은 사전계획.md 그대로: 샤프(96_c 월 빈티지, 만기 36/60 따로), 과제식 ROE(20_test_split_sim), AUC. 운영 지점 25%, 참고 4%·53%.
출력: 산출물/테스트평가/표/*.csv, 결과.json"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
K = importlib.import_module("98_common"); C = K.C; log = K.log
OUT = C.ROOT / "산출물" / "테스트평가"; TB = OUT / "표"; TB.mkdir(exist_ok=True)
S = pd.read_parquet(OUT / "test_scores.parquet"); Lt = pd.read_parquet(OUT / "loans_test.parquet"); Lt["id"] = Lt.id.astype(str)
V2 = OUT / "test_scores_v2.parquet"
if V2.exists(): S = S.merge(pd.read_parquet(V2)[["id", "score_M6v2", "pd_blend"]], on="id", validate="1:1")     # 117: 부도확률 3모형 평균 v2
M = Lt.merge(S.drop(columns=["term_m"]), on="id", how="left", validate="1:1"); assert M.score_M6.notna().all() and len(M) == len(Lt)
SG = sorted(M.sub_grade.unique()); M["score_GRADE"] = M.sub_grade.map({g: i for i, g in enumerate(SG)}) + 0.001 * np.random.RandomState(0).rand(len(M))
STRAT = {"M6": "위험 대비 수익 순위(주 전략)", "M5": "기대수익 순위", "B0": "부도확률 XGBoost", "GRADE": "세부등급 규칙"}
if "score_M6v2" in M: STRAT = {"M6v2": "위험 대비 수익 순위 v2(부도확률 3모형 평균)", **STRAT}
V3 = C.ROOT / "산출물" / "최종보고서" / "test_scores_v3.parquet"                      # 125: 최종 v3(이득 목표 = 과제 잣대 수익, 124 학습 OOF 판정)
if V3.exists(): M = M.merge(pd.read_parquet(V3)[["id", "score_V3"]], on="id", validate="1:1"); STRAT = {"V3": "최종 v3(이득 목표 = 과제 잣대 수익)", **STRAT}
POINTS = (0.04, 0.25, 0.53); IDX = {a: int(np.argmin(np.abs(K.AG - a))) for a in POINTS}
res = dict(n_test_rows=len(S), n_resolved=len(M), default_rate=float(M.default.mean()))

# AUC (부도 순위)
res["AUC"] = {k: float(roc_auc_score(M.default, M["score_" + k])) for k in STRAT}
for t in (36, 60): res[f"AUC_{t}"] = {k: float(roc_auc_score(M.default[M.term == t], M["score_" + k][M.term == t])) for k in STRAT}

# 샤프 곡선(만기별) + e 행렬
E, rows = {}, []
for t in (36, 60):
    m = (M.term == t).to_numpy(); d = M[m].reset_index(drop=True)
    for k in STRAT:
        e = K.series(K.month_aggs(d, d["score_" + k].to_numpy(), t)); E[(k, t)] = e; sr = e.mean(0) / e.std(0, ddof=1); xs = e.mean(0)
        rows += [dict(strategy=k, term=t, rate=a, SR=sr[i], mean_excess=xs[i]) for i, a in enumerate(K.AG)]
SR = pd.DataFrame(rows); SR.to_csv(TB / "샤프_곡선.csv", index=False)

# 과제식 ROE 곡선(+연도군)
rates = tuple(sorted(set(np.round(np.arange(0.05, 1.0, 0.05), 2)) | set(POINTS)))
roe = {k: K.course_roe(M, M["score_" + k].to_numpy(), rates=rates) for k in STRAT}
RO = pd.DataFrame([dict(strategy=k, group=g, rate=int(q[1:]) / 100, ROE=v) for k in STRAT for g, r in roe[k].items() for q, v in r.items() if q.startswith("r")]); RO.to_csv(TB / "ROE_곡선.csv", index=False)
res["ROE_ref"] = {g: {q: roe["M6"][g][q] for q in ("all_in", "treasury", "oracle", "n")} for g in roe["M6"]}

# 운영 지점 요약 + 학습 교차검증 대비
CV = pd.read_csv(OUT / "cv_곡선_학습전용.csv"); key = []
for k in STRAT:
    for a in POINTS:
        row = dict(strategy=k, name=STRAT[k], rate=a, ROE_test=roe[k]["all"][f"r{int(round(a * 100))}"])
        for t in (36, 60): row[f"SR{t}_test"] = float(SR[(SR.strategy == k) & (SR.term == t) & np.isclose(SR.rate, a)].SR.iloc[0])
        if k in ("B0", "M5", "M6", "M6v2"):
            c = CV[(CV.model == k) & np.isclose(CV.rate, a)]
            row["ROE_cv"] = float(c[c.term == 0].ROE.iloc[0]) if (c.term == 0).any() else np.nan
            for t in (36, 60): row[f"SR{t}_cv"] = float(c[c.term == t].SR.iloc[0])
        key.append(row)
KEY = pd.DataFrame(key); KEY.to_csv(TB / "운영지점_요약.csv", index=False, encoding="utf-8-sig")

# 쌍체 차이: 샤프(발행월 블록 부트스트랩, 96_c boot_idx) · ROE(대출 부트스트랩 300회)
D2 = []
rs = np.random.RandomState(0); n = len(M); bidx = [rs.randint(0, n, n) for _ in range(200)]; Ms = M[["A", "ann", "tb", "roe_ok", "issue_year"]]
PAIRS = [("M6", "B0"), ("M6", "GRADE")] + ([("M6v2", "M6"), ("M6v2", "B0"), ("M6v2", "GRADE")] if "M6v2" in STRAT else []) + ([("V3", "M6v2"), ("V3", "B0"), ("V3", "GRADE")] if "V3" in STRAT else [])
for top, base in PAIRS:
    for a in POINTS:
        r = dict(compare=f"{top} − {base}", rate=a)
        for t in (36, 60): mu, lo, hi = K.paired_sr_delta(E[(top, t)], E[(base, t)], IDX[a]); r.update({f"dSR{t}": mu, f"dSR{t}_lo": lo, f"dSR{t}_hi": hi})
        q = f"r{int(round(a * 100))}"; r["dROE"] = roe[top]["all"][q] - roe[base]["all"][q]
        if a == 0.25:
            dd = [(lambda Mb: K.course_roe(Mb, M["score_" + top].to_numpy()[b], rates=(a,), groups=False)["all"][q] - K.course_roe(Mb, M["score_" + base].to_numpy()[b], rates=(a,), groups=False)["all"][q])(Ms.iloc[b].reset_index(drop=True)) for b in bidx]
            r["dROE_lo"], r["dROE_hi"] = np.percentile(dd, [2.5, 97.5])
        D2.append(r); log(r)
pd.DataFrame(D2).to_csv(TB / "쌍체차이.csv", index=False, encoding="utf-8-sig")

# 왜 다른가: 승인 25%에서 M6 와 B0 가 다르게 고른 대출
def approved(score, a=0.25):
    o = np.argsort(score, kind="stable"); cA = np.cumsum(M.A.to_numpy()[o]) / M.A.sum(); sel = np.zeros(len(M), bool); sel[o[cA <= a + 1e-12]] = True; return sel
MAIN = "V3" if "V3" in STRAT else ("M6v2" if "M6v2" in STRAT else "M6")      # 최종 모델(125 v3 > 117 v2 > 110 v1)
a6, a0, ag = approved(M["score_" + MAIN].to_numpy()), approved(M.score_B0.to_numpy()), approved(M.score_GRADE.to_numpy())
M["exret"] = np.where(M.roe_ok, M.ann / M.A - M.tb, np.nan)
def prof(sel, name): s = M[sel]; return dict(그룹=name, 건수=int(sel.sum()), 평균금리=float(s.int_rate.mean()), 부도율=float(s.default.mean()), 평균대출액=float(s.A.mean()),
                                         초과수익_금액가중=float(np.nansum(s.exret * s.A) / s.A[s.roe_ok].sum() * 100), 만기60_비중=float((s.term == 60).mean()))
WHY = pd.DataFrame([prof(a6 & a0, "둘 다 승인"), prof(a6 & ~a0, "M6만 승인"), prof(~a6 & a0, "B0만 승인"), prof(a6, "M6 승인 전체"), prof(a0, "B0 승인 전체"), prof(ag, "등급규칙 승인 전체")])
WHY.to_csv(TB / "다르게_고른_대출.csv", index=False, encoding="utf-8-sig")
GR = pd.DataFrame({"등급": M.sub_grade.str[0], "M6": a6, "B0": a0, "등급규칙": ag}).groupby("등급")[["M6", "B0", "등급규칙"]].sum(); GR.to_csv(TB / "승인_등급구성.csv", encoding="utf-8-sig")
json.dump(res, open(OUT / "결과.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, indent=1, default=float)); print(KEY.round(3).to_string()); print(pd.DataFrame(D2).round(3).to_string()); print(WHY.round(3).to_string()); print(GR.to_string())
