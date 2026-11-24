"""121 팀 공유 보고서의 잣대로 최종 모델(v2)과 비교 대상을 평가. 승인 비율은 학습 파일 교차검증 예측(OOF)에서만 고르고, 시험 파일은 그 비율로 한 번 채점한다.
잣대(그 보고서의 정의 그대로)
 T1 과제 정의 포트폴리오 Sharpe — 「LendingClub_Sharpe_최종보고서」 2장. 대상: 결과 확정 대출(Fully Paid·Charged Off·Default).
    r = (total_pymnt / funded_amnt)^(12/T) − 1(계약만기 연율화), 승인 x = r, 거절 x = rf, SR = (mean x − mean rf) / sd(x).
 T2 기회풀 Sharpe — 「LENDINGCLUB_CREDIT_MODEL_FINAL_REPORT」 2장. 대상: 만기 기준 코호트(36m 발급 ≤ 2017-03, 60m ≤ 2015-03, 같은 세 상태).
    r = (max(P + I + Lf + R, 0) / F)^(12/T) − 1, opp = mean_N(1_S·(r − rf)) / sd_N(1_S·(r − rf)).
 rf: FRED DGS3·DGS5 발급월 월평균(36개월→3년, 60개월→5년, 119와 같은 표).
승인 규칙(사전 고정): 점수가 좋은 순으로 전체 풀의 상위 q(건수 비율)를 승인(과제 안내의 '기준선' 규칙). T2는 전체 풀에서 판정한 뒤 코호트로 자른다.
 q = 학습 OOF에서 같은 잣대를 최대로 하는 값(0.5% 격자). 시험 결과를 본 뒤 바꾸지 않는다.
비교 대상(같은 규칙, 각자 학습에서 고른 q): 전부 승인, 세부등급 순서, 부도확률 XGBoost(B0), 기대 초과수익(M5), 위험 대비 수익 v1(M6), 최종 v2(M6v2). 완전 예측(Oracle, 부도 아닌 대출만 승인)은 상한 참고.
구간: 발급월 블록 부트스트랩 1,000회(쌍체 차이), 대출 부트스트랩 300회(Sharpe 보고서와 같은 방식, 단일 값).
강건성(사전 고정, 각 전략의 T1·T2 q 그대로): 금액 가중, 연 1% 관리비, rf 0–3% 고정, 만기별, 발급연도군별, 월별 할당 규칙, 만기 코호트 4종(T2).
출력: 산출물/최종보고서/표/*.csv, 결과.json"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R_ = importlib.import_module("121_rulers"); K = R_.K; C = K.C; log = K.log
frame, grade_score, order_rank, sel, t1_curve, t2_curve, QS = R_.frame, R_.grade_score, R_.order_rank, R_.sel, R_.t1_curve, R_.t2_curve, R_.QS
OUT = C.ROOT / "산출물" / "최종보고서"; TB = OUT / "표"; TB.mkdir(parents=True, exist_ok=True)
STRAT = {"V3": "최종 모델 v3(이득 목표 = 과제 잣대 수익)", "M6v2": "v2(위험 대비 수익, 부도확률 3모형 평균)", "M6": "v1(위험 대비 수익, 부도확률 XGBoost)", "M5": "기대 초과수익 순위", "B0": "부도확률 XGBoost 순위", "GRADE": "세부등급 순서", "ALL": "전부 승인"}
FINAL = "V3"                                                                # 124 판정(학습 OOF)으로 채택한 최종 모델

# ---- 학습: 확정 대출(98_e 순서) × OOF(튜닝 보류 10% 제외)
Ltr = K.load_loans(); ids = Ltr.id.astype(str).to_numpy()
W2 = C.ROOT / "산출물" / "대안모델" / "_work"; OOF = {"V3": "oof_V3_t1", "M6v2": "oof_AUD_blend", "M6": "oof_M6_full", "M5": "oof_M5_full", "B0": "oof_B0_full"}
o_sc = {k: np.load(W2 / f"{f}.npz")["score"] for k, f in OOF.items()}; keep = ~np.isnan(o_sc["M6v2"])
assert all((np.isnan(v) == ~keep).all() for v in o_sc.values())
dtr = frame(C.ROOT / "lending_club_2020_train.csv").set_index("id").loc[ids[keep]].reset_index()
SC_TR = {k: v[keep] for k, v in o_sc.items()}; SC_TR["GRADE"] = grade_score(dtr); SC_TR["ALL"] = np.zeros(len(dtr))
log("train OOF rows", len(dtr), "default", round(dtr.dflt.mean(), 4))

# ---- 시험: 결과 확정 대출 × 학습 파일만으로 만든 점수(110·117)
dte = frame(C.ROOT / "lending_club_2020_test.csv")
S = pd.read_parquet(C.ROOT / "산출물/테스트평가/test_scores.parquet")[["id", "score_B0", "score_M5", "score_M6", "pd"]].merge(pd.read_parquet(C.ROOT / "산출물/테스트평가/test_scores_v2.parquet")[["id", "score_M6v2", "pd_blend"]], on="id", validate="1:1")
S = S.merge(pd.read_parquet(OUT / "test_scores_v3.parquet")[["id", "score_V3"]], on="id", validate="1:1")
dte = dte.merge(S, on="id", how="left", validate="1:1"); assert dte.score_M6v2.notna().all() and len(dte) == 744608
SC_TE = {k: dte[f"score_{k}"].to_numpy() for k in OOF}; SC_TE["GRADE"] = grade_score(dte); SC_TE["ALL"] = np.zeros(len(dte))

# ---- 잣대 계산(상위 q 승인, 정렬은 전체 풀)
ORD_TR = {k: order_rank(v) for k, v in SC_TR.items()}; ORD_TE = {k: order_rank(v) for k, v in SC_TE.items()}
REF_TR, REF_TE = dtr.ref.to_numpy(), dte.ref.to_numpy()
QSTAR, CUR = {}, []
for k in STRAT:
    if k == "ALL": QSTAR[k] = {"T1": 1.0, "T2": 1.0}; continue
    c1 = t1_curve(dtr, ORD_TR[k][0])[0]; c2 = t2_curve(dtr, ORD_TR[k][0], REF_TR); e1 = t1_curve(dte, ORD_TE[k][0])[0]; e2 = t2_curve(dte, ORD_TE[k][0], REF_TE)
    QSTAR[k] = {"T1": float(QS[np.nanargmax(c1)]), "T2": float(QS[np.nanargmax(c2)])}
    CUR += [dict(strategy=k, q=q, T1_train=a, T2_train=b, T1_test=c, T2_test=e) for q, a, b, c, e in zip(QS, c1, c2, e1, e2)]
pd.DataFrame(CUR).to_csv(TB / "곡선_T1_T2.csv", index=False); log("q* (학습 OOF)", QSTAR)

def summary(d, k, rk, ruler, q):
    m = sel(rk, q); r = dict(ruler=ruler, strategy=k, q=q, 승인율=m.mean())
    if ruler == "T1":
        sr, mx, sd = t1_curve(d, np.argsort(rk), qs=np.array([q])); r.update(SR=sr[0], 연수익률=mx[0], 변동성=sd[0], 승인부도율=d.dflt[m].mean() if m.any() else np.nan)
    else:
        ref = d.ref.to_numpy(); r.update(SR=t2_curve(d, np.argsort(rk), ref, qs=np.array([q]))[0], 코호트승인율=m[ref].mean(), 승인부도율=d.dflt[m & ref].mean() if (m & ref).any() else np.nan,
                                        순현금수익률=float(((d.total_pymnt - d.funded_amnt) / d.funded_amnt)[m & ref].mean()) if (m & ref).any() else np.nan)
    return r
rows = []
for ruler in ("T1", "T2"):
    for k in STRAT:
        q = QSTAR[k][ruler]; a = summary(dtr, k, ORD_TR[k][1], ruler, q); b = summary(dte, k, ORD_TE[k][1], ruler, q)
        rows.append(dict(ruler=ruler, strategy=k, name=STRAT[k], q=q, SR_train=a["SR"], **{f"{c}_test": v for c, v in b.items() if c not in ("ruler", "strategy", "q")}))
# 기준선 재현(그 보고서에 인쇄된 값과 대조): Oracle(부도 아닌 대출만), 전부 국채
orc = (~dte.dflt).to_numpy(); o_or = np.argsort(~orc, kind="stable"); q_or = orc.mean()
sr_or = t1_curve(dte, o_or, qs=np.array([q_or]))[0][0]
MAIN = pd.DataFrame(rows); MAIN.to_csv(TB / "주요결과_T1_T2.csv", index=False, encoding="utf-8-sig"); print(MAIN.round(4).to_string())
REP = dict(T1_all_test=float(MAIN[(MAIN.ruler == "T1") & (MAIN.strategy == "ALL")].SR_test.iloc[0]), T1_all_train=float(MAIN[(MAIN.ruler == "T1") & (MAIN.strategy == "ALL")].SR_train.iloc[0]),
           T1_oracle_test=float(sr_or), oracle_share=float(q_or), mean_rf_test=float(dte.rf.mean()), mean_r1_test=float(dte.r1.mean()), default_test=float(dte.dflt.mean()), n_test=len(dte),
           T2_all_ref=float(MAIN[(MAIN.ruler == "T2") & (MAIN.strategy == "ALL")].SR_test.iloc[0]), n_ref=int(REF_TE.sum()), co_ref=float(dte.dflt[REF_TE].mean()), sh60_ref=float((dte["T"][REF_TE] == 60).mean()), rf_ref=float(dte.rf[REF_TE].mean()))

# ---- 만기 코호트 4종(T2), rf 고정·관리비(T1·T2), 금액 가중·만기·연도군·월별 할당(T1)
COH = {"1_만기기준(36≤17-03, 60≤15-03)": REF_TE, "2_순수만기(둘 다 ≤15-03)": (dte.ym <= 201503).to_numpy(), "3_완결(둘 다 ≤14-12)": (dte.ym <= 201412).to_numpy(), "4_평탄(둘 다 ≤17-12)": (dte.ym <= 201712).to_numpy()}
YG = {"2007–11": (2007, 2011), "2012–16": (2012, 2016), "2017–20": (2017, 2020)}
def approve_monthly(d, score, a):
    """우리 운영 규칙: 발급월·만기 안에서 점수 순 금액 누적 비율 ≤ a."""
    o = np.lexsort((score, d["T"].to_numpy(), d.mi.to_numpy())); g = pd.DataFrame(dict(m=d.mi.to_numpy()[o], t=d["T"].to_numpy()[o], A=d.funded_amnt.to_numpy()[o]))
    cum = g.groupby(["m", "t"]).A.transform(lambda s: s.cumsum() / s.sum()).to_numpy(); ap = np.zeros(len(d), bool); ap[o] = cum <= a + 1e-12; return ap
def sr_mask(d, m, kind="T1", sub=None, fee=0.0, rfc=None, w=None):
    """승인 마스크 m 을 직접 받아 T1/T2 계산(월별 할당 규칙용)."""
    o = np.argsort(~m, kind="stable"); q = np.array([m.mean()])
    return (t1_curve(d, o, sub=sub, w=w, fee=fee, rfc=rfc, qs=q)[0] if kind == "T1" else t2_curve(d, o, sub, fee=fee, rfc=rfc, qs=q))[0]
ROB = []
for k in STRAT:
    q1, q2 = QSTAR[k]["T1"], QSTAR[k]["T2"]; m1, m2 = sel(ORD_TE[k][1], q1), sel(ORD_TE[k][1], q2)
    r = dict(strategy=k, q_T1=q1, q_T2=q2, T1=sr_mask(dte, m1), T1_금액가중=sr_mask(dte, m1, w=dte.funded_amnt.to_numpy()), T1_관리비1pct=sr_mask(dte, m1, fee=0.01))
    for c in (0.0, 0.01, 0.02, 0.03): r[f"T1_rf{int(c * 100)}"] = sr_mask(dte, m1, rfc=c); r[f"T2_rf{int(c * 100)}"] = sr_mask(dte, m2, "T2", sub=REF_TE, rfc=c)
    r["T2_관리비1pct"] = sr_mask(dte, m2, "T2", sub=REF_TE, fee=0.01)
    for t in (36, 60): r[f"T1_{t}m"] = sr_mask(dte, m1, sub=(dte["T"] == t).to_numpy())
    for g, (a_, b_) in YG.items(): r[f"T1_{g}"] = sr_mask(dte, m1, sub=((dte.yr >= a_) & (dte.yr <= b_)).to_numpy())
    for c, cm in COH.items(): r[f"T2_{c}"] = sr_mask(dte, m2, "T2", sub=cm); r[f"n_{c}"] = int(cm.sum())
    if k != "ALL":
        mm = approve_monthly(dte, SC_TE[k], q1); r["T1_월별할당"] = sr_mask(dte, mm); r["월별할당_승인율"] = mm.mean()
        thr = np.sort(SC_TR[k])[int(round(q1 * len(SC_TR[k]))) - 1]; mf = SC_TE[k] <= thr      # 학습 OOF 상위 q1 의 점수선을 시험에 그대로(발급 시점에 구현 가능한 규칙)
        r["T1_고정점수선"] = sr_mask(dte, mf); r["고정점수선_승인율"] = mf.mean()
    for g, (a_, b_) in YG.items(): r[f"승인율_{g}"] = m1[((dte.yr >= a_) & (dte.yr <= b_)).to_numpy()].mean()     # 발급 시기 집중 진단
    ROB.append(r)
ROB = pd.DataFrame(ROB); ROB.to_csv(TB / "강건성.csv", index=False, encoding="utf-8-sig"); print(ROB.round(3).T.to_string())

# ---- 구간: 발급월 블록 부트스트랩(쌍체), 대출 부트스트랩(v2 단일 값)
cnt, mpos, nm = R_.month_boot(dte); rf = dte.rf.to_numpy()
def agg_t2(m):
    y = np.where(m & REF_TE, dte.r2.to_numpy() - rf, 0.0); return np.stack([np.bincount(mpos, REF_TE * 1.0, nm), np.bincount(mpos, y, nm), np.bincount(mpos, y * y, nm)])
def boot(kind, m):
    if kind == "T1": return R_.t1_boot(dte, m, cnt, mpos, nm)
    n, s1, s2 = (cnt @ v for v in agg_t2(m)); mu = s1 / n; return mu / np.sqrt(s2 / n - mu * mu)
CI = []
for kind in ("T1", "T2"):
    B = {k: boot(kind, sel(ORD_TE[k][1], QSTAR[k][kind])) for k in STRAT}
    for k in STRAT:
        r = dict(ruler=kind, strategy=k, SR_lo=np.percentile(B[k], 2.5), SR_hi=np.percentile(B[k], 97.5))
        if k != FINAL: dd = B[FINAL] - B[k]; r.update(d_final=dd.mean(), d_lo=np.percentile(dd, 2.5), d_hi=np.percentile(dd, 97.5), final_win=(dd > 0).mean())
        CI.append(r)
r1 = dte.r1.to_numpy()
for k in (FINAL, "M6v2"):
    rs = np.random.RandomState(0); mv = sel(ORD_TE[k][1], QSTAR[k]["T1"]); lb = []
    for _ in range(300):
        b = rs.randint(0, len(dte), len(dte)); x = np.where(mv[b], r1[b], rf[b]); lb.append((x.mean() - rf[b].mean()) / x.std(ddof=1))
    CI.append(dict(ruler="T1_대출부트스트랩", strategy=k, SR_lo=np.percentile(lb, 2.5), SR_hi=np.percentile(lb, 97.5)))
CI = pd.DataFrame(CI); CI.to_csv(TB / "구간.csv", index=False, encoding="utf-8-sig"); print(CI.round(4).to_string())

# ---- 최종 포트폴리오(T1 지점): 등급별 승인율·부도율, 만기 구성 / 부도확률 보정(10분위)
mv = sel(ORD_TE[FINAL][1], QSTAR[FINAL]["T1"]); g = dte.sub_grade.str[0]
GR = pd.DataFrame(dict(등급=g, 승인=mv, 부도=dte.dflt)).groupby("등급")[["승인", "부도"]].apply(lambda x: pd.Series(dict(대출=len(x), 승인=int(x.승인.sum()), 승인율=x.승인.mean(), 등급부도율=x.부도.mean(), 승인부도율=x.부도[x.승인].mean() if x.승인.sum() >= 100 else np.nan)))
GR.to_csv(TB / "등급구성_final.csv", encoding="utf-8-sig"); print(GR.round(3).to_string())
CAL = []
for col, nm_ in (("pd_blend", "3모형 평균"), ("pd", "XGBoost")):
    p = dte[col].to_numpy(); dec = pd.qcut(p, 10, labels=False); y = dte.dflt.to_numpy()
    t = pd.DataFrame(dict(dec=dec, p=p, y=y)).groupby("dec").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index(); t["model"] = nm_; CAL.append(t)
    REP[f"ECE_{col}"] = float((t.n * (t.pred - t.obs).abs()).sum() / t.n.sum()); REP[f"AUC_{col}"] = float(roc_auc_score(y, p))
pd.concat(CAL).to_csv(TB / "보정_10분위.csv", index=False, encoding="utf-8-sig")
REP["term60_share_final"] = float((dte["T"][mv] == 60).mean()); REP["QSTAR"] = QSTAR; REP["FINAL"] = FINAL
REP["n_train_confirmed"] = len(Ltr); REP["default_train"] = float(Ltr.default.mean())   # 130이 가공 데이터 없이 표 3을 만들도록 저장
json.dump(REP, open(OUT / "결과_T1_T2.json", "w"), ensure_ascii=False, indent=1, default=float); print(json.dumps(REP, ensure_ascii=False, indent=1, default=float))
