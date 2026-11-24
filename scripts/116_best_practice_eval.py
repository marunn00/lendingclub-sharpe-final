"""116 우수사례 1·3 평가 방식으로 최종 모델(M6)과 비교 대상(B0 부도확률 XGB, 세부등급 규칙)을 평가.
우수사례3(p.9–17, 21–22): 대출별 IRR(월 현금흐름: 0월 −loan_amnt, 매월 +installment, 부도 시 마지막 납입 다음 달 +recoveries −collection_recovery_fee) → 연환산 − 무위험(발행월 36m→3년, 60m→5년 국채)
  = 초과수익. Sharpe = 승인 대출 초과수익의 평균 ÷ 표준편차(대출 단위). 100 seed 반복: seed마다 검증 표본에서 Sharpe 최대 기준선(승인 비율) 선택 → 시험 표본 Sharpe. 기준선=1(전부 승인)도 함께.
  IRR 계산 불가(받은 돈 0) 규칙: R0 초과수익 0(원 발표), R1 IRR −100%(발표의 수정안 1), R2 계산 불가·음수 IRR 모두 초과수익 0(수정안 2). 대표 = R1.
  검증 표본 = 학습 파일 교차검증 예측(OOF, 시험 파일 미사용), 시험 표본 = 시험 파일 결과 확정 대출. 표본 크기는 각각 시험 확정 대출의 20%(6:2:2의 2).
우수사례1(p.75–98): 혼동행렬(양성 = 부도, 예측 양성 = 거절) Accuracy/Precision/Recall/F1/AUC, 금액 위험
  loss_amnt = installment×term − total_pymnt (실제 부도인데 승인), loss_amnt_implicit = total_pymnt − loan_amnt (실제 상환인데 거절; 슬라이드 설명문 기준), Total Risk = 둘의 합, LendingClub(전부 승인) 행 포함.
  운영 지점: (가) 사전 고정 승인 25%(test_scores.approve_25, B0·등급은 같은 월별 규칙), (나) 거절 비율 = 기저 부도율.
출력: 산출물/우수사례평가/표/*.csv, fig/*.png"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, json, numpy as np, pandas as pd, polars as pl
from sklearn.metrics import roc_auc_score
K = importlib.import_module("98_common"); C = K.C; log = K.log
OUT = C.ROOT / "산출물" / "우수사례평가"; TB = OUT / "표"; FG = OUT / "fig"; [p.mkdir(parents=True, exist_ok=True) for p in (TB, FG)]
COLS = ["id", "issue_d", "term", "loan_status", "loan_amnt", "installment", "total_pymnt", "recoveries", "collection_recovery_fee", "last_pymnt_d"]
MODELS = {"V3": "최종 v3(과제 잣대 목표)", "M6v2": "우리 방식 v2(3모형 부도확률)", "M6": "우리 방식(위험 대비 수익)", "B0": "부도확률 XGBoost", "GRADE": "세부등급 규칙"}

def raw(csv):
    d = pl.read_csv(csv, columns=COLS, infer_schema_length=0, null_values=["", "NA"]).filter(pl.col("issue_d").is_not_null() & pl.col("loan_status").is_in(["Fully Paid", "Charged Off", "Default"])).to_pandas()
    for c in COLS[4:9]: d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["T"] = d.term.str.strip().str[:2].astype(int); d["co"] = d.loan_status.isin(["Charged Off", "Default"]); mi = C.mcode_str(d.issue_d); lp = C.mcode_str(d.last_pymnt_d)
    d["Kp"] = np.where(lp >= 0, np.maximum(lp - mi, 0), 0); return d

def irr_annual(d, chunk=100000):
    """우수사례3 현금흐름의 월 IRR(이분법, 단일 부호 변화) → 연환산. 받은 돈이 0이면 NaN(계산 불가)."""
    out = np.full(len(d), np.nan)
    for s in range(0, len(d), chunk):
        x = d.iloc[s:s + chunk]; n = len(x); Kp = x.Kp.to_numpy(); inst = x.installment.to_numpy(); tp = x.total_pymnt.to_numpy(); rec = x.recoveries.to_numpy(); fee = x.collection_recovery_fee.to_numpy(); co = x.co.to_numpy()
        W = int(Kp.max()) + 3; j = np.arange(W)[None, :]; F = np.zeros((n, W)); F[:, 0] = -x.loan_amnt.to_numpy()
        paid = np.where(co, tp - rec, tp); Kc = np.maximum(Kp, 1)
        # 상환 기간: 1..K−1월 installment, K월 = 남은 금액(음수가 되면 균등 분배)
        last = paid - inst * (Kc - 1); uni = last < 0; per = np.where(uni, paid / Kc, inst); lastv = np.where(uni, paid / Kc, last)
        F += np.where((j >= 1) & (j < Kc[:, None]), per[:, None], 0) + np.where(j == Kc[:, None], lastv[:, None], 0) * (paid[:, None] > 0)
        F += np.where(co[:, None] & (j == (Kc + 1)[:, None]), np.maximum(rec - fee, 0)[:, None], 0)       # 부도: 다음 달 회수액 − 회수 수수료
        out[s:s + n] = (1 + C.irr_monthly(F)) ** 12 - 1
    return out

def excess(irr, rf, rule):
    if rule == "R0": return np.where(np.isnan(irr), 0.0, irr - rf)
    if rule == "R1": return np.where(np.isnan(irr), -1.0, irr) - rf
    return np.where(np.isnan(irr) | (irr < 0), 0.0, irr - rf)

# ---------- 데이터: 시험(채점) / 학습 OOF(검증)
Lt = pd.read_parquet(C.ROOT / "산출물/테스트평가/loans_test.parquet"); St = pd.read_parquet(C.ROOT / "산출물/테스트평가/test_scores.parquet")
Rt = raw(C.ROOT / "lending_club_2020_test.csv"); assert (Rt.id.astype(str).to_numpy() == Lt.id.astype(str).to_numpy()).all()
St = St.merge(pd.read_parquet(C.ROOT / "산출물/테스트평가/test_scores_v2.parquet")[["id", "score_M6v2"]], on="id", validate="1:1").merge(pd.read_parquet(C.ROOT / "산출물/최종보고서/test_scores_v3.parquet")[["id", "score_V3"]], on="id", validate="1:1")
Te = Lt.assign(id=Lt.id.astype(str)).merge(St.drop(columns=["term_m", "loan_amnt"]), on="id", validate="1:1"); Te = pd.concat([Te, Rt[["loan_amnt", "installment", "total_pymnt", "Kp", "T", "recoveries", "collection_recovery_fee", "co"]].reset_index(drop=True)], axis=1)
SG = sorted(Te.sub_grade.unique()); g_rank = lambda s: s.map({g: i for i, g in enumerate(SG)}).to_numpy() + 0.001 * np.random.RandomState(0).rand(len(s))
Te["score_GRADE"] = g_rank(Te.sub_grade)
Lr = K.load_loans(); Rr = raw(C.ROOT / "lending_club_2020_train.csv"); assert (Rr.id.astype(str).to_numpy() == Lr.id.astype(str).to_numpy()).all()
Tr = pd.concat([Lr.reset_index(drop=True), Rr[["loan_amnt", "installment", "total_pymnt", "Kp", "T", "recoveries", "collection_recovery_fee", "co"]].reset_index(drop=True)], axis=1)
OO = C.ROOT / "산출물/대안모델/_work"; Tr["score_M6"] = np.load(OO / "oof_M6_full.npz")["score"]; Tr["score_B0"] = np.load(OO / "oof_B0_full.npz")["score"]; Tr["score_M6v2"] = np.load(OO / "oof_AUD_blend.npz")["score"]; Tr["score_V3"] = np.load(OO / "oof_V3_t1.npz")["score"]   # v2(115)·v3(124) 검증용 OOF
sgt = pd.read_parquet(C.ROOT / "산출물/간결테이블/train_간결.parquet", columns=["sub_grade", "default"]); Tr["sub_grade"] = sgt[sgt.default.notna()].sub_grade.to_numpy(); Tr["score_GRADE"] = g_rank(Tr.sub_grade)
Tr = Tr[Tr.score_M6.notna()].reset_index(drop=True); log("val pool", len(Tr), "test pool", len(Te))
log("IRR test"); Te["irr"] = irr_annual(Te); log("IRR train"); Tr["irr"] = irr_annual(Tr)
for D_ in (Te, Tr): D_["rf"] = D_.tb.to_numpy()
irr_sum = pd.DataFrame([dict(표본=nm, 건수=len(D_), IRR_계산불가=int(D_.irr.isna().sum()), IRR_음수=int((D_.irr < 0).sum()), IRR_중앙값=float(D_.irr.median()), 무위험_평균=float(D_.rf.mean()))
                        for nm, D_ in (("시험", Te), ("검증(학습 OOF)", Tr))]); irr_sum.to_csv(TB / "IRR_요약.csv", index=False, encoding="utf-8-sig"); print(irr_sum.to_string())

# ---------- 우수사례3: 100 seed 기준선 선택 → 시험 Sharpe
FR = np.round(np.arange(0.01, 1.0001, 0.01), 2); nS = int(round(0.2 * len(Te)))
def sharpe_curve(x, score):
    o = np.argsort(score, kind="stable"); xs = x[o]; c1, c2 = np.cumsum(xs), np.cumsum(xs * xs); k = np.maximum((FR * len(xs)).astype(int), 2)
    m = c1[k - 1] / k; v = (c2[k - 1] - k * m * m) / (k - 1); return m / np.sqrt(np.maximum(v, 1e-18))
rows, curves = [], []
for rule in ("R0", "R1", "R2"):
    xv_all, xt_all = excess(Tr.irr.to_numpy(), Tr.rf.to_numpy(), rule), excess(Te.irr.to_numpy(), Te.rf.to_numpy(), rule)
    for seed in range(100):
        rs = np.random.RandomState(seed); iv = rs.choice(len(Tr), nS, replace=False); it = rs.choice(len(Te), nS, replace=False)
        for m in MODELS:
            cv = sharpe_curve(xv_all[iv], Tr["score_" + m].to_numpy()[iv]); ct = sharpe_curve(xt_all[it], Te["score_" + m].to_numpy()[it]); b = int(np.argmax(cv))
            rows.append(dict(rule=rule, seed=seed, model=m, best_frac=FR[b], val_SR=cv[b], test_SR=ct[b], test_SR_at25=ct[24], test_SR_all=ct[-1]))
            if seed == 0: curves += [dict(rule=rule, model=m, frac=f, val_SR=cv[i], test_SR=ct[i]) for i, f in enumerate(FR)]
R3 = pd.DataFrame(rows); R3.to_csv(TB / "우3_seed별.csv", index=False); pd.DataFrame(curves).to_csv(TB / "우3_곡선_seed0.csv", index=False)
q = lambda s: pd.Series({"중앙값": s.median(), "5%": s.quantile(.05), "95%": s.quantile(.95)})
S3 = R3.groupby(["rule", "model"]).agg(최적승인비율_중앙=("best_frac", "median"), 검증SR_중앙=("val_SR", "median"), 시험SR_중앙=("test_SR", "median"), 시험SR_5=("test_SR", lambda s: s.quantile(.05)),
                                       시험SR_95=("test_SR", lambda s: s.quantile(.95)), 승인25_시험SR_중앙=("test_SR_at25", "median"), 전부승인_시험SR=("test_SR_all", "median")).reset_index()
win = R3.pivot_table(index=["rule", "seed"], columns="model", values="test_SR").reset_index(); S3w = win.groupby("rule")[list(MODELS)].apply(lambda g: pd.Series({"M6>B0_seed수": int((g.M6 > g.B0).sum()), "M6>GRADE_seed수": int((g.M6 > g.GRADE).sum()), "M6v2>B0_seed수": int((g.M6v2 > g.B0).sum()), "M6v2>M6_seed수": int((g.M6v2 > g.M6).sum()),
                                                                                              "V3>B0_seed수": int((g.V3 > g.B0).sum()), "V3>M6v2_seed수": int((g.V3 > g.M6v2).sum()), "V3>GRADE_seed수": int((g.V3 > g.GRADE).sum())}))
S3.to_csv(TB / "우3_요약.csv", index=False, encoding="utf-8-sig"); S3w.to_csv(TB / "우3_seed승률.csv", encoding="utf-8-sig"); print(S3.round(3).to_string()); print(S3w)

# ---------- 우수사례1: 혼동행렬·금액 위험 (시험 확정 대출 전부)
y = Te.default.to_numpy().astype(int); base = y.mean()
def approve_monthly(score, a=0.25):
    mc = Te.issue_month.to_numpy(); t = Te.term.to_numpy(); A = Te.A.to_numpy(); o = np.lexsort((score, t, mc)); df = pd.DataFrame(dict(g1=mc[o], g2=t[o], A=A[o]))
    cum = df.groupby(["g1", "g2"]).A.transform(lambda s: s.cumsum() / s.sum()).to_numpy(); ap = np.zeros(len(score), bool); ap[o] = cum <= a + 1e-12; return ap
def approve_count(score, frac): o = np.argsort(score, kind="stable"); ap = np.zeros(len(score), bool); ap[o[:int(round(frac * len(score)))]] = True; return ap
lam = (Te.installment * Te["T"] - Te.total_pymnt).to_numpy(); imp = (Te.total_pymnt - Te.loan_amnt).to_numpy()
def u1(ap, score, name, point):
    pr = (~ap).astype(int); TP = int(((pr == 1) & (y == 1)).sum()); FP = int(((pr == 1) & (y == 0)).sum()); FN = int(((pr == 0) & (y == 1)).sum()); TN = int(((pr == 0) & (y == 0)).sum())
    P_ = TP / max(TP + FP, 1); R_ = TP / max(TP + FN, 1)
    la = float(lam[(y == 1) & (pr == 0)].sum()); li = float(imp[(y == 0) & (pr == 1)].sum())
    return dict(지점=point, 모델=name, 승인건수=int(ap.sum()), TN=TN, FP=FP, FN=FN, TP=TP, Accuracy=(TP + TN) / len(y), Precision=P_, Recall=R_, F1=2 * P_ * R_ / max(P_ + R_, 1e-12),
                AUC=roc_auc_score(y, score) if score is not None else np.nan, loss_amnt=la, loss_amnt_implicit=li, total_risk=la + li), pr
U, PR = [], {}
for point, fn in (("가) 승인 25%(사전 고정)", lambda s: approve_monthly(s)), ("나) 거절 비율 = 부도율", lambda s: approve_count(s, 1 - base))):
    for m in MODELS:
        s = Te["score_" + m].to_numpy(); r, pr = u1(fn(s), s, MODELS[m], point); U.append(r); PR[(point, m)] = pr
    U.append(u1(np.ones(len(y), bool), None, "LendingClub(전부 승인)", point)[0])
U1 = pd.DataFrame(U)
# 공통 판단 제거(우1 p.92–95): 세 모델의 예측이 모두 같은 대출을 빼고 다시 합산
for point in U1.지점.unique():
    same = np.all([PR[(point, m)] == PR[(point, "M6")] for m in ("M6", "B0", "GRADE")], axis=0)        # 우1 원 비교 대상 3개 기준
    for m in MODELS:
        pr = PR[(point, m)]; i = U1.index[(U1.지점 == point) & (U1.모델 == MODELS[m])][0]
        U1.loc[i, "total_risk_공통제외"] = float(lam[(y == 1) & (pr == 0) & ~same].sum() + imp[(y == 0) & (pr == 1) & ~same].sum())
U1.to_csv(TB / "우1_혼동행렬_금액위험.csv", index=False, encoding="utf-8-sig"); print(U1.round(4).to_string())
# 금액 위험 곡선(승인 비율별, 건수 기준)
cur = []
for f in np.round(np.arange(0.05, 1.0001, 0.05), 2):
    for m in MODELS:
        ap = approve_count(Te["score_" + m].to_numpy(), f); pr = ~ap
        cur.append(dict(model=m, frac=f, loss_amnt=lam[(y == 1) & ~pr].sum(), implicit=imp[(y == 0) & pr].sum()))
CU = pd.DataFrame(cur); CU["total_risk"] = CU.loss_amnt + CU.implicit; CU.to_csv(TB / "우1_금액위험_곡선.csv", index=False)
json.dump(dict(n_test=len(Te), n_val_pool=len(Tr), sample_n=nS, base_rate=base), open(OUT / "meta.json", "w"), ensure_ascii=False, indent=1)
