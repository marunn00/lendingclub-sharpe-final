"""134 소표본 실험실(최종 보고서 v4 Part Ⅰ의 근거): 정규화 회귀(Ridge·Lasso)·분류(ROC·임계값 거리·소프트마진)·1계조건(FOC) 규칙을 작은 표본에서 구현해 보고, 쓸지 말지를 판별한다.
표본·분할(실행 전 고정): 학습 파일(lending_club_2020_train.csv)의 결과 확정 대출에서 N=20,000건을 무작위로 뽑아 학습 60% / 검증 20% / 점검 20%로 나눔. 시험 파일은 쓰지 않음. 20회 반복(시드 0~19, 표본·분할 모두 다시 뽑음); 시드 0은 곡선·표 상세에 사용.
 - 학습(train): 계수 추정·결측 대체값·표준화 평균/표준편차(학습 구간으로만) / 검증(valid): 하이퍼파라미터(α, C) 선택 / 점검(hold): 한 번만 채점.
 - 특성: 숫자형 특성 중 코드값 범주(k__ 열)는 순서가 없어 제외하고 세부등급 코드(k__sub_grade)만 유지, 결측 비율 10% 초과 열은 결측 표지 열 추가.
 - 목표 z = r1 − rf (과제 정의 잣대의 대출 수익 − 발급월 만기 맞춤 국채, 121_rulers.frame). 부도 y = 완납이 아닌 상태.
 - 잣대: 점검 구간 Sharpe = (평균 x − 평균 rf) ÷ 표준편차 x, 승인 x = r1, 거절 x = rf, 승인율 q = 26%(운영 승인율, 사전 고정).
 - 아이디어 사다리(회귀): 가격만 OLS → +신용이력 → +재무 → +시장·시기(전부 OLS) → Ridge → Lasso → 2단계(부도확률×조건부 이득) → 비선형(HistGB) → v3-lite(HistGB 2단계, 비율 순위).
 - 분류: 로지스틱(정규화 없음·L2·L1), 선형 SVM 소프트마진(C), |z|로 가중한 소프트마진. 임계값 거리 d = (점수 − τ) ÷ 점수 표준편차, τ = 상위 26% 경계.
 - FOC: 승인 ⇔ E[z] − λ·{Var + (E[r] − Rp)² − (rf − Rp)²} > 0, λ는 검증 구간 Sharpe 최대, Rp는 승인 포트폴리오 평균 수익(고정점).
출력: 산출물/최종보고서/표/랩_*.csv, 랩_요약.json, fig/lab*.png"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, json, time, warnings
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, roc_curve, log_loss
from sklearn.svm import LinearSVC
warnings.filterwarnings("ignore"); plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False})
M = importlib.import_module("102_common"); R = importlib.import_module("121_rulers"); log = M.log
OUT = M.K.C.ROOT / "산출물" / "최종보고서"; TB = OUT / "표"; FG = OUT / "fig"
N, NSEED, Q = 20000, 20, 0.26

# ---- 자료: 학습 파일 확정 대출(102와 같은 순서), 숫자형 특성
d = R.frame(M.K.C.ROOT / "lending_club_2020_train.csv").set_index("id").loc[M.L.id.astype(str)].reset_index()
assert (d.dflt.to_numpy() == (M.y_all == 1)).all()
Z = (d.r1 - d.rf).to_numpy(); Y = d.dflt.to_numpy().astype(int); R1 = d.r1.to_numpy(); RF = d.rf.to_numpy()
XA = M.B.X.drop(columns=[c for c in M.B.X.columns if c.startswith("k__") and c != "k__sub_grade"]); COLS = list(XA.columns)
PRICE = ["term_m", "int_rate", "rate_premium_subgrade", "k__sub_grade", "installment", "loan_amnt"]
FIN = ["annual_inc", "dti", "emp_length_yr", "tot_cur_bal", "avg_cur_bal", "total_bal_ex_mort", "total_il_high_credit_limit", "total_rev_hi_lim", "total_bc_limit", "pti", "lti", "hc__zip3_logfreq", "hc__emp_title_norm_logfreq"]
MKT = ["issue_year", "tsy10_at_issue", "tsy_matched_at_issue", "spread_10y", "spread_30y", "spread_matched", "spread_fedfunds", "st_hpi_yoy", "st_beta_hpi", "shortemp_x_st_ur_chg6", "d30_vol3"]
CRED = [c for c in COLS if c not in PRICE + FIN + MKT]; assert set(PRICE + FIN + MKT) <= set(COLS)
XV = XA.to_numpy(np.float64); ix = {c: i for i, c in enumerate(COLS)}

class Prep:
    """학습 구간으로만 맞추는 전처리: 중앙값 대체 + 결측 표지(결측 10% 초과 열) + 표준화."""
    def __init__(s, X):
        s.med = np.nanmedian(X, 0); s.miss = np.where(np.isnan(X).mean(0) > 0.10)[0]
        T = s._f(X); s.mu, s.sd = T.mean(0), T.std(0) + 1e-9
    def _f(s, X): return np.hstack([np.where(np.isnan(X), s.med, X), np.isnan(X[:, s.miss]).astype(float)])
    def __call__(s, X): return (s._f(X) - s.mu) / s.sd
    def names(s): return COLS + [f"결측_{COLS[j]}" for j in s.miss]

def sr(r1, rf, m): x = np.where(m, r1, rf); return float((x.mean() - rf.mean()) / x.std(ddof=1))
def top(score, q): m = np.zeros(len(score), bool); m[np.argsort(-score, kind="stable")[: int(round(q * len(score)))]] = True; return m
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
def mz_slope(z, p): return float(np.polyfit(p, z, 1)[0])
def risk_of(p, a, dd, sa, sd): return np.sqrt(p * (1 - p) * (a - dd) ** 2 + (1 - p) * sa ** 2 + p * sd ** 2)
def foc_mask(ev, rk, rf, lam):
    Rp = rf.mean()
    for _ in range(200):
        m = ev - lam * (rk ** 2 + (ev + rf - Rp) ** 2 - (rf - Rp) ** 2) > 0; Rn = np.where(m, ev + rf, rf).mean()
        if abs(Rn - Rp) < 1e-13: break
        Rp = Rn
    return m
ALPHA_R, ALPHA_L = np.logspace(-2, 4.5, 28), np.logspace(-5, -1, 28)

def one_seed(seed, keep=False):
    rs = np.random.RandomState(seed); idx = rs.choice(len(Z), N, replace=False); k = int(N * 0.6); j = int(N * 0.8)
    tr, va, ho = idx[:k], idx[k:j], idx[j:]; P = Prep(XV[tr]); Xt, Xv, Xh = P(XV[tr]), P(XV[va]), P(XV[ho]); zt, zv, zh = Z[tr], Z[va], Z[ho]
    out, det = [], {}
    def add(name, step, pt, pv, ph, score=None, note="", p=np.nan):
        sc = ph if score is None else score
        out.append(dict(seed=seed, model=name, step=step, p=p, rmse_train=rmse(zt, pt), rmse_valid=rmse(zv, pv), rmse_hold=rmse(zh, ph), r2_hold=1 - np.sum((zh - ph) ** 2) / np.sum((zh - zt.mean()) ** 2),
                        mz_hold=mz_slope(zh, ph), SR26=sr(R1[ho], RF[ho], top(sc, Q)), note=note))
    base = np.full(len(zt), zt.mean()); add("평균만(기준)", 0, base, np.full(len(zv), zt.mean()), np.full(len(zh), zt.mean()), score=rs.rand(len(zh)))
    # 1) OLS 사다리: 변수 군을 하나씩 더함(과적합 간격 = 검증 − 학습)
    ladder = [("가격·약정 OLS", PRICE), ("+신용이력 OLS", PRICE + CRED), ("+재무 OLS", PRICE + CRED + FIN), ("+시장·시기 = 전체 OLS", COLS)]
    names = P.names(); col_of = lambda g: [i for i, c in enumerate(names) if (c in g) or (c.startswith("결측_") and c[3:] in g)]
    for s, (nm, g) in enumerate(ladder, 1):
        c = col_of(g); m = LinearRegression().fit(Xt[:, c], zt); add(nm, s, m.predict(Xt[:, c]), m.predict(Xv[:, c]), m.predict(Xh[:, c]), note=f"p={len(c)}", p=len(c))
    ols = sm.OLS(zt, sm.add_constant(Xt)).fit(); pv_ = ols.pvalues[1:]; pcount = dict(p_total=len(pv_), p05=int((pv_ < 0.05).sum()), p_bonf=int((pv_ < 0.05 / len(pv_)).sum()))
    # 2) Ridge·Lasso: 검증 구간으로 α 선택
    rr = [(a, Ridge(alpha=a).fit(Xt, zt)) for a in ALPHA_R]; rv = [rmse(zv, m.predict(Xv)) for _, m in rr]; ir = int(np.argmin(rv)); ar, mr = rr[ir]
    add("Ridge(α 검증 선택)", 5, mr.predict(Xt), mr.predict(Xv), mr.predict(Xh), note=f"α={ar:.3g}", p=Xt.shape[1])
    ll = [(a, Lasso(alpha=a, max_iter=4000).fit(Xt, zt)) for a in ALPHA_L]; lv = [rmse(zv, m.predict(Xv)) for _, m in ll]; il = int(np.argmin(lv)); al, ml = ll[il]
    add("Lasso(α 검증 선택)", 6, ml.predict(Xt), ml.predict(Xv), ml.predict(Xh), note=f"α={al:.3g}, 비영 {int((ml.coef_ != 0).sum())}", p=int((ml.coef_ != 0).sum()))
    # 3) 2단계: 부도확률(로지스틱 L2) × 조건부 이득(Ridge)
    Cs = np.logspace(-3, 1, 9); lg = [(c, LogisticRegression(C=c, max_iter=300).fit(Xt, Y[tr])) for c in Cs]; lgv = [log_loss(Y[va], m.predict_proba(Xv)[:, 1]) for _, m in lg]; cL2, mL2 = lg[int(np.argmin(lgv))]
    def two_stage(Pm, regs, Xs):
        p = Pm.predict_proba(Xs)[:, 1]; a, dd = regs[0].predict(Xs), regs[1].predict(Xs); return p, a, dd
    regs = []
    for g in (0, 1):
        mt, mv = Y[tr] == g, Y[va] == g; fits = [(a, Ridge(alpha=a).fit(Xt[mt], zt[mt])) for a in ALPHA_R]; ev_ = [rmse(zv[mv], m.predict(Xv[mv])) for _, m in fits]; regs.append(fits[int(np.argmin(ev_))][1])
    sa, sd_ = zt[Y[tr] == 0].std(), zt[Y[tr] == 1].std(); T2 = {n_: two_stage(mL2, regs, X_) for n_, X_ in (("t", Xt), ("v", Xv), ("h", Xh))}
    E2 = {n_: (1 - p) * a + p * dd for n_, (p, a, dd) in T2.items()}; ph, ah, dh = T2["h"]
    add("2단계(부도확률×조건부 이득, Ridge)", 7, E2["t"], E2["v"], E2["h"], note=f"C={cL2:.3g}", p=Xt.shape[1])
    # 4) 비선형: HistGB 1단계, v3-lite(HistGB 2단계 + 비율 순위)
    Xr = {"t": XV[tr], "v": XV[va], "h": XV[ho]}
    hg = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=seed).fit(Xr["t"], zt)
    add("비선형(HistGB 1단계)", 8, hg.predict(Xr["t"]), hg.predict(Xr["v"]), hg.predict(Xr["h"]), p=XV.shape[1])
    hc = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=seed).fit(Xr["t"], Y[tr])
    hr = [HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=seed).fit(Xr["t"][Y[tr] == g], zt[Y[tr] == g]) for g in (0, 1)]
    H = {}
    for n_ in "tvh":
        p = hc.predict_proba(Xr[n_])[:, 1]; a, dd = hr[0].predict(Xr[n_]), hr[1].predict(Xr[n_]); ev = (1 - p) * a + p * dd; H[n_] = (p, ev, risk_of(p, a, dd, sa, sd_))
    add("v3-lite(HistGB 2단계 + 비율 순위)", 9, H["t"][1], H["v"][1], H["h"][1], score=H["h"][1] / H["h"][2], note="점수 = E/위험", p=XV.shape[1])
    # 5) FOC 규칙(v3-lite 예측): λ 는 검증 구간 Sharpe 최대
    LAM = np.exp(np.linspace(np.log(0.05), np.log(500), 120)); sv = [sr(R1[va], RF[va], foc_mask(H["v"][1], H["v"][2], RF[va], l)) for l in LAM]; lam = float(LAM[int(np.argmax(sv))])
    mh = foc_mask(H["h"][1], H["h"][2], RF[ho], lam); mrank = top(H["h"][1] / H["h"][2], Q); mlam_v = foc_mask(H["v"][1], H["v"][2], RF[va], lam)
    sig_h = float(np.where(mh, R1[ho], RF[ho]).std(ddof=1)); foc = dict(seed=seed, lam=lam, q_foc=float(mh.mean()), SR_foc=sr(R1[ho], RF[ho], mh), SR_rank26=sr(R1[ho], RF[ho], mrank), SR_rank_qmatch=sr(R1[ho], RF[ho], top(H["h"][1] / H["h"][2], mh.mean())), lam_theory=float(sr(R1[ho], RF[ho], mh) / (2 * sig_h)))
    # 6) 분류: 로지스틱(없음·L2·L1), 선형 SVM 소프트마진(무가중·|z| 가중)
    cls = []; tpos = (zt > 0).astype(int); tv, th = (zv > 0).astype(int), (zh > 0).astype(int)
    def cadd(name, C, dv, dh, nz=""):
        """dv·dh = 부도 점수(클수록 부도 위험). 승인 점수 = −dh."""
        cls.append(dict(seed=seed, model=name, C=C, auc_valid=roc_auc_score(Y[va], dv), auc_hold=roc_auc_score(Y[ho], dh), auc_sign_hold=roc_auc_score(th, -dh), SR26=sr(R1[ho], RF[ho], top(-dh, Q)), nonzero=nz))
    m0 = LogisticRegression(C=1e6, max_iter=500).fit(Xt, Y[tr]); cadd("로지스틱(정규화 없음)", 1e6, m0.decision_function(Xv), m0.decision_function(Xh), int((m0.coef_ != 0).sum()))
    cadd("로지스틱 L2(C 검증 선택)", cL2, mL2.decision_function(Xv), mL2.decision_function(Xh), int((mL2.coef_ != 0).sum()))
    l1 = [(c, LogisticRegression(C=c, penalty="l1", solver="liblinear", max_iter=300).fit(Xt, Y[tr])) for c in Cs]; l1v = [roc_auc_score(Y[va], m.decision_function(Xv)) for _, m in l1]; c1, m1 = l1[int(np.argmax(l1v))]
    cadd("로지스틱 L1(C 검증 선택)", c1, m1.decision_function(Xv), m1.decision_function(Xh), int((m1.coef_ != 0).sum()))
    CS = np.logspace(-4, 0, 9)
    for nm, w in (("선형 SVM 소프트마진(무가중)", None), ("선형 SVM 소프트마진(|z| 가중)", np.abs(zt))):
        fits = [(c, LinearSVC(C=c, loss="hinge", dual=True, max_iter=3000).fit(Xt, tpos, sample_weight=w)) for c in CS]; vv = [roc_auc_score(tv, m.decision_function(Xv)) for _, m in fits]; c_, m_ = fits[int(np.argmax(vv))]
        cadd(nm, c_, -m_.decision_function(Xv), -m_.decision_function(Xh), int((m_.coef_ != 0).sum()))
    fits = [(c, LogisticRegression(C=c, max_iter=300).fit(Xt, tpos)) for c in Cs]; vv = [roc_auc_score(tv, m.decision_function(Xv)) for _, m in fits]; c_, m_ = fits[int(np.argmax(vv))]    # 대조군: SVM과 같은 목표(z>0)로 학습한 로지스틱
    cadd("로지스틱 L2(z>0 목표, 대조군)", c_, -m_.decision_function(Xv), -m_.decision_function(Xh), int((m_.coef_ != 0).sum()))
    res = dict(rows=out, foc=foc, cls=cls, pcount=dict(seed=seed, **pcount, lasso_nz=int((ml.coef_ != 0).sum()), ridge_alpha=ar, lasso_alpha=al))
    # 점검 구간 마진 자료(풀링용): v3-lite 비율 점수
    sc = H["h"][1] / H["h"][2]; ho_d = pd.DataFrame(dict(seed=seed, score=sc, z=zh, y=Y[ho])); tau = np.sort(sc)[::-1][int(round(Q * len(sc))) - 1]; ho_d["d"] = (sc - tau) / sc.std(); res["margin"] = ho_d
    if keep:
        res["curves"] = dict(alpha_r=ALPHA_R, tr_r=[rmse(zt, m.predict(Xt)) for _, m in rr], va_r=rv, alpha_l=ALPHA_L, tr_l=[rmse(zt, m.predict(Xt)) for _, m in ll], va_l=lv, nz_l=[int((m.coef_ != 0).sum()) for _, m in ll],
                             lam=LAM, sr_lam=sv, roc={n_: roc_curve(Y[ho], s_) for n_, s_ in (("로지스틱 정규화 없음", m0.decision_function(Xh)), ("로지스틱 L2", mL2.decision_function(Xh)), ("로지스틱 L1", m1.decision_function(Xh)))},
                             coefs=dict(names=names, ridge=mr.coef_, lasso=ml.coef_, ols=ols.params[1:]),
                             resid={g: dict(train=(zt - E2["t"])[Y[tr] == g], hold=(zh - E2["h"])[Y[ho] == g], one_train=(zt - mr.predict(Xt))[Y[tr] == g], one_hold=(zh - mr.predict(Xh))[Y[ho] == g]) for g in (0, 1)})
    return res

t0 = time.time(); RES = []
for s in range(NSEED):
    RES.append(one_seed(s, keep=(s == 0))); log("seed", s, round(time.time() - t0), "s")

# ---------------- 집계 표
D = pd.DataFrame([r for x in RES for r in x["rows"]]); D.to_csv(TB / "랩_회귀반복.csv", index=False, encoding="utf-8-sig"); G = D.groupby(["step", "model"], sort=True)
LAD = G.agg(rmse_train=("rmse_train", "median"), rmse_valid=("rmse_valid", "median"), rmse_hold=("rmse_hold", "median"), r2_hold=("r2_hold", "median"), mz_hold=("mz_hold", "median"), SR26=("SR26", "median"), SR26_q25=("SR26", lambda v: v.quantile(.25)), SR26_q75=("SR26", lambda v: v.quantile(.75)), 반복=("seed", "count")).reset_index()
LAD["과적합간격"] = LAD.rmse_valid - LAD.rmse_train; LAD["직전대비_점검RMSE"] = LAD.rmse_hold.diff(); LAD["직전대비_SR26"] = LAD.SR26.diff()
wins = []; piv = D.pivot(index="seed", columns="step", values="rmse_hold"); sp = D.pivot(index="seed", columns="step", values="SR26")
for s in LAD.step: wins.append(np.nan if s == 0 else float((piv[s] < piv[s - 1]).mean()))
LAD["점검RMSE_직전보다_작은_비율"] = wins; LAD["SR26_직전보다_큰_비율"] = [np.nan if s == 0 else float((sp[s] > sp[s - 1]).mean()) for s in LAD.step]
LAD.to_csv(TB / "랩_회귀사다리.csv", index=False, encoding="utf-8-sig"); print(LAD.round(4).to_string())
PC = pd.DataFrame([x["pcount"] for x in RES]); PC.to_csv(TB / "랩_p값_선택변수.csv", index=False, encoding="utf-8-sig"); print(PC.median().round(2).to_string())
C_ = pd.DataFrame([r for x in RES for r in x["cls"]]); C_.to_csv(TB / "랩_분류반복.csv", index=False, encoding="utf-8-sig"); CL = C_.groupby("model", sort=False).agg(C=("C", "median"), auc_valid=("auc_valid", "median"), auc_hold=("auc_hold", "median"), auc_sign_hold=("auc_sign_hold", "median"), SR26=("SR26", "median"), nonzero=("nonzero", "median"), 반복=("seed", "count")).reset_index()
CL.to_csv(TB / "랩_분류.csv", index=False, encoding="utf-8-sig"); print(CL.round(4).to_string())
FC = pd.DataFrame([x["foc"] for x in RES]); FC.to_csv(TB / "랩_FOC_반복.csv", index=False, encoding="utf-8-sig"); FCS = dict(lam_median=float(FC.lam.median()), lam_theory_median=float(FC.lam_theory.median()), q_foc_median=float(FC.q_foc.median()), SR_foc=float(FC.SR_foc.median()), SR_rank26=float(FC.SR_rank26.median()), SR_rank_qmatch=float(FC.SR_rank_qmatch.median()),
                                                                                                                             diff_median=float((FC.SR_foc - FC.SR_rank26).median()), win_vs_rank26=float((FC.SR_foc > FC.SR_rank26).mean()), win_vs_qmatch=float((FC.SR_foc > FC.SR_rank_qmatch).mean())); print(FCS)
# 임계값 거리(마진) 구간
MG = pd.concat([x["margin"] for x in RES], ignore_index=True); MG["측"] = np.where(MG.d >= 0, "승인측(τ 이상)", "거절측(τ 미만)"); MG["|d|"] = pd.cut(MG.d.abs(), [0, 0.25, 0.5, 1, 2, np.inf], right=False, labels=["0~0.25", "0.25~0.5", "0.5~1", "1~2", "2 이상"])
MG["맞음"] = np.where(MG.d >= 0, MG.z > 0, MG.z <= 0)
MB = MG.groupby(["측", "|d|"], observed=True).agg(대출=("z", "size"), 맞음비율=("맞음", "mean"), 평균z=("z", "mean"), 평균절대z=("z", lambda v: np.abs(v).mean()), 부도율=("y", "mean")).reset_index(); MB["비중"] = MB.대출 / len(MG)
MB.to_csv(TB / "랩_마진구간.csv", index=False, encoding="utf-8-sig"); print(MB.round(3).to_string())
c0 = RES[0]["curves"]
A0 = pd.concat([pd.DataFrame(dict(모형="Ridge", alpha=c0["alpha_r"], 학습RMSE=c0["tr_r"], 검증RMSE=c0["va_r"], 비영계수=np.nan)), pd.DataFrame(dict(모형="Lasso", alpha=c0["alpha_l"], 학습RMSE=c0["tr_l"], 검증RMSE=c0["va_l"], 비영계수=c0["nz_l"]))]); A0.to_csv(TB / "랩_알파곡선_시드0.csv", index=False, encoding="utf-8-sig")
rd = []
for g, nm in ((0, "완납"), (1, "부도")):
    for key, lab in (("one", "1단계 Ridge"), ("", "2단계(부도확률×조건부)")):
        tr_, ho_ = c0["resid"][g][(key + "_train") if key else "train"], c0["resid"][g][(key + "_hold") if key else "hold"]
        rd.append(dict(집단=nm, 모형=lab, 학습_평균잔차=tr_.mean(), 학습_잔차SD=tr_.std(), 점검_평균잔차=ho_.mean(), 점검_잔차SD=ho_.std(), 대출_점검=len(ho_)))
pd.DataFrame(rd).to_csv(TB / "랩_잔차진단_시드0.csv", index=False, encoding="utf-8-sig"); print(pd.DataFrame(rd).round(4).to_string())
JS = dict(N=N, 학습=int(N * .6), 검증=int(N * .2), 점검=N - int(N * .8), 반복=NSEED, Q=Q, 특성수=len(COLS), 특성군={"가격·약정": len(PRICE), "신용이력": len(CRED), "재무": len(FIN), "시장·시기": len(MKT)}, FOC=FCS,
          p값=dict(전체변수=float(PC.p_total.median()), p05=float(PC.p05.median()), 본페로니=float(PC.p_bonf.median()), lasso_비영=float(PC.lasso_nz.median())), 소요초=round(time.time() - t0))
json.dump(JS, open(TB / "랩_요약.json", "w"), ensure_ascii=False, indent=1)

# ---------------- 그림(제목은 보고서 캡션에)
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "legend.frameon": False})
NAVY, BLUE, ORANGE, GRAY, RED = "#1F4E79", "#5B9BD5", "#ED7D31", "#A5A5A5", "#C00000"
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, nm, a, t_, v_ in ((axs[0], "Ridge", c0["alpha_r"], c0["tr_r"], c0["va_r"]), (axs[1], "Lasso", c0["alpha_l"], c0["tr_l"], c0["va_l"])):
    ax.semilogx(a, t_, color=BLUE, label="학습(in-sample)"); ax.semilogx(a, v_, color=NAVY, label="검증(out-of-sample)"); i = int(np.argmin(v_)); ax.axvline(a[i], color=RED, ls=":"); ax.set_xlabel(f"{nm} 규제 강도 α"); ax.set_ylabel("RMSE (z)"); ax.legend(); ax.text(a[i], np.mean(ax.get_ylim()), f" α*={a[i]:.3g}", color=RED)
axs[2].semilogx(c0["alpha_l"], c0["nz_l"], color=NAVY); axs[2].set_xlabel("Lasso 규제 강도 α"); axs[2].set_ylabel("비영 계수 수(p)")
fig.tight_layout(); fig.savefig(FG / "lab1_알파곡선.png", dpi=200); plt.close(fig)
fig, ax = plt.subplots(figsize=(6.2, 4.2)); st = LAD[LAD.step >= 1]; x = np.arange(len(st)); ax.bar(x - 0.2, st.rmse_train, 0.4, color=BLUE, label="학습(in-sample)"); ax.bar(x + 0.2, st.rmse_hold, 0.4, color=NAVY, label="점검(out-of-sample)")
ax.set_xticks(x); ax.set_xticklabels([f"{int(s)}" for s in st.step]); ax.set_xlabel("아이디어 사다리 단계(표 F3)"); ax.set_ylabel("RMSE (z), 20회 중앙값"); ax.set_ylim(st.rmse_train.min() * 0.9, st.rmse_hold.max() * 1.02); ax.legend()
fig.tight_layout(); fig.savefig(FG / "lab2_잔차사다리.png", dpi=200); plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for (nm, (fp, tp, _)), c in zip(c0["roc"].items(), (GRAY, NAVY, ORANGE)): axs[0].plot(fp, tp, color=c, label=nm)
axs[0].plot([0, 1], [0, 1], color="#888", ls=":"); axs[0].set_xlabel("FPR(완납을 거절하는 비율)"); axs[0].set_ylabel("TPR(부도를 거절하는 비율)"); axs[0].legend(loc="lower right")
mm = MB.pivot(index="|d|", columns="측", values="맞음비율"); axs[1].plot(range(len(mm)), mm["승인측(τ 이상)"], "o-", color=NAVY, label="승인측"); axs[1].plot(range(len(mm)), mm["거절측(τ 미만)"], "s-", color=ORANGE, label="거절측")
axs[1].set_xticks(range(len(mm))); axs[1].set_xticklabels(mm.index); axs[1].set_xlabel("임계값 거리 |d| (점수 표준편차 단위)"); axs[1].set_ylabel("결정이 맞은 비율"); axs[1].legend()
fig.tight_layout(); fig.savefig(FG / "lab3_ROC_마진.png", dpi=200); plt.close(fig)
log("done", JS["소요초"], "s")
