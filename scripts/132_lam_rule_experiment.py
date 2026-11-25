"""132 탐색 실험(최종 보고서 본문 미반영, 활용 예시): 「LendingClub_Sharpe_최종보고서」 5.4절 1계조건 규칙을 v3 예측에 적용.
규칙: 승인 ⇔ E[z] − λ·ΔV > 0, ΔV = Var + (E[r] − Rp)² − (rf − Rp)², E[r] = E[z] + rf, Rp = 예측 기준 포트폴리오 평균(고정점).
λ는 학습 OOF(124와 같은 5겹)에서 과제 정의 Sharpe 최대로 고르고, 시험 파일은 채점에만 쓴다."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, numpy as np, pandas as pd
M = importlib.import_module("102_common"); MD = importlib.import_module("102_models"); R = importlib.import_module("121_rulers")
ROOT = M.K.C.ROOT

d = R.frame(ROOT / "lending_club_2020_train.csv").set_index("id").loc[M.L.id.astype(str)].reset_index()
z1 = (d.r1 - d.rf).to_numpy(); y = M.y_all; p_oof = np.load(M.W2 / "oof_AUD_blend.npz")["pd"]
EV, RK = np.full(M.n, np.nan), np.full(M.n, np.nan)
for k in range(5):
    tr, te, Xtr, Xte = M.fold_data(k, M.IX, "full"); yt, zt, w = y[tr], z1[tr], np.ones(len(tr))
    a, _ = MD._xgb_reg(Xtr[yt == 0], zt[yt == 0], w[yt == 0], Xte); dd, _ = MD._xgb_reg(Xtr[yt == 1], zt[yt == 1], w[yt == 1], Xte)
    sa, sd = np.sqrt(np.var(zt[yt == 0])), np.sqrt(np.var(zt[yt == 1])); p = p_oof[te]
    EV[te] = (1 - p) * a + p * dd; RK[te] = np.sqrt(p * (1 - p) * (a - dd) ** 2 + (1 - p) * sa ** 2 + p * sd ** 2)
S3 = np.load(M.W2 / "oof_V3_t1.npz")["score"]; keep = ~np.isnan(S3)
gap = np.nanmax(np.abs(-EV[keep] / RK[keep] - S3[keep])); print("OOF 재현 최대 차이", gap); assert gap < 1e-4

def lam_mask(ev, rk, rf, lam):
    Rp = rf.mean()
    for _ in range(200):
        m = ev - lam * (rk ** 2 + (ev + rf - Rp) ** 2 - (rf - Rp) ** 2) > 0
        Rn = np.where(m, ev + rf, rf).mean()
        if abs(Rn - Rp) < 1e-13: break
        Rp = Rn
    return m
def sr(D, m):
    x = np.where(m, D.r1.to_numpy(), D.rf.to_numpy()); s = x.std(ddof=1); return (x.mean() - D.rf.mean()) / s, s

D = d[keep].reset_index(drop=True); ev, rk, rf = EV[keep], RK[keep], D.rf.to_numpy()
LAM = np.exp(np.linspace(np.log(0.05), np.log(500), 161)); rows = []
for lam in LAM:
    m = lam_mask(ev, rk, rf, lam); s, sg = sr(D, m); rows.append(dict(lam=lam, SR=s, sigma=sg, q=m.mean()))
TR = pd.DataFrame(rows); i = int(TR.SR.idxmax()); lam_s = float(TR.lam[i])
v3_tr = sr(D, R.sel(R.order_rank(S3[keep])[1], 0.26))[0]
print("학습 OOF: λ* %.3f SR %.4f 승인율 %.3f | 이론 SR/(2σ) %.3f | v3(q=0.26) SR %.4f" % (lam_s, TR.SR[i], TR.q[i], TR.SR[i] / (2 * TR.sigma[i]), v3_tr))
print(TR.iloc[max(i - 12, 0):i + 13:3].round(4).to_string(index=False))

T = pd.read_parquet(ROOT / "산출물/최종보고서/test_scores_v3.parquet")[["id", "exp_gain_v3", "risk_v3", "score_V3"]]
dte = R.frame(ROOT / "lending_club_2020_test.csv").merge(T, on="id", how="left", validate="1:1"); assert len(dte) == 744608 and dte.score_V3.notna().all()
m_v = R.sel(R.order_rank(dte.score_V3.to_numpy())[1], 0.26); m_l = lam_mask(dte.exp_gain_v3.to_numpy(), dte.risk_v3.to_numpy(), dte.rf.to_numpy(), lam_s)
(sv, _), (sl, sgl) = sr(dte, m_v), sr(dte, m_l); assert abs(sv - 0.14616601679834823) < 1e-9
cnt, mpos, nm = R.month_boot(dte); bd = R.t1_boot(dte, m_l, cnt, mpos, nm) - R.t1_boot(dte, m_v, cnt, mpos, nm)
print("시험: v3 SR %.4f(승인 26.0%%) | 1계조건 SR %.4f 승인율 %.3f 부도율 %.3f σ %.4f | 이론 λ %.3f" % (sv, sl, m_l.mean(), dte.dflt[m_l].mean(), sgl, sl / (2 * sgl)))
print("차이(1계조건 − v3) %.4f, 발급월 블록 95%% 구간 [%.4f, %.4f], 1계조건 우세 비율 %.3f" % (sl - sv, *np.percentile(bd, [2.5, 97.5]), (bd > 0).mean()))
print("승인 집합 겹침(자카드) %.3f" % ((m_v & m_l).sum() / (m_v | m_l).sum()))

# ---- 보고서 v4용 저장: λ 곡선(학습 OOF가 선택 근거, 시험 곡선은 사후 참고), 요약, 같은 승인율의 비율 순위와 비교
import json
TB = ROOT / "산출물" / "최종보고서" / "표"; TR.rename(columns={"lam": "λ", "SR": "학습OOF_Sharpe", "sigma": "학습OOF_σ", "q": "승인율"}).to_csv(TB / "FOC_학습곡선.csv", index=False, encoding="utf-8-sig")
ts = []
for lam in LAM[::2]:
    mk = lam_mask(dte.exp_gain_v3.to_numpy(), dte.risk_v3.to_numpy(), dte.rf.to_numpy(), lam); s_, g_ = sr(dte, mk); ts.append(dict(λ=lam, 시험_Sharpe=s_, 시험_σ=g_, 승인율=float(mk.mean())))
pd.DataFrame(ts).to_csv(TB / "FOC_시험곡선_사후참고.csv", index=False, encoding="utf-8-sig")
m_q = R.sel(R.order_rank(dte.score_V3.to_numpy())[1], float(m_l.mean())); s_q, _ = sr(dte, m_q); bq = R.t1_boot(dte, m_l, cnt, mpos, nm) - R.t1_boot(dte, m_q, cnt, mpos, nm)
json.dump(dict(lam_star=lam_s, 학습_SR_foc=float(TR.SR[i]), 학습_승인율_foc=float(TR.q[i]), 학습_이론_lam=float(TR.SR[i] / (2 * TR.sigma[i])), 학습_SR_v3_q26=float(v3_tr),
               시험_SR_v3_q26=float(sv), 시험_SR_foc=float(sl), 시험_승인율_foc=float(m_l.mean()), 시험_부도율_foc=float(dte.dflt[m_l].mean()), 시험_σ_foc=float(sgl), 시험_이론_lam=float(sl / (2 * sgl)),
               차이_foc_v3=float(sl - sv), 차이_구간=[float(x) for x in np.percentile(bd, [2.5, 97.5])], foc_우세비율=float((bd > 0).mean()), 자카드=float((m_v & m_l).sum() / (m_v | m_l).sum()),
               시험_SR_순위_같은승인율=float(s_q), 차이_foc_순위_같은승인율=float(sl - s_q), 차이_같은승인율_구간=[float(x) for x in np.percentile(bq, [2.5, 97.5])], 시험_부도율_v3_q26=float(dte.dflt[m_v].mean())),
          open(TB / "FOC_요약.json", "w"), ensure_ascii=False, indent=1)
print("saved FOC tables")
