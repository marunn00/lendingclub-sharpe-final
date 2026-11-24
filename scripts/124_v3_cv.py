"""124 v3 후보 판정(학습 파일 교차검증만, 시험 파일 미사용).
v3 = v2와 같은 부도확률(3모형 평균 OOF, oof_AUD_blend)·같은 점수 구조(−기대이득 ÷ 위험)에서, 이득 회귀의 목표만 과제 잣대(T1)의 대출 수익 정의로 바꾼 것.
 목표 z = (total_pymnt / funded_amnt)^(12/T) − 1 − rf(발급월 만기 맞춤 국채), 대출 동일 가중(T1이 대출 건수 기준이므로).
 v2 목표는 실제 상환기간 연환산 초과수익(ann/A − tb, [−3, 0.5] 절단, 금액 가중).
채택 규칙(실행 전 고정): 학습 OOF에서 각자 고른 q*의 T1 차이(v3 − v2)의 발급월 블록 부트스트랩 95% 구간 하한 > 0 이면 v3 채택, 아니면 v2 유지.
동기 공개: 이 후보는 팀 공유 「LendingClub_Sharpe_최종보고서」와 같은 잣대(T1)의 시험 비교(우리 0.140 대 그 보고서 0.147)를 본 뒤에 만들었다.
출력: 산출물/대안모델/_work/oof_V3_t1.npz, 산출물/최종보고서/표/v3_판정.csv"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, time, numpy as np, pandas as pd
M = importlib.import_module("102_common"); MD = importlib.import_module("102_models"); R = importlib.import_module("121_rulers"); log = M.log
TB = M.K.C.ROOT / "산출물" / "최종보고서" / "표"; TB.mkdir(parents=True, exist_ok=True)
d = R.frame(M.K.C.ROOT / "lending_club_2020_train.csv").set_index("id").loc[M.L.id.astype(str)].reset_index()
assert (d.dflt.to_numpy() == (M.y_all == 1)).all()
z1 = (d.r1 - d.rf).to_numpy(); y = M.y_all
V2 = np.load(M.W2 / "oof_AUD_blend.npz"); p_oof, s_v2 = V2["pd"], V2["score"]
S3 = np.full(M.n, np.nan); t0 = time.time()
for k in range(5):
    tr, te, Xtr, Xte = M.fold_data(k, M.IX, "full"); yt, zt, w = y[tr], z1[tr], np.ones(len(tr))
    a, ia = MD._xgb_reg(Xtr[yt == 0], zt[yt == 0], w[yt == 0], Xte); dd, idd = MD._xgb_reg(Xtr[yt == 1], zt[yt == 1], w[yt == 1], Xte)
    sa, sd = np.sqrt(np.var(zt[yt == 0])), np.sqrt(np.var(zt[yt == 1])); p = p_oof[te]; assert not np.isnan(p).any()
    S3[te] = -((1 - p) * a + p * dd) / np.sqrt(p * (1 - p) * (a - dd) ** 2 + (1 - p) * sa ** 2 + p * sd ** 2)
    log("fold", k, "rounds", ia, idd, "sa %.4f sd %.4f" % (sa, sd), round(time.time() - t0), "s")
np.savez_compressed(M.W2 / "oof_V3_t1.npz", score=S3, pd=p_oof)
keep = ~np.isnan(S3); assert (keep == ~np.isnan(s_v2)).all(); D = d[keep].reset_index(drop=True); sc = {"M6v2": s_v2[keep], "V3": S3[keep]}
cur = {k: R.t1_curve(D, R.order_rank(v)[0])[0] for k, v in sc.items()}; qs = {k: float(R.QS[np.nanargmax(c)]) for k, c in cur.items()}
cur2 = {k: R.t2_curve(D, R.order_rank(v)[0], D.ref.to_numpy()) for k, v in sc.items()}
# 발급월 블록 부트스트랩(학습 OOF, 1,000회): 각자 q* 승인 고정
months = np.sort(D.mi.unique()); mpos = np.searchsorted(months, D.mi.to_numpy()); nm = len(months); rf = D.rf.to_numpy(); r1 = D.r1.to_numpy()
idx = np.random.RandomState(0).randint(0, nm, (1000, nm)); cnt = np.stack([np.bincount(i, minlength=nm) for i in idx]).astype(float)
def boot(m):
    x = np.where(m, r1, rf); n, s1, s2, sf = (cnt @ np.bincount(mpos, v, nm) for v in (np.ones(len(x)), x, x * x, rf)); mx = s1 / n; return (mx - sf / n) / np.sqrt((s2 / n - mx * mx) * n / (n - 1))
b = {k: boot(R.sel(R.order_rank(v)[1], qs[k])) for k, v in sc.items()}; dl = b["V3"] - b["M6v2"]; lo, hi = np.percentile(dl, [2.5, 97.5])
out = pd.DataFrame([dict(model=k, q_star=qs[k], T1_train=float(np.nanmax(cur[k])), T2_train_at_T1q=float(cur2[k][np.nanargmax(cur[k])]), T2_train_max=float(np.nanmax(cur2[k])),
                         q_T2=float(R.QS[np.nanargmax(cur2[k])])) for k in sc])
out["d_T1_v3_minus_v2"], out["d_lo"], out["d_hi"], out["v3_win_share"] = float(dl.mean()), lo, hi, float((dl > 0).mean()); out["adopt_v3"] = bool(lo > 0)
out.to_csv(TB / "v3_판정.csv", index=False, encoding="utf-8-sig"); print(out.round(4).to_string()); log("판정", "v3 채택" if lo > 0 else "v2 유지")
