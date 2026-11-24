"""115 검수: 이전에 '좋았다'고 판정했지만 최종 M6에 빠진 두 요소를 넣으면 좋아지는지, 학습 파일 교차검증(102와 같은 5겹·튜닝 보류 10% 제외)으로만 판정.
- V_blend: M6의 부도확률을 XGB·LightGBM·CatBoost 평균(98_c T8_avg와 같은 OOF 확률)으로 교체
- V_shock: 충격 변수 8열(산출물/충격변수/충격변수.parquet)을 PD·이득 모델 입력에 추가
지표는 102와 같음(102_common.evaluate → 산출물/대안모델/_work/results.csv 에 AUD_* 행). 시험 파일은 쓰지 않는다."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, time, numpy as np, pandas as pd
M = importlib.import_module("102_common"); MD = importlib.import_module("102_models"); B, K, log = M.B, M.K, M.log
OOFD = K.WORK; lgb_o, cat_o = (np.load(OOFD / f"oof_T8_{e}.npz")["pd"] for e in ("lgb", "cat"))
SH = pd.read_parquet(K.C.ROOT / "산출물/충격변수/충격변수.parquet"); SH["id"] = SH.id.astype(str)
SH = SH.set_index("id").loc[M.L.id.astype(str)].reset_index(drop=True).astype(np.float32); assert len(SH) == M.n
def m6(Xtr, Xte, tr, te, blend=False):
    """102_models.m_xgb_pd_twostage 의 M6 와 같은 계산. blend=True 이면 같은 이득 예측에 부도확률만 (XGB+LGB+Cat)/3 로 바꾼 점수도 반환."""
    pdv, _ = B.fit_pred(M.SP, Xtr, M.y_all[tr], None, None, Xte, 600, params=M.BP); y = M.y_all[tr]; z = M.Z_TR[tr]; w = M.A_all[tr] / M.A_all[tr].mean()
    a, _ = MD._xgb_reg(Xtr[y == 0], z[y == 0], w[y == 0], Xte); d, _ = MD._xgb_reg(Xtr[y == 1], z[y == 1], w[y == 1], Xte)
    sa, sd = np.sqrt(np.average((z[y == 0] - z[y == 0].mean()) ** 2)), np.sqrt(np.average((z[y == 1] - z[y == 1].mean()) ** 2))
    def score(p): ev = (1 - p) * a + p * d; return -ev / np.sqrt(p * (1 - p) * (a - d) ** 2 + (1 - p) * sa ** 2 + p * sd ** 2), p
    out = {"": score(pdv)}
    if blend: out["blend"] = score((pdv + lgb_o[te] + cat_o[te]) / 3)
    return out
S = {k: np.full(M.n, np.nan) for k in ("AUD_M6", "AUD_blend", "AUD_shock")}; P = {k: np.full(M.n, np.nan) for k in S}; t0 = time.time()
for k in range(5):
    tr, te, Xtr, Xte = M.fold_data(k, M.IX, "full")
    o = m6(Xtr, Xte, tr, te, blend=True); log("fold", k, "base+blend done", round(time.time() - t0))
    S["AUD_M6"][te], P["AUD_M6"][te] = o[""]; S["AUD_blend"][te], P["AUD_blend"][te] = o["blend"]
    Xtr2, Xte2 = pd.concat([Xtr, SH.iloc[tr].reset_index(drop=True)], axis=1), pd.concat([Xte, SH.iloc[te].reset_index(drop=True)], axis=1)
    o2 = m6(Xtr2, Xte2, tr, te); S["AUD_shock"][te], P["AUD_shock"][te] = o2[""]; log("fold", k, "shock done", round(time.time() - t0))
    M.guard("AUD_shock", k, __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(M.y_all[te], o2[""][1]))
for mid in S: M.evaluate(mid, mid, M.IX, S[mid], P[mid], sec=time.time() - t0, note="audit"); M.save_oof(mid, S[mid], P[mid])
