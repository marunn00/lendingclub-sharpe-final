"""102 실행기: python3 scripts/102_run.py <그룹> [--full]. 그룹: xgb2(B0+M5+M6) | reg(M2a) | rank(M2b) | ebm | woe | mlp ...
1차 패스 = 30만 행(SUB), --full = 보고 겹 전체(IX; 겹 캐시 태그 full). 결과: 산출물/대안모델/_work/results.csv, oof_<id>_<tag>.npz"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, time; M = importlib.import_module("102_common"); log = M.log
import numpy as np
G = sys.argv[1]; full = "--full" in sys.argv; idx = M.IX if full else M.SUB; tag = "full" if full else "sub"
MD = importlib.import_module("102_models")
FN = {"xgb2": "m_xgb_pd_twostage", "reg": "m_xgb_profit_reg", "rank": "m_lgb_rank", "rank2": "m_lgb_rank2", "ebm": "m_ebm", "woe": "m_woe", "mlp": "m_mlp", "surv": "m_surv", "ft": "m_ft"}
if G in ("ebm", "woe"): MD = importlib.import_module("102_glass"); 
if G == "ft": MD = importlib.import_module("102_ft")
if G == "surv": MD = importlib.import_module("102_surv")
if G == "mlp": MD = importlib.import_module("102_nn")
fn = getattr(MD, FN[G]); S, P, T = {}, {}, {}; t0 = time.time()
for k in range(5):
    t1 = time.time(); tr, te, Xtr, Xte = M.fold_data(k, idx, tag); out = fn(tr, te, Xtr, Xte); dt = time.time() - t1
    for mid, (s, p) in out.items():
        S.setdefault(mid, np.full(M.n, np.nan))[te] = s; P.setdefault(mid, np.full(M.n, np.nan))[te] = np.nan if p is None else p; T[mid] = T.get(mid, 0) + dt
        from sklearn.metrics import roc_auc_score; M.guard(f"{G}:{mid}", k, roc_auc_score(M.y_all[te], s))
for mid in S:
    M.evaluate(f"{mid}_{tag}", mid, idx, S[mid], P[mid] if not np.isnan(P[mid][idx]).any() else None, sec=time.time() - t0 if len(S) == 1 else T[mid] / len(S) * 0 + (time.time() - t0), note=G); M.save_oof(f"{mid}_{tag}", S[mid], P[mid])
