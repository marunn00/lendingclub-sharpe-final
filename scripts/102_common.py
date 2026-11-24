"""102 공통: 대안 모델·학습법 탐색. 98_b_sweep(데이터·겹·열 레시피), 98_c_ens.merged(채택 레시피), 98_common(ROE·Sharpe 정의)를 그대로 재사용.
- SUB: 1차 패스용 고정 무작위 30만 행(보고 겹 행에서, 시드 0). FULL: 보고 겹 전체(IX).
- fold_data(k, idx): 겹 k 의 (학습, 시험) 특성행렬(98_b.build 결과, float32) — 디스크 캐시.
- evaluate(): 지표 고정(AUC, logloss, ROE@10/30/50/70, Sharpe@53% 36m, 소요시간) 행을 CSV 에 추가."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, json, time, pickle
E = importlib.import_module("98_c_ens"); B = E.B; K = B.K; log = K.log
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
OUT2 = K.C.ROOT / "산출물" / "대안모델"; W2 = OUT2 / "_work"; W2.mkdir(parents=True, exist_ok=True)
KEPT = "T3_hc_both,T3_state_te,T7_year,T4_credit,T4_spread_ff,T4_tsy_ref".split(",")
SP = E.merged(KEPT, "alt", "alt"); BP = json.load(open(K.WORK / "best_params.json"))["best"]
L, n, y_all, FOLD, IX = B.L, B.n, B.y_all, E.FOLD, E.IX
SUB = np.sort(np.random.RandomState(K.SEED).choice(IX, 300000, replace=False))
A_all = L.A.to_numpy(np.float64); TB = L.tb.to_numpy(np.float64); OK = L.roe_ok.to_numpy()
Z_RAW = np.where(OK, L.ann.to_numpy() / A_all - TB, np.nan)              # 승인 시 이득률(초과 연환산 수익), 평가용 원값
Z_TR = np.clip(np.nan_to_num(Z_RAW), -3.0, 0.5)                           # 학습용(사전 고정: [-3, 0.5] 절단)
RATE = B.X.int_rate.to_numpy(np.float32) / 100
CSV = W2 / "results.csv"

def fold_data(k, idx, tag):
    """겹 k 의 학습/시험 인덱스·특성(캐시). 반환 tr, te, Xtr, Xte(DataFrame)."""
    tr, te = idx[FOLD[idx] != k], idx[FOLD[idx] == k]; p = W2 / f"fd_{tag}_{k}.pkl"
    if p.exists(): Xtr, Xte = pickle.load(open(p, "rb"))
    else:
        Xtr, Xte = B.build(SP, tr, te, y_all[tr]); Xtr, Xte = Xtr.astype(np.float32), Xte.astype(np.float32); pickle.dump((Xtr, Xte), open(p, "wb"), protocol=4)
    return tr, te, Xtr, Xte

def evaluate(mid, name, idx, score, pdv=None, sec=0, note=""):
    """score: 전체 길이, 작을수록 먼저 승인(idx 밖 NaN). pdv: 확률(없으면 NaN)."""
    s = np.full(n, np.nan); s[idx] = score[idx]; yy = y_all[idx]
    roe = K.course_roe(L, s, rates=(0.1, 0.3, 0.5, 0.7), groups=False)["all"]
    m = ~np.isnan(s) & (L.term.to_numpy() == 36); sh, _ = K.sharpe_eval(L[m], s[m], 36)
    fa = [roc_auc_score(yy[FOLD[idx] == k], s[idx][FOLD[idx] == k]) for k in np.unique(FOLD[idx])]
    ll = log_loss(yy, np.clip(pdv[idx], 1e-6, 1 - 1e-6)) if pdv is not None else np.nan
    row = dict(id=mid, name=name, rows=len(idx), AUC=roc_auc_score(yy, s[idx]), fold_auc_sd=float(np.std(fa)), logloss=ll, ROE10=roe["r10"], ROE30=roe["r30"], ROE50=roe["r50"], ROE70=roe["r70"],
               SR53_36=sh["SR_r53"], sec=round(sec), note=note)
    pd.DataFrame([row]).to_csv(CSV, mode="a", header=not CSV.exists(), index=False, encoding="utf-8-sig")
    log(mid, "AUC %.4f ll %.4f ROE30 %.3f ROE50 %.3f SR53 %.3f sec %d" % (row["AUC"], ll, row["ROE30"], row["ROE50"], row["SR53_36"], sec)); return row

def guard(mid, k, auc):
    """학습 중 감시(원장 3단계): 겹 AUC 0.95 이상이면 중단."""
    log(mid, "fold", k, "AUC %.4f" % auc)
    if auc >= 0.95: raise SystemExit(f"STOP {mid}: fold AUC {auc:.3f} >= 0.95 (누수 의심)")

def save_oof(mid, score, pdv):
    np.savez_compressed(W2 / f"oof_{mid}.npz", score=score, pd=pdv)
