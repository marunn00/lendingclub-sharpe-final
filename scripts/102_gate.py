"""102 데이터 관문(학습 전 자동 검사). 실패 시 SystemExit — 학습 시작 금지. 통과하면 1차 패스 30만 행의 5겹 특성행렬 캐시를 만든다."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; M = importlib.import_module("102_common"); B, K, log = M.B, M.K, M.log
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
F = importlib.import_module("98_features"); POST = F.POST
fail = []
chk = lambda ok, msg: (log("OK  " if ok else "FAIL", msg), None if ok else fail.append(msg))
cols = list(M.SP["cols"])
chk(not any(c.startswith("lk__") for c in cols), "누수(lk__) 열 없음")
chk(not (set(cols) & POST), "사후 결과 열(원장) 없음: %s" % sorted(set(cols) & POST))
chk(not (set(cols) & {"default", "loan_status", "id", "funded_amnt", "total_pymnt", "recoveries"}), "라벨·식별·결과 열 없음")
chk(len(B.D) == len(B.L) and (B.D.id.to_numpy() == B.L.id.to_numpy()).all(), "행 수·키 정렬 %d" % len(B.D))
chk(len(B.D) == 1116156, "확정 대출 1,116,156건")
chk(set(np.unique(M.FOLD)) == {-1, 0, 1, 2, 3, 4}, "겹 구성 -1..4"); chk(abs((M.FOLD == -1).mean() - 0.10) < 0.005, "보류 10%% 분리 %.4f" % (M.FOLD == -1).mean())
chk(len(M.SUB) == 300000 and (M.FOLD[M.SUB] >= 0).all(), "30만 부분표본 보류 제외")
cnt = np.bincount(M.FOLD[M.SUB]); chk(cnt.min() > 55000, "겹 크기 %s" % cnt.tolist())
dr = [B.y_all[M.SUB][M.FOLD[M.SUB] == k].mean() for k in range(5)]; chk(max(dr) - min(dr) < 0.01, "겹별 부도율 %s" % np.round(dr, 4).tolist())
X = B.X; ys = B.y_all[M.SUB]; worst = []
for c in cols:
    v = X[c].to_numpy()[M.SUB]; ok = ~np.isnan(v)
    if ok.sum() > 1000 and len(np.unique(ys[ok])) == 2: a = roc_auc_score(ys[ok], v[ok]); worst.append((max(a, 1 - a), c))
worst.sort(reverse=True); chk(worst[0][0] < 0.80, "단일 열 최대 AUC %.3f (%s)" % worst[0])
chk(np.isfinite(M.Z_TR).all() and np.isnan(M.Z_RAW).mean() < 0.01, "수익 목표값 결측 %.4f(roe_ok 거짓 행)" % np.isnan(M.Z_RAW).mean())
for k in range(5):
    tr, te, Xtr, Xte = M.fold_data(k, M.SUB, "sub")
    chk(np.isfinite(Xtr.to_numpy()[~np.isnan(Xtr.to_numpy())]).all() and Xtr.isna().to_numpy().any() and list(Xtr.columns) == list(Xte.columns) and len(Xtr) == len(tr), "겹%d 행렬 NaN 보존·무한값 없음 %s" % (k, Xtr.shape))
chk(not (set(tr) & set(te)), "학습·시험 겹침 없음")
log("GATE", "FAIL" if fail else "PASS")
if fail: raise SystemExit("GATE FAIL: " + "; ".join(fail))
