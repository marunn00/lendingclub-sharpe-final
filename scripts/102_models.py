"""102 모델 구현. 각 함수: (tr, te, Xtr, Xte) → {모델id: (score(te 길이, 작을수록 먼저 승인), pd 또는 None)}.
결측: 트리/EBM 은 NaN 그대로. WoE·MLP 는 학습 겹 안 중앙값 대치 + 결측 표지(모형 요구 인코딩)."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; M = importlib.import_module("102_common"); B, K, log = M.B, M.K, M.log
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.linear_model import LogisticRegression
CATC = lambda X: [c for c in X.columns if c in B.NOMINAL or c in B.ORD]       # 범주 코드 열(k__*)

def _val(n, seed): return np.random.RandomState(seed).rand(n) < 0.10

def _xgb_reg(Xtr, ztr, wtr, Xte, seed=0, rounds=600):
    va = _val(len(Xtr), seed); p = dict(B.P0, **M.BP, seed=seed, objective="reg:squarederror", eval_metric="rmse")
    b = xgb.train(p, xgb.DMatrix(Xtr[~va], ztr[~va], weight=wtr[~va]), rounds, evals=[(xgb.DMatrix(Xtr[va], ztr[va], weight=wtr[va]), "va")], early_stopping_rounds=30, verbose_eval=False)
    return b.predict(xgb.DMatrix(Xte), iteration_range=(0, b.best_iteration + 1)), b.best_iteration + 1

def m_xgb_pd_twostage(tr, te, Xtr, Xte):
    """B0(PD XGB), M5(2단계: (1−PD)·a + PD·d), M6(M5 기대수익 ÷ 위험 표준편차)."""
    pdv, it = B.fit_pred(M.SP, Xtr, M.y_all[tr], None, None, Xte, 600, params=M.BP); y = M.y_all[tr]; z = M.Z_TR[tr]; w = M.A_all[tr] / M.A_all[tr].mean()
    a, _ = _xgb_reg(Xtr[y == 0], z[y == 0], w[y == 0], Xte); d, _ = _xgb_reg(Xtr[y == 1], z[y == 1], w[y == 1], Xte)
    ev = (1 - pdv) * a + pdv * d
    sa, sd = np.sqrt(np.average((z[y == 0] - z[y == 0].mean()) ** 2)), np.sqrt(np.average((z[y == 1] - z[y == 1].mean()) ** 2))   # 조건부 잔차 표준편차(상수)
    var = pdv * (1 - pdv) * (a - d) ** 2 + (1 - pdv) * sa ** 2 + pdv * sd ** 2
    return {"B0": (pdv, pdv), "M5": (-ev, pdv), "M6": (-ev / np.sqrt(var), pdv)}

def m_xgb_profit_reg(tr, te, Xtr, Xte):
    """M2a: 초과 연환산 수익 z 를 직접 회귀(대출액 가중). 학습 z 는 [−3, 0.5] 절단."""
    w = M.A_all[tr] / M.A_all[tr].mean(); p, it = _xgb_reg(Xtr, M.Z_TR[tr], w, Xte)
    return {"M2a": (-p, None)}

def m_lgb_rank(tr, te, Xtr, Xte):
    """M2b: LambdaRank — 발행월 그룹 안에서 z 의 십분위(0–9, 높을수록 이득)를 순위 학습."""
    import lightgbm as lgb
    mo = M.B.D.issue_month_code.to_numpy()[tr]; o = np.argsort(mo, kind="stable"); X_ = Xtr.iloc[o]; z = M.Z_TR[tr][o]; mo = mo[o]
    lab = pd.Series(z).groupby(mo).rank(pct=True, method="first").mul(10).clip(upper=9.999).astype(int).to_numpy()
    grp = np.bincount(mo - mo.min()); grp = grp[grp > 0]
    return {"M2b": _rank(Xtr, Xte, lab, grp, o)}

def m_lgb_rank2(tr, te, Xtr, Xte):
    """M2b2: 1차 실패(십분위 라벨이 비부도 대출 안의 금리 순서를 학습) 후 사전 고정한 절대 구간 라벨 — 손실 쪽을 세분(z 절단 구간 9단계)."""
    mo = M.B.D.issue_month_code.to_numpy()[tr]; o = np.argsort(mo, kind="stable"); z = M.Z_TR[tr][o]; mo = mo[o]
    lab = np.digitize(z, [-1.5, -0.9, -0.4, -0.1, 0.03, 0.06, 0.09, 0.12]); grp = np.bincount(mo - mo.min()); grp = grp[grp > 0]
    return {"M2b2": _rank(Xtr, Xte, lab, grp, o)}

def _rank(Xtr, Xte, lab, grp, o):
    import lightgbm as lgb; X_ = Xtr.iloc[o]
    p = dict(objective="lambdarank", learning_rate=0.1, num_leaves=31, min_child_samples=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=10, seed=0,
             verbose=-1, label_gain=list(range(10)), lambdarank_truncation_level=200, max_bin=127)
    b = lgb.train(p, lgb.Dataset(X_, lab, group=grp), 150)
    return -b.predict(Xte), None
