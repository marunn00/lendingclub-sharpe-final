"""102 M3 경쟁위험(구간화 이산시간): 11 클래스 다항 LightGBM = {부도, 조기상환} × 나이 구간(12개월) + 만기완납(102_events.py).
구간별 위험 p_c(x) → 기대 이득률 E[z|x] = Σ_c p_c(x)·μ_c(만기, 등급문자), μ_c = 학습 겹의 클래스별 평균 z(대출액 가중; 복원 현금흐름의 연환산 이득). PD = Σ_{c≤4} p_c."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; M = importlib.import_module("102_common"); B, K, log = M.B, M.K, M.log
import numpy as np, pandas as pd, lightgbm as lgb
EV = np.load(M.W2 / "events.npy"); NC = 11

def m_surv(tr, te, Xtr, Xte):
    ev = EV[tr]; va = np.random.RandomState(0).rand(len(tr)) < 0.10
    p = dict(objective="multiclass", num_class=NC, learning_rate=0.1, num_leaves=31, min_child_samples=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=10, seed=0, verbose=-1)
    b = lgb.train(p, lgb.Dataset(Xtr[~va], ev[~va]), 300, valid_sets=[lgb.Dataset(Xtr[va], ev[va])], callbacks=[lgb.early_stopping(20, verbose=False)])
    P = b.predict(Xte, num_iteration=b.best_iteration); log("  M3 iters", b.best_iteration)
    gl = lambda X: (X["k__sub_grade"].fillna(0).to_numpy() // 5).astype(int); tm = lambda X: (X["term_m"].to_numpy() == 60).astype(int)
    gtr, ttr, gte, tte = gl(Xtr), tm(Xtr), gl(Xte), tm(Xte); z, w = M.Z_TR[tr], M.A_all[tr]
    mu = np.zeros((NC, 2, 7)); 
    for c in range(NC):
        for t in (0, 1):
            m0 = (ev == c) & (ttr == t); mc = np.average(z[m0], weights=w[m0]) if m0.sum() >= 30 else np.nan; mu[c, t] = mc
            for g in range(7):
                m1 = m0 & (gtr == g); mu[c, t, g] = np.average(z[m1], weights=w[m1]) if m1.sum() >= 30 else mc
    mu = np.where(np.isnan(mu), 0.0, mu)
    ez = np.zeros(len(te))
    for c in range(NC): ez += P[:, c] * mu[c, tte, gte]
    pdv = P[:, :5].sum(1)
    return {"M3": (-ez, pdv)}
