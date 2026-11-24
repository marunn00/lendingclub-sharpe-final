"""117 최종 모델 v2: M6 의 부도확률을 XGB(3시드, 110 과 같은 모델)·LightGBM·CatBoost 평균으로 교체.
채택 근거는 학습 파일 교차검증만(115 AUD_blend: ROE@30 +0.035%p [0.014, 0.049], Sharpe@50% +0.06 [0.02, 0.16]). 시험 파일의 결과 열은 쓰지 않는다.
LightGBM·CatBoost 설정은 98_b_sweep.fit_other / 98_c_ens 'ens' 단계와 같다(LGB: 같은 71열, Cat: TE·빈도 열 대신 명목형 6개+zip3·직함을 범주로).
이득 모델·상수는 110 이 저장한 것(model_gain_*.json)을 그대로 쓴다. 출력: 산출물/테스트평가/test_scores_v2.parquet"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, json, numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier, Pool
A = importlib.import_module("98_apply_test"); K = A.K; C = A.C; log = K.log; G = importlib.import_module("98_gate")
OUT = C.ROOT / "산출물" / "테스트평가"; TRAIN, TEST = C.ROOT / "lending_club_2020_train.csv", C.ROOT / "lending_club_2020_test.csv"
rc = json.load(open(A.MD / "recipe.json")); gc = json.load(open(OUT / "model_gain_const.json")); SEED = K.SEED
CATC = ["k__" + c for c in A.NOMINAL] + ["k__zip3", "k__emp_title_norm"]
def cat_frame(X, D):
    cd = A.codes(D, rc["cmaps"]); Z = X.drop(columns=[c for c in X.columns if c.startswith(("hc__", "te__"))]).copy()
    Z["k__zip3"], Z["k__emp_title_norm"] = cd["zip3"], cd["emp_title_norm"]
    for c in CATC: Z[c] = pd.Series(np.nan_to_num(Z[c].to_numpy(np.float64), nan=-1)).astype(int).astype(str).to_numpy()
    return Z

D = A.raw_frame(TRAIN, rc["table_cols"])
G.enforce(G.gate_table(D, rc["columns"]) + G.gate_te(A.te_oof) + G.gate_source(D.id, TRAIN), "데이터 관문(학습, v2)")
L = D[D.default.notna()].reset_index(drop=True); y = L.default.to_numpy(); X = A.featurize(L, rc, train_y=y); va = np.random.RandomState(SEED).rand(len(X)) < 0.1; log("train X", X.shape)
pl_ = dict(objective="binary", learning_rate=0.08, num_leaves=31, min_child_samples=100, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=10, seed=SEED, verbose=-1)
mL = lgb.train(pl_, lgb.Dataset(X[~va], y[~va]), 1500, valid_sets=[lgb.Dataset(X[va], y[va])], callbacks=[lgb.early_stopping(40, verbose=False)]); mL.save_model(str(OUT / "model_lgb.txt")); log("lgb iters", mL.best_iteration)
Xc = cat_frame(X, L)
mC = CatBoostClassifier(iterations=800, learning_rate=0.15, depth=6, allow_writing_files=False, l2_leaf_reg=5, border_count=64, thread_count=10, random_seed=SEED, verbose=0, early_stopping_rounds=40)
mC.fit(Pool(Xc[~va], y[~va], cat_features=CATC), eval_set=Pool(Xc[va], y[va], cat_features=CATC), use_best_model=True); mC.save_model(str(OUT / "model_cat.cbm")); log("cat iters", mC.get_best_iteration())
del D, L, X, Xc

T = A.raw_frame(TEST, rc["table_cols"]); G.enforce(G.gate_table(T, rc["columns"], labeled=False) + G.gate_source(T.id, TEST), "데이터 관문(시험 적용, v2)")
Xt = A.featurize(T, rc); d = xgb.DMatrix(Xt)
p_x = np.mean([xgb.Booster(model_file=str(A.MD / f"xgb_s{s}.json")).predict(d) for s in range(rc["seeds"])], 0)
p_l = mL.predict(Xt, num_iteration=mL.best_iteration); p_c = mC.predict_proba(Pool(cat_frame(Xt, T), cat_features=CATC))[:, 1]; p = (p_x + p_l + p_c) / 3
ga, gd = (b.predict(d, iteration_range=(0, b.best_iteration + 1)) for b in (xgb.Booster(model_file=str(OUT / f"model_gain_{k}.json")) for k in ("repaid", "default")))
ev = (1 - p) * ga + p * gd; risk = np.sqrt(p * (1 - p) * (ga - gd) ** 2 + (1 - p) * gc["sa"] ** 2 + p * gc["sd"] ** 2)
S1 = pd.read_parquet(OUT / "test_scores.parquet"); assert (S1.id.to_numpy() == T.id.astype(str).to_numpy()).all() and np.allclose(S1.pd.to_numpy(), p_x, atol=1e-6)
out = pd.DataFrame(dict(id=T.id.astype(str), pd_xgb=p_x, pd_lgb=p_l, pd_cat=p_c, pd_blend=p, exp_gain_v2=ev, risk_v2=risk, score_M6v2=-ev / risk))
mc = T.issue_month_code.to_numpy(); tm = T.term_m.to_numpy(); o = np.lexsort((out.score_M6v2.to_numpy(), tm, mc)); amt = T.loan_amnt.to_numpy()
cum = pd.Series(amt[o]).groupby([mc[o], tm[o]]).transform(lambda a: a.cumsum() / a.sum()).to_numpy(); ap = np.zeros(len(out), int); ap[o] = (cum <= 0.25).astype(int); out["approve_25_v2"] = ap
out.to_parquet(OUT / "test_scores_v2.parquet", index=False); log("saved v2", out.shape, "approve share", round(ap.mean(), 3), "corr pd_x/pd_l/pd_c", np.round(np.corrcoef([p_x, p_l, p_c])[0, 1:], 3))
