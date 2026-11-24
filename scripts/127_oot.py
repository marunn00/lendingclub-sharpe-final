"""127 시간 외 검증(OOT): 학습 파일 중 LAST년 이하 발급분만으로 최종 모델(v3) 전 과정을 다시 학습 → 시험 파일 LAST+1년 이후 발급분에 적용.
사용: python3 scripts/127_oot.py [LAST=2015]. 2015 = 「CREDIT_MODEL_FINAL_REPORT」 OOT(적합 ≤2015)와 같은 학습 기간, 2016 = 「Sharpe 최종보고서」 6.4절(학습 762,059건 = 2016년 이하)과 같은 학습 기간.
- 학습: 98_apply_test.fit(전처리·TE·XGBoost 3시드, 2016년 이후 학습 id 제외) + LightGBM·CatBoost(117 설정) + 이득 회귀(125 설정, T1 목표).
- 비교: 같은 2015년 이전 학습의 부도확률 XGBoost(B0), 세부등급 순서, 전부 승인.
- 승인 비율(실행 전 고정, 시험을 보고 바꾸지 않음): 본 분석의 학습 OOF q*(121·124) 그대로, 평가 구간의 대출 풀 안에서 상위 q 승인.
- 평가 구간: (가) 2017–2020 발급, T1 — 「Sharpe 최종보고서」 6.4절과 같은 구간 / (나) 2016–2017 발급, T2 — 「CREDIT_MODEL_FINAL_REPORT」 OOT_MATURED와 같은 구간.
- 같은 승인율 비교(사후 참고, 시험 결과를 본 뒤 추가): 「Sharpe 최종보고서」 OOT 승인율 6.4%에서의 값도 함께 적는다(순위 품질 비교용, 운영 규칙 아님).
출력: 산출물/최종보고서/oot_LAST/test_scores_oot.parquet, 표/OOT_LAST.csv, 표/OOT_곡선_LAST.csv"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, json, types, numpy as np, pandas as pd, polars as pl, xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier, Pool
A = importlib.import_module("98_apply_test"); K = A.K; C = A.C; log = K.log; G = importlib.import_module("98_gate"); R = importlib.import_module("121_rulers")
LAST = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
OUT = C.ROOT / "산출물" / "최종보고서"; OD = OUT / f"oot_{LAST}"; MD = OD / "model"; MD.mkdir(parents=True, exist_ok=True); TB = OUT / "표"
TRAIN, TEST = C.ROOT / "lending_club_2020_train.csv", C.ROOT / "lending_club_2020_test.csv"; SEED = K.SEED
CATC = ["k__" + c for c in A.NOMINAL] + ["k__zip3", "k__emp_title_norm"]
def cat_frame(X, D, rc):                                                    # 117 cat_frame 과 같은 변환
    cd = A.codes(D, rc["cmaps"]); Z = X.drop(columns=[c for c in X.columns if c.startswith(("hc__", "te__"))]).copy()
    Z["k__zip3"], Z["k__emp_title_norm"] = cd["zip3"], cd["emp_title_norm"]
    for c in CATC: Z[c] = pd.Series(np.nan_to_num(Z[c].to_numpy(np.float64), nan=-1)).astype(int).astype(str).to_numpy()
    return Z

# ---- 1) 2016년 이후 학습 id 제외 후 98_apply_test.fit
ids = pl.read_csv(TRAIN, columns=["id", "issue_d"], infer_schema_length=0).filter(pl.col("issue_d").is_not_null()).to_pandas()
ex = ids.id[ids.issue_d.str[-4:].astype(int) > LAST].astype(str); (OD / "exclude_ids.txt").write_text("\n".join(ex)); log(f"제외({LAST + 1}년 이후 발급) 학습 행", len(ex))
if not (MD / "recipe.json").exists(): A.fit(types.SimpleNamespace(train=str(TRAIN), exclude_ids=str(OD / "exclude_ids.txt"), model_dir=str(MD), seeds=3, rounds=0))
rc = json.load(open(MD / "recipe.json"))
D = A.raw_frame(TRAIN, rc["table_cols"]); D = D[~D.id.astype(str).isin(set(ex))]; L = D[D.default.notna()].reset_index(drop=True); y = L.default.to_numpy()
assert L.issue_d.str[-4:].astype(int).max() <= LAST and len(L) == rc["n_train"]; log("OOT 학습 확정 대출", len(L))
X = A.featurize(L, rc, train_y=y); va = np.random.RandomState(SEED).rand(len(X)) < 0.1
pl_ = dict(objective="binary", learning_rate=0.08, num_leaves=31, min_child_samples=100, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=10, seed=SEED, verbose=-1)
mL = lgb.train(pl_, lgb.Dataset(X[~va], y[~va]), 1500, valid_sets=[lgb.Dataset(X[va], y[va])], callbacks=[lgb.early_stopping(40, verbose=False)]); log("lgb iters", mL.best_iteration)
mC = CatBoostClassifier(iterations=800, learning_rate=0.15, depth=6, allow_writing_files=False, l2_leaf_reg=5, border_count=64, thread_count=10, random_seed=SEED, verbose=0, early_stopping_rounds=40)
mC.fit(Pool(cat_frame(X, L, rc)[~va], y[~va], cat_features=CATC), eval_set=Pool(cat_frame(X, L, rc)[va], y[va], cat_features=CATC), use_best_model=True); log("cat iters", mC.get_best_iteration())
f = R.frame(TRAIN).set_index("id").loc[L.id.astype(str)]; assert (f.dflt.to_numpy() == (y == 1)).all(); z = (f.r1 - f.rf).to_numpy()
P = dict(rc["params"], objective="reg:squarederror", eval_metric="rmse", seed=0)
def reg(m):
    v = np.random.RandomState(0).rand(int(m.sum())) < 0.10; Xm, zm = X[m], z[m]
    return xgb.train(P, xgb.DMatrix(Xm[~v], zm[~v]), 600, evals=[(xgb.DMatrix(Xm[v], zm[v]), "va")], early_stopping_rounds=30, verbose_eval=False)
ba, bd = reg(y == 0), reg(y == 1); sa, sd = float(np.std(z[y == 0])), float(np.std(z[y == 1])); log("gain rounds", ba.best_iteration + 1, bd.best_iteration + 1)
del D, X

# ---- 2) 시험 파일 적용(입력 열만)
T = A.raw_frame(TEST, rc["table_cols"]); G.enforce(G.gate_table(T, rc["columns"], labeled=False) + G.gate_source(T.id, TEST), "데이터 관문(OOT 시험 적용)")
Xt = A.featurize(T, rc); dm = xgb.DMatrix(Xt)
p_x = np.mean([xgb.Booster(model_file=str(MD / f"xgb_s{s}.json")).predict(dm) for s in range(rc["seeds"])], 0)
p = (p_x + mL.predict(Xt, num_iteration=mL.best_iteration) + mC.predict_proba(Pool(cat_frame(Xt, T, rc), cat_features=CATC))[:, 1]) / 3
ga, gd = (b.predict(dm, iteration_range=(0, b.best_iteration + 1)) for b in (ba, bd))
ev = (1 - p) * ga + p * gd; S = pd.DataFrame(dict(id=T.id.astype(str), pd_xgb_oot=p_x, pd_blend_oot=p, score_V3_oot=-ev / np.sqrt(p * (1 - p) * (ga - gd) ** 2 + (1 - p) * sa ** 2 + p * sd ** 2)))
S.to_parquet(OD / "test_scores_oot.parquet", index=False); log("saved OOT scores", S.shape)

# ---- 3) 채점(결과 확정 시험 대출, 평가 구간 풀 안에서 상위 q)
d = R.frame(TEST).merge(S, on="id", validate="1:1"); Q = json.load(open(OUT / "결과_T1_T2.json"))["QSTAR"]; V3 = pd.read_csv(TB / "v3_판정.csv").set_index("model")
QQ = {"V3": {"T1": float(V3.loc["V3", "q_star"]), "T2": float(V3.loc["V3", "q_T2"])}, "B0": Q["B0"], "GRADE": Q["GRADE"], "ALL": {"T1": 1.0, "T2": 1.0}}
rows, cur = [], []
SLICES = [s for s in (("2017–2020 발급(T1)", (2017, 2020), "T1"), ("2016–2017 발급(T2)", (2016, 2017), "T2")) if s[1][0] > LAST]
for nm, (a_, b_), ruler in SLICES:
    s = d[(d.yr >= a_) & (d.yr <= b_)].reset_index(drop=True); sc = {"V3": s.score_V3_oot.to_numpy(), "B0": s.pd_xgb_oot.to_numpy(), "GRADE": R.grade_score(s), "ALL": np.zeros(len(s))}
    allm = np.ones(len(s), bool)
    for k, v in sc.items():
        o, rk = R.order_rank(v); q = QQ[k][ruler]; m = R.sel(rk, q)
        val = R.t1_curve(s, o, qs=np.array([q]))[0][0] if ruler == "T1" else R.t2_curve(s, o, allm, qs=np.array([q]))[0]
        q64 = np.array([0.064]); v64 = R.t1_curve(s, o, qs=q64)[0][0] if ruler == "T1" else R.t2_curve(s, o, allm, qs=q64)[0]
        rows.append(dict(구간=nm, ruler=ruler, strategy=k, q=q, n=len(s), SR=val, SR_승인6_4=v64, 승인부도율=s.dflt[m].mean() if m.any() else np.nan, 구간부도율=s.dflt.mean(), 평균rf=s.rf.mean()))
        c = R.t1_curve(s, o)[0] if ruler == "T1" else R.t2_curve(s, o, allm); cur += [dict(구간=nm, strategy=k, q=qq, SR=x) for qq, x in zip(R.QS, c)]
OOT = pd.DataFrame(rows); OOT.to_csv(TB / f"OOT_{LAST}.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(cur).to_csv(TB / f"OOT_곡선_{LAST}.csv", index=False); print(OOT.round(4).to_string())
from sklearn.metrics import roc_auc_score
for nm, m in (("2016+", d.yr >= 2016), ("2017+", d.yr >= 2017)): log("OOT AUC", nm, "blend %.4f xgb %.4f" % (roc_auc_score(d.dflt[m], d.pd_blend_oot[m]), roc_auc_score(d.dflt[m], d.pd_xgb_oot[m])), "n", int(m.sum()))
