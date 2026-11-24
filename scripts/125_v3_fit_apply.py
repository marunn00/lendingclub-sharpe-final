"""125 최종 모델 v3 학습·시험 적용(124 판정으로 채택). 학습 파일만으로 적합하고, 시험 파일의 결과 열은 쓰지 않는다.
- 부도확률: v2와 같은 3모형 평균(117 산출 test_scores_v2.pd_blend).
- 이득 회귀: 과제 잣대(T1)의 대출 초과수익 z = (total_pymnt/funded_amnt)^(12/T) − 1 − rf(발급월 만기 맞춤 국채), 대출 동일 가중, 비부도·부도 따로. XGBoost 설정·10% 조기종료는 110과 같음.
- 점수(작을수록 먼저 승인) = −기대이득 ÷ 위험(110·117과 같은 식).
출력: 산출물/최종보고서/test_scores_v3.parquet, model/model_gain3_*.json"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, json, numpy as np, pandas as pd, xgboost as xgb
A = importlib.import_module("98_apply_test"); K = A.K; C = A.C; log = K.log; G = importlib.import_module("98_gate"); R = importlib.import_module("121_rulers")
OUT = C.ROOT / "산출물" / "최종보고서"; MO = OUT / "model"; MO.mkdir(parents=True, exist_ok=True); TRAIN, TEST = C.ROOT / "lending_club_2020_train.csv", C.ROOT / "lending_club_2020_test.csv"
rc = json.load(open(A.MD / "recipe.json"))

D = A.raw_frame(TRAIN, rc["table_cols"])
G.enforce(G.gate_table(D, rc["columns"]) + G.gate_te(A.te_oof) + G.gate_source(D.id, TRAIN), "데이터 관문(학습, v3)")
L = D[D.default.notna()].reset_index(drop=True); y = L.default.to_numpy(); assert len(L) == rc["n_train"], (len(L), rc["n_train"])
f = R.frame(TRAIN).set_index("id").loc[L.id.astype(str)]; assert (f.dflt.to_numpy() == (y == 1)).all()
z = (f.r1 - f.rf).to_numpy(); X = A.featurize(L, rc, train_y=y); log("train X", X.shape)
P = dict(rc["params"], objective="reg:squarederror", eval_metric="rmse", seed=0)
def reg(m, name):
    va = np.random.RandomState(0).rand(int(m.sum())) < 0.10; Xm, zm = X[m], z[m]
    b = xgb.train(P, xgb.DMatrix(Xm[~va], zm[~va]), 600, evals=[(xgb.DMatrix(Xm[va], zm[va]), "va")], early_stopping_rounds=30, verbose_eval=False)
    b.save_model(str(MO / f"model_gain3_{name}.json")); log(name, "rounds", b.best_iteration + 1); return b
ba, bd = reg(y == 0, "repaid"), reg(y == 1, "default")
sa, sd = float(np.std(z[y == 0])), float(np.std(z[y == 1])); json.dump(dict(sa=sa, sd=sd, n_train=len(L)), open(MO / "model_gain3_const.json", "w"))
del D, L, X

T = A.raw_frame(TEST, rc["table_cols"]); G.enforce(G.gate_table(T, rc["columns"], labeled=False) + G.gate_source(T.id, TEST), "데이터 관문(시험 적용, v3)")
d = xgb.DMatrix(A.featurize(T, rc)); ga, gd = (b.predict(d, iteration_range=(0, b.best_iteration + 1)) for b in (ba, bd))
V2 = pd.read_parquet(C.ROOT / "산출물/테스트평가/test_scores_v2.parquet"); assert (V2.id.to_numpy() == T.id.astype(str).to_numpy()).all(); p = V2.pd_blend.to_numpy()
ev = (1 - p) * ga + p * gd; risk = np.sqrt(p * (1 - p) * (ga - gd) ** 2 + (1 - p) * sa ** 2 + p * sd ** 2)
out = pd.DataFrame(dict(id=T.id.astype(str), pd_blend=p, gain_repaid=ga, gain_default=gd, exp_gain_v3=ev, risk_v3=risk, score_V3=-ev / risk))
out.to_parquet(OUT / "test_scores_v3.parquet", index=False); log("saved v3", out.shape, "corr(score_V3, score_M6v2) %.3f" % np.corrcoef(out.score_V3, V2.score_M6v2)[0, 1])
