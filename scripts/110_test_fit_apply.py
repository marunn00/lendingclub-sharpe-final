"""110 시험 적용: 학습 파일만으로 모형을 만들고 시험 파일에 점수를 낸다. 시험 파일의 결과 열(loan_status·상환액)은 쓰지 않는다.
- PD: 산출물/앙상블_v1/model 의 XGBoost 3시드(학습 확정 대출 1,116,156건으로 적합, recipe.json).
- 이득 회귀: 102_models.m_xgb_pd_twostage 와 같은 정의. 초과 연환산 수익 z = ann/A − tb([−3, 0.5] 절단), 대출액 가중, 비부도·부도 대출 따로, 10% 조기종료.
- 점수(작을수록 먼저 승인): B0 = PD, M5 = −기대이득, M6 = −기대이득/위험(사전계획.md 주 전략).
출력: 산출물/테스트평가/test_scores.parquet (시험 전 행), model_gain_*.json"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, json, numpy as np, pandas as pd, xgboost as xgb
A = importlib.import_module("98_apply_test"); K = A.K; C = A.C; log = K.log; G = importlib.import_module("98_gate")
OUT = C.ROOT / "산출물" / "테스트평가"; OUT.mkdir(exist_ok=True); TRAIN, TEST = C.ROOT / "lending_club_2020_train.csv", C.ROOT / "lending_club_2020_test.csv"
rc = json.load(open(A.MD / "recipe.json"))

# ---- 학습(학습 파일만)
D = A.raw_frame(TRAIN, rc["table_cols"])
G.enforce(G.gate_table(D, rc["columns"]) + G.gate_te(A.te_oof) + G.gate_source(D.id, TRAIN), "데이터 관문(학습)")
L = D[D.default.notna()].reset_index(drop=True); y = L.default.to_numpy(); assert len(L) == rc["n_train"], (len(L), rc["n_train"])
R = K.load_loans(); R.index = R.id.astype(str); r = R.loc[L.id.astype(str)]
z = np.clip(np.nan_to_num(np.where(r.roe_ok, r.ann / r.A - r.tb, np.nan)), -3.0, 0.5); w = (r.A / r.A.mean()).to_numpy()
X = A.featurize(L, rc, train_y=y); log("train X", X.shape)
P = dict(rc["params"], objective="reg:squarederror", eval_metric="rmse", seed=0)
def reg(m, name):
    va = np.random.RandomState(0).rand(int(m.sum())) < 0.10; Xm, zm, wm = X[m], z[m], w[m]
    b = xgb.train(P, xgb.DMatrix(Xm[~va], zm[~va], weight=wm[~va]), 600, evals=[(xgb.DMatrix(Xm[va], zm[va], weight=wm[va]), "va")], early_stopping_rounds=30, verbose_eval=False)
    b.save_model(str(OUT / f"model_gain_{name}.json")); log(name, "rounds", b.best_iteration + 1); return b
ba, bd = reg(y == 0, "repaid"), reg(y == 1, "default")
sa, sd = (float(np.sqrt(np.average((z[m] - z[m].mean()) ** 2))) for m in (y == 0, y == 1))
json.dump(dict(sa=sa, sd=sd, n_train=len(L)), open(OUT / "model_gain_const.json", "w"))
del D, L, X

# ---- 시험 적용(입력 열만)
T = A.raw_frame(TEST, rc["table_cols"])
G.enforce(G.gate_table(T, rc["columns"], labeled=False) + G.gate_source(T.id, TEST), "데이터 관문(시험 적용)")
d = xgb.DMatrix(A.featurize(T, rc))
pdv = np.mean([xgb.Booster(model_file=str(A.MD / f"xgb_s{s}.json")).predict(d) for s in range(rc["seeds"])], 0)
ga, gd = (b.predict(d, iteration_range=(0, b.best_iteration + 1)) for b in (ba, bd))
ev = (1 - pdv) * ga + pdv * gd; risk = np.sqrt(pdv * (1 - pdv) * (ga - gd) ** 2 + (1 - pdv) * sa ** 2 + pdv * sd ** 2)
S = pd.DataFrame(dict(id=T.id.astype(str), issue_d=T.issue_d, term_m=T.term_m, sub_grade=T.sub_grade, int_rate=T.int_rate, loan_amnt=T.loan_amnt,
                      pd=pdv, exp_gain=ev, risk=risk, score_B0=pdv, score_M5=-ev, score_M6=-ev / risk))
mc = T.issue_month_code.to_numpy(); o = np.lexsort((S.score_M6.to_numpy(), S.term_m.to_numpy(), mc)); S["approve_25"] = 0
cum = S.iloc[o].groupby([mc[o], S.term_m.to_numpy()[o]]).loan_amnt.transform(lambda a: a.cumsum() / a.sum())
S.loc[cum.index, "approve_25"] = (cum <= 0.25).astype(int)            # 발행월·만기 안에서 M6 점수 순 금액 누적 25% (사전 운영 지점)
S.to_parquet(OUT / "test_scores.parquet", index=False); log("saved", S.shape, "approve_25 share", round(S.approve_25.mean(), 3))
