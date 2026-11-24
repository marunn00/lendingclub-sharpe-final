"""98_c: 채택 기법 결합(T8_kept) → Optuna 튜닝 → 시드 배깅 · XGB+LGBM+CatBoost 앙상블(평균/순위평균/스태킹) · 수익회귀 혼합 → SHAP 가지치기·겹 간 안정성.
사용: python3 scripts/98_c_ens.py --stage kept|tune|fit|ens|prune --kept T3_hc_both,T4_credit,... [--trials 30]
출력: _work/results_ens.csv(98_b와 같은 열), best_params.json, optuna_trials.csv, prune_stability_<k>.json, oof_*.npz"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; B = importlib.import_module("98_b_sweep"); K = B.K; log = K.log
import argparse, json, time, numpy as np, pandas as pd, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
W = K.WORK; RES = W / "results_ens.csv"; n = B.n; FOLD = B.folds_for(5); IX = np.where(FOLD >= 0)[0]; RATE = B.X.int_rate.to_numpy(np.float32) / 100
def merged(ids, sid, name, **extra):
    sp = dict(id=sid, grp="8 결합·튜닝·앙상블", name=name, cols=list(B.BASE))
    for i in ids:
        s = B.S[i]; sp["cols"] = list(dict.fromkeys(sp["cols"] + s["cols"]))
        for k in ("te", "onehot", "native"): sp[k] = list(dict.fromkeys(sp.get(k, []) + s.get(k, [])))
        for k in ("impute", "flags", "mono", "weight", "target", "dropsparse"):
            if s.get(k): sp[k] = s[k]
    if "T3_te_nominal" in ids: sp["cols"] = [c for c in sp["cols"] if c not in B.NOMINAL]
    if "T4_swap_rate" in ids: sp["cols"] = [c for c in sp["cols"] if c != "int_rate"]
    sp.update(extra); return sp
def rec_all(sp, fold, derived=()):
    t0 = time.time(); base = dict(id=sp["id"], grp=sp["grp"], name=sp["name"], quick=0, folds=int(fold.max()) + 1)
    pdv, pr, its = B.run_cv(sp, fold, 600, IX); np.savez_compressed(W / f"oof_{sp['id']}.npz", pd=pdv, pr=pr)
    B.record(base, pdv, pr, IX, True, fold[IX], its, t0, RES)
    for dk in derived:
        d_pd, d_pr = B.derive(dk, pdv[IX], fold[IX], B.y_all[IX], B.v_all[IX], RATE[IX]); pp, rr = np.full(n, np.nan), np.full(n, np.nan); pp[IX] = d_pd; rr[IX] = d_pr
        B.record(dict(base, id=sp["id"] + "+" + dk), pp, rr, IX, not dk.startswith("ev"), fold[IX], its, t0, RES)
def stage_tune(kept, trials):
    """튜닝은 보고 폴드와 분리된 보류 10%(FOLD==-1)에서만: 그 안에서 85/15 학습/검증. 보고 폴드(0–4)는 튜닝에 쓰이지 않음."""
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
    sp = merged(kept, "T8_tune", "tune"); s = np.where(FOLD == -1)[0]; np.random.RandomState(K.SEED).shuffle(s); cut = int(0.85 * len(s)); tr, va = s[:cut], s[cut:]
    Xtr, Xva = B.build(sp, tr, va, B.y_all[tr]); w = B.weights(sp, tr, B.y_all[tr], B.v_all[tr], Xtr)
    dtr, dva = xgb.DMatrix(Xtr, B.y_all[tr], weight=w), xgb.DMatrix(Xva, B.y_all[va])
    def obj(t):
        p = dict(B.P0, objective="binary:logistic", eval_metric="logloss", max_depth=t.suggest_int("max_depth", 3, 8), learning_rate=t.suggest_float("learning_rate", 0.03, 0.15, log=True),
                 min_child_weight=t.suggest_float("min_child_weight", 5, 300, log=True), subsample=t.suggest_float("subsample", 0.5, 1.0), colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                 reg_lambda=t.suggest_float("reg_lambda", 1, 100, log=True), reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True), gamma=t.suggest_float("gamma", 0, 5))
        b = xgb.train(p, dtr, 1500, evals=[(dva, "va")], early_stopping_rounds=40, verbose_eval=False); t.set_user_attr("iters", b.best_iteration + 1); return b.best_score
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=K.SEED)); st.optimize(obj, n_trials=trials, timeout=1500)
    st.trials_dataframe().to_csv(W / "optuna_trials.csv", index=False, encoding="utf-8-sig")
    json.dump(dict(best=st.best_params, logloss=st.best_value, n_trials=len(st.trials)), open(W / "best_params.json", "w"), indent=1); log("optuna best", st.best_value, st.best_params)
def blends(bags, extra_rank=None):
    """bags {이름: [OOF id...]}: 시드 배깅 평균 → 모형 평균 · 순위평균 · 스태킹(교차적합 로지스틱, logit 입력)."""
    fold = FOLD; ld = lambda i: np.load(W / f"oof_{i}.npz")["pd"]; P = {k: np.mean([ld(i) for i in v], 0) for k, v in bags.items()}; cmb = {}
    for k, p in P.items(): cmb[f"T8_bag_{k}"] = p
    if len(P) > 1:
        cmb["T8_avg"] = np.mean(list(P.values()), 0); cmb["T8_rankavg"] = np.mean([np.where(fold >= 0, rankdata(np.nan_to_num(p)) / n, np.nan) for p in P.values()], 0)
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); Z = np.c_[[lg(p) for p in P.values()]].T; st = np.full(n, np.nan)
        for k in range(5): o, t = (fold != k) & (fold >= 0), fold == k; st[t] = LogisticRegression(C=1.0).fit(Z[o], B.y_all[o]).predict_proba(Z[t])[:, 1]
        cmb["T8_stack"] = st
    for nm, p in cmb.items():
        B.record(dict(id=nm, grp="8 결합·튜닝·앙상블", name=nm + " (" + ",".join(P) + ")", quick=0, folds=5), np.where(nm == "T8_rankavg", np.nan, p), p, IX, nm != "T8_rankavg", fold[IX], [0], time.time(), RES)
    return P
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--kept", default=""); ap.add_argument("--trials", type=int, default=30)
    a = ap.parse_args(); G = importlib.import_module("98_gate"); G.enforce(G.gate_table(B.D_ALL, list(B.X.columns)) + G.gate_te(lambda ci, y: B.te_cols(ci.astype(np.float64), y, ci[:1].astype(np.float64))[0]) + G.gate_folds(FOLD) + G.gate_source(B.D_ALL.id, G.TRAIN_CSV, resolved_n=len(B.D), D=B.D_ALL), "데이터 관문(98_c)"); kept = [k for k in a.kept.split(",") if k]; fold = FOLD; bp = json.load(open(W / "best_params.json"))["best"] if (W / "best_params.json").exists() else None
    if a.stage == "kept": rec_all(merged(kept, "T8_kept", "채택 기법 결합(기본 모수)"), fold, derived=["iso", "ev", "ev_iso"])
    if a.stage == "tune": stage_tune(kept, a.trials)
    if a.stage == "fit": rec_all(merged(kept, "T8_tuned", "채택 기법 + Optuna 튜닝", params=bp), fold, derived=["iso", "ev", "ev_iso"])
    if a.stage == "shock":                                                    # 충격변수 토글: 튜닝 모형(T8_tuned)에 8열 추가, 같은 겹·모수·시드
        sp = merged(kept, "T10_shock", "T8_tuned + 충격변수 8열", params=bp); sp["cols"] = list(dict.fromkeys(sp["cols"] + B.SHOCK)); rec_all(sp, fold)
    if a.stage == "ens":
        ids = []
        for s in range(3): sp = merged(kept, f"T8_xgb_s{s}", f"XGB 튜닝 시드{s}", params=bp, seed=s); rec_all(sp, fold); ids.append(sp["id"])
        rec_all(merged(kept, "T8_lgb", "LightGBM", engine="lgb", params={}), fold)
        hc = [c for c in B.HIGHCARD]; ck = [k for k in kept if not k.startswith("T3_hc") and k not in ("T3_te_nominal", "T3_onehot", "T3_native_cat")]
        sc = merged(ck, "T8_cat", "CatBoost(순서형 TE 범주: 명목형+zip3·직함)", engine="cat", cat_cols=B.NOMINAL + hc, te=[], onehot=[], native=[]); sc["cols"] = list(dict.fromkeys(sc["cols"] + B.NOMINAL + hc))
        rec_all(sc, fold); P = blends({"xgb": ids, "lgb": ["T8_lgb"], "cat": ["T8_cat"]})
        if (W / "oof_T6_reg.npz").exists():                                      # 수익회귀(기본 모수) + 분류 앙상블 순위 혼합
            reg = np.load(W / "oof_T6_reg.npz")["pr"]; pa = np.mean(list(P.values()), 0); mix = np.where(FOLD >= 0, (rankdata(np.nan_to_num(pa)) + rankdata(np.nan_to_num(reg))) / (2 * n), np.nan)
            B.record(dict(id="T8_mix_clf_reg", grp="6 목표", name="앙상블 PD 순위 + 수익회귀 순위 평균", quick=0, folds=5), np.full(n, np.nan), mix, IX, False, fold[IX], [0], time.time(), RES)
    if a.stage == "prune":
        for k in (20, 30, 40):
            B.PRUNE_LOG.clear(); rec_all(merged(kept, f"T9_prune{k}", f"SHAP 상위 {k}개 가지치기(겹 학습셋 안에서 선택)", params=bp, prune_k=k), fold)
            keeps = [set(p["keep"]) for p in B.PRUNE_LOG]; jac = [len(x & y) / len(x | y) for i, x in enumerate(keeps) for y in keeps[i + 1:]]
            json.dump(dict(k=k, mean_jaccard=float(np.mean(jac)), min_jaccard=float(np.min(jac)), always=sorted(set.intersection(*keeps)), imp=B.PRUNE_LOG[0]["imp"], keep_by_fold=[sorted(x) for x in keeps]), open(W / f"prune_stability_{k}.json", "w"), ensure_ascii=False, indent=1)
