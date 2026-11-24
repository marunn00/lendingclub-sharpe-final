"""98_b: 간결 테이블(train_간결.parquet) 위에서 기법별 토글 스윕 — XGBoost 기준선(T0) 대비.
프로토콜: 확정 대출(36m 2012–16, 60m 2012–14; 618,750행) 층화 5-겹 CV(부도×만기 층화, 시드 0) — 시험 파일이 같은 모집단의 무작위 ~40% 표본이므로 무작위 분할이 대응 프로토콜.
평가: AUC(겹별 + 전체 OOF), logloss, 36m 대출의 발행월 Sharpe·초과수익(96_c month_aggs/series 그대로; 무위험 = 만기일치 국채 M−1 월평균).
사용: python3 scripts/98_b_sweep.py --list | --only T0,T2_impute [--folds 5] [--quick 60000] [--force] [--out results_raw.csv]
출력: _work/results_raw.csv, _work/oof_<id>.npz(pd, pr), _work/e_<id>.npy(월×승인율 초과수익 행렬)"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; K = importlib.import_module("98_common"); C = K.C; log = K.log
import argparse, json, time, numpy as np, pandas as pd, polars as pl, xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
W = K.WORK; TABLE = C.ROOT / "산출물" / "간결테이블" / "train_간결.parquet"
# ---- 데이터 ----------------------------------------------------------------------------------------------------------
D_ALL = pd.read_parquet(TABLE)
D = D_ALL[D_ALL.default.notna()].reset_index(drop=True); L = K.load_loans()                       # 확정 대출 전체(모든 발행연도·두 만기)
assert len(D) == len(L) and (D.id.to_numpy() == L.id.to_numpy()).all() and (D.default.to_numpy() == L.default.to_numpy()).all()
_yr = D.issue_d.str[4:8].astype(int)
CATS = ["sub_grade", "home_ownership", "verification_status", "purpose", "addr_state", "initial_list_status", "application_type", "zip3", "emp_title_norm"]
IDC = {"id", "issue_d", "issue_month_code", "loan_status", "default"} | set(CATS)
NUMERIC = [c for c in D.columns if c not in IDC]
X = pd.DataFrame({c: D[c].to_numpy(np.float32) for c in NUMERIC})
for c in CATS: X["k__" + c] = D[c].astype("category").cat.codes.to_numpy().astype(np.float32); X.loc[X["k__" + c] < 0, "k__" + c] = np.nan
X["issue_year"] = D.issue_d.str[4:8].astype(int).to_numpy().astype(np.float32)
for c in ("zip3", "emp_title_norm"):                                      # 비지도 빈도(전체 파일 점유율, 로그) — 시험 파일에는 저장된 표를 적용
    sh = D_ALL[c].value_counts(normalize=True); X["hc__" + c + "_logfreq"] = np.log10(D[c].map(sh).astype(float).fillna(1e-7).to_numpy()).astype(np.float32)
SHOCK = ["st_hpi_yoy", "st_beta_hpi", "shortemp_x_st_ur_chg6", "d30_vol3", "st_hpi_dd36", "st_beta_ur", "hpi_nat_dd36", "mortacc_x_hpi_dd"]   # 충격변수 8열(토글 T10 에서만 사용)
_sh = pd.read_parquet(C.ROOT / "산출물" / "충격변수" / "충격변수.parquet").set_index("id").reindex(D.id.to_numpy())
for c in SHOCK: X[c] = _sh[c].to_numpy(np.float32)
y_all = D.default.to_numpy().astype(np.float32); v_all = (L.NPV_short / L.A).to_numpy(np.float32); n = len(X)
DERIVED = ["lti", "spread_10y", "spread_30y", "spread_matched", "spread_fedfunds", "rate_premium_subgrade", "tsy10_at_issue", "tsy_matched_at_issue", "pti", "credit_hist_years", "open_acc_ratio", "bal_per_acc", "new_acc_share24"]
SPREADS = ["spread_10y", "spread_30y", "spread_matched", "spread_fedfunds"]; TSY = ["tsy10_at_issue", "tsy_matched_at_issue"]
BURDEN = ["pti", "bal_per_acc"]; CREDIT = ["credit_hist_years", "open_acc_ratio", "new_acc_share24"]
ORIG = [c for c in NUMERIC if c not in DERIVED]                                  # 원본 상장시점 수치(+fico_mid, term_m)
ORD = ["k__sub_grade"]; NOMINAL = ["k__home_ownership", "k__verification_status", "k__purpose", "k__addr_state", "k__initial_list_status", "k__application_type"]
BASE = ORIG + ORD + NOMINAL; HIGHCARD = ["k__zip3", "k__emp_title_norm"]
HCF = ["hc__zip3_logfreq", "hc__emp_title_norm_logfreq"]
MONO = {"fico_mid": -1, "dti": 1, "int_rate": 1}
S = {}
def add(i, grp, name, **kw): S[i] = dict(id=i, grp=grp, name=name, cols=kw.pop("cols", BASE), **kw)
add("T0", "기준선", "XGB 기준선(원본 상장시점 열, NaN 기본, 순서형 범주코드)")
add("T1_leak", "1 누수점검", "누수(사후 결과) 열 포함 — 사용 불가 대조군", cols=BASE, leak=True)
add("T2_impute", "2 결측", "중앙값 대치(학습 겹 중앙값)", impute=True)
add("T2_impute_flag", "2 결측", "중앙값 대치 + 결측 표지(결측률>1% 열)", impute=True, flags=True)
add("T2_native_flag", "2 결측", "NaN 기본 + 결측 표지·결측 개수", flags=True)
add("T2_drop_sparse", "2 결측", "결측률>50% 열 제거(시대 열)", dropsparse=True)
add("T3_onehot", "3 범주", "원-핫(명목형 6개)", onehot=NOMINAL)
add("T3_ordinal_only", "3 범주", "명목형 범주 제거(sub_grade만 유지)", cols=ORIG + ORD)
add("T3_te_nominal", "3 범주", "명목형 → OOF 타깃인코딩", cols=ORIG + ORD, te=NOMINAL)
add("T3_native_cat", "3 범주", "XGB 네이티브 범주형(명목형 6개)", native=NOMINAL)
add("T3_hc_freq", "3 범주", "고유값 빈도(zip3·emp_title)", cols=BASE + HCF)
add("T3_hc_te", "3 범주", "zip3·emp_title OOF TE", te=HIGHCARD)
add("T3_hc_both", "3 범주", "빈도 + OOF TE (zip3·emp_title)", cols=BASE + HCF, te=HIGHCARD)
add("T3_state_te", "3 범주", "addr_state·purpose OOF TE 추가", te=["k__addr_state", "k__purpose"])
add("T4_spread_matched", "4 파생·스프레드", "+ spread_matched", cols=BASE + ["spread_matched"])
add("T4_spread_10y", "4 파생·스프레드", "+ spread_10y", cols=BASE + ["spread_10y"])
add("T4_spread_30y", "4 파생·스프레드", "+ spread_30y", cols=BASE + ["spread_30y"])
add("T4_spread_ff", "4 파생·스프레드", "+ spread_fedfunds", cols=BASE + ["spread_fedfunds"])
add("T4_spreads_all", "4 파생·스프레드", "+ 스프레드 4개", cols=BASE + SPREADS)
add("T4_tsy_ref", "4 파생·스프레드", "+ 국채금리 2개(tsy10, 만기일치)", cols=BASE + TSY)
add("T4_swap_rate", "4 파생·스프레드", "int_rate → spread_matched 교체", cols=[c for c in BASE if c != "int_rate"] + ["spread_matched"])
add("T4_prem", "4 파생·스프레드", "+ rate_premium_subgrade", cols=BASE + ["rate_premium_subgrade"])
add("T4_burden", "4 파생·부담", "+ pti, bal_per_acc", cols=BASE + BURDEN)
add("T4_credit", "4 파생·신용", "+ 신용이력·계좌비율·신규비중", cols=BASE + CREDIT)
add("T4_lti", "4 파생·부담", "+ lti", cols=BASE + ["lti"])
add("T4_all_derived", "4 파생·전체", "+ 파생 12개 전체", cols=BASE + DERIVED)
add("T5_spw", "5 불균형", "scale_pos_weight=neg/pos", weight="spw", derived=["iso", "platt"])
add("T5_spw_sqrt", "5 불균형", "scale_pos_weight=√(neg/pos)", weight="spw_sqrt")
add("T5_focal", "5 불균형", "focal 유사 가중 |y−p|(1단계 얕은 모형)", weight="focal")
add("T6_reg", "6 목표", "수익률(NPV/대출액) 회귀", target="reg")
add("T6_profit_w", "6 목표", "분류 + 손실크기 가중(수익 인지)", weight="profit")
add("T7_recency", "7 시간", "최근성 가중 exp(−Δ월/24)", weight="recency")
add("T7_year", "7 시간", "발행연도 열 추가", cols=BASE + ["issue_year"])
add("T7_mono", "7 시간", "단조 제약(FICO−, dti+, int_rate+)", mono=True)
add("T0_cal", "5 보정·6 기대수익", "기준선 PD 보정(교차적합 isotonic/Platt)·EV=f(PD, 금리)", reuse="T0", derived=["iso", "platt", "ev", "ev_iso"])
# ---- 누수 열(대조군) -----------------------------------------------------------------------------------------------
def leak_frame():
    led = pd.read_csv(C.ROOT / "산출물" / "전처리_v5" / "열_원장.csv"); post = [c for c in led[led.A_내생보존_reason.astype(str).str.contains("사후")].column if not c.startswith("hardship")]
    raw = pl.read_csv(C.ROOT / "lending_club_2020_train.csv", columns=["id"] + post, infer_schema_length=0).to_pandas().set_index("id").loc[D.id]
    out = {}
    for c in post:
        v = pd.to_numeric(raw[c].astype(str).str.replace("%", "", regex=False), errors="coerce")
        out["lk__" + c] = v.to_numpy(np.float32) if v.notna().sum() > 0.5 * raw[c].notna().sum() else (C.mcode_str(raw[c]) - D.issue_month_code.to_numpy()).astype(np.float32)
    return pd.DataFrame(out)
# ---- 모형 · 인코딩 -----------------------------------------------------------------------------------------------------
P0 = dict(tree_method="hist", max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.6, min_child_weight=50, reg_lambda=5.0, max_bin=128, nthread=10, seed=K.SEED)
def te_cols(ci, ytr, cte_i, inner=5, m=30.0, seed=0):
    f = lambda c: np.where(np.isnan(c), -1, c).astype(np.int64) + 1
    a, b = f(ci), f(cte_i); nc = int(max(a.max(), b.max())) + 1; pri = ytr.mean()
    enc = lambda ca, ya, cb: ((np.bincount(ca, ya, nc) + m * pri) / (np.bincount(ca, minlength=nc) + m))[cb]
    tr = np.zeros(len(a), np.float32)
    for i, j in StratifiedKFold(inner, shuffle=True, random_state=seed).split(a, ytr): tr[j] = enc(a[i], ytr[i], a[j])
    return tr, enc(a, ytr, b).astype(np.float32)
LEAK = None; PRUNE_LOG = []
def build(sp, tr, te, ytr):
    global LEAK
    cols = list(sp["cols"])
    if sp.get("dropsparse"): cols = [c for c in cols if X[c].isna().mean() <= 0.5]
    Z = X
    if sp.get("leak"):
        if LEAK is None: LEAK = leak_frame()
        Z = pd.concat([X[cols], LEAK], axis=1); cols = cols + list(LEAK.columns)
    Xtr, Xte = Z.loc[tr, cols].reset_index(drop=True), Z.loc[te, cols].reset_index(drop=True)
    if sp.get("flags"):
        sparse = [c for c in cols if 0.01 < X[c].isna().mean()] if not sp.get("leak") else []
        for c in sparse: Xtr["na__" + c] = Xtr[c].isna().astype(np.float32); Xte["na__" + c] = Xte[c].isna().astype(np.float32)
        Xtr["na__count"] = Xtr[sparse].isna().sum(axis=1).astype(np.float32); Xte["na__count"] = Xte[sparse].isna().sum(axis=1).astype(np.float32)
    for c in sp.get("te", []):
        a, b = te_cols(X[c].to_numpy()[tr], ytr, X[c].to_numpy()[te]); Xtr["te__" + c[3:]] = a; Xte["te__" + c[3:]] = b
    for c in sp.get("onehot", []):
        lv = pd.Series(X[c].to_numpy()[tr]).value_counts(); lv = lv[lv >= 200].index
        for l in lv: Xtr[f"oh__{c[3:]}_{l:g}"] = (X[c].to_numpy()[tr] == l).astype(np.float32); Xte[f"oh__{c[3:]}_{l:g}"] = (X[c].to_numpy()[te] == l).astype(np.float32)
        Xtr, Xte = Xtr.drop(columns=c), Xte.drop(columns=c)
    if sp.get("impute"):
        med = Xtr.median(); Xtr, Xte = Xtr.fillna(med).fillna(0), Xte.fillna(med).fillna(0)
    if sp.get("native"):
        for c in sp["native"]: Xtr[c] = pd.Categorical(Xtr[c].fillna(-1).astype(int)); Xte[c] = Xte[c].fillna(-1).astype(int).astype(pd.CategoricalDtype(Xtr[c].cat.categories))
    if sp.get("prune_k"):                                              # SHAP 가지치기(학습 겹 안에서만 중요도 계산)
        rs = np.random.RandomState(K.SEED); s1 = rs.choice(len(Xtr), min(150000, len(Xtr)), replace=False); s2 = s1[:20000]
        bst = xgb.train(dict(P0, objective="binary:logistic"), xgb.DMatrix(Xtr.iloc[s1], ytr[s1]), 120)
        sh = np.abs(bst.predict(xgb.DMatrix(Xtr.iloc[s2]), pred_contribs=True)[:, :-1]).mean(0)
        keep = list(Xtr.columns[np.argsort(-sh)[:sp["prune_k"]]]); PRUNE_LOG.append(dict(id=sp["id"], imp=dict(zip(Xtr.columns, sh.tolist())), keep=keep)); Xtr, Xte = Xtr[keep], Xte[keep]
    return Xtr, Xte
def weights(sp, tr, ytr, vtr, Xtr):
    w = sp.get("weight")
    if w is None: return None
    pos = ytr.mean()
    if w == "spw": ww = np.where(ytr == 1, (1 - pos) / pos, 1.0)
    elif w == "spw_sqrt": ww = np.where(ytr == 1, np.sqrt((1 - pos) / pos), 1.0)
    elif w == "profit": loss = np.clip(-vtr, 0.05, None); ww = np.where(ytr == 1, loss / loss[ytr == 1].mean(), 1.0)
    elif w == "recency": mi = D.issue_month_code.to_numpy()[tr]; ww = np.exp(-(mi.max() - mi) / 24.0)
    elif w == "focal":
        b = xgb.train(dict(P0, max_depth=3, objective="binary:logistic"), xgb.DMatrix(Xtr, ytr), 60); ww = np.abs(ytr - b.predict(xgb.DMatrix(Xtr)))
    return (ww / ww.mean()).astype(np.float32)
def fit_other(sp, Xtr, ytr, Xte, seed, va):
    if sp["engine"] == "lgb":
        import lightgbm as lgb
        p = dict(objective="binary", learning_rate=0.08, num_leaves=31, min_child_samples=100, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=10, seed=seed, verbose=-1, **(sp.get("params") or {}))
        m = lgb.train(p, lgb.Dataset(Xtr[~va], ytr[~va]), 1500, valid_sets=[lgb.Dataset(Xtr[va], ytr[va])], callbacks=[lgb.early_stopping(40, verbose=False)])
        return m.predict(Xte, num_iteration=m.best_iteration), m.best_iteration
    from catboost import CatBoostClassifier, Pool
    cc = sp.get("cat_cols", []); f = lambda Z: Z.assign(**{c: Z[c].fillna(-1).astype(int).astype(str) for c in cc})
    p = dict(iterations=800, learning_rate=0.15, depth=6, allow_writing_files=False, l2_leaf_reg=5, border_count=64, thread_count=10, random_seed=seed, verbose=0, early_stopping_rounds=40, **(sp.get("params") or {}))
    m = CatBoostClassifier(**p).fit(Pool(f(Xtr[~va]), ytr[~va], cat_features=cc), eval_set=Pool(f(Xtr[va]), ytr[va], cat_features=cc), use_best_model=True)
    return m.predict_proba(Pool(f(Xte), cat_features=cc))[:, 1], m.get_best_iteration()
def fit_pred(sp, Xtr, ytr, vtr, wtr, Xte, rounds, seed=K.SEED, params=None):
    seed = sp.get("seed", seed); params = params or sp.get("params")
    if sp.get("engine"): return fit_other(sp, Xtr, ytr, Xte, seed, np.random.RandomState(seed).rand(len(Xtr)) < 0.1)
    cat = bool(sp.get("native")); va = np.random.RandomState(seed).rand(len(Xtr)) < 0.1
    reg = sp.get("target") == "reg"; w_is = wtr is not None; p = dict(P0, **(params or {}), seed=seed)
    p.update(objective="reg:squarederror", eval_metric="rmse") if reg else p.update(objective="binary:logistic", eval_metric="auc" if w_is else "logloss")
    if sp.get("mono"): p["monotone_constraints"] = "(" + ",".join(str(MONO.get(c, 0)) for c in Xtr.columns) + ")"
    t = vtr if reg else ytr
    dtr = xgb.DMatrix(Xtr[~va], t[~va], weight=None if wtr is None else wtr[~va], enable_categorical=cat); dva = xgb.DMatrix(Xtr[va], t[va], enable_categorical=cat)   # 검증은 무가중
    b = xgb.train(p, dtr, rounds, evals=[(dva, "va")], early_stopping_rounds=30, verbose_eval=False)
    return b.predict(xgb.DMatrix(Xte, enable_categorical=cat), iteration_range=(0, b.best_iteration + 1)), b.best_iteration + 1
def derive(kind, pdv, fold, y, v, rate):
    """교차적합 보정/EV. 겹 k 의 변환은 다른 겹 행으로만 적합. EV = (1−PD)(a+b·금리) + PD·손실평균 (a,b: 비부도 수익률 OLS, 손실평균: 부도 수익률 평균; 모두 다른 겹)."""
    out = np.zeros(len(pdv)); ev = np.zeros(len(pdv))
    for k in np.unique(fold):
        o, t = fold != k, fold == k; pc = pdv[t]
        if kind in ("iso", "ev_iso"): pc = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(pdv[o], y[o]).predict(pdv[t])
        elif kind == "platt":
            lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))[:, None]
            pc = LogisticRegression(C=1e6).fit(lg(pdv[o]), y[o]).predict_proba(lg(pdv[t]))[:, 1]
        out[t] = pc
        if kind.startswith("ev"):
            g = o & (y == 0); A = np.c_[np.ones(g.sum()), rate[g]]; ab = np.linalg.lstsq(A, v[g], rcond=None)[0]; ld = v[o & (y == 1)].mean()
            ev[t] = -((1 - pc) * (ab[0] + ab[1] * rate[t]) + pc * ld)
    return out, (ev if kind.startswith("ev") else out)
def folds_for(n_folds, tune_frac=0.10):
    """fold=-1: Optuna 튜닝 전용 보류(10%, 보고 폴드와 분리). 나머지 90%를 (부도×만기×발행연도군) 층화 n_folds-겹."""
    rs = np.random.RandomState(K.SEED + 1); f = np.full(n, -1); idx = np.where(rs.rand(n) >= tune_frac)[0]
    yg = np.digitize(_yr.to_numpy(), [2012, 2017]); st = (y_all * 6 + (D.term_m.to_numpy() == 60) * 3 + yg).astype(int)[idx]
    for k, (_, t) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=K.SEED).split(idx, st)): f[idx[t]] = k
    return f
def run_cv(sp, fold, rounds, idx_sub):
    pdv, pr = np.full(n, np.nan), np.full(n, np.nan); its = []
    for k in np.unique(fold[idx_sub]):
        tr, te = idx_sub[fold[idx_sub] != k], idx_sub[fold[idx_sub] == k]
        Xtr, Xte = build(sp, tr, te, y_all[tr]); w = weights(sp, tr, y_all[tr], v_all[tr], Xtr)
        pred, it = fit_pred(sp, Xtr, y_all[tr], v_all[tr], w, Xte, rounds); its.append(it)
        if sp.get("target") == "reg": pr[te] = -pred
        else: pdv[te] = pred; pr[te] = pred
        log(sp["id"], "fold", k, "iters", it)
    return pdv, pr, its
def flat(sc, roe, aucg):
    r = {}
    for k, (m, e) in sc.items():
        if k == "36": r.update({k_: v for k_, v in m.items() if k_ not in ("n", "months")}); r.update(n36=m["n"], months36=m["months"])
        else: r.update({f"{q}@{k}": m[q] for q in ("SR_r30", "SR_r53", "SR_r70", "SR_r90", "SR_best", "a_best", "XS_r53", "n", "months")})
    for g, v in roe.items(): r.update({f"ROE_{q}@{g}": x for q, x in v.items() if g == "all" or q in ("r53", "r70", "all_in", "treasury")})
    r.update(aucg); return r
def record(base, pp, rr, ix, prob, fold, its, t0, resf):
    """AUC·logloss·Sharpe(36/60 × 발행연도군)·코스 ROE 한 행 추가. rr = 우선순위(클수록 위험; 낮은 값부터 승인, ix 밖은 NaN), pp = PD(없으면 NaN)."""
    sc = K.scoped_sharpe(L, rr); roe = K.course_roe(L, rr); yy = y_all[ix]; yrs = _yr.to_numpy()[ix]; tm = D.term_m.to_numpy()[ix]
    fa = [round(roc_auc_score(yy[fold == k], rr[ix][fold == k]), 5) for k in np.unique(fold)]
    aucg = {f"AUC@{g}": roc_auc_score(yy[(yrs >= a_) & (yrs <= b_)], rr[ix][(yrs >= a_) & (yrs <= b_)]) for g, (a_, b_) in K.YG.items()}
    aucg.update({f"AUC@t{t}": roc_auc_score(yy[tm == t], rr[ix][tm == t]) for t in (36, 60)})
    ok = prob and not np.isnan(pp[ix]).any()
    row = dict(base, AUC=roc_auc_score(yy, rr[ix]), LOGLOSS=log_loss(yy, np.clip(pp[ix], 1e-6, 1 - 1e-6)) if ok else np.nan, fold_auc=json.dumps(fa), n=len(ix), iters=int(np.mean(its)), sec=round(time.time() - t0), **flat(sc, roe, aucg))
    for k, (m, e) in sc.items(): np.save(W / f"e_{base['id']}.npy" if k == "36" else W / f"e{k}_{base['id']}.npy", e)
    pd.DataFrame([row]).to_csv(resf, mode="a", header=not resf.exists(), index=False, encoding="utf-8-sig")
    log(base["id"], "AUC %.4f SR53 %.3f SRbest %.3f ROE53 %.2f" % (row["AUC"], row["SR_r53"], row["SR_best"], row["ROE_r53@all"])); return row
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--list", action="store_true"); ap.add_argument("--only", default=""); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--quick", type=int, default=0); ap.add_argument("--force", action="store_true"); ap.add_argument("--rounds", type=int, default=600); ap.add_argument("--out", default="results_raw.csv"); a = ap.parse_args()
    if a.list: [print(i, s["grp"], s["name"]) for i, s in S.items()]; sys.exit()
    G = importlib.import_module("98_gate"); G.enforce(G.gate_table(D_ALL, [c for c in X.columns]) + G.gate_te(lambda ci, y: te_cols(ci.astype(np.float64), y, ci[:1].astype(np.float64))[0]) + G.gate_folds(folds_for(a.folds), a.folds) + G.gate_source(D_ALL.id, G.TRAIN_CSV, resolved_n=len(D), D=D_ALL), "데이터 관문(98_b)")
    ids = a.only.split(",") if a.only else [i for i in S if i != "T1_leak"]   # T1_leak(대조군)는 이전 창 결과로 갈음, 재실행 안 함
    fold = folds_for(a.folds); idx_sub = np.where(fold >= 0)[0]
    if a.quick: idx_sub = np.sort(np.random.RandomState(K.SEED).choice(idx_sub, a.quick, replace=False))
    rate = X.int_rate.to_numpy(np.float32) / 100; resf = W / a.out
    done = set(pd.read_csv(resf).id) if resf.exists() and not a.force else set()
    for i in ids:
        sp = S[i]; t0 = time.time(); base = dict(id=i, grp=sp["grp"], name=sp["name"], quick=a.quick, folds=a.folds)
        if i in done and not sp.get("derived"): continue
        if sp.get("reuse"): z = np.load(W / f"oof_{sp['reuse']}.npz"); pdv, pr, its = z["pd"], z["pr"], [0]
        else:
            if i in done: continue
            pdv, pr, its = run_cv(sp, fold, a.rounds, idx_sub); np.savez_compressed(W / f"oof_{i}.npz", pd=pdv, pr=pr)
        ix = idx_sub[~np.isnan(pr[idx_sub])]; prob = sp.get("target") != "reg"
        outs = [] if sp.get("reuse") else [("", pdv, pr, prob)]
        for dk in sp.get("derived", []):
            if i + "+" + dk in done: continue
            d_pd, d_pr = derive(dk, pdv[ix], fold[ix], y_all[ix], v_all[ix], rate[ix]); pp, rr = np.full(n, np.nan), np.full(n, np.nan); pp[ix] = d_pd; rr[ix] = d_pr
            outs.append(("+" + dk, pp, rr, not dk.startswith("ev")))
        for suf, pp, rr, prob_ in outs:
            if i + suf not in done: record(dict(base, id=i + suf), pp, rr, ix, prob_, fold[ix], its, t0, resf)
