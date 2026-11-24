"""98_apply_test: 학습 파일로 전처리·모형을 적합해 산출물/앙상블_v1/model/ 에 저장(fit) → 같은 형식 CSV에 적용해 행별 PD·승인 결정을 출력(apply).
사용:
  python3 scripts/98_apply_test.py fit   --train lending_club_2020_train.csv [--exclude-ids 파일] [--model-dir DIR] [--seeds 3]
  python3 scripts/98_apply_test.py apply --csv 시험.csv --out 예측.csv [--model-dir DIR] [--approve-rate 0.53]
  python3 scripts/98_apply_test.py validate   # 학습 파일의 무작위 40%를 떼어 60%로 fit → 40%에 apply → AUC/Sharpe를 CV 수준과 비교
레시피(recipe.json): 간결 테이블 열 + 범주 코드표 + zip3·emp_title 로그빈도표 + 타깃인코딩(5겹 OOF로 학습, 전체 학습으로 적용 지도) + issue_year + XGBoost 튜닝 모수 시드 평균.
승인 결정: 발행월 안에서 PD 오름차순 순위 → 하위 approve-rate 비율 승인(96_c와 같은 월별 우선순위 규칙). 라벨(loan_status)이 없어도 동작."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, argparse, json, time, numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import StratifiedKFold
F = importlib.import_module("98_features"); K = importlib.import_module("98_common"); C = K.C; log = K.log
MD = C.ROOT / "산출물" / "앙상블_v1" / "model"; TABLE = C.ROOT / "산출물" / "간결테이블" / "train_간결.parquet"
CATS = ["sub_grade", "home_ownership", "verification_status", "purpose", "addr_state", "initial_list_status", "application_type", "zip3", "emp_title_norm"]
NOMINAL = ["home_ownership", "verification_status", "purpose", "addr_state", "initial_list_status", "application_type"]
IDC = {"id", "issue_d", "issue_month_code", "loan_status", "default"} | set(CATS)
DERIVED = ["lti", "spread_10y", "spread_30y", "spread_matched", "spread_fedfunds", "rate_premium_subgrade", "tsy10_at_issue", "tsy_matched_at_issue", "pti", "credit_hist_years", "open_acc_ratio", "bal_per_acc", "new_acc_share24"]
TE_COLS = ["zip3", "emp_title_norm", "addr_state", "purpose"]; HC = ["zip3", "emp_title_norm"]
EXTRA_DERIVED = ["spread_fedfunds", "tsy10_at_issue", "tsy_matched_at_issue", "credit_hist_years", "open_acc_ratio", "new_acc_share24"]   # 98_run_ens.sh 채택 파생과 동일
def raw_frame(csv, cols):
    """원본 형식 CSV → 간결 테이블 열만 남긴 DataFrame (학습 테이블과 같은 98_features 경로)."""
    D = F.build_raw_table(csv); return D[[c for c in cols if c in D.columns]].copy()
def te_fit(ci, ytr, m=30.0):
    """범주 코드(정수, 결측=-1) → (smoothed 평균 표, 사전평균). 학습 전체로 적합한 최종 맵."""
    a = ci + 1; nc = a.max() + 2; pri = float(ytr.mean())
    return ((np.bincount(a, ytr, nc) + m * pri) / (np.bincount(a, minlength=nc) + m)), pri
def te_oof(ci, ytr, inner=5, m=30.0, seed=0):
    a = ci + 1; nc = a.max() + 2; pri = float(ytr.mean()); tr = np.zeros(len(a), np.float32)
    for i, j in StratifiedKFold(inner, shuffle=True, random_state=seed).split(a, ytr): tr[j] = ((np.bincount(a[i], ytr[i], nc) + m * pri) / (np.bincount(a[i], minlength=nc) + m))[a[j]]
    return tr
def te_apply(ci, tab, pri):
    a = ci + 1; v = np.full(len(a), pri, np.float32); ok = a < len(tab); v[ok] = tab[a[ok]]; return v
def codes(D, cmaps):
    """범주 열 → 저장된 범주 목록 기준 정수코드(미등록=-1)."""
    return {c: pd.Categorical(D[c], categories=cmaps[c]).codes.astype(np.int64) for c in CATS}
def featurize(D, rc, train_y=None):
    """D(간결 테이블 열) → 모형 입력 행렬. train_y가 주어지면(학습) TE는 5겹 OOF, 아니면 저장된 최종 맵 적용."""
    cd = codes(D, rc["cmaps"]); X = pd.DataFrame({c: D[c].to_numpy(np.float32) for c in rc["numeric"]})
    X["k__sub_grade"] = np.where(cd["sub_grade"] < 0, np.nan, cd["sub_grade"]).astype(np.float32)
    for c in NOMINAL: X["k__" + c] = np.where(cd[c] < 0, np.nan, cd[c]).astype(np.float32)
    X["issue_year"] = D.issue_d.str[4:8].astype(int).to_numpy().astype(np.float32)
    for c in HC: X["hc__" + c + "_logfreq"] = np.log10(D[c].map(rc["freq"][c]).astype(float).fillna(1e-7).to_numpy()).astype(np.float32)
    for c in TE_COLS:
        X["te__" + c] = te_oof(cd[c], train_y) if train_y is not None else te_apply(cd[c], np.array(rc["te"][c]["tab"]), rc["te"][c]["pri"])
    return X[rc["columns"]]
def fit(a):
    t0 = time.time(); MD = Path_md(a); MD.mkdir(parents=True, exist_ok=True); cols = list(pd.read_parquet(TABLE).columns)
    D = raw_frame(a.train, cols); log("train rows", len(D)); G = importlib.import_module("98_gate"); nex = 0
    if a.exclude_ids: ex = set(open(a.exclude_ids).read().split()); D = D[~D.id.astype(str).isin(ex)]; nex = len(ex); log("after exclude", len(D))
    G.enforce(G.gate_table(D, [c for c in D.columns if c not in IDC]) + G.gate_te(te_oof) + G.gate_source(D.id, a.train, expected_removed=nex), "데이터 관문(fit)")
    freq = {c: D[c].value_counts(normalize=True).to_dict() for c in HC}                      # 비지도 빈도: 해결 여부와 무관한 전체 학습 행
    cmaps = {c: sorted(D[c].dropna().unique().tolist()) for c in CATS}
    L = D[D.default.notna()].reset_index(drop=True); y = L.default.to_numpy(np.float64)
    numeric = [c for c in L.columns if c not in IDC and (c not in DERIVED or c in EXTRA_DERIVED)]
    rc = dict(cmaps=cmaps, freq=freq, numeric=numeric); rc["columns"] = numeric + ["k__sub_grade"] + ["k__" + c for c in NOMINAL] + ["issue_year"] + ["hc__" + c + "_logfreq" for c in HC] + ["te__" + c for c in TE_COLS]
    cd = codes(L, cmaps); rc["te"] = {}
    for c in TE_COLS: tab, pri = te_fit(cd[c], y); rc["te"][c] = dict(tab=tab.tolist(), pri=pri)
    Xtr = featurize(L, rc, train_y=y); log("X", Xtr.shape)
    bp = json.load(open(K.WORK / "best_params.json"))["best"]; params = dict(tree_method="hist", max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.6, min_child_weight=50, reg_lambda=5.0, max_bin=128, nthread=10, objective="binary:logistic", eval_metric="logloss"); params.update(bp)
    rounds = a.rounds or 600   # CV 5겹 조기종료 반복수가 565–599(상한 600)였음 → CV와 같은 600. (Optuna 보류 10% 표본의 307회는 학습량이 작아 과소적합이었음)
    d = xgb.DMatrix(Xtr, y.astype(np.float32))
    for s in range(a.seeds): xgb.train(dict(params, seed=s), d, rounds).save_model(str(MD / f"xgb_s{s}.json")); log("seed", s, "done")
    json.dump(dict(rc, table_cols=cols, params=params, rounds=rounds, seeds=a.seeds, n_train=len(L)), open(MD / "recipe.json", "w"), ensure_ascii=False)
    D[["issue_month_code", "term_m", "tsy10_at_issue", "tsy_matched_at_issue"]].drop_duplicates(["issue_month_code", "term_m"]).sort_values(["issue_month_code", "term_m"]).to_csv(MD / "treasury_lookup.csv", index=False)
    log("fit done", round(time.time() - t0), "s")
def apply(a):
    rc = json.load(open(Path_md(a) / "recipe.json")); D = raw_frame(a.csv, rc["table_cols"]); G = importlib.import_module("98_gate")
    G.enforce(G.gate_table(D, rc["columns"], labeled=False) + G.gate_source(D.id, a.csv), "데이터 관문(apply)"); X = featurize(D, rc); d = xgb.DMatrix(X)
    pdv = np.mean([xgb.Booster(model_file=str(Path_md(a) / f"xgb_s{s}.json")).predict(d) for s in range(rc["seeds"])], 0)
    out = pd.DataFrame(dict(id=D.id, issue_d=D.issue_d, term_m=D.term_m, pd=pdv)); out["month_code"] = D.issue_month_code.to_numpy()
    out["pd_rank_in_month"] = out.groupby(["month_code", "term_m"]).pd.rank(method="first", pct=True)       # 발행월·만기 안 순위(0~1, 낮을수록 안전)
    out["approve"] = (out.pd_rank_in_month <= a.approve_rate).astype(int); out.drop(columns="month_code").to_csv(a.out, index=False); log("saved", a.out, len(out)); return out
def Path_md(a): return __import__("pathlib").Path(a.model_dir)
def validate(a):
    """학습 파일의 무작위 40%(id 기준, 시드 0)를 시험처럼 격리: 60%로 fit → 40% CSV에 apply → 확정 대출의 AUC·Sharpe."""
    import polars as pl; from sklearn.metrics import roc_auc_score
    W = K.WORK; T = pd.read_parquet(TABLE, columns=["id", "default", "term_m"]); rs = np.random.RandomState(0); hold = T.id[rs.rand(len(T)) < 0.4].astype(str)
    (W / "hold_ids.txt").write_text("\n".join(hold)); raw = pl.read_csv(C.ROOT / "lending_club_2020_train.csv", infer_schema_length=0, null_values=["", "NA"]); raw = raw.filter(pl.col("issue_d").is_not_null())
    raw.filter(pl.col("id").is_in(hold.tolist())).write_csv(W / "hold40.csv"); del raw
    a.train = str(C.ROOT / "lending_club_2020_train.csv"); a.exclude_ids = str(W / "hold_ids.txt"); a.csv = str(W / "hold40.csv"); a.out = str(W / "hold40_pred.csv"); a.model_dir = str(W / "model_val"); fit(a); o = apply(a)
    L = K.load_loans()
    L = L.assign(sid=L.id.astype(str)); o = o.assign(sid=o.id.astype(str)); M = L.merge(o[["sid", "pd"]], on="sid"); res = dict(n_hold_resolved=len(M), AUC=float(roc_auc_score(M.default, M.pd)))
    for t in (36, 60):
        s = M[M.term == t].reset_index(drop=True)
        if len(s): res[f"AUC_{t}"] = float(roc_auc_score(s.default, s.pd)); res[f"n_{t}"] = len(s)
    s36 = M[M.term == 36].reset_index(drop=True); m36, _ = K.sharpe_eval(s36, s36.pd.to_numpy()); res.update({k: float(v) for k, v in m36.items()})
    M = M.reset_index(drop=True); roe = K.course_roe(M, M.pd.to_numpy())
    for g, v in roe.items(): res.update({f"ROE_{q}@{g}": float(x) for q, x in v.items() if q.startswith("r") or q in ("all_in", "treasury")})
    json.dump(res, open(W / "validate_apply.json", "w"), indent=1); log("VALIDATE", res)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--train"); ap.add_argument("--exclude-ids", dest="exclude_ids"); ap.add_argument("--model-dir", dest="model_dir", default=str(MD))
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--rounds", type=int, default=0); ap.add_argument("--csv"); ap.add_argument("--out"); ap.add_argument("--approve-rate", dest="approve_rate", type=float, default=0.53)
    a = ap.parse_args(); {"fit": fit, "apply": apply, "validate": validate}[a.mode](a)
