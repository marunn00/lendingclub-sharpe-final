"""98_gate: 학습·적용 전 데이터 관문 (tasks/요구사항_원장.md '탐색 루프 절차' 2단계). 하나라도 실패하면 SystemExit — 학습을 시작하지 않는다.
검사 4묶음: (1) 내생·사후 결과 열 부재 (2) 결측을 설계대로(NaN 유지, 대치 없음) 처리 (3) 의사결정 시점에 알 수 없는 정보 없음(시장금리 월 단위 일정, 신용이력 ≥ 0,
    타깃 인코딩은 폴드 안에서만: 잡음 범주 열로 누수 시험, 튜닝 보류·보고 겹 분리) (4) 행 수·키가 원본 CSV와 일치.
사용: python3 scripts/98_gate.py table                # 현재 간결 테이블
      python3 scripts/98_gate.py split                # validate 격리 분할(_work/hold40.csv, hold_ids.txt)
모듈: gate_table(D, features, ...)·gate_te(te_fn)·gate_folds(fold)·gate_source(ids, csv, ...) → enforce(results)"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, numpy as np, pandas as pd
F = importlib.import_module("98_features"); K = importlib.import_module("98_common"); C = K.C
FORBID = set(F.POST) | {"funded_amnt", "funded_amnt_inv", "url", "policy_code", "pymnt_plan", "loan_status", "default", "grade", "title", "earliest_cr_line", "last_pymnt_d"}
FORBID_PREFIX = ("nv_", "hardship_", "settlement_", "debt_settlement")
MARKET = ["tsy10_at_issue", "tsy_matched_at_issue"]; TABLE = C.ROOT / "산출물" / "간결테이블" / "train_간결.parquet"; TRAIN_CSV = C.ROOT / "lending_club_2020_train.csv"
def _r(name, ok, detail): return (name, bool(ok), detail)
def gate_table(D, features, labeled=True):
    """D: 간결 테이블형 DataFrame(id, issue_d, issue_month_code, term_m, 수치 열…), features: 모형이 쓰는 열 이름 목록(범주 k__/hc__/te__ 접두 포함 가능)."""
    out = []; cols = list(D.columns); base = lambda c: c.split("__", 1)[1] if "__" in c else c
    isbad = lambda c: base(c) in FORBID or c.startswith(FORBID_PREFIX)
    bad = sorted(c for c in cols if isbad(c) and c not in ("loan_status", "default")); bad_feat = sorted(c for c in features if isbad(c))
    out.append(_r("1a 내생·사후 결과 열 부재(테이블 열: 라벨용 loan_status·default 제외)", not bad, bad or "없음"))
    out.append(_r("1b 모형 특성에 라벨·사후·nv_ 열 없음", not bad_feat, bad_feat or f"특성 {len(features)}개 모두 허용"))
    num = [c for c in cols if c in features and pd.api.types.is_numeric_dtype(D[c])]
    if labeled and "default" in D:
        L = D[D.default.notna()]; s = L.sample(min(150000, len(L)), random_state=0); y = s.default.to_numpy()
        from sklearn.metrics import roc_auc_score
        a = {c: max(roc_auc_score(y, v), 1 - roc_auc_score(y, v)) for c in num for v in [np.nan_to_num(s[c].to_numpy(np.float64), nan=-1e9)] if np.isfinite(v).all()}
        top = max(a, key=a.get) if a else None
        out.append(_r("1c 단일 열 AUC < 0.95 (누수 신호)", top is None or a[top] < 0.95, f"최대 {top} {a[top]:.4f}" if top else "수치 열 없음"))
    inf = [c for c in num if np.isinf(D[c].to_numpy(np.float64)).any()]
    out.append(_r("2a 무한대 값 없음", not inf, inf or "없음"))
    allnan = [c for c in num if D[c].isna().all()]
    out.append(_r("2b 전부 결측인 열 없음", not allnan, allnan or "없음"))
    if "mths_since_last_record" in D: nr = D.mths_since_last_record.isna().mean(); out.append(_r("2c 구조적 결측 유지(대치 없음): mths_since_last_record 결측률 > 50%", nr > 0.5, f"{nr:.3f}"))
    if "credit_hist_years" in D: nr = D.credit_hist_years.isna().mean(); out.append(_r("2d credit_hist_years 결측률 < 2% (2005-01 이전 개설 포함)", nr < 0.02, f"{nr:.4f}"))
    mk = [c for c in MARKET if c in D]; g = D.groupby(["issue_month_code", "term_m"])
    nuniq = {c: int((g[c].nunique(dropna=True) > 1).sum()) for c in mk}
    out.append(_r("3a 시장금리는 (발행월, 만기)당 하나의 값(발행 시점 정보)", all(v == 0 for v in nuniq.values()), nuniq or "시장금리 열 없음"))
    if "credit_hist_years" in D: mn = float(D.credit_hist_years.min()); out.append(_r("3b 신용이력 ≥ 0 (미래 개설일 없음)", mn >= 0, f"최솟값 {mn:.3f}"))
    return out
def gate_te(te_fn, n=120000, seed=0):
    """te_fn(범주코드 int64, y)→같은 길이 OOF 인코딩. 라벨과 무관한 잡음 범주(약 n/4종)를 넣었을 때 인코딩이 라벨과 무상관이어야 한다(전체적합 인코딩은 크게 상관)."""
    rs = np.random.RandomState(seed); y = (rs.rand(n) < 0.2).astype(np.float64); ci = rs.randint(0, n // 4, n).astype(np.int64)
    corr = float(np.corrcoef(y, te_fn(ci, y))[0, 1]); a = ci + 1; pri = y.mean(); full = ((np.bincount(a, y) * 0 + np.bincount(a, y) + 30 * pri) / (np.bincount(a) + 30))[a]; cf = float(np.corrcoef(y, full)[0, 1])
    return [_r("3c 타깃 인코딩은 폴드 안(OOF)만: 잡음 범주의 라벨 상관 |r| < 0.02", abs(corr) < 0.02 and cf > 0.05, f"OOF r={corr:+.4f} (대조 전체적합 r={cf:+.3f})")]
def gate_folds(fold, n_folds=5):
    f = np.asarray(fold); rep = [int((f == k).sum()) for k in range(n_folds)]; tu = int((f == -1).sum()); ok = all(r > 0 for r in rep) and set(np.unique(f)) <= set(range(-1, n_folds)) and 0.05 < tu / len(f) < 0.15
    return [_r("3d 튜닝 보류(−1)와 보고 겹 0..4 분리·분할 완전", ok, f"보고 겹 {rep}, 튜닝 {tu} ({tu / len(f):.3f})")]
def gate_source(ids, csv, expected_removed=0, resolved_n=None, D=None):
    """ids: 처리된 표의 id 배열(순서 그대로). csv: 원본. 원본의 issue_d 존재 행(= 처리 대상)과 행 수·키·순서가 같아야 한다(expected_removed 만큼 제외 허용)."""
    import polars as pl
    src = pl.read_csv(csv, infer_schema_length=0, null_values=["", "NA"], columns=["id", "issue_d"]).filter(pl.col("issue_d").is_not_null()).get_column("id").to_numpy().astype(str)
    ids = np.asarray(ids).astype(str); uniq = len(set(ids)) == len(ids); exp = len(src) - expected_removed
    out = [_r("4a 행 수 = 원본(발행월 있는 행) − 제외", len(ids) == exp, f"표 {len(ids):,} / 원본 {len(src):,} − 제외 {expected_removed:,}"), _r("4b id 고유", uniq, f"고유 {len(set(ids)):,}")]
    keep = src if not expected_removed else src[np.isin(src, ids)]
    out.append(_r("4c id 순서·키가 원본과 동일", len(keep) == len(ids) and bool((keep == ids).all()), "원본 순서 유지" if len(keep) == len(ids) else f"원본 매칭 {len(keep):,}"))
    if resolved_n is not None and D is not None and "default" in D:
        n = int(D.default.notna().sum()); out.append(_r("4d 확정 대출 수", n == resolved_n, f"{n:,} (기대 {resolved_n:,})"))
    return out
def enforce(results, title="데이터 관문"):
    print(f"[{title}]"); [print(f"  {'통과' if ok else '실패'}  {n}: {d}") for n, ok, d in results]
    if not all(ok for _, ok, _ in results): raise SystemExit(f"{title} 실패 — 학습/적용을 시작하지 않습니다. 처리 방식을 고친 뒤 다시 실행하십시오.")
    print(f"  → {title} 전체 통과")
def te_fn_apply(ci, y): return importlib.import_module("98_apply_test").te_oof(ci, y)
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "table"
    if mode == "table":
        D = pd.read_parquet(TABLE); IDC = {"id", "issue_d", "issue_month_code", "loan_status", "default"}
        feats = [c for c in D.columns if c not in IDC]
        res = gate_table(D, feats) + gate_te(te_fn_apply) + gate_source(D.id, TRAIN_CSV, resolved_n=1116156, D=D)
        enforce(res, "데이터 관문(현재 간결 테이블)")
    if mode == "split":
        import json; W = K.WORK; M = json.load(open(MD := W / "model_val" / "recipe.json")); raw = F.build_raw_table(str(W / "hold40.csv")); hold = set((W / "hold_ids.txt").read_text().split())
        T = pd.read_parquet(TABLE, columns=["id", "default"]); trn = T[~T.id.astype(str).isin(hold)]
        res = gate_table(raw[[c for c in M["table_cols"] if c in raw.columns]], M["columns"], labeled=True)
        res += gate_source(raw.id, W / "hold40.csv") + gate_source(trn.id, TRAIN_CSV, expected_removed=len(hold))
        res += [_r("4e 격리 40%와 학습 60% id 교집합 없음", not (hold & set(trn.id.astype(str))), f"격리 {len(hold):,} / 학습 {len(trn):,}"), _r("4f 학습 모형이 격리 id를 보지 않음(recipe n_train = 학습 60% 확정 대출)", M["n_train"] == int(trn.default.notna().sum()), f"{M['n_train']:,} vs {int(trn.default.notna().sum()):,}")]
        enforce(res, "데이터 관문(validate 분할)")
