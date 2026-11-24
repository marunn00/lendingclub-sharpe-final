"""98_features: 원본 형식 CSV(학습/시험 동일 141열) → 간결 테이블 전열 후보 DataFrame(열 선택 전). 98_a_clean_table.py(학습 테이블 생성)와 98_apply_test.py(시험 적용)가 공유.
라벨을 쓰지 않음. 시장금리는 발행월 M−1 월평균(산출물/raw_econ, raw_macro)."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; K = importlib.import_module("98_common"); C = K.C; log = K.log
import numpy as np, pandas as pd, polars as pl
ledger = pd.read_csv(C.ROOT / "산출물" / "전처리_v5" / "열_원장.csv")
POST = set(ledger[ledger.A_내생보존_reason.astype(str).str.contains("사후")].column)
NOT_LIST = POST | {"id", "url", "loan_status", "policy_code", "pymnt_plan", "funded_amnt", "funded_amnt_inv", "issue_d", "term", "sec_app_earliest_cr_line", "earliest_cr_line", "emp_length",
                   "emp_title", "title", "zip_code", "grade", "verification_status_joint"}
CATS = ["sub_grade", "home_ownership", "verification_status", "purpose", "addr_state", "initial_list_status", "application_type"]

def build_raw_table(csv):
    raw = pl.read_csv(csv, infer_schema_length=0, null_values=["", "NA"]); raw = raw.filter(pl.col("issue_d").is_not_null()); log("csv rows", raw.height)
    pdf = raw.to_pandas(); del raw; n = len(pdf)
    num = lambda c: pd.to_numeric(pdf[c].astype(str).str.replace("%", "", regex=False), errors="coerce").to_numpy(np.float64)
    mi = C.mcode_str(pdf.issue_d); term = pdf.term.str.strip().str[:2].astype(int).to_numpy()
    T = {"id": pdf.id.to_numpy(), "issue_d": pdf.issue_d.to_numpy(), "issue_month_code": mi, "term_m": term, "loan_status": pdf.loan_status.to_numpy()}
    T["default"] = np.where(pdf.loan_status.isin(["Charged Off", "Default"]), 1.0, np.where(pdf.loan_status.isin(["Fully Paid"]), 0.0, np.nan))
    for c in pdf.columns:                                                                      # 상장시점 원본 수치 열
        if c in NOT_LIST or c in CATS: continue
        v = num(c)
        if np.isfinite(v).sum() > 0 and np.nanstd(v) > 0: T[c] = v
    emp = pdf.emp_length.astype(str); T["emp_length_yr"] = np.where(emp.str.contains("<"), 0, pd.to_numeric(emp.str.extract(r"(\d+)")[0], errors="coerce"))
    for c in CATS: T[c] = pdf[c].to_numpy()
    T["zip3"] = pdf.zip_code.astype(str).str[:3].where(pdf.zip_code.notna()).to_numpy()
    T["emp_title_norm"] = pdf.emp_title.astype(str).str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip().where(pdf.emp_title.notna()).to_numpy()
    ecm = C.mcode_str(pdf.earliest_cr_line); rate = T["int_rate"]
    # ---- 시장금리 (발행월 M−1)
    def monthly_prev(path):
        d = pd.read_csv(path); d.columns = ["d", "v"]; d["v"] = pd.to_numeric(d.v, errors="coerce"); d = d.dropna()
        d["m"] = (d.d.str[:4].astype(int) - 2005) * 12 + d.d.str[5:7].astype(int) - 1       # 관측월 코드(2005-01=0)
        mm = d.groupby("m").v.mean(); arr = np.full(12 * 25, np.nan); mm = mm[(mm.index >= 0) & (mm.index < len(arr))]; arr[mm.index.to_numpy()] = mm.to_numpy(); return arr
    r10, r30 = monthly_prev(C.ROOT / "산출물/raw_econ/DGS10.csv"), monthly_prev(C.ROOT / "산출물/raw_econ/DGS30.csv")
    r3, r5, rff = (monthly_prev(C.MACRO / f"{k}.csv") for k in ("DGS3", "DGS5", "FEDFUNDS"))
    p = mi - 1                                                                                 # 발행월 M−1 (96_common.treasury_prev_month 와 같은 시점 규칙)
    t10, t30, ff = r10[p], r30[p], rff[p]; tm = np.where(term == 36, r3[p], r5[p])
    T["tsy10_at_issue"], T["tsy_matched_at_issue"] = t10, tm
    T["spread_10y"], T["spread_30y"], T["spread_matched"], T["spread_fedfunds"] = rate - t10, rate - t30, rate - tm, rate - ff
    g = pd.DataFrame({"m": mi, "t": term, "s": pdf.sub_grade.to_numpy(), "r": rate}).groupby(["m", "t", "s"]).r.transform("median").to_numpy()
    T["rate_premium_subgrade"] = rate - g
    # ---- 부담·신용 파생
    with np.errstate(all="ignore"):
        dv = lambda a, b: np.where(np.abs(b) > 1e-9, a / b, np.nan)
        T["pti"] = dv(T["installment"] * 12, T["annual_inc"]); T["lti"] = dv(T["loan_amnt"], T["annual_inc"])
        # credit_hist_years: ecm=-1 만 결측(2005-01 이전 개설은 음수 코드 → ecm>=0 마스크는 오류였음)
        T["credit_hist_years"] = np.where(ecm != -1, (mi - ecm) / 12.0, np.nan); T["fico_mid"] = (T["fico_range_low"] + T["fico_range_high"]) / 2
        T["open_acc_ratio"] = dv(T["open_acc"], T["total_acc"]); T["bal_per_acc"] = dv(T["tot_cur_bal"], T["total_acc"])
        T["new_acc_share24"] = dv(T["acc_open_past_24mths"], T["total_acc"]); T["revol_to_limit"] = dv(T["revol_bal"], T["total_rev_hi_lim"])
    return pd.DataFrame(T)
