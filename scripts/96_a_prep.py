"""96_a: 새 테이블 → 확정 대출(완납·상각; 36m 2012–2016, 60m 2012–2014) 대출별 지표 + 모형용 특성 + 빈티지×나이 집계.
저장: 기술IR_v5/_work/loans.parquet(지표·키), feats.parquet(특성), vintage_age.npz."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; C = importlib.import_module("96_common")
import numpy as np, pandas as pd, polars as pl
log = C.log
POST = {"id","url","emp_title","title","zip_code","loan_status","pymnt_plan","out_prncp","out_prncp_inv","total_pymnt","total_pymnt_inv","total_rec_prncp",
 "total_rec_int","total_rec_late_fee","recoveries","collection_recovery_fee","last_pymnt_d","last_pymnt_amnt","next_pymnt_d","last_credit_pull_d",
 "last_fico_range_high","last_fico_range_low","orig_projected_additional_accrued_interest","debt_settlement_flag","policy_code","funded_amnt_inv","funded_amnt","issue_d",
 "payment_plan_start_date","deferral_term","term","sec_app_earliest_cr_line"} | {c for c in pl.read_parquet_schema(C.TABLE) if c.startswith("hardship")}
sch = list(pl.read_parquet_schema(C.TABLE))
RAW, NV = sch[:141], [c for c in sch if c.startswith("nv_")]
assert len(RAW) == 141 and len(NV) == 56
CAT = ["grade","sub_grade","home_ownership","verification_status","purpose","addr_state","initial_list_status","application_type","verification_status_joint","emp_length"]
LIST = [c for c in RAW if c not in POST]
d = pl.read_parquet(C.TABLE).filter(pl.col("issue_d").is_not_null())
d = d.filter(pl.col("loan_status").is_in(["Fully Paid", "Charged Off", "Default"]))
d = d.with_columns(pl.col("term").str.strip_chars().str.slice(0, 2).cast(pl.Int32).alias("term_m"), pl.col("issue_d").str.slice(4, 4).cast(pl.Int32).alias("iy"))
d = d.filter(((pl.col("term_m") == 36) & pl.col("iy").is_between(2012, 2016)) | ((pl.col("term_m") == 60) & pl.col("iy").is_between(2012, 2014)))
log("resolved rows", d.height)
pdf = d.to_pandas(); del d
n = len(pdf); log("pandas", n)
num = lambda c: pd.to_numeric(pdf[c].str.replace("%", "", regex=False), errors="coerce").to_numpy(float)
f = num("funded_amnt"); rate = num("int_rate"); term = pdf.term_m.to_numpy(int)
mi0 = C.mcode_str(pdf.issue_d); lp = C.mcode_str(pdf.last_pymnt_d); lp = np.where(lp < 0, mi0, lp)
K = np.maximum(lp - mi0, 0); co = pdf.loan_status.isin(["Charged Off", "Default"]).to_numpy()   # "Default" 상태(창 내 50건)도 부도로 포함(구 outcome_class 정의)
TRP, TRI, TLF, REC, CF = (np.nan_to_num(num(c)) for c in ["total_rec_prncp","total_rec_int","total_rec_late_fee","recoveries","collection_recovery_fee"])
TP = num("total_pymnt")
tsy = {36: C.treasury_prev_month("DGS3"), 60: C.treasury_prev_month("DGS5")}
res_month = np.where(co, K + C.L_MAIN, K) + mi0
out = dict(row=np.arange(n), A=f, term=term, issue_month=mi0, issue_year=pdf.iy.to_numpy(int), grade=pdf.grade.to_numpy(), sub_grade=pdf.sub_grade.to_numpy(),
           default=co.astype(np.int8), int_rate=rate, K=K, res_month=res_month, total_pymnt=TP)
cols = {k: np.zeros(n) for k in ["NPV_short","NPV_tsy","TW_short","cash_total","S","rT","fin_loss"]}
VA = {}
for t in (36, 60):
    ix = np.where(term == t)[0]; W = int(K[ix].max()) + C.L_MAIN + 1; H = t
    for s0 in range(0, len(ix), 25000):
        s = ix[s0:s0 + 25000]
        net, wo, prin = C.recon(f[s], rate[s], term[s], K[s], co[s], TRP[s], TRI[s], TLF[s], REC[s], CF[s], W)
        j = np.arange(W)[None, :]; m0 = mi0[s]
        npv = (net * C.G[m0[:, None]] / C.G[m0[:, None] + j]).sum(1) - f[s]
        rT = tsy[t][m0 - 1] / 100
        cols["NPV_short"][s] = npv; cols["TW_short"][s] = (net * C.G[m0[:, None] + H] / C.G[m0[:, None] + j]).sum(1)
        cols["NPV_tsy"][s] = (net * (1 + rT[:, None]) ** (-j / 12)).sum(1) - f[s]
        cols["cash_total"][s] = net.sum(1); cols["S"][s] = C.G[m0 + H] / C.G[m0]; cols["rT"][s] = rT
        cols["fin_loss"][s] = wo.sum(1) - (net.sum(1) - ((f[s] * 0)))  # placeholder replaced below
        cols["fin_loss"][s] = wo.sum(1)
        for yy in range(2012, 2017):
            sel = pdf.iy.to_numpy()[s] == yy
            if sel.any():
                key = (t, yy)
                a = VA.setdefault(key, dict(A=0.0, net=np.zeros(W), wo=np.zeros(W), prin=np.zeros(W), n=0))
                a["A"] += f[s][sel].sum(); a["net"] += net[sel].sum(0); a["wo"] += wo[sel].sum(0); a["prin"] += prin[sel].sum(0); a["n"] += sel.sum()
        log(f"{t}m chunk {s0//25000}")
out.update({k: v for k, v in cols.items()})
loans = pd.DataFrame(out); loans.to_parquet(C.WORK / "loans.parquet", index=False)
np.savez(C.WORK / "vintage_age.npz", **{f"{t}_{y}_{k}": v for (t, y), a in VA.items() for k, v in a.items()})
# ---- 특성
fe = pd.DataFrame(index=np.arange(n))
for c in LIST:
    if c in CAT:
        fe["raw__" + c] = pdf[c].astype("category").cat.codes.astype("int16") if c != "emp_length" else pd.to_numeric(pdf[c].str.extract(r"(\d+)")[0], errors="coerce") + (pdf[c].str.contains("<", na=False) * 0)
    elif c == "earliest_cr_line":
        fe["raw__earliest_cr_age_mo"] = mi0 - C.mcode_str(pdf[c])
    elif c in ("int_rate", "revol_util", "loan_amnt"):
        fe["raw__" + c] = num(c)
    else:
        v = pd.to_numeric(pdf[c], errors="coerce")
        if v.notna().sum() >= 0.5 * pdf[c].notna().sum() and v.notna().sum() > 0: fe["raw__" + c] = v.astype(float).to_numpy()
fe["raw__term"] = term; fe["raw__issue_mo_of_year"] = (mi0 % 12)
for c in NV: fe[c] = pdf[c].astype(float).to_numpy()
fe = fe.loc[:, fe.notna().any()]
fe.to_parquet(C.WORK / "feats.parquet", index=False)
log("feats", fe.shape, "raw", sum(c.startswith("raw__") for c in fe), "nv", sum(c.startswith("nv_") for c in fe))
C.meta("_work/prep", n=n, ass="K+6 상각, 1% 수수료, FEDFUNDS 재투자; 확정=Fully Paid+Charged Off", excluded_post=sorted(POST))
