"""98_e: 확정 대출 전체(Fully Paid·Charged Off·Default, 모든 발행연도·두 만기)의 대출별 수익 지표 → 산출물/앙상블_v1/_work/loans_all.parquet.
현금흐름 재구성·가정은 96_a_prep.py / 96_common.recon 그대로(부도월 = 마지막 상환+6, 수수료 1%, 상각·회수 반영). 원본 CSV에서 읽음(96 은 창 2012–16 한정이라 사용 불가).
컬럼: id, A(대출액), term, issue_month, issue_year, default, NPV_tsy(만기일치 국채 M−1 할인 NPV), rT, S=(1+rT)^(term/12), TW_short=(NPV_tsy+A)·S, NPV_short=NPV_tsy,
      ROE용: ann=(total_pymnt−funded)/상환기간(년), tb=발행월 DGS3/DGS5 월평균(20_test_split_sim 정의), roe_ok=(상환기간>0)"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; K = importlib.import_module("98_common"); C = K.C; log = K.log
import numpy as np, pandas as pd, polars as pl
COLS = ["id", "issue_d", "term", "loan_status", "funded_amnt", "int_rate", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "total_pymnt"]
SRC = sys.argv[1] if len(sys.argv) > 1 else C.ROOT / "lending_club_2020_train.csv"; DST = sys.argv[2] if len(sys.argv) > 2 else K.WORK / "loans_all.parquet"   # 시험 파일도 같은 정의로 계산
raw = pl.read_csv(SRC, columns=COLS, infer_schema_length=0, null_values=["", "NA"]).filter(pl.col("issue_d").is_not_null())
log("rows", raw.height, raw["loan_status"].value_counts().sort("count", descending=True).head(8).to_dict(as_series=False))
pdf = raw.filter(pl.col("loan_status").is_in(["Fully Paid", "Charged Off", "Default"])).to_pandas(); del raw; n = len(pdf); log("resolved", n)
num = lambda c: pd.to_numeric(pdf[c].astype(str).str.replace("%", "", regex=False), errors="coerce").to_numpy(float)
f, rate = num("funded_amnt"), num("int_rate"); term = pdf.term.str.strip().str[:2].astype(int).to_numpy()
mi0 = C.mcode_str(pdf.issue_d); lp = C.mcode_str(pdf.last_pymnt_d); lp_ok = lp >= 0; lp = np.where(lp < 0, mi0, lp)
Kk = np.maximum(lp - mi0, 0); co = pdf.loan_status.isin(["Charged Off", "Default"]).to_numpy()
TRP, TRI, TLF, REC, CF = (np.nan_to_num(num(c)) for c in ["total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee"]); TP = num("total_pymnt")
tsy = {36: C.treasury_prev_month("DGS3"), 60: C.treasury_prev_month("DGS5")}
NPV = np.zeros(n); rT = np.zeros(n)
for t in (36, 60):
    ix = np.where(term == t)[0]; W = int(Kk[ix].max()) + C.L_MAIN + 1
    for s0 in range(0, len(ix), 25000):
        s = ix[s0:s0 + 25000]; net, wo, prin = C.recon(f[s], rate[s], term[s], Kk[s], co[s], TRP[s], TRI[s], TLF[s], REC[s], CF[s], W)
        j = np.arange(W)[None, :]; r_ = tsy[t][mi0[s] - 1] / 100; rT[s] = r_; NPV[s] = (net * (1 + r_[:, None]) ** (-j / 12)).sum(1) - f[s]
    log(f"{t}m done")
S = (1 + rT) ** (term / 12)
def tb_month(name):
    t = pd.read_csv(C.MACRO / f"{name}.csv"); t.columns = ["d", "v"]; t["v"] = pd.to_numeric(t.v, errors="coerce"); t = t.dropna()
    m = (t.d.str[:4].astype(int) - 2005) * 12 + t.d.str[5:7].astype(int) - 1; a = np.full(216, np.nan); g = t.groupby(m).v.mean(); g = g[(g.index >= 0) & (g.index < 216)]; a[g.index.to_numpy()] = g.to_numpy() / 100; return a
t3, t5 = tb_month("DGS3"), tb_month("DGS5"); tb = np.where(term == 36, t3[mi0], t5[mi0])
pt = np.where(lp_ok, (lp - mi0) / 12.0, np.nan); roe_ok = pt > 0; ann = np.where(roe_ok, (TP - f) / np.where(roe_ok, pt, 1), 0.0)
L = pd.DataFrame(dict(id=pdf.id.to_numpy(), A=f, term=term, issue_month=mi0, issue_year=pdf.issue_d.str[4:8].astype(int).to_numpy(), default=co.astype(np.int8), NPV_tsy=NPV, rT=rT, S=S,
                      TW_short=(NPV + f) * S, NPV_short=NPV, ann=ann, tb=tb, roe_ok=roe_ok))
assert not L[["NPV_tsy", "rT", "tb"]].isna().any().any(), L[["NPV_tsy", "rT", "tb"]].isna().sum()
L.to_parquet(DST, index=False); log("saved", L.shape)
print(L.groupby(["issue_year", "term"]).size().unstack(fill_value=0).to_string()); print("default rate", L.groupby(["issue_year", "term"]).default.mean().unstack().round(3).to_string())
