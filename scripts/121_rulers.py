"""121 공용: 팀 공유 보고서 잣대 계산(정의는 121_team_metrics.py 머리말). 시험·학습 파일 모두 같은 함수로 읽고 계산한다."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, numpy as np, pandas as pd, polars as pl
K = importlib.import_module("98_common"); C = K.C; log = K.log
RES = ["Fully Paid", "Charged Off", "Default"]
COLS = ["id", "issue_d", "term", "loan_status", "sub_grade", "funded_amnt", "total_pymnt", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries"]
QS = np.round(np.arange(0.005, 1.00001, 0.005), 3)

def tb_month(name):
    t = pd.read_csv(C.MACRO / f"{name}.csv"); t.columns = ["dd", "v"]; t["v"] = pd.to_numeric(t.v, errors="coerce"); t = t.dropna()
    m = (t.dd.str[:4].astype(int) - 2005) * 12 + t.dd.str[5:7].astype(int) - 1; a = np.full(216, np.nan); g = t.groupby(m).v.mean(); g = g[(g.index >= 0) & (g.index < 216)]; a[g.index.to_numpy()] = g.to_numpy() / 100; return a
t3, t5 = tb_month("DGS3"), tb_month("DGS5")

def frame(csv):
    """결과 확정 대출만. r1 = T1 대출 수익(계약만기 연율화), r2 = T2 대출 수익, rf = 발급월 만기 맞춤 국채, ref = T2 만기 기준 코호트."""
    d = pl.read_csv(csv, columns=COLS, infer_schema_length=0, null_values=["", "NA"]).filter(pl.col("loan_status").is_in(RES)).to_pandas()
    for c in COLS[5:]: d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["id"] = d.id.astype(str); d["T"] = d.term.str.strip().str[:2].astype(int); d["mi"] = C.mcode_str(d.issue_d)
    dt = pd.to_datetime(d.issue_d, format="%b-%Y"); d["ym"] = (dt.dt.year * 100 + dt.dt.month).to_numpy(); d["yr"] = dt.dt.year.to_numpy()
    d["rf"] = np.where(d["T"] == 36, t3[d.mi], t5[d.mi]); assert not np.isnan(d.rf).any(); d["dflt"] = d.loan_status != "Fully Paid"
    d["r1"] = (d.total_pymnt / d.funded_amnt) ** (12 / d["T"]) - 1
    tot = d.total_rec_prncp + d.total_rec_int + d.total_rec_late_fee + d.recoveries; d["r2"] = (np.maximum(tot, 0) / d.funded_amnt) ** (12 / d["T"]) - 1
    d["ref"] = ((d["T"] == 36) & (d.ym <= 201703)) | ((d["T"] == 60) & (d.ym <= 201503))
    log(csv.name, "rows", len(d), "|total_pymnt − (P+I+Lf+R)| 중앙값", round(float(np.median(np.abs(d.total_pymnt - tot))), 4)); return d

def grade_score(d): sg = sorted(d.sub_grade.unique()); return d.sub_grade.map({g: i for i, g in enumerate(sg)}).to_numpy() + 0.001 * np.random.RandomState(0).rand(len(d))
def order_rank(score):
    o = np.argsort(score, kind="stable"); rk = np.empty(len(o), np.int64); rk[o] = np.arange(len(o)); return o, rk
def sel(rk, q): return rk < int(round(q * len(rk)))

def t1_curve(d, o, sub=None, w=None, fee=0.0, rfc=None, qs=QS):
    """T1 값(q 격자). o: 승인 우선순위(전체 풀), sub: 평가 범위, w: 가중치, fee: 승인 대출 연 관리비, rfc: rf 고정값. 반환 (SR, 평균 x, sd x)."""
    n_all = len(d); sub = np.ones(n_all, bool) if sub is None else sub; w = np.ones(n_all) if w is None else w; w = w * sub
    rf = d.rf.to_numpy() if rfc is None else np.full(n_all, rfc); r = d.r1.to_numpy() - fee
    cs = lambda v: np.concatenate([[0.0], np.cumsum(v[o])])
    k = np.round(qs * n_all).astype(int); W = w.sum(); A1, A2, F1, F2 = (cs(w * v)[k] for v in (r, r * r, rf, rf * rf))
    R1, R2 = (w * rf).sum() - F1, (w * rf * rf).sum() - F2; mx = (A1 + R1) / W; ex2 = (A2 + R2) / W; sd = np.sqrt(np.maximum(ex2 - mx * mx, 0) * W / (W - 1))
    return (mx - (w * rf).sum() / W) / sd, mx, sd
def month_boot(d, B=1000, seed=0):
    """발급월 블록 부트스트랩 재표본: 월별 복원추출 횟수 행렬 cnt(B×월), 대출의 월 위치 mpos, 월 수 nm."""
    months = np.sort(d.mi.unique()); mpos = np.searchsorted(months, d.mi.to_numpy()); nm = len(months)
    idx = np.random.RandomState(seed).randint(0, nm, (B, nm)); return np.stack([np.bincount(i, minlength=nm) for i in idx]).astype(float), mpos, nm
def t1_boot(d, m, cnt, mpos, nm, r=None):
    """T1 형 Sharpe(승인 x = r, 거절 x = rf)의 부트스트랩 분포. r 기본값 = d.r1."""
    rf = d.rf.to_numpy(); x = np.where(m, d.r1.to_numpy() if r is None else r, rf)
    n, s1, s2, sf = (cnt @ v for v in (np.bincount(mpos, None, nm), np.bincount(mpos, x, nm), np.bincount(mpos, x * x, nm), np.bincount(mpos, rf, nm)))
    mx = s1 / n; return (mx - sf / n) / np.sqrt((s2 / n - mx * mx) * n / (n - 1))
def t2_curve(d, o, sub, fee=0.0, rfc=None, qs=QS):
    n_all = len(d); rf = d.rf.to_numpy() if rfc is None else np.full(n_all, rfc); y = (d.r2.to_numpy() - fee - rf) * sub
    k = np.round(qs * n_all).astype(int); N = sub.sum(); s1, s2 = (np.concatenate([[0.0], np.cumsum(v[o])])[k] for v in (y, y * y))
    mu = s1 / N; return mu / np.sqrt(np.maximum(s2 / N - mu * mu, 1e-300))
