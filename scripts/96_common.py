"""96 공통: 새 테이블(train_새테이블.parquet) 전용 입력, 현금흐름 재구성, 안전금리(FEDFUNDS) 지수, 포트폴리오 집계.
입력 계산은 오직 산출물/최종테이블/train_새테이블.parquet. 거시(FEDFUNDS, DGS3/DGS5)는 산출물/raw_macro/ 원자료(국채·무위험).
가정: 상각월 = 마지막 납입월 K + 6, 서비스 수수료 1%(납입×1%), 회수금은 회수수수료 차감 후 상각월에 일시 유입,
      재투자 = 순현금을 실현 FEDFUNDS 월복리 경로로 H(만기)까지 굴림, 미선택 자본은 안전금리."""
import time, json, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "산출물" / "최종테이블" / "train_새테이블.parquet"
MACRO = ROOT / "산출물" / "raw_macro"
OUT = ROOT / "산출물" / "기술IR_v5"
WORK = OUT / "_work"; FIG = OUT / "fig"
for p in (OUT, WORK, FIG): p.mkdir(parents=True, exist_ok=True)
FEE, L_MAIN, SEED = 0.01, 6, 0
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
MON = {m: i for i, m in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])}
def mcode_str(s):
    """'Apr-2020' -> 2005-01 기준 월 코드(int, 결측=-1)."""
    s = pd.Series(s)
    y = pd.to_numeric(s.str[4:8], errors="coerce"); m = s.str[:3].map(MON)
    c = (y - 2005) * 12 + m
    return c.fillna(-1).astype(int).to_numpy()
def safe_index():
    ff = pd.read_csv(MACRO / "FEDFUNDS.csv"); ff.columns = ["d", "v"]
    ff = ff[(ff.d >= "2005-01-01") & (ff.d < "2023-01-01")]
    return np.cumprod(1 + ff.v.to_numpy(float) / 1200.0)
G = safe_index()
def treasury_prev_month(name):
    d = pd.read_csv(MACRO / f"{name}.csv"); d.columns = ["d", "v"]
    d["v"] = pd.to_numeric(d.v, errors="coerce"); d = d.dropna()
    d["m"] = (d.d.str[:4].astype(int) - 2005) * 12 + d.d.str[5:7].astype(int) - 1
    mm = d.groupby("m").v.mean(); mm = mm[(mm.index >= 0) & (mm.index < 216)]
    arr = np.full(12 * 18, np.nan); arr[mm.index.to_numpy()] = mm.to_numpy()
    return arr
def recon(f, rate_pct, term, K, co, TRP, TRI, TLF, REC, CF, W):
    """대출×나이 순현금 행렬 (n×W, 열 0=발행시점). 비례배분·완납 일시금 보정·상각 K+6 (85_cf_common 정의의 재구현)."""
    n = len(f); r = rate_pct / 1200.0
    pmt = f * r / (1 - (1 + r) ** (-term)); j = np.arange(W)[None, :]
    tm, rr, ff, pm = term[:, None], r[:, None], f[:, None], pmt[:, None]
    on = (j >= 1) & (j <= tm)
    b = np.where(on, (pm - ff * rr) * (1 + rr) ** (j - 1), 0.0); a = np.where(on, pm - b, 0.0)
    cb, ca = np.cumsum(b, 1), np.cumsum(a, 1); idx = np.arange(n)
    Kc = np.maximum(K, 1); Kp = np.minimum(Kc, term); m = np.minimum(Kc - 1, term)
    intK, prinK = TRI - ca[idx, m], TRP - cb[idx, m]
    bK = np.where(Kc <= term, b[idx, Kc], 0.0)
    payoff = (~co) & (intK >= 0) & (prinK >= 0)
    prepK = np.maximum(prinK - bK, 0.0); spK = prinK - prepK; Kcc = Kc[:, None]
    int_p = np.where(j <= m[:, None], a, 0.0) + np.where(j == Kcc, intK[:, None], 0.0)
    sp_p = np.where(j <= m[:, None], b, 0.0) + np.where(j == Kcc, spK[:, None], 0.0)
    pp_p = np.where(j == Kcc, prepK[:, None], 0.0)
    inK = (j >= 1) & (j <= Kp[:, None])
    int_q = np.where(inK, TRI[:, None] * a / np.maximum(ca[idx, Kp], 1e-9)[:, None], 0.0)
    sp_q = np.where(inK, TRP[:, None] * b / np.maximum(cb[idx, Kp], 1e-9)[:, None], 0.0)
    pr = (~payoff)[:, None]
    intr, sprin, prepay = np.where(pr, int_q, int_p), np.where(pr, sp_q, sp_p), np.where(pr, 0.0, pp_p)
    Kl = np.where(~payoff, Kp, Kc)[:, None]
    late = np.where((j >= 1) & (j <= Kl), TLF[:, None] / Kl, 0.0)
    atwo = co[:, None] & (j == (K + L_MAIN)[:, None])
    wo = np.where(atwo, (f - TRP)[:, None], 0.0)
    rec = np.where(atwo, REC[:, None], 0.0); cfee = np.where(atwo, CF[:, None], 0.0)
    pay = intr + sprin + prepay + late
    net = pay - FEE * pay + rec - cfee
    return net, wo, sprin + prepay
def irr_monthly(F, it=80):
    """현금흐름 행렬 F(n×W, 열 j = j월, 0월 = 투자)의 월 IRR(이분법, 단일 부호 변화 가정). 받은 돈이 0이면 NaN."""
    F = np.atleast_2d(F); j = np.arange(F.shape[1])[None, :]; n = len(F); ok = F[:, 1:].sum(1) > 0; lo, hi = np.full(n, -0.9999), np.full(n, 1.0)
    for _ in range(it):
        mid = (lo + hi) / 2; pos = (F * (1 + mid[:, None]) ** (-j)).sum(1) > 0; lo = np.where(pos, mid, lo); hi = np.where(pos, hi, mid)
    return np.where(ok, (lo + hi) / 2, np.nan)
def month_str(code): return str(pd.Period("2005-01", freq="M") + int(code))
def ann(x, H): return np.power(np.maximum(x, 1e-12), 12.0 / H) - 1
def meta(name, **kw):
    (OUT / f"{name}.meta.json").write_text(json.dumps(dict(script="scripts/96_*.py", input="산출물/최종테이블/train_새테이블.parquet", seed=SEED,
        created=dt.datetime.now().isoformat(timespec="seconds"), **kw), ensure_ascii=False, indent=1))
