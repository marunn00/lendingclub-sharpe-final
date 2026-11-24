"""98 공통: 앙상블/가공 기법 탐색용. 수익 계산은 96_common(recon)·96_a가 만든 loans.parquet, Sharpe 정의는 96_c_strategies.py의
month_aggs/series/pooled/boot_idx 함수 소스를 AST로 그대로 가져와 재사용(재구현 없음).
Sharpe = 발행월별 초과수익 e_m(a)의 mean/sd (a = 승인율, 발행월 안에서 점수순 금액 누적), 거절 자본 = 만기 일치 국채(36m→DGS3, 60m→DGS5)의 발행 전월(M−1) 월평균 수익(load_loans 에서 TW_short/S 를 이 값으로 구성; 96 원본의 FEDFUNDS 지수는 쓰지 않음)."""
import sys, ast, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import importlib; C = importlib.import_module("96_common")
import numpy as np, pandas as pd
ROOT = C.ROOT
OUT = ROOT / "산출물" / "앙상블_v1"; WORK = OUT / "_work"; FIG = OUT / "fig"
for p in (OUT, WORK, FIG): p.mkdir(parents=True, exist_ok=True)
SEED = 0; log = C.log
LOANS = WORK / "loans_all.parquet"        # 98_e_returns.py 산출: 확정 대출 전체(모든 발행연도·만기)
_ns = {"np": np, "pd": pd, "C": C}
_src = (Path(__file__).parent / "96_c_strategies.py").read_text()
for _n in ast.parse(_src).body:
    if isinstance(_n, ast.FunctionDef) and _n.name in {"month_aggs", "series", "pooled", "boot_idx"}: exec(compile(ast.Module([_n], []), "96_c", "exec"), _ns)
    if isinstance(_n, ast.Assign) and ast.unparse(_n.targets[0]) in {"AG", "NA", "(B, BL)"}: exec(compile(ast.Module([_n], []), "96_c", "exec"), _ns)
AG, month_aggs, series, pooled, boot_idx = _ns["AG"], _ns["month_aggs"], _ns["series"], _ns["pooled"], _ns["boot_idx"]
RATES = {"r30": 29, "r53": 52, "r70": 69, "r90": 89}

def sharpe_eval(d, pr, H=36):
    """d: loans 하위집합(발행월·A·TW_short·NPV_short·S 포함), pr: 우선순위(작을수록 먼저 승인). 반환 (지표 dict, e 행렬 M×100)."""
    ag = month_aggs(d.reset_index(drop=True), np.asarray(pr, float), H)
    e = series(ag); M = e.shape[0]
    sr = e.mean(0) / e.std(0, ddof=1)
    R0, _ = pooled(ag, np.ones((1, M)))
    R0 = R0[0]; ib = int(np.nanargmax(sr))
    m = {}
    for k, i in RATES.items(): m["SR_" + k] = sr[i]; m["XS_" + k] = R0[i]
    m.update(SR_best=sr[ib], XS_best=R0[ib], a_best=AG[ib])
    return m, e

def paired_sr_delta(e_t, e_b, idx_a, nb=1000):
    """월 블록 부트스트랩(96_c의 boot_idx) 쌍체 ΔSharpe @ 승인율 인덱스 idx_a → (평균, 2.5%, 97.5%)."""
    idx = boot_idx(e_t.shape[0], np.random.RandomState(SEED))[:nb]
    f = lambda e: e[idx][:, :, idx_a].mean(1) / e[idx][:, :, idx_a].std(1, ddof=1)
    dl = f(e_t) - f(e_b); return float(dl.mean()), *np.percentile(dl, [2.5, 97.5])

def load_loans():
    """98_e_returns.py 산출(확정 대출 전체, 간결 테이블 확정 행과 같은 순서). 무위험·거절 자금 = 만기 일치 국채(36m→DGS3, 60m→DGS5) 발행월 M−1 월평균.
    NPV_tsy = Σ 순현금·(1+rT)^(−j/12) − 대출액 → 만기 H 종가치 TW_short=(NPV_tsy+A)(1+rT)^(H/12), 안전자산 S=(1+rT)^(H/12) (98_e 에서 계산해 저장)."""
    return pd.read_parquet(LOANS)

YG = {"y0711": (2007, 2011), "y1216": (2012, 2016), "y1720": (2017, 2020)}
def scoped_sharpe(L, pr, fold=None):
    """pr: 전체 L 길이 우선순위(작을수록 먼저 승인; NaN 행 제외). 범위: 만기 36/60 × (전체, 발행연도 3군). 각 범위는 96_c month_aggs/series 그대로(H=만기).
    반환 {scope: (지표 dict, e 행렬)}; scope = '36','60','36_y0711',... 지표에 n(대출 수), months."""
    out = {}; ok = ~np.isnan(pr)
    for t in (36, 60):
        for g, (a, b) in [("", (2007, 2020))] + list(YG.items()):
            m = ok & (L.term.to_numpy() == t) & L.issue_year.between(a, b).to_numpy()
            if m.sum() < 500: continue
            mm, e = sharpe_eval(L[m], pr[m], t); mm.update(n=int(m.sum()), months=int(e.shape[0])); out[f"{t}" + ("_" + g if g else "")] = (mm, e)
    return out

def course_roe(L, pr, rates=(0.10, 0.30, 0.50, 0.53, 0.70, 0.90), groups=True):
    """20_test_split_sim.py 8기 ROE 정의: 대출별 ann=(총수령−투자금)/실제상환기간(년)을 합산 ÷ 투자금 합; 거절 자금은 발행월 3년/5년 국채(tb)로 같은 기간 수익 F·tb.
    승인 = 점수 낮은 순으로 금액(funded) 누적 비율 ≤ a (전체 풀 정렬). 상환기간>0 대출만. 반환 {scope: {r30:%,..., all_in:%, treasury:%, oracle:%}}."""
    out = {}; ok = ~np.isnan(pr) & L.roe_ok.to_numpy()
    for g, (a_, b_) in [("all", (2007, 2020))] + (list(YG.items()) if groups else []):
        m = ok & L.issue_year.between(a_, b_).to_numpy(); d = L[m]; p = pr[m]; o = np.argsort(p, kind="stable"); F = d.A.to_numpy()[o]; ann = d.ann.to_numpy()[o]; inc = F * d.tb.to_numpy()[o]
        cA = np.cumsum(F); tot = cA[-1]; ca, ci = np.concatenate([[0], np.cumsum(ann)]), np.concatenate([[0], np.cumsum(inc)])
        r = {}
        for a in rates:
            k = np.searchsorted(cA / tot, a + 1e-12, side="right"); r[f"r{int(round(a * 100))}"] = (ca[k] + ci[-1] - ci[k]) / tot * 100
        r.update(all_in=ca[-1] / tot * 100, treasury=ci[-1] / tot * 100, oracle=np.maximum(ann, inc).sum() / tot * 100, n=int(m.sum())); out[g] = r
    return out
