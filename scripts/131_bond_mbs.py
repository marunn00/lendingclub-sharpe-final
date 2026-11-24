"""131 채권·MBS 형식 평가: 「채권분석」(2025-2, 조규성) 4·5·6·7.5·10장의 가격·수익률 형식을 대출 자료에 적용. 최종 모델(v3)·승인 규칙은 바꾸지 않는 평가·진단 분석.
기준(실행 전 고정)
 표본: P = 시험 결과 확정 대출 744,608건(121 frame, 과제 정의 잣대와 같은 표본), R = P 중 만기 기준 코호트(36개월 ≤2017-03, 60개월 ≤2015-03 발급),
       H = 시험 파일 전체 1,170,198건(상환 중 포함, 126과 같은 사건 정의: 부도 = 첫 미납월, 완납 = 마지막 납입월, 그 밖 = 마지막 납입월 중도절단).
 승인: P·R은 전략별 과제 정의 잣대 승인 집합(121 q*, 결과 확정 풀 상위 q), H는 126과 같은 시험 전체 상위 q*(최종 모델).
 현금흐름: 96_common.recon(설계상 허용 가정: 서비스 수수료 1%, 상각 = 마지막 납입 + 6개월, 회수금은 회수수수료 차감 후 상각월 유입).
 A 수익률(4.1·4.3·4.4·7.5절)
   약정수익률 = 약정금리(월복리 명목, 강의안의 y×m 관례), 풀 실현 현금흐름 수익률 = 12 × IRR([−Σ대출액, Σ순현금(나이별)…]),
   약정 스프레드 = 금액가중 약정금리 − 금액가중 rf(발급월 만기 맞춤 국채), 실현 스프레드(실현 유효 가산금리) = 풀 수익률 − 금액가중 rf, 보존율 = 실현 ÷ 약정.
   보유기간수익률 HPR = (계약만기 T 시점에서 중간 현금흐름을 발급월 국채금리 rf로 월복리 재투자한 미래가치 ÷ 대출액)^(12/T) − 1.
   새 잣대 「보유기간수익률 Sharpe」 = 과제 정의 잣대에서 승인 대출 수익만 HPR로 바꿈(승인 x = HPR, 거절 x = rf). 승인 집합 = 전략별 q* 그대로.
   판정: 최종 − 부도확률 순위의 발급월 블록 부트스트랩(1,000회) 95% 구간 하한 > 0 이면 우세.
 B MBS 조기상환·부도(10.1절, H): 나이 t별 SMM_t = Σ(t월 만기 전 완납 대출의 예정잔액 MB_t) ÷ Σ(t월 위험집합의 MB_t), CPR = 1 − (1 − SMM)^12, MB_t = 강의안 p.88 원금잔액 식.
   부도 MDR_t = Σ(t월 부도의 MB_{t−1}) ÷ Σ(위험집합 MB_{t−1}), CDR = 1 − (1 − MDR)^12. PSA 속도 k: 위험집합 잔액 가중 최소제곱 CPR_t ≈ k·min(0.2%·t, 6%), t < T.
   조기상환 결정요인(10.1절 ①④⑤): (발급월, 나이, 만기, 등급) 셀의 위험집합·만기 전 완납 건수로 이항 GLM(logit, Pearson 척도).
   설명변수 = 금리 하락 폭 ΔTsy(발급월 국채 − 그 달 국채, 만기 맞춤, %p), 나이 구간, 달력 월(계절), 등급, 만기, 발급연도. 사전 가설: ΔTsy 계수 > 0, 달력 월 결합 유의.
   손실률(ABS 관례, 강의안 밖) = 1 − Σ(회수금 − 회수수수료) ÷ Σ(대출액 − 상환원금), R의 부도 대출.
 C 자동이전증권·순차 CMO(10.2절, R): 전략별 승인 대출·등급별 대출을 나이 기준으로 합친 풀 현금흐름 → 풀 수익률, 스프레드, 실현 WAL = Σ(t/12)·원금_t ÷ Σ원금_t,
   누적 순손실 CNL = Σ(상각 원금 − 순회수) ÷ Σ대출액. 순차 지급 CMO(예시 구조 A 70%·B 20%·C 10%, 36개월 풀): 원금 회수(예정+조기+순회수)를 A→B→C 순 지급,
   순손실(상각 − 순회수)은 C→B→A 순 차감(신용 손실 배분은 민간 ABS 관례, 강의안 CMO는 손실 없는 기관 MBS). 초과 이자(excess spread)는 손실 보전에 쓰지 않음(보수적).
 D 유효 듀레이션·볼록성(5.6·6.4절, 그림 10-1): 36개월 대표 풀(승인·전체)의 기대 현금흐름 모형 = C의 금액가중 금리, B의 나이별 SMM·MDR, A의 손실률(회수 6개월 후).
   가격 P(y) = 기대 현금흐름의 y 할인 현재가치, 기준 y = C의 풀 수익률, ±50bp. 조기상환 SMM_t(Δ) = min(SMM_t·exp(β̂·Δ), 1), Δ = −수익률 변화(%p), β̂ = B의 ΔTsy 계수. 부도 고정.
   ED = (P− − P+) ÷ ((y+ − y−)·P), ECV = (P+ + P− − 2P) ÷ ((0.5(y+ − y−))²·P). 조기상환 고정(β = 0)과 비교, 자동이전증권·IO(이자)·PO(원금·회수) 각각.
 E 같은 시기 시장 스프레드(7.5절): FRED 월평균 Aaa·Baa(Moody's 장기 회사채) − DGS10, 30년 모기지 − DGS10, 은행 24개월 개인대출(분기, 다음 관측까지 유지) − DGS2.
   (1) 발급연도별 위험 사다리: 시장 스프레드 대 LC 약정 스프레드(H: 등급 A·C, 전체, 승인)와 실현 스프레드(C의 36개월 풀: 전체·승인).
   (2) 동조성: 월별 LC 약정 스프레드(H 전체, 금액가중, 월 300건 이상, 2008~2018) 1차 차분을 Baa·모기지 스프레드 1차 차분에 각각 회귀(Newey–West 6).
   (3) 예측: 발급월 36개월 풀(R, 월 300건 이상)의 실현 스프레드를 발급월 Baa 스프레드에 회귀(Newey–West 6), 전체·승인 풀 각각.
데이터 관문: P 744,608건, 최종 모델 승인율 = q*, 과제 정의 Sharpe 재계산 = 121 값, Σ(원금 회수 + 상각) = Σ대출액(상대오차 < 0.1%), H 점수 순서 = 파일 순서.
독립 검수 반영(첫 실행 뒤 수정, 판정 기준은 그대로): 실현 스프레드·보존율은 약정금리와 같은 수수료 전 기준(수수료 후 값 병기), 부도 달은 조기상환 위험집합에서 제외,
 GLM 표준오차는 달력 월 군집(ΔTsy가 달력 월 공통 충격), PSA·평균 CPR은 30개월 이하로도 계산, D는 약정금리 흐름을 수수료 전 풀 수익률로 할인하고 β = 0·+|β̂|(가상 MBS형) 민감도 병기,
 E(2)의 1차 차분은 연속된 달끼리만.
출력: 산출물/최종보고서/표/채권_*.csv, 채권_요약.json"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, json, urllib.request, numpy as np, pandas as pd, polars as pl, scipy.sparse as sp, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import chi2
R = importlib.import_module("121_rulers"); K = R.K; CC = K.C; log = K.log
OUT = CC.ROOT / "산출물" / "최종보고서"; TB = OUT / "표"; TEST = CC.ROOT / "lending_club_2020_test.csv"
SUMM = {}

# ---- E 시장 금리(FRED 원자료는 raw_macro 에 보관하고 출처를 manifest 에 기록)
FRED = {"AAA": "Moody's Aaa 회사채 수익률 (월)", "BAA": "Moody's Baa 회사채 수익률 (월)", "MORTGAGE30US": "30년 고정 모기지 금리 (주)",
        "TERMCBPER24NS": "상업은행 24개월 개인대출 금리 (분기)", "DGS2": "2년 국채수익률 (일)"}
for s, desc in FRED.items():
    p = CC.MACRO / f"{s}.csv"
    if not p.exists():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"; urllib.request.urlretrieve(url, p); t = pd.read_csv(p).dropna()
        with open(CC.MACRO / "_manifest.csv", "a", encoding="utf-8") as fh: fh.write(f"{s},{desc},국가,{url},200,성공,{t.iloc[0, 0]},{t.iloc[-1, 0]},{len(t)},131에서 수집\n")
mk = lambda n: pd.Series(R.tb_month(n)).ffill().to_numpy()                       # 2005-01 기준 월평균(소수), 분기 자료는 다음 관측까지 유지
MKT = {"Aaa 회사채": mk("AAA") - mk("DGS10"), "Baa 회사채": mk("BAA") - mk("DGS10"), "30년 모기지": mk("MORTGAGE30US") - mk("DGS10"), "24개월 개인대출": mk("TERMCBPER24NS") - mk("DGS2")}

# ---- P: 결과 확정 대출 × 점수 × 승인 집합
d = R.frame(TEST)
ex = pl.read_csv(TEST, columns=["id", "int_rate", "collection_recovery_fee", "last_pymnt_d", "grade"], infer_schema_length=0, null_values=["", "NA"]).to_pandas(); ex["id"] = ex.id.astype(str)
d = d.merge(ex, on="id", validate="1:1")
for f_, c in (("test_scores_v3.parquet", ["score_V3"]), ("../테스트평가/test_scores_v2.parquet", ["score_M6v2"]), ("../테스트평가/test_scores.parquet", ["score_B0"])):
    d = d.merge(pd.read_parquet(OUT / f_)[["id"] + c], on="id", validate="1:1")
assert len(d) == 744608
d["rate"] = pd.to_numeric(d.int_rate.str.strip().str.replace("%", ""), errors="coerce") / 100; d["cfee"] = pd.to_numeric(d.collection_recovery_fee, errors="coerce").fillna(0.0)
lp = CC.mcode_str(d.last_pymnt_d); d["K"] = np.where(lp >= 0, np.maximum(lp - d.mi.to_numpy(), 0), 0); assert d.rate.notna().all()
MAIN = pd.read_csv(TB / "주요결과_T1_T2.csv"); QT1 = MAIN[MAIN.ruler == "T1"].set_index("strategy")
SC = {"V3": d.score_V3.to_numpy(), "M6v2": d.score_M6v2.to_numpy(), "B0": d.score_B0.to_numpy(), "GRADE": R.grade_score(d), "ALL": np.zeros(len(d))}
ORD = {k: R.order_rank(v) for k, v in SC.items()}; APP = {k: R.sel(ORD[k][1], float(QT1.loc[k, "q"])) for k in SC}
for k in SC:                                                                     # 관문: 121 과 같은 승인 집합·같은 값
    sr = R.t1_curve(d, ORD[k][0], qs=np.array([QT1.loc[k, "q"]]))[0][0]; assert abs(sr - QT1.loc[k, "SR_test"]) < 1e-9, (k, sr)
log("관문 통과: P", len(d), "최종 승인율", round(APP["V3"].mean(), 4))

# ---- A·C: recon 현금흐름 → 대출 HPR, 풀(전략·등급 × 만기 × 발급월) 나이별 합계
GRP = {"V3": APP["V3"], "V3_거절": ~APP["V3"], "M6v2": APP["M6v2"], "B0": APP["B0"], "GRADE": APP["GRADE"], "ALL": APP["ALL"]}
GRP.update({f"등급{g}": (d.grade == g).to_numpy() for g in "ABCDEFG"})
ref, Tm, mi, f, rf = d.ref.to_numpy(), d["T"].to_numpy(), d.mi.to_numpy(), d.funded_amnt.to_numpy(float), d.rf.to_numpy()
Kk, co = d.K.to_numpy(), d.dflt.to_numpy(); TRP, TRI, TLF, REC, CFE = (d[c].to_numpy(float) for c in ("total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "cfee"))
WM = int(Kk.max()) + CC.L_MAIN + 1; G = len(GRP); gk = list(GRP)
ACC = {a: np.zeros((G, 2, 216, WM)) for a in ("net", "prin", "wo", "rec")}; SCL = {a: np.zeros((G, 2, 216)) for a in ("n", "f", "frf", "frate")}
HPR = np.zeros(len(d)); tot_prin = tot_wo = 0.0
o = np.argsort(Kk, kind="stable")
for s0 in range(0, len(o), 25000):
    s = o[s0:s0 + 25000]; W = int(Kk[s].max()) + CC.L_MAIN + 1
    net, wo, prin = CC.recon(f[s], d.rate.to_numpy()[s] * 100, Tm[s], Kk[s], co[s], TRP[s], TRI[s], TLF[s], REC[s], CFE[s], W)
    j = np.arange(W)[None, :]; rec = np.where(co[s][:, None] & (j == (Kk[s] + CC.L_MAIN)[:, None]), (REC[s] - CFE[s])[:, None], 0.0)
    HPR[s] = (np.maximum((net * (1 + rf[s][:, None] / 12) ** (Tm[s][:, None] - j)).sum(1), 0) / f[s]) ** (12 / Tm[s]) - 1
    tot_prin += prin.sum(); tot_wo += wo.sum()
    for gi, g in enumerate(gk):
        for ti, t in enumerate((36, 60)):
            m = GRP[g][s] & ref[s] & (Tm[s] == t); nm_ = int(m.sum())
            if nm_ == 0: continue
            Gm = sp.csr_matrix((np.ones(nm_), (mi[s][m], np.arange(nm_))), shape=(216, nm_))               # 발급월 합계(희소 행렬 곱)
            for a, M in (("net", net), ("prin", prin), ("wo", wo), ("rec", rec)): ACC[a][gi, ti][:, :W] += Gm @ M[m]
            for a, v in (("n", np.ones(nm_)), ("f", f[s][m]), ("frf", f[s][m] * rf[s][m]), ("frate", f[s][m] * d.rate.to_numpy()[s][m])): SCL[a][gi, ti] += np.bincount(mi[s][m], v, 216)
gap = abs(tot_prin + tot_wo - f.sum()) / f.sum(); log("Σ(원금+상각)/Σ대출액 − 1 =", gap); assert gap < 1e-3

def pool(gi, ti, months):
    """풀 지표: 나이별 합계를 발급월 집합 months 에서 더한 뒤 계산."""
    S = lambda a: ACC[a][gi, ti][months].sum(0); n, F, FRF, FR = (SCL[a][gi, ti][months].sum() for a in ("n", "f", "frf", "frate"))
    net, prin, wo, rec = S("net"), S("prin"), S("wo"), S("rec"); gross = (net - rec) / (1 - CC.FEE) + rec      # 수수료 전 = 납입액 + 순회수(약정금리와 같은 기준)
    y, yg = (12 * float(CC.irr_monthly(np.r_[x[0] - F, x[1:]])[0]) for x in (net, gross)); t = np.arange(len(prin)) / 12
    return dict(n=int(n), 대출액=F, 약정금리=FR / F, rf=FRF / F, 풀수익률_전=yg, 풀수익률_후=y, 약정스프레드=FR / F - FRF / F, 실현스프레드_전=yg - FRF / F, 실현스프레드_후=y - FRF / F,
                WAL=float((t * prin).sum() / prin.sum()), CNL=float((wo.sum() - rec.sum()) / F)), (prin, wo, rec)
ALLM = np.arange(216)
rows = []
for gi, g in enumerate(gk):
    for ti, t in enumerate((36, 60)):
        if SCL["n"][gi, ti].sum() == 0: continue
        r_, _ = pool(gi, ti, ALLM); rows.append(dict(풀=g, 만기=t, **r_))
POOL = pd.DataFrame(rows); POOL["보존율"] = POOL.실현스프레드_전 / POOL.약정스프레드; POOL.to_csv(TB / "채권_풀.csv", index=False, encoding="utf-8-sig"); print(POOL.round(4).to_string())
yr_m = lambda y: np.arange((y - 2005) * 12, (y - 2004) * 12)
POOLY = pd.DataFrame([dict(풀=g, 발급연도=y, **pool(gk.index(g), 0, yr_m(y))[0]) for g in ("ALL", "V3") for y in range(2009, 2018) if SCL["n"][gk.index(g), 0][yr_m(y)].sum() > 0])
POOLY.to_csv(TB / "채권_풀_연도별.csv", index=False, encoding="utf-8-sig")
POOLM = pd.DataFrame([dict(풀=g, mi=m, **pool(gk.index(g), 0, [m])[0]) for g in ("ALL", "V3") for m in range(216) if SCL["n"][gk.index(g), 0][m] >= 300])

# ---- A 새 잣대: 보유기간수익률 Sharpe
cnt, mpos, nm = R.month_boot(d); dH = d.assign(r1=HPR); HR = []; BT = {}
for k in SC:
    q = float(QT1.loc[k, "q"]); HR.append(dict(strategy=k, q=q, T1=float(QT1.loc[k, "SR_test"]), HPR_SR=float(R.t1_curve(dH, ORD[k][0], qs=np.array([q]))[0][0]),
                                               승인_HPR평균=float(HPR[APP[k]].mean()), 승인_초과HPR=float((HPR - rf)[APP[k]].mean())))
    BT[k] = R.t1_boot(d, APP[k], cnt, mpos, nm, r=HPR)
HR = pd.DataFrame(HR).set_index("strategy")
for k in SC:
    lo_, hi_ = np.percentile(BT[k], [2.5, 97.5]); HR.loc[k, "SR_lo"], HR.loc[k, "SR_hi"] = lo_, hi_
    if k != "V3": dd = BT["V3"] - BT[k]; HR.loc[k, "d_V3"], HR.loc[k, "d_lo"], HR.loc[k, "d_hi"] = HR.loc["V3", "HPR_SR"] - HR.loc[k, "HPR_SR"], *np.percentile(dd, [2.5, 97.5])
HR.to_csv(TB / "채권_HPR잣대.csv", encoding="utf-8-sig"); print(HR.round(4).to_string())

# ---- A 손실률(R 부도 대출)
sev = lambda m: float(1 - (REC - CFE)[m & co & ref].sum() / (f - TRP)[m & co & ref].sum())
SEV = {"V3": sev(APP["V3"] & (Tm == 36)), "ALL": sev(Tm == 36)}; SUMM["손실률_36"] = SEV

# ---- B: H 전체(상환 중 포함) 나이별 SMM·MDR, PSA, 조기상환 GLM
h = pl.read_csv(TEST, columns=["id", "issue_d", "term", "loan_status", "last_pymnt_d", "int_rate", "funded_amnt", "grade"], infer_schema_length=0).filter(pl.col("issue_d").is_not_null()).to_pandas()
S3 = pd.read_parquet(OUT / "test_scores_v3.parquet")[["id", "score_V3"]]; assert (S3.id.to_numpy() == h.id.astype(str).to_numpy()).all()
qv = float(pd.read_csv(TB / "v3_판정.csv").set_index("model").loc["V3", "q_star"]); ap = np.zeros(len(h), bool); ap[np.argsort(S3.score_V3.to_numpy(), kind="stable")[:int(round(qv * len(h)))]] = True
hmi = CC.mcode_str(h.issue_d); hlp = CC.mcode_str(h.last_pymnt_d); mm_ = np.where(hlp >= 0, np.maximum(hlp - hmi, 0), 0); st = h.loan_status.fillna("")
ev = np.where(st.str.contains("Charged Off|Default"), 1, np.where(st.str.contains("Fully Paid"), 2, 0)); tt = np.where(ev == 1, mm_ + 1, np.where(ev == 2, np.maximum(mm_, 1), mm_)).astype(int)
hT = h.term.str.strip().str[:2].astype(int).to_numpy(); hr = pd.to_numeric(h.int_rate.str.strip().str.replace("%", ""), errors="coerce").to_numpy() / 1200; hf = pd.to_numeric(h.funded_amnt).to_numpy(float); assert not np.isnan(hr).any()
gi_ = h.grade.map({g: i for i, g in enumerate("ABCDEFG")}).to_numpy(); assert not np.isnan(gi_.astype(float)).any(); gi_ = gi_.astype(int)
cell = ((hT == 60).astype(int) * 2 + ap.astype(int)) * 7 + gi_                    # 만기 × 승인 × 등급 = 28칸
AGE = np.arange(1, 61)[None, :]; RP, EP, RD, ED_ = (np.zeros((28, 60)) for _ in range(4)); NC = 216 * 61 * 2 * 7; GR_N, GR_E = np.zeros(NC), np.zeros(NC)
for s0 in range(0, len(h), 100000):
    s = slice(s0, s0 + 100000); r, nT, ttc, evc = hr[s][:, None], hT[s][:, None], tt[s][:, None], ev[s][:, None]
    mb = lambda k: hf[s][:, None] * np.where(k <= nT, (1 - (1 + r) ** (k - nT)) / (1 - (1 + r) ** (-nT)), 0.0)     # 강의안 p.88 원금잔액
    MBt, MBt1 = mb(AGE), mb(AGE - 1)
    rp = (AGE <= ttc) & (AGE < nT) & ~((evc == 1) & (AGE == ttc)); ep = rp & (evc == 2) & (AGE == ttc); rd = (AGE <= ttc) & (AGE <= nT); edm = rd & (evc == 1) & (AGE == ttc)   # 부도 달에는 조기상환할 수 없음
    Gm = sp.csr_matrix((np.ones(len(hf[s])), (cell[s], np.arange(len(hf[s])))), shape=(28, len(hf[s])))
    RP += Gm @ (MBt * rp); EP += Gm @ (MBt * ep); RD += Gm @ (MBt1 * rd); ED_ += Gm @ (MBt1 * edm)
    base = ((hmi[s] * 61)[:, None] + AGE) * 14 + ((hT[s] == 60).astype(int) * 7 + gi_[s])[:, None]
    GR_N += np.bincount(base[rp], minlength=NC); GR_E += np.bincount(base[ep], minlength=NC)
def curve(ix):
    smm, mdr = EP[ix].sum(0) / np.maximum(RP[ix].sum(0), 1e-12), ED_[ix].sum(0) / np.maximum(RD[ix].sum(0), 1e-12); return smm, mdr, RP[ix].sum(0)
CELLS = {(t, a): [((t == 60) * 2 + a) * 7 + g for g in range(7)] for t in (36, 60) for a in (0, 1)}
GROUPS = {f"{t}_{n}": ix for t in (36, 60) for n, ix in (("전체", CELLS[(t, 0)] + CELLS[(t, 1)]), ("승인", CELLS[(t, 1)]), ("거절", CELLS[(t, 0)]))}
GROUPS.update({f"{t}_등급{g}": [((t == 60) * 2 + a) * 7 + gi for a in (0, 1)] for t in (36, 60) for gi, g in enumerate("ABCDEFG")})
CUR, PSA = [], []
for nmg, ix in GROUPS.items():
    t = int(nmg[:2]); smm, mdr, w = curve(ix); cpr, cdr = 1 - (1 - smm) ** 12, 1 - (1 - mdr) ** 12; age = np.arange(1, 61); ok = age < t; psa = np.minimum(0.002 * age, 0.06)
    k = float((w * cpr * psa)[ok].sum() / (w * psa * psa)[ok].sum())
    CUR += [dict(집단=nmg, 만기=t, 나이=a, SMM=smm[a - 1], CPR=cpr[a - 1], MDR=mdr[a - 1], CDR=cdr[a - 1], 위험잔액=w[a - 1]) for a in range(1, t + 1)]
    o30 = ok & (age <= 30); k30 = float((w * cpr * psa)[o30].sum() / (w * psa * psa)[o30].sum())          # 만기 직전 기록 차이를 뺀 점검
    PSA.append(dict(집단=nmg, 만기=t, PSA속도=k, 평균CPR=float((w * cpr)[ok].sum() / w[ok].sum()), 평균CDR=float((w * cdr)[age <= t].sum() / w[age <= t].sum()),
                    PSA속도_30=k30, 평균CPR_30=float((w * cpr)[o30].sum() / w[o30].sum())))
CURV = pd.DataFrame(CUR); CURV.to_csv(TB / "채권_CPR.csv", index=False, encoding="utf-8-sig"); PSAT = pd.DataFrame(PSA); PSAT.to_csv(TB / "채권_PSA.csv", index=False, encoding="utf-8-sig"); print(PSAT.round(4).to_string())
nz = np.nonzero(GR_N)[0]; ix_ = nz; g7 = ix_ % 7; tl = (ix_ // 7) % 2; age_ = (ix_ // 14) % 61; imi = ix_ // (14 * 61)
tsy = np.where(tl == 1, R.t5[np.minimum(imi + age_, 215)], R.t3[np.minimum(imi + age_, 215)]); tsy0 = np.where(tl == 1, R.t5[imi], R.t3[imi])
CEL = pd.DataFrame(dict(n=GR_N[nz], e=GR_E[nz], age=age_, dtsy=(tsy0 - tsy) * 100, moy=(imi + age_) % 12 + 1, grade=np.array(list("ABCDEFG"))[g7], term=np.where(tl == 1, 60, 36), iy=2005 + imi // 12, cm=imi + age_))
CEL = CEL[imi + age_ <= 215]; CEL["prop"] = CEL.e / CEL.n; CEL["agebin"] = pd.cut(CEL.age, [0, 3, 6, 12, 18, 24, 36, 60]).astype(str)
GLM = smf.glm("prop ~ C(agebin) + dtsy + C(moy) + C(grade) + C(term) + C(iy)", data=CEL, family=sm.families.Binomial(), var_weights=CEL.n).fit(scale="X2")
WT = GLM.wald_test_terms(skip_single=False, scalar=True).table
# 달력 월 군집 샌드위치 표준오차: ΔTsy·계절은 달력 월의 공통 충격이라 셀 독립 가정(Pearson 척도)은 표준오차를 과소평가
X = GLM.model.exog; mu = np.asarray(GLM.mu); nn, ee = CEL.n.to_numpy(), CEL.e.to_numpy()
Ai = np.linalg.inv((X * (nn * mu * (1 - mu))[:, None]).T @ X); Sg = pd.DataFrame(X * (ee - nn * mu)[:, None]).groupby(CEL.cm.to_numpy()).sum().to_numpy(); Gc = len(Sg)
VC = Ai @ (Sg.T @ Sg) @ Ai * Gc / (Gc - 1); SEC = pd.Series(np.sqrt(np.diag(VC)), index=GLM.params.index)
im = [i for i, nme in enumerate(GLM.params.index) if "moy" in nme]; bm = GLM.params.to_numpy()[im]; chi_cl = float(bm @ np.linalg.solve(VC[np.ix_(im, im)], bm))
GT = pd.DataFrame(dict(계수=GLM.params, 표준오차=GLM.bse, z=GLM.tvalues, 군집표준오차=SEC, 군집z=GLM.params / SEC)); GT.to_csv(TB / "채권_GLM.csv", encoding="utf-8-sig")
SUMM["GLM"] = dict(beta_dtsy=float(GLM.params["dtsy"]), se_dtsy=float(GLM.bse["dtsy"]), z_dtsy=float(GLM.tvalues["dtsy"]), se_cl_dtsy=float(SEC["dtsy"]), z_cl_dtsy=float(GLM.params["dtsy"] / SEC["dtsy"]),
                   군집=int(Gc), 계절_chi2_cl=chi_cl, 계절_p_cl=float(chi2.sf(chi_cl, len(im))), 셀=int(len(CEL)), 위험월=float(CEL.n.sum()), 조기상환=float(CEL.e.sum()),
                   scale=float(GLM.scale), 계절_chi2=float(WT.loc["C(moy)", "statistic"]), 계절_df=int(WT.loc["C(moy)", "df_constraint"]), 계절_p=float(WT.loc["C(moy)", "pvalue"]),
                   나이=GLM.params.filter(like="agebin").to_dict(), 달력월=GLM.params.filter(like="moy").to_dict())
print(GT.loc[["dtsy"] + [i for i in GT.index if "agebin" in i or "moy" in i]].round(4).to_string())

# ---- C 순차 지급 CMO(36개월 풀, 예시 구조)
def cmo(prin, wo, rec, F, split=(0.7, 0.2, 0.1)):
    bal = np.array(split) * F; orig = bal.copy(); paid = np.zeros((3, len(prin))); loss = np.zeros(3)
    for jj in range(len(prin)):
        c = prin[jj] + rec[jj]; l = wo[jj] - rec[jj]
        for i in range(3):
            x = min(c, bal[i]); bal[i] -= x; paid[i, jj] += x; c -= x
        for i in (2, 1, 0):
            x = min(max(l, 0.0), bal[i]); bal[i] -= x; loss[i] += x; l -= x
    t = np.arange(len(prin)) / 12; return [dict(트랜치=nmx, 비중=split[i], 손실률=loss[i] / orig[i], WAL=float((t * paid[i]).sum() / max(paid[i].sum(), 1e-9))) for i, nmx in enumerate("ABC")]
CM = []
for g in ("V3", "B0", "ALL"):
    r_, (pr_, wo_, rc_) = pool(gk.index(g), 0, ALLM); CM += [dict(풀=g, CNL=r_["CNL"], **x) for x in cmo(pr_, wo_, rc_, r_["대출액"])]
CMO = pd.DataFrame(CM); CMO.to_csv(TB / "채권_CMO.csv", index=False, encoding="utf-8-sig"); print(CMO.round(4).to_string())

# ---- D 유효 듀레이션·볼록성(36개월 대표 풀)
def flows(c, smm, mdr, sv, n=36, lag=CC.L_MAIN):
    B = 1.0; W = n + lag + 1; I, PO = np.zeros(W), np.zeros(W); rr = c / 12
    for t in range(1, n + 1):
        Dd = B * mdr[t - 1]; B1 = B - Dd; PO[t + lag] += (1 - sv) * Dd; I[t] = rr * B1
        sch = B1 * rr / ((1 + rr) ** (n - t + 1) - 1); Bs = B1 - sch; pre = Bs * smm[t - 1]; PO[t] += sch + pre; B = Bs - pre
    return I, PO
price = lambda cf, y: float((cf * (1 + y / 12) ** (-np.arange(len(cf)))).sum())
DUR, PYR, PP = [], [], {}; b_ = SUMM["GLM"]["beta_dtsy"]
CASES = (("추정 반응(β̂)", b_), ("고정(β=0)", 0.0), ("가상 MBS형(β=+|β̂|)", abs(b_)))           # β̂는 연관 추정치라 부호 가정별 민감도를 함께 보고
def secs(c, smm, mdr, sv, beta, dy):
    I, PO = flows(c, np.minimum(smm[:36] * np.exp(beta * (-dy * 100)), 1.0), mdr[:36], sv); return {"자동이전증권": I + PO, "IO": I, "PO": PO}
for g, hg in (("V3", "36_승인"), ("ALL", "36_전체")):
    pr_ = POOL[(POOL.풀 == g) & (POOL.만기 == 36)].iloc[0]; c, y0 = float(pr_.약정금리), float(pr_.풀수익률_전)     # 약정금리(수수료 전) 흐름 ↔ 수수료 전 풀 수익률로 할인
    smm, mdr, _ = curve(GROUPS[hg]); sv = SEV[g]
    for case, beta in CASES:
        P = {dy: {k: price(v, y0 + dy) for k, v in secs(c, smm, mdr, sv, beta, dy).items()} for dy in (-0.005, 0.0, 0.005)}; PP[(g, case)] = P
        for sec in ("자동이전증권", "IO", "PO"):
            pm, p0, pp = P[-0.005][sec], P[0.0][sec], P[0.005][sec]
            DUR.append(dict(풀=g, 경우=case, 증권=sec, 기준수익률=y0, 가격=p0, ED=(pm - pp) / (0.01 * p0), ECV=(pp + pm - 2 * p0) / (0.005 ** 2 * p0)))
        if g == "V3": PYR += [dict(풀=g, 수익률변화=dy, 경우=case, **{k: price(v, y0 + dy) for k, v in secs(c, smm, mdr, sv, beta, dy).items()}) for dy in np.round(np.arange(-0.03, 0.0301, 0.0025), 4)]
DURT = pd.DataFrame(DUR); DURT.to_csv(TB / "채권_듀레이션.csv", index=False, encoding="utf-8-sig"); print(DURT.round(3).to_string())
pd.DataFrame(PYR).to_csv(TB / "채권_가격곡선.csv", index=False, encoding="utf-8-sig")
SUMM["가격차_50bp"] = {g: max(abs(PP[(g, CASES[0][0])][dy]["자동이전증권"] - PP[(g, CASES[1][0])][dy]["자동이전증권"]) for dy in (-0.005, 0.005)) / PP[(g, CASES[1][0])][0.0]["자동이전증권"] for g in ("V3", "ALL")}

# ---- E 시장 스프레드: 사다리·동조성·예측
hrf = np.where(hT == 36, R.t3[hmi], R.t5[hmi]); hs = hr * 12 - hrf; hy = 2005 + hmi // 12
def wavg(m, v=hs): return float((hf * v)[m].sum() / hf[m].sum()) if m.any() else np.nan
LAD = []
for y in range(2009, 2018):
    ym = hy == y; mo = yr_m(y); row = dict(발급연도=y, **{k: float(np.nanmean(v[mo])) for k, v in MKT.items()})
    row.update({"LC 약정 A등급": wavg(ym & (gi_ == 0)), "LC 약정 C등급": wavg(ym & (gi_ == 2)), "LC 약정 전체": wavg(ym), "LC 약정 승인": wavg(ym & ap)})
    for g, lab in (("ALL", "LC 실현 전체(36개월)"), ("V3", "LC 실현 승인(36개월)")):
        s_ = POOLY[(POOLY.풀 == g) & (POOLY.발급연도 == y)]; row[lab] = float(s_.실현스프레드_전.iloc[0]) if len(s_) else np.nan
    LAD.append(row)
LADT = pd.DataFrame(LAD); LADT.to_csv(TB / "채권_사다리.csv", index=False, encoding="utf-8-sig"); print(LADT.round(4).to_string())
mon = pd.DataFrame(dict(mi=hmi, w=hf, ws=hf * hs)).groupby("mi").agg(n=("w", "size"), w=("w", "sum"), ws=("ws", "sum")); mon = mon[(mon.n >= 300) & (mon.index >= 36) & (mon.index < 168)]
mon["lc"] = mon.ws / mon.w * 100
for k in MKT: mon[k] = MKT[k][mon.index] * 100
REG = []
def nw(y, X, lab, kind):
    r_ = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 6}); REG.append(dict(회귀=kind, 설명변수=lab, 관측=int(r_.nobs), 계수=float(r_.params[1]), 표준오차=float(r_.bse[1]), t=float(r_.tvalues[1]), R2=float(r_.rsquared)))
dm = mon.diff()[np.r_[False, np.diff(mon.index.to_numpy()) == 1]]                            # 연속된 달끼리만 차분
for k in ("Baa 회사채", "30년 모기지"): nw(dm.lc.to_numpy(), dm[k].to_numpy(), k, "동조성: Δ LC 약정 스프레드")
for g in ("ALL", "V3"):
    pm_ = POOLM[POOLM.풀 == g]; nw(pm_.실현스프레드_전.to_numpy() * 100, MKT["Baa 회사채"][pm_.mi.to_numpy()] * 100, "Baa 회사채", f"예측: 발급월 실현 스프레드({'전체' if g == 'ALL' else '승인'} 풀)")
REGT = pd.DataFrame(REG); REGT.to_csv(TB / "채권_시장회귀.csv", index=False, encoding="utf-8-sig"); print(REGT.round(4).to_string())
SUMM.update(HPR=HR.round(6).reset_index().to_dict("records"), 표본=dict(P=len(d), R=int(ref.sum()), H=len(h), 월_동조성=int(len(dm)), 월_예측=int((POOLM.풀 == "ALL").sum())))
json.dump(SUMM, open(TB / "채권_요약.json", "w"), ensure_ascii=False, indent=1, default=float); log("done")
