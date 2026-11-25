"""논문형 보고서 3.3 「1계조건 규칙의 성과」와 3.4 「같은 잣대 비교와 강건성」(절 모듈).
수치는 산출물/최종보고서 표·json, 산출물/우수사례평가, 산출물/테스트평가의 저장 결과에서 읽는다."""
import json, re
import numpy as np
import pandas as pd


def build(P):
    T, O = P.tables, P.out
    BP, TP = P.root / "산출물" / "우수사례평가", P.root / "산출물" / "테스트평가" / "표"
    rd = pd.read_csv
    js = lambda p: json.load(open(p, encoding="utf-8"))
    MAIN = rd(T / "주요결과_T1_T2.csv"); CI = rd(T / "구간.csv"); ROB = rd(T / "강건성.csv").set_index("strategy")
    REP = js(O / "결과_T1_T2.json"); CAL = js(T / "보정검정.json"); PS = js(T / "교수_요약.json"); FJ = js(T / "FOC_요약.json")
    PR = rd(T / "팀보고서_인쇄값.csv"); O15 = rd(T / "OOT_2015.csv"); O16 = rd(T / "OOT_2016.csv"); CUR = rd(T / "곡선_T1_T2.csv")
    GR = rd(T / "등급구성_final.csv"); CIF = rd(T / "누적부도_CIF.csv"); WVT = rd(T / "누적부도_발급연도내.csv"); CZ = rd(T / "중도절단_연도별.csv").set_index("발급연도")
    HRT = rd(T / "채권_HPR잣대.csv").set_index("strategy"); FCR = rd(T / "랩_FOC_반복.csv"); MO = rd(T / "교수_D_모먼트.csv")
    LP = rd(BP / "이수용/표/정책표.csv").set_index("정책"); LC = rd(BP / "이수용/표/신뢰구간.csv"); LB = rd(BP / "이수용/표/만기_연도별.csv").set_index("구분")
    U1 = rd(BP / "표/우1_혼동행렬_금액위험.csv"); U3 = rd(BP / "표/우3_요약.csv"); U3W = rd(BP / "표/우3_seed승률.csv").set_index("rule")
    KP = rd(TP / "운영지점_요약.csv"); PAIR = rd(TP / "쌍체차이.csv")

    # ---------- 값 꺼내기와 표기
    def mv(k, c="SR_test", ru="T1"): return float(MAIN[(MAIN.ruler == ru) & (MAIN.strategy == k)][c].iloc[0])
    def ci(ru, k, c): return float(CI[(CI.ruler == ru) & (CI.strategy == k)][c].iloc[0])
    def pv(r, k): return float(PR[(PR.잣대 == r) & (PR.항목 == k)].값.iloc[0])
    def pr_row(r, k): return PR[(PR.잣대 == r) & (PR.항목 == k)].iloc[0]
    def oot(D, k, c="SR"): return float(D[D.strategy == k][c].iloc[0])
    def u3(m, c="시험SR_중앙"): return float(U3[(U3.rule == "R1") & (U3.model == m)][c].iloc[0])
    def u1(pt, m): return float(U1[U1.지점.str.startswith(pt) & U1.모델.str.startswith(m)].total_risk.iloc[0])
    def kp(k, c, a=0.25): return float(KP[(KP.strategy == k) & np.isclose(KP.rate, a)][c].iloc[0])
    def pair(n, c, a=0.25): return float(PAIR[(PAIR["compare"] == n) & np.isclose(PAIR.rate, a)][c].iloc[0])
    def lc(p, c, rf="국채"): return float(LC[(LC.rf == rf) & (LC.정책 == p)][c].iloc[0])
    def cif(t, g, m): return float(CIF[(CIF.만기 == t) & (CIF.집단 == g) & (CIF.개월 == m)].누적부도.iloc[0])
    def cn(t, g): return float(CIF[(CIF.만기 == t) & (CIF.집단 == g) & (CIF.개월 == 0)].n.iloc[0])
    wvv = lambda y, g: float(WVT[(WVT.발급연도 == y) & (WVT.집단 == g)].누적부도_36개월.iloc[0])
    f3 = lambda x: f"{round(x, 3):.3f}".replace("-", "−").replace("−0.000", "0.000")
    f4 = lambda x: f"{x:.4f}".replace("-", "−")
    f2 = lambda x: f"{x:.2f}".replace("-", "−")
    def d(x, n=3):
        r = round(x, n)
        return f"{0:.{n}f}" if r == 0 else ("+" if r > 0 else "△") + f"{abs(r):.{n}f}"
    pc = lambda x, n=1: f"{x * 100:.{n}f}%".replace("-", "−")
    num = lambda x: f"{int(round(x)):,}"
    usd = lambda x: f"${x / 1e6:,.0f}M"

    # ---------- 3.3에서 쓰는 핵심 값
    SR = mv("V3"); SR_LO, SR_HI = ci("T1", "V3", "SR_lo"), ci("T1", "V3", "SR_hi")
    LN_LO, LN_HI = ci("T1_대출부트스트랩", "V3", "SR_lo"), ci("T1_대출부트스트랩", "V3", "SR_hi")
    DEFF = (SR_HI - SR_LO) / (LN_HI - LN_LO)
    S8 = pv("T1", "제안 모형(S8)")
    QV = float(MAIN[(MAIN.ruler == "T1") & (MAIN.strategy == "V3")].q.iloc[0]); QPS = f"{QV * 100:.0f}%"
    APR, APD = mv("V3", "승인율_test"), mv("V3", "승인부도율_test")
    ARET, AVOL = mv("V3", "연수익률_test"), mv("V3", "변동성_test")
    ORC, ORC_SH = REP["T1_oracle_test"], REP["oracle_share"]

    cur = lambda k: CUR[(CUR.strategy == k) & (CUR.q < 1)].set_index("q").T1_test
    cv3, cb0, cgr = cur("V3"), cur("B0"), cur("GRADE")
    NQ, NQW = len(cv3), int(((cv3 > cb0) & (cv3 > cgr)).sum())
    TOPQ = cv3[cv3 >= cv3.max() - 0.005].index

    fo = FCR
    FOC_D = float((fo.SR_foc - fo.SR_rank26).median()); FOC_W = float((fo.SR_foc > fo.SR_rank26).mean())
    FOC_DQ = float((fo.SR_foc - fo.SR_rank_qmatch).median()); FOC_WQ = float((fo.SR_foc > fo.SR_rank_qmatch).mean())
    FOC_SR, RANK_SR, RANK_Q_SR = FJ["시험_SR_foc"], FJ["시험_SR_v3_q26"], FJ["시험_SR_순위_같은승인율"]
    FOC_Q, FOC_DEF, RANK_DEF = FJ["시험_승인율_foc"], FJ["시험_부도율_foc"], FJ["시험_부도율_v3_q26"]
    DIFF, DIFF_LO, DIFF_HI = FJ["차이_foc_v3"], *FJ["차이_구간"]
    DIFF_Q, DIFF_Q_LO, DIFF_Q_HI = FJ["차이_foc_순위_같은승인율"], *FJ["차이_같은승인율_구간"]
    LAM, TH_TR, TH_TE, JAC = FJ["lam_star"], FJ["학습_이론_lam"], FJ["시험_이론_lam"], FJ["자카드"]

    M = {k: mv(k) for k in ("ALL", "GRADE", "B0", "M5", "M6", "M6v2", "V3")}
    MON = {k: float(ROB.loc[k, "T1_월별할당"]) for k in ("V3", "M6v2", "B0", "GRADE")}
    FIX, FIXQ = float(ROB.loc["V3", "T1_고정점수선"]), float(ROB.loc["V3", "고정점수선_승인율"])
    YA = lambda k, g="2017–20": float(ROB.loc[k, f"승인율_{g}"])

    # ---------- 3.4에서 쓰는 핵심 값
    OOT15 = oot(O15[O15.ruler == "T2"], "V3"); OOT16 = oot(O16[O16.ruler == "T1"], "V3", "SR_승인6_4")
    M4 = pv("T2", "M4_TwoStage(주력)"); SR2 = mv("V3", ru="T2"); NREF = REP["n_ref"]
    RAROC = pv("이수용_국채", "RAROC 모델(최종)"); LSY20 = LP.loc["우리 방식 20%", "국채"]
    raroc = pr_row("이수용_국채", "RAROC 모델(최종)")
    RAROC_APR = float(raroc.승인율); RAROC_DEF = float(re.search(r"부도 ([\d.]+)%", raroc.비고).group(1)) / 100
    RAROC_RET = float(re.search(r"연수익 ([\d.]+)%", raroc.비고).group(1)) / 100
    EVAL_N = int(re.search(r"평가 ([\d,]+)건", raroc.비고).group(1).replace(",", ""))
    LYRS = [i for i in LB.index if str(i).isdigit()]
    LSYRF = sum(LP.loc["우리 방식 20%", c] > LP.loc["A등급만", c] for c in ("rf 0%", "rf 1%", "rf 2%", "rf 3%", "국채"))
    LSYYR = sum(LB.loc[y, "우리 방식 25%"] > LB.loc[y, "A등급만"] for y in LYRS)
    T2V = [("만기 기준 코호트(36개월 ≤2017.3, 60개월 ≤2015.3)", "1_만기기준(36≤17-03, 60≤15-03)", "T2", "M4_TwoStage(주력)"),
           ("순수만기 코호트(둘 다 ≤2015.3)", "2_순수만기(둘 다 ≤15-03)", "T2_2_순수만기", "M4_TwoStage"),
           ("완결 코호트(둘 다 ≤2014.12)", "3_완결(둘 다 ≤14-12)", "T2_3_완결", "M4_TwoStage"),
           ("평탄 코호트(둘 다 ≤2017.12)", "4_평탄(둘 다 ≤17-12)", "T2_4_평탄", "M4_TwoStage")]
    T2V += [(f"무위험수익률 {x}% 고정", f"rf{x}", f"T2_rf{x}", "M4_TwoStage") for x in range(4)]
    T2V += [("연 1% 관리비 차감", "관리비1pct", "T2_관리비1pct", "M4_TwoStage")]
    T2WIN = sum(ROB.loc["V3", "T2_" + c] > pv(r, k) for _, c, r, k in T2V)
    rep_rows = [("과제 정의", "전부 승인 Sharpe", pv("T1", "전체 승인"), REP["T1_all_test"]),
                ("과제 정의", "완전 예측(Oracle) Sharpe", pv("T1", "Oracle(부도 완전 예측)"), ORC),
                ("과제 정의", "시험 부도율", pv("T1", "시험 부도율"), REP["default_test"]),
                ("과제 정의", "평균 국채금리", pv("T1", "평균 만기매칭 국채금리"), REP["mean_rf_test"]),
                ("과제 정의(시간 외)", "2017~20 전부 승인", pv("T1_OOT_2017-2020", "전체 승인"), oot(O16, "ALL")),
                ("기회풀", "만기 기준 코호트 전부 승인", pv("T2", "전부 승인"), REP["T2_all_ref"]),
                ("기회풀", "순수만기 코호트 전부 승인", pv("T2_2_순수만기", "전부 승인"), ROB.loc["ALL", "T2_2_순수만기(둘 다 ≤15-03)"]),
                ("기회풀", "완결 코호트 전부 승인", pv("T2_3_완결", "전부 승인"), ROB.loc["ALL", "T2_3_완결(둘 다 ≤14-12)"]),
                ("기회풀", "평탄 코호트 전부 승인", pv("T2_4_평탄", "전부 승인"), ROB.loc["ALL", "T2_4_평탄(둘 다 ≤17-12)"])]
    rep_rows += [("기회풀", f"rf {x}% 전부 승인", pv(f"T2_rf{x}", "전부 승인"), ROB.loc["ALL", f"T2_rf{x}"]) for x in range(4)]
    rep_rows += [("기회풀", "관리비 1% 전부 승인", pv("T2_관리비1pct", "전부 승인"), ROB.loc["ALL", "T2_관리비1pct"]),
                 ("기회풀(시간 외)", "2016~17 전부 승인", pv("T2_OOT_2016-2017", "전부 승인"), oot(O15[O15.ruler == "T2"], "ALL")),
                 ("이수용", "전부 승인 Sharpe(국채)", pv("이수용_국채", "전부 승인"), LP.loc["전부 승인", "국채"]),
                 ("이수용", "A등급만 Sharpe(국채)", pv("이수용_국채", "A등급만"), LP.loc["A등급만", "국채"])]
    MAXGAP = max(abs(a - b) for g, _, a, b in rep_rows if not g.startswith("이수용"))
    LAB_GAP = [b - a for g, _, a, b in rep_rows if g.startswith("이수용")]

    # 아홉 개 잣대
    HV, HB = HRT.loc["V3"], HRT.loc["B0"]
    SCORE = [
        ("과제 정의 포트폴리오 Sharpe", "Sharpe 보고서 2장", f"시험 자료 {num(REP['n_test'])}건", f"Sharpe 보고서 제안 {f3(S8)}",
         f"<b>{f3(SR)}</b><br><small>[{f3(SR_LO)}, {f3(SR_HI)}]</small>", d(SR - S8), "동등"),
        ("기회풀 Sharpe(만기 코호트)", "M4 보고서 2장", f"만기 코호트 {num(NREF)}건", f"M4 주력 {f3(M4)}",
         f"<b>{f3(SR2)}</b><br><small>[{f3(ci('T2', 'V3', 'SR_lo'))}, {f3(ci('T2', 'V3', 'SR_hi'))}]</small>", d(SR2 - M4), "우세"),
        ("대출 단위 Sharpe(국채 rf)", "이수용 보고서 5장", f"만기 지난 대출 {num(LP.loc['전부 승인', '승인건수'])}건", f"이수용 RAROC {f3(RAROC)}",
         f"<b>{f3(LSY20)}</b><br><small>[{f3(lc('우리 방식 20%', 'SR_lo'))}, {f3(lc('우리 방식 20%', 'SR_hi'))}]</small>", d(LSY20 - RAROC), "우세"),
        ("대출 IRR Sharpe(100회 반복)", "우수사례 3", "무작위 20% 표본 × 100회", f"부도확률 순위 {f3(u3('B0'))}<br><small>(본 연구 재채점)</small>",
         f"<b>{f3(u3('V3'))}</b><br><small>우세 반복 R0 {int(U3W.loc['R0', 'V3>B0_seed수'])}·R1 {int(U3W.loc['R1', 'V3>B0_seed수'])}·R2 {int(U3W.loc['R2', 'V3>B0_seed수'])}/100</small>", d(u3("V3") - u3("B0")), "혼재"),
        ("금액 위험(Total Risk)", "우수사례 1", f"시험 자료 {num(REP['n_test'])}건", f"부도확률 순위 {usd(u1('가', '부도확률'))}<br><small>(본 연구 재채점)</small>",
         f"<b>{usd(u1('가', '최종'))}</b><br><small>거절 비율 = 부도율 지점 {usd(u1('나', '최종'))}</small>", f"{usd(u1('가', '부도확률') - u1('가', '최종'))} 감소", "혼재"),
        ("발급월 현금흐름 Sharpe(36개월)", "본 연구 운영 지점", f"시험 자료 {num(REP['n_test'])}건", f"부도확률 순위 {f2(kp('B0', 'SR36_test'))}",
         f"<b>{f2(kp('V3', 'SR36_test'))}</b><br><small>차이 [{f2(pair('V3 − B0', 'dSR36_lo'))}, {f2(pair('V3 − B0', 'dSR36_hi'))}]</small>", d(kp("V3", "SR36_test") - kp("B0", "SR36_test"), 2), "우세"),
        ("보유기간수익률 Sharpe(국채 재투자)", "채권분석 강의안 4.3절", f"시험 자료 {num(REP['n_test'])}건", f"부도확률 순위 {f3(HB.HPR_SR)}<br><small>(본 연구 재채점)</small>",
         f"<b>{f3(HV.HPR_SR)}</b><br><small>[{f3(HV.SR_lo)}, {f3(HV.SR_hi)}]</small>", d(HB.d_V3), "우세" if HB.d_lo > 0 else "동등"),
        ("시간 외: 기회풀 Sharpe 2016~17", "M4 보고서 시간 외", f"{num(oot(O15[O15.ruler == 'T2'], 'V3', 'n'))}건(2015년까지 학습)",
         f"M4 보고서 최고 {f3(pv('T2_OOT_2016-2017', 'M2_PD_Decile_Mix(승인 30%)'))}", f"<b>{f3(OOT15)}</b>", d(OOT15 - pv("T2_OOT_2016-2017", "M2_PD_Decile_Mix(승인 30%)")), "우세"),
        ("시간 외: 과제 정의 Sharpe 2017~20", "Sharpe 보고서 6.4절", f"{num(oot(O16, 'V3', 'n'))}건(2016년까지 학습)",
         f"Sharpe 보고서 제안 {f3(pv('T1_OOT_2017-2020', '제안 모형'))}<br><small>(승인 6.4%)</small>", f"<b>{f3(OOT16)}</b><br><small>(같은 승인율 6.4%)</small>",
         d(OOT16 - pv("T1_OOT_2017-2020", "제안 모형")), "열세"),
    ]
    TAG = {w: sum(s[-1] == w for s in SCORE) for w in ("우세", "동등", "혼재", "열세")}
    TAG_TXT = "·".join(f"{TAG[w]}개 {w}" for w in ("우세", "동등", "혼재", "열세") if TAG[w])

    # 강건성
    MOM = PS["모먼트_상관"]
    m_sd = MO[(MO.모먼트 == "표준편차") & (MO.분위 == 1)].iloc[0]
    rb_cols = [("T1", "기본"), ("T1_금액가중", "금액 가중"), ("T1_관리비1pct", "관리비 1%"), ("T1_rf0", "rf 0%"), ("T1_rf3", "rf 3%"),
               ("T1_36m", "36개월"), ("T1_60m", "60개월"), ("T1_2007–11", "2007~11"), ("T1_2012–16", "2012~16"), ("T1_2017–20", "2017~20")]
    rb_order_ok = all(ROB.loc["V3", c] > ROB.loc["B0", c] > ROB.loc["ALL", c] for c, _ in rb_cols)
    OOT_O15T2 = O15[O15.ruler == "T2"]; OOT_O16T1 = O16[O16.ruler == "T1"]
    M4_OOT = [pv("T2_OOT_2016-2017", "M4_MeanVar_Util(자유 임계값)"), pv("T2_OOT_2016-2017", "M4_TwoStage(자유 임계값)"), pv("T2_OOT_2016-2017", "M2_PD_Decile_Mix(승인 30%)")]
    OOT_RATE = {"2018": CZ.loc[2018, "결과확정비율"], "2019": CZ.loc[2019, "결과확정비율"]}
    N36, N60 = cn(36, "전체 신청"), cn(60, "전체 신청")
    CAL_SLOPE, CAL_SE = CAL["slope"], CAL["slope_se"]
    CAL_Z = (CAL_SLOPE - 1) / CAL_SE

    H = []
    # ================= 3.3
    H.append(P.h3("3.3 1계조건 규칙의 성과"))
    H.append(P.p(f"본 절은 2.2절에서 정의한 1계조건 규칙과 비교 규칙들의 시험 자료 성과를 보고한다. 최종 모형은 학습 자료의 교차검증에서 Sharpe를 가장 높게 준 승인율 {QPS}를 채택하여, 점수 상위 {QPS}의 대출을 승인하고 나머지 자금은 만기 맞춤 국채에 두었다. 여기서 Sharpe는 포트폴리오 수익률 x의 평균에서 국채금리 평균({pc(REP['mean_rf_test'], 2)})을 뺀 값을 x의 표준편차로 나눈 값이다. 시험 자료에서 이 전략의 Sharpe는 {f3(SR)}이고, 발급월 블록 부트스트랩으로 구한 95% 구간은 [{f3(SR_LO)}, {f3(SR_HI)}]이다."))
    H.append(P.p("이하에서는 승인율을 바꿔 가며 본 성과 곡선, 1계조건 규칙을 직접 구현한 결과, 예측 단계별 기여, 승인 포트폴리오의 구성, 발급 시기 집중의 순서로 살핀다."))

    H.append(P.h4("가. 승인율에 따른 성과"))
    H.append(P.p(f"[[f:ru_curve]]은 승인율을 0%에서 100%까지 바꿔 가며 시험 자료의 Sharpe를 그린 것이다. 승인율 100% 미만의 격자 {NQ}개 지점 가운데 {'모든 지점' if NQW == NQ else str(NQW) + '개 지점'}에서 최종 모형이 부도확률 순위와 세부등급 순서보다 높았다. 최댓값과의 차이가 0.005 이내인 승인율은 {TOPQ.min() * 100:.1f}~{TOPQ.max() * 100:.1f}% 구간에 모여 있으며, 학습 자료가 고른 {QPS}는 이 구간 안에 있다."))
    H.append(P.p(f"세부등급 순서의 최고 Sharpe는 {f3(cgr.max())}이며 0을 넘지 못하였다. 그 최적 지점은 승인율 {cgr.idxmax() * 100:.1f}%, 즉 거의 전부를 거절하는 지점이었다. 등급 정보만으로 대출을 고르면 국채를 이기기 어렵다는 뜻이며, 이는 Sharpe 보고서의 결론과 같다."))
    H.append(P.fig("ru_curve", "fig02_승인율곡선.png", "승인율별 시험 자료 Sharpe. 빈 원은 Sharpe 보고서에 보고된 값이고, 점선은 학습 자료 교차검증 곡선이다."))

    H.append(P.h4("나. 1계조건 규칙의 직접 구현과 비율 순위 비교"))
    H.append(P.p(f"1계조건 규칙은 위험회피 계수 λ에 따라 승인 집합이 달라진다. λ가 커지면 위험에 민감해져 승인이 줄고, 작아지면 기대 이득만 보고 승인이 늘어난다. 비율 순위 규칙은 이 곡선 위의 한 점(승인율 26%)을 점수 순위로 정하는 방식이다. λ는 격자 160개(0.05~500)에서 학습 자료 5겹 교차검증의 Sharpe를 최대로 하는 값을 골랐고, 그 값은 {LAM:.2f}이었다. 같은 지점에서 이론값 SR/(2σ<sub>p</sub>)은 학습 자료 기준 {TH_TR:.2f}이며, 선택값보다 {(TH_TR / LAM - 1) * 100:.0f}% 컸다([[t:ru_lambda]], [[f:ru_foccurve]])."))
    H.append(P.tab(label="ru_lambda", caption="위험회피 계수 λ의 선택값과 이론값", head=["구분", "λ", "이론값 SR/(2σ<sub>p</sub>)", "이론값 ÷ λ"],
                   rows=[["학습 자료 교차검증 선택(λ*)", f"{LAM:.2f}", f"{TH_TR:.2f}", f"{TH_TR / LAM:.2f}배"],
                         ["시험 자료 기준 이론값", "—", f"{TH_TE:.2f}", "—"],
                         ["소표본 20회의 중앙값", f"{fo.lam.median():.2f}", f"{fo.lam_theory.median():.2f}", "—"]],
                   widths=["40%", "15%", "25%", "20%"]))
    H.append(P.p(f"λ의 이론값이 선택값보다 큰 원인은 두 가지로 볼 수 있다. 첫째, 이론은 E[r]과 Var(r)을 정확히 안다고 가정하는데 예측 오차가 있으면 이득 추정치가 과장되어 같은 승인 집합을 얻으려면 더 작은 λ가 필요할 수 있다. 둘째, 대출 간 공분산을 0으로 둔 근사가 σ<sub>p</sub> 추정에 영향을 준다. 두 원인의 크기는 직접 검증하지 않았다. 그래서 λ는 고정값이 아니라 학습 교차검증으로 정하였고, 이론값은 보조 비교치로만 제시한다."))
    H.append(P.p(f"[[t:ru_foc]]에서 1계조건 규칙의 시험 자료 Sharpe는 {f4(FOC_SR)}, 비율 순위 26%는 {f4(RANK_SR)}이다. 차이는 {d(DIFF, 4)}이고 발급월 군집 구간 [{d(DIFF_LO, 4)}, {d(DIFF_HI, 4)}]은 0을 포함한다. 같은 승인율로 맞춘 비율 순위와 비교해도 차이는 {d(DIFF_Q, 4)} [{d(DIFF_Q_LO, 4)}, {d(DIFF_Q_HI, 4)}]이다. 두 규칙은 통계적으로 구분되지 않는다. 1계조건 규칙은 승인율을 {pc(FOC_Q)}까지 줄여 승인 대출의 부도율을 {pc(FOC_DEF)}까지 낮추었고, 비율 순위는 26.0%를 승인하면서 부도율 {pc(RANK_DEF)}를 보였다. 두 규칙이 승인한 집합의 자카드 지수는 {JAC:.2f}로 대부분 겹친다."))
    H.append(P.tab(label="ru_foc", caption="1계조건 규칙과 비율 순위의 시험 자료 비교", head=["항목", "1계조건 규칙(FOC)", "비율 순위 26%", "비율 순위(같은 승인율)"],
                   rows=[["학습 자료 교차검증 Sharpe · 승인율", f"{f3(FJ['학습_SR_foc'])} · {pc(FJ['학습_승인율_foc'])}", f"{f3(FJ['학습_SR_v3_q26'])} · 26.0%", "—"],
                         (["시험 자료 Sharpe", f"<b>{f4(FOC_SR)}</b>", f4(RANK_SR), f4(RANK_Q_SR)], "hl"),
                         ["시험 자료 승인율 · 승인 대출 부도율", f"{pc(FOC_Q)} · {pc(FOC_DEF)}", f"26.0% · {pc(RANK_DEF)}", f"{pc(FOC_Q)} · —"],
                         ["Sharpe 차이 [95% 구간, 발급월 군집]", "—", f"{d(DIFF, 4)} [{d(DIFF_LO, 4)}, {d(DIFF_HI, 4)}]", f"{d(DIFF_Q, 4)} [{d(DIFF_Q_LO, 4)}, {d(DIFF_Q_HI, 4)}]"],
                         ["승인 집합 겹침(자카드 지수)", f"{JAC:.2f}", "—", "—"]],
                   widths=["34%", "22%", "22%", "22%"], foot="λ*와 승인율 26%는 모두 학습 자료에서 정하였으며, 시험 자료에는 이 값을 한 번 적용하였다."))
    H.append(P.fig("ru_foccurve", "lab4_FOC곡선.png", "λ에 따른 학습 자료 교차검증 Sharpe(왼쪽)와 시험 자료 승인율(오른쪽). 왼쪽 곡선이 λ*의 선택 근거이며, 점선은 이론값 SR/(2σ<sub>p</sub>)이다.", h="64mm"))
    H.append(P.p(f"[[t:ru_small]]은 점검 자료에서 4,000건 안팎의 소표본을 20회 반복한 결과다. 소표본에서는 경량 버전의 예측(HistGB 2단계)을 썼다. FOC는 같은 승인율의 비율 순위보다 {pc(FOC_WQ, 0)}의 반복에서 높았고 중앙값 차이는 {d(FOC_DQ)}이다. 26% 비율 순위와 비교하면 FOC가 높은 반복은 {pc(FOC_W, 0)}이고, 중앙값 차이는 {d(FOC_D)}이다. 두 비교는 방향만 시사할 뿐, 표본 규모를 감안하면 우열을 가리기 어렵다."))
    H.append(P.tab(label="ru_small", caption="소표본 20회 반복에서 1계조건 규칙과 비율 순위 비교", head=["항목", "1계조건 규칙(FOC)", "비율 순위 26%", "비율 순위(같은 승인율)"],
                   rows=[["Sharpe 중앙값", f3(fo.SR_foc.median()), f3(fo.SR_rank26.median()), f3(fo.SR_rank_qmatch.median())],
                         ["승인율 중앙값", pc(fo.q_foc.median()), "26.0%", pc(fo.q_foc.median())],
                         ["FOC − 해당 규칙(반복별 차이의 중앙값)", "—", d(FOC_D), d(FOC_DQ)],
                         ["FOC가 더 높았던 반복 비율", "—", pc(FOC_W, 0), pc(FOC_WQ, 0)]],
                   widths=["38%", "20%", "20%", "22%"]))

    H.append(P.h4("다. 예측 단계별 기여"))
    LAD = [("ALL", "전부 승인"), ("GRADE", "세부등급 순서"), ("B0", "부도확률 순위"), ("M5", "+ 조건부 수익(1차 모먼트)"),
           ("M6", "+ 위험(2차 모먼트) = v1"), ("M6v2", "+ 부도확률 3모형 평균 = v2"), ("V3", "+ 학습 목표 정렬 = 최종 모형 v3")]
    rows, prev = [], None
    for k, lab in LAD:
        v = M[k]
        row = [lab, f"{mv(k, 'q') * 100:.1f}", f3(v), d(v - prev) if prev is not None else "—",
               "—" if k == "V3" else f"{d(ci('T1', k, 'd_final'))} [{d(ci('T1', k, 'd_lo'))}, {d(ci('T1', k, 'd_hi'))}]"]
        rows.append((row, "hl" if k == "V3" else "")); prev = v
    H.append(P.p(f"[[t:ru_step]]은 예측 구조를 한 단계씩 더했을 때 Sharpe가 어떻게 바뀌는지 보여준다. 가장 큰 개선은 부도확률 순위에서 조건부 기대 초과수익 순위로 옮길 때 나타났다({d(M['M5'] - M['B0'])}). 위험(2차 모먼트)을 더한 v1의 추가 효과는 {d(M['M6'] - M['M5'])}에 그쳐 작았고, 세 부도확률 모형의 평균(v2, {d(M['M6v2'] - M['M6'])})과 학습 목표를 과제 잣대 수익에 맞춘 최종 모형({d(M['V3'] - M['M6v2'])})이 그 뒤를 이었다. 단계 사이의 차이는 발급월 군집 쌍체 부트스트랩 구간과 함께 제시하였다."))
    H.append(P.tab(label="ru_step", caption="단계별 과제 정의 Sharpe(시험 자료)", head=["단계", "승인율(%)", "Sharpe", "직전 단계 대비", "최종 모형 − 해당 단계 [95% 구간]"],
                   rows=rows, widths=["30%", "11%", "11%", "15%", "33%"],
                   foot="구간은 발급월 블록 부트스트랩 1,000회의 쌍체 차이이다."))

    H.append(P.h4("라. 승인 포트폴리오의 구성"))
    PF = [("ALL", "전부 승인"), ("GRADE", "세부등급 순서"), ("B0", "부도확률 순위"), ("M6v2", "v2"), ("V3", "최종 모형 v3")]
    H.append(P.p(f"[[t:ru_main]]은 주요 전략의 포트폴리오 특성을 정리한 것이다. 최종 모형은 승인율 {pc(APR)}에서 승인 대출의 부도율 {pc(APD)}, 연 수익률 {pc(ARET, 2)}, 변동성 {pc(AVOL, 2)}를 보였다. 전부 승인의 연 수익률은 {pc(mv('ALL', '연수익률_test'), 2)}, 부도율은 {pc(REP['default_test'])}이었다. 부도 대출만 거절하는 완전 예측 기준선은 Sharpe {f3(ORC)}이며, 승인율은 {pc(ORC_SH)}이다. 이 기준선은 도달할 수 없는 이론 상한이다."))
    main_rows = [([lab, pc(mv(k, "승인율_test")), pc(mv(k, "연수익률_test"), 2), pc(mv(k, "변동성_test"), 2), pc(mv(k, "승인부도율_test")), f3(mv(k)),
                   f"[{f3(ci('T1', k, 'SR_lo'))}, {f3(ci('T1', k, 'SR_hi'))}]"], "hl" if k == "V3" else "") for k, lab in PF]
    main_rows.append((["완전 예측 기준선(부도 아닌 대출만 승인)", pc(ORC_SH), "—", "—", "0.0%", f3(ORC), "이론 상한"], "ref"))
    H.append(P.tab(label="ru_main", caption="전략별 포트폴리오 특성과 과제 정의 Sharpe(시험 자료)",
                   head=["전략", "승인율", "연 수익률", "변동성", "승인 대출 부도율", "Sharpe", "95% 구간(발급월 군집)"], rows=main_rows,
                   widths=["28%", "10%", "12%", "11%", "13%", "9%", "17%"],
                   foot=f"연 수익률과 변동성은 포트폴리오 수익률 x의 평균과 표준편차이며, 국채 평균은 {pc(REP['mean_rf_test'], 2)}이다."))
    H.append(P.p(f"승인 Sharpe의 발급월 군집 구간 [{f3(SR_LO)}, {f3(SR_HI)}]은 대출 단위 구간 [{f3(LN_LO)}, {f3(LN_HI)}]보다 {DEFF:.1f}배 넓다. 같은 발급월의 대출은 금리와 경기 충격을 공유하므로, 대출을 독립으로 보고 계산한 구간은 불확실성을 과소평가한다."))
    GR_i = GR.set_index("등급"); GRR = (GR.승인부도율 / GR.등급부도율).dropna()
    H.append(P.p(f"등급별로는 A등급의 {pc(GR_i.loc['A', '승인율'], 0)}, B등급의 {pc(GR_i.loc['B', '승인율'], 0)}를 승인하고 C등급 이하에서는 일부만 고른다. 승인 대출의 부도율은 등급 전체 부도율의 {GRR.min() * 100:.0f}~{GRR.max() * 100:.0f}% 수준이었다(A~F등급). G등급은 승인이 100건에 못 미쳐 승인 부도율을 따로 보고하지 않았다."))

    H.append(P.h4("마. 발급 시기 집중과 매달 같은 비율 승인"))
    H.append(P.p(f"승인 규칙은 시험 자료 전체에서 점수 상위 {QPS}를 고르므로, 점수가 높은 시기에 발급된 대출이 많이 승인되고 낮은 시기의 대출은 거의 승인되지 않는다. [[t:ru_issue]]에 따르면 최종 모형은 2017~20년 발급분의 {pc(YA('V3'))}만 승인하였다. v2는 {pc(YA('M6v2'))}, 부도확률 순위는 {pc(YA('B0'))}를 승인하였다. 학습 자료의 교차검증 점수선을 그대로 적용하면(승인율 {pc(FIXQ)}) Sharpe는 {f3(FIX)}으로 기본 규칙과 같아, 이 시기 선택은 발급 시점에 구현할 수 있는 규칙이다."))
    H.append(P.p(f"발급월과 만기마다 같은 비율을 승인하는 방식으로 바꾸면 최종 모형의 Sharpe는 {f3(MON['V3'])}까지 낮아진다. 그래도 v2({f3(MON['M6v2'])}), 부도확률 순위({f3(MON['B0'])}), 세부등급 순서({f3(MON['GRADE'])})보다는 높아서 같은 달 안의 선별력은 유지된다. 기본 규칙과의 차이({d(SR - MON['V3'])})는 발급 시기 선택의 효과로 해석한다. 다만 이 해석은 과제 잣대가 계약만기로 연율화되어 조기 종료 대출의 수익을 낮게 잡는 특성의 영향을 받을 수 있다. 2017년 이후 발급분은 결과가 확정된 대출이 적어 이 영향이 특히 클 것으로 보인다."))
    VK = [("V3", "최종 모형 v3"), ("M6v2", "v2"), ("B0", "부도확률 순위"), ("GRADE", "세부등급 순서")]
    issue_rows = [([lab, pc(ROB.loc[k, "승인율_2007–11"]), pc(ROB.loc[k, "승인율_2012–16"]), pc(YA(k)), f3(mv(k)),
                    f"{f3(ROB.loc[k, 'T1_고정점수선'])}<br><small>(승인 {pc(ROB.loc[k, '고정점수선_승인율'])})</small>", f3(MON[k])], "hl" if k == "V3" else "")
                   for k, lab in VK]
    issue_rows.append((["전부 승인", "100.0%", "100.0%", "100.0%", f3(M["ALL"]), f3(M["ALL"]), f3(M["ALL"])], "ref"))
    H.append(P.tab(label="ru_issue", caption="발급연도군별 승인율과 승인 규칙별 Sharpe(시험 자료)",
                   head=["전략", "승인율 2007~11", "승인율 2012~16", "승인율 2017~20", "기본 규칙 Sharpe", "학습 점수선 고정 Sharpe", "매달 같은 비율 Sharpe"],
                   rows=issue_rows, widths=["18%", "12%", "12%", "12%", "13%", "17%", "16%"],
                   foot="기본 규칙은 전체 시험 자료에서 상위 q를 고른다. 학습 점수선 고정은 학습 교차검증 점수의 상위 q 경계를 시험 자료에 그대로 적용한 값이다."))

    # ================= 3.4
    H.append(P.h3("3.4 같은 잣대 비교와 강건성"))
    H.append(P.p("다른 보고서의 숫자와 비교하려면 기준선이 같은 정의로 계산되어야 한다. 이 절은 먼저 기준선의 정의를 맞춘 뒤, 아홉 개 잣대에서의 결과를 종합하고, 잣대별 세부 결과와 강건성 분석을 제시한다."))

    H.append(P.h4("가. 기준선의 정의 비교"))
    H.append(P.p(f"[[t:ru_base]]는 전부 승인, 완전 예측, 시험 부도율, 국채금리, 시간 외 구간, 기회풀 코호트 기준선을 같은 정의로 다시 계산한 값과 원보고서에 보고된 값을 나란히 놓은 것이다. 과제 정의와 기회풀 잣대의 기준선은 보고값과의 차이가 모두 {MAXGAP:.3f} 이하였다. 이수용 잣대만 다시 계산한 값이 보고값보다 {min(LAB_GAP):.3f}~{max(LAB_GAP):.3f} 높았다. 평가 대상 건수도 원보고서({num(EVAL_N)}건)보다 {num(LP.loc['전부 승인', '승인건수'] - EVAL_N)}건 많다. 이 차이만으로 값의 차이를 설명하지 못하였고 원인은 확인하지 못했다. 그러므로 이수용 잣대는 근사적 비교로 보고하며, 우열은 같은 표본 안에서 최종 모형과 A등급만 규칙을 비교하여 판단한다."))
    base_rows = []
    for g, lab, a, b in rep_rows:
        rate = ("부도율" in lab) or ("금리" in lab)
        fmt = (lambda x: pc(x, 2)) if rate else f3
        base_rows.append([g, lab, fmt(a), fmt(b), d(b - a, 4 if rate else 3)])
    H.append(P.tab(label="ru_base", caption="기준선의 보고값과 같은 정의 계산값 비교", head=["잣대", "항목", "보고값", "같은 정의 계산값", "차이"],
                   rows=base_rows, widths=["18%", "34%", "16%", "16%", "16%"], foot="보고값은 Sharpe 보고서, M4 보고서, 이수용 보고서에 실린 값이다."))

    H.append(P.h4("나. 아홉 개 잣대의 종합"))
    H.append(P.p(f"[[t:ru_scorecard]]는 아홉 개 잣대에서 최종 모형을 보고된 비교 모형과 견준 결과다. 전체는 {TAG_TXT}이다. 과제 정의 Sharpe는 Sharpe 보고서의 제안 모형과 {f3(abs(SR - S8))} 차이로 구간 안에서 동등하였다. 기회풀, 이수용, 발급월 현금흐름, 보유기간수익률, 시간 외 2016~17년 구간에서는 우세하였다. IRR과 금액 위험 기준은 혼재로 판정되었는데, 손실 처리 방식이나 거절 비율 지점에 따라 우열이 바뀌기 때문이다. 시간 외 2017~20년 구간은 열세다."))
    H.append(P.tab(label="ru_scorecard", caption="아홉 개 잣대에서 최종 모형의 성과",
                   head=["잣대", "정의 보고서", "표본", "비교 대상", "최종 모형 [95% 구간]", "차이", "판정"],
                   rows=[list(s) for s in SCORE], widths=["18%", "13%", "15%", "16%", "15%", "11%", "8%"],
                   foot="R0·R1·R2는 IRR 계산 불가 처리 규칙이다(R0 원 발표 규칙, R1 계산 불가를 −100%로 처리, R2 음의 IRR을 초과수익 0으로 처리). 구간은 발급월 블록 부트스트랩 또는 쌍체 차이의 구간이다."))
    H.append(P.p(f"시간 외 2017~20년 구간의 열세는 이 구간의 결과 확정 대출이 조기 종료 위주이고, 평균 국채금리가 {pc(float(O16.iloc[0]['평균rf']))}으로 올라 대출의 초과수익이 낮았던 국면이라는 점과 맞물려 있을 것으로 추정한다. 원인을 분리하여 검증하지는 않았다."))

    H.append(P.h4("다. 잣대별 상세"))
    H.append(P.p(f"[[t:ru_cohort]]는 기회풀 Sharpe를 코호트 정의, 무위험수익률, 관리비 가정을 바꿔 계산한 결과다. 이 아홉 가지 변형 가운데 {T2WIN}개에서 최종 모형이 M4 주력 모형보다 높았다. 무위험수익률이 3%이면 최종 모형은 {f3(ROB.loc['V3', 'T2_rf3'])}, M4 주력 모형은 {f3(pv('T2_rf3', 'M4_TwoStage'))}로 두 모형 모두 음수가 되지만 최종 모형의 손실이 작다. 만기 코호트 안에서 최종 모형의 승인 비율은 {pc(mv('V3', '코호트승인율_test', 'T2'))}였다."))
    coh_rows = []
    for lab, c, r, k in T2V:
        col = "T2_" + c
        coh_rows.append([lab, f3(ROB.loc["ALL", col]), f3(pv(r, k)), f3(ROB.loc["V3", col]), d(ROB.loc["V3", col] - pv(r, k))])
    H.append(P.tab(label="ru_cohort", caption="기회풀 Sharpe의 코호트·무위험수익률·비용 변형",
                   head=["변형", "전부 승인(같은 정의 계산)", "M4 주력 보고값", "최종 모형", "차이(최종 − M4)"], rows=coh_rows,
                   widths=["36%", "16%", "16%", "16%", "16%"]))
    H.append(P.p(f"[[t:ru_policy]]는 만기가 지난 대출만을 대상으로 한 이수용 방식의 대출 단위 Sharpe다. 이수용 보고서에서는 A등급만 고르는 규칙이 RAROC 모델을 이겼다. 최종 모형(승인율 20%)은 무위험수익률 다섯 수준 {'모두에서' if LSYRF == 5 else str(LSYRF) + '개 수준에서'} A등급만 규칙보다 높았고, 발급연도 {len(LYRS)}개 해 {'모두에서' if LSYYR == len(LYRS) else str(LSYYR) + '개 해에서'} 승인율 25% 기준으로 우세하였다. 국채 기준 차이는 {d(lc('우리 방식 20%', '−A등급만'))} [{d(lc('우리 방식 20%', '−A등급만_lo'))}, {d(lc('우리 방식 20%', '−A등급만_hi'))}]이다."))
    pol = [("전부 승인", "전부 승인"), ("A등급만", "A등급만"), ("부도확률 낮은 순 20%", "부도확률 낮은 순 20%"), ("v2 20%", "v2(승인율 20%)"),
           ("우리 방식 20%", "최종 모형(승인율 20%)"), ("우리 방식 25%", "최종 모형(승인율 25%)")]
    pol_rows = [([lab, pc(LP.loc[p, "승인율"]), pc(LP.loc[p, "부도율"]), pc(LP.loc[p, "연수익률"], 2)] + [f3(LP.loc[p, c]) for c in ("rf 0%", "rf 1%", "rf 2%", "rf 3%", "국채")],
                 "hl" if p == "우리 방식 20%" else "") for p, lab in pol]
    pol_rows.append((["이수용 RAROC(보고값)", pc(RAROC_APR), pc(RAROC_DEF), pc(RAROC_RET, 2)] + [f3(pv(f"이수용_rf{x}", "RAROC 모델(최종)")) for x in range(4)] + [f3(RAROC)], "ref"))
    H.append(P.tab(label="ru_policy", caption="이수용 방식의 대출 단위 Sharpe(무위험수익률별)",
                   head=["정책", "승인율", "부도율", "연 수익률", "rf 0%", "rf 1%", "rf 2%", "rf 3%", "국채"], rows=pol_rows,
                   widths=["22%", "9%", "9%", "11%", "9%", "9%", "9%", "9%", "9%"],
                   foot=f"평가 대상은 36개월 대출 중 2017년 3월까지, 60개월 대출 중 2015년 3월까지 발급되어 상태와 무관하게 포함된 {num(LP.loc['전부 승인', '승인건수'])}건이다."))
    mm = [("V3", "최종", "최종 모형 v3"), ("M6v2", "우리 방식 v2", "v2"), ("B0", "부도확률", "부도확률 순위"), ("GRADE", "세부등급", "세부등급 순서")]
    H.append(P.p(f"[[t:ru_u13]]은 우수사례 1과 3의 방식으로 최종 모형과 비교 모형을 평가한 값이다. 우수사례 3의 IRR Sharpe는 100회 반복의 중앙값이며, 계산 불가 처리 규칙에 따라 우열이 달라진다. R1 기준으로 최종 모형은 {f3(u3('V3'))}, 부도확률 순위는 {f3(u3('B0'))}이었다. 우수사례 1의 Total Risk는 놓친 부도 손실과 놓친 수익의 합으로 작을수록 좋다. 승인율 25%에서 최종 모형은 부도확률 순위보다 작은 위험을 보였으나, 거절 비율을 기저 부도율(19.5%)에 맞춘 지점에서는 v2보다 큰 위험({usd(u1('나', '최종'))} 대 {usd(u1('나', '우리 방식 v2'))})을 보였다."))
    u13_rows = [([lab, f3(u3(k)), f3(u3(k, "승인25_시험SR_중앙")), usd(u1("가", m)), usd(u1("나", m)), f2(kp(k, "SR36_test")), f2(kp(k, "SR60_test")), f"{kp(k, 'ROE_test'):.2f}%"],
                 "hl" if k == "V3" else "") for k, m, lab in mm]
    H.append(P.tab(label="ru_u13", caption="우수사례 1·3 방식과 현금흐름 지표",
                   head=["모형", "우3 IRR Sharpe", "우3 승인 25%", "우1 Total Risk(승인 25%)", "우1 Total Risk(거절 = 부도율)", "발급월 Sharpe 36개월", "발급월 Sharpe 60개월", "연 수익률(ROE)"],
                   rows=u13_rows, widths=["16%", "11%", "11%", "14%", "14%", "11%", "11%", "12%"],
                   foot="우3 IRR Sharpe는 100회 반복 중앙값이며 계산 불가는 −100%로 처리한 R1 기준이다. 우1 Total Risk는 작을수록 좋다. 발급월 Sharpe와 ROE는 승인율 25%의 값이다."))

    H.append(P.h4("라. 목표 정렬의 효과"))
    H.append(P.p(f"v3는 이득 목표를 과제 잣대 수익으로 두었고, v2는 실제 보유기간 연환산 수익으로 두었다. 과제 정의 Sharpe에서 v3는 v2보다 {d(ci('T1', 'M6v2', 'd_final'), 4)} [{d(ci('T1', 'M6v2', 'd_lo'), 4)}, {d(ci('T1', 'M6v2', 'd_hi'), 4)}] 높았다. 반면 현금흐름 ROE에서는 v3가 v2보다 {d(pair('V3 − M6v2', 'dROE'))}%p [{d(pair('V3 − M6v2', 'dROE_lo'))}, {d(pair('V3 − M6v2', 'dROE_hi'))}] 낮았다. 두 결과는 모형이 무엇을 맞히도록 학습했는지가 어떤 잣대에서 이기는지를 결정한다는 점을 보여준다. 이 때문에 최종 모형은 과제가 정한 목표인 포트폴리오 Sharpe에 맞춘 v3로 정하고, 다른 잣대에서의 손실 크기를 함께 보고하였다."))
    H.append(P.p(f"다만 v3의 과제 정의 Sharpe 우위 일부는 2017~20년 발급분을 적게 승인한 데서 나온다. 같은 비율을 매달 승인하면 v3({f3(MON['V3'])})가 v2({f3(MON['M6v2'])})보다 여전히 높지만 차이는 작다. 평가 잣대 자체도 가정을 담고 있다. 과제 정의 Sharpe는 계약만기 연율화로 분할상환과 조기상환을 반영하지 못한다. 잣대가 바뀌면 최적 모형도 바뀔 수 있으므로 여러 잣대의 결과를 함께 읽어야 한다."))

    H.append(P.h4("마. 강건성 분석"))
    H.append(P.p(f"예측 평균과 예측 표준편차가 실현값과 얼마나 맞는지를 10분위 기준으로 보면, 평균의 상관은 {MOM['평균']:.3f}, 표준편차의 상관은 {MOM['표준편차']:.3f}이었다([[f:ru_moment]]). 두 모먼트 모두 크기 순서를 잘 맞추었다. 위험 예측은 하위 분위에서 실제보다 크게 잡혔다(1분위 예측 {m_sd.예측:.2f} 대 실현 {m_sd.실현:.2f}). 이는 위험이 낮은 대출을 과도하게 우대하지 않는 보수적 방향이다."))
    H.append(P.fig("ru_moment", "fig07_모먼트.png", "예측과 실현의 비교. 왼쪽은 기대 초과수익, 오른쪽은 초과수익의 표준편차이며, 10분위별로 나타냈다."))
    H.append(P.p(f"[[t:ru_robust]]은 승인 집합을 고정한 채 계산 방식과 평가 범위를 바꾼 결과다. 모든 열에서 최종 모형, 부도확률 순위, 전부 승인의 순서가 {'유지되었다' if rb_order_ok else '항상 유지되지는 않았다'}. 무위험수익률이 3%이면 최종 모형은 {f3(ROB.loc['V3', 'T1_rf3'])}, 부도확률 순위는 {f3(ROB.loc['B0', 'T1_rf3'])}, 전부 승인은 {f3(ROB.loc['ALL', 'T1_rf3'])}이 되어 모두 낮아진다([[f:ru_rf]]). 이 표는 승인 집합을 고정한 계산이므로, 금리에 따라 승인율을 조정하는 규칙의 효과는 별도로 검증해야 한다."))
    kk = [("V3", "최종 모형 v3"), ("B0", "부도확률 순위"), ("ALL", "전부 승인")]
    H.append(P.tab(label="ru_robust", caption="과제 정의 Sharpe의 강건성 변형", head=["전략"] + [b for _, b in rb_cols],
                   rows=[([lab] + [f3(ROB.loc[k, c]) for c, _ in rb_cols], "hl" if k == "V3" else "") for k, lab in kk],
                   widths=["16%"] + ["8.4%"] * 10, foot="승인 집합은 기본 규칙과 같게 고정하고 계산 방식과 평가 범위만 바꾸었다. 기간별 열은 해당 발급 범위의 대출로 계산하였다."))
    H.append(P.fig("ru_rf", "fig09_rf민감도.png", "무위험수익률 수준별 Sharpe. 왼쪽은 과제 정의 잣대, 오른쪽은 기회풀 잣대이다."))

    H.append(P.p(f"시간 외 검증은 과거 자료만으로 학습한 모형을 이후 발급분에 적용하는 방식이다([[t:ru_oot]], [[f:ru_oot_fig]]). 승인율은 전체 학습 자료 교차검증의 값을 그대로 쓰고, 시간 외 학습 구간에서 다시 고르지 않았다. 1계조건 규칙은 시간 외 구간에서 다시 계산하지 않았다. 2016~17년 발급분(2015년까지 학습)에서 최종 모형의 기회풀 Sharpe는 {f3(OOT15)}으로, M4 보고서의 세 방식({f3(min(M4_OOT))}~{f3(max(M4_OOT))})보다 높았고 부도확률 순위({f3(oot(OOT_O15T2, 'B0'))})보다도 높았다."))
    H.append(P.p(f"그러나 2017~20년 발급분(2016년까지 학습)은 결과 확정 대출 기준으로 두 승인율 모두에서 모든 전략의 Sharpe가 음수였다. 승인율 6.4%에서 최종 모형은 {f3(OOT16)}이고, Sharpe 보고서의 제안 모형({f3(pv('T1_OOT_2017-2020', '제안 모형'))})보다 낮다. 이 구간의 판정은 열세이며, 이전 보고서의 결론보다 약하다. 이 구간의 결과 확정 대출은 2018년 {pc(OOT_RATE['2018'], 0)}, 2019년 {pc(OOT_RATE['2019'], 0)}만 확정된 조기 종료 표본이다."))
    oot_rows = [
        (["2016~17 발급, 기회풀", "2015년까지", num(oot(OOT_O15T2, "V3", "n")), f3(oot(OOT_O15T2, "ALL")), f3(oot(OOT_O15T2, "GRADE")), f3(oot(OOT_O15T2, "B0")), f3(OOT15),
          f"M4 방식 {f3(pv('T2_OOT_2016-2017', 'M4_MeanVar_Util(자유 임계값)'))}·{f3(pv('T2_OOT_2016-2017', 'M4_TwoStage(자유 임계값)'))}, M2 {f3(pv('T2_OOT_2016-2017', 'M2_PD_Decile_Mix(승인 30%)'))}"], "hl"),
        (["2017~20 발급, 과제 정의(승인율 26%)", "2016년까지", num(oot(OOT_O16T1, "V3", "n")), f3(oot(OOT_O16T1, "ALL")), f3(oot(OOT_O16T1, "GRADE")), f3(oot(OOT_O16T1, "B0")), f3(oot(OOT_O16T1, "V3")), "—"], ""),
        (["2017~20 발급, 과제 정의(승인율 6.4%)", "2016년까지", num(oot(OOT_O16T1, "V3", "n")), "—", f3(oot(OOT_O16T1, "GRADE", "SR_승인6_4")), f3(oot(OOT_O16T1, "B0", "SR_승인6_4")), f3(OOT16),
          f"Sharpe 보고서 {f3(pv('T1_OOT_2017-2020', '제안 모형'))}"], ""),
    ]
    H.append(P.tab(label="ru_oot", caption="시간 외 검증 결과",
                   head=["구간", "학습 기간", "대출", "전부 승인", "세부등급", "부도확률 순위", "최종 모형", "보고된 비교값"], rows=oot_rows,
                   widths=["22%", "9%", "9%", "9%", "9%", "11%", "9%", "22%"],
                   foot="승인율은 학습 자료 교차검증 값을 그대로 쓴 것이며, 6.4%는 Sharpe 보고서의 시간 외 승인율에 맞춘 비교이다."))
    H.append(P.fig("ru_oot_fig", "fig10_시간외검증.png", "시간 외 검증에서 승인율별 Sharpe. 빈 원은 Sharpe 보고서에 보고된 값이고, 세로 점선은 운영 승인율이다."))

    H.append(P.p(f"결과 확정 대출만으로 평가하면 상환 중인 대출이 빠진다. 이 문제를 줄이기 위해 상환 중인 대출을 중도절단 관측으로 포함하고, 경쟁위험 누적부도율(Aalen–Johansen)로 승인 집단과 거절 집단을 비교하였다. 시험 자료 신청은 {num(N36 + N60)}건(36개월 {num(N36)}건, 60개월 {num(N60)}건)이다. [[t:ru_cif]]에 따르면 36개월 대출의 36개월 누적부도율은 승인 {pc(cif(36, '최종 모형 승인', 36))}, 거절 {pc(cif(36, '최종 모형 거절', 36))}이었고, 60개월 대출의 60개월 누적부도율은 승인 {pc(cif(60, '최종 모형 승인', 60))}, 거절 {pc(cif(60, '최종 모형 거절', 60))}이었다."))
    cif_rows = []
    for t, ms in ((36, (12, 24, 36)), (60, (12, 36, 60))):
        for m in ms:
            a, b = cif(t, "최종 모형 승인", m), cif(t, "최종 모형 거절", m)
            cif_rows.append([f"{t}개월", f"{m}개월", pc(cif(t, "전체 신청", m)), pc(a), pc(b), d((b - a) * 100, 1) + "%p"])
    H.append(P.tab(label="ru_cif", caption="승인·거절 집단의 누적부도율(경쟁위험 기준)",
                   head=["만기", "경과", "전체 신청", "최종 모형 승인", "최종 모형 거절", "거절 − 승인"], rows=cif_rows,
                   widths=["14%", "14%", "18%", "18%", "18%", "18%"],
                   foot="부도는 사건, 완납은 경쟁 사건, 상환 중·연체·유예 대출은 마지막 납입 월에서 중도절단하였다. 승인은 시험 자료 신청 중 점수 상위 26%이다."))
    H.append(P.p(f"승인 집단은 2016년 이전 발급분에 많이 몰려 있어 위 비교에는 발급 시기의 차이가 섞인다. 같은 발급연도 안에서 다시 비교해도 선별력은 유지되었다. 2015년에는 승인 {pc(wvv(2015, '승인'))} 대 거절 {pc(wvv(2015, '거절'))}, 2016년에는 {pc(wvv(2016, '승인'))} 대 {pc(wvv(2016, '거절'))}, 2017년에는 {pc(wvv(2017, '승인'))} 대 {pc(wvv(2017, '거절'))}이었다(36개월 누적부도율)."))

    H.append(P.p(f"부도확률의 보정은 Cox 보정 회귀 logit P(부도) = a + b·logit(p̂)로 검정하였다. 기울기 b는 {f3(CAL_SLOPE)}(표준오차 {f3(CAL_SE)}), 절편 a는 {f3(CAL['intercept'])}({f3(CAL['intercept_se'])})였다. 기울기는 1에 가깝지만 통계적으로는 1과 다르다(z = {CAL_Z:.1f}). b가 1보다 약간 커서 예측 확률이 다소 완만하게 잡혀 있다는 뜻이다. 평균 예측 부도확률은 {pc(CAL['mean_pred'], 2)}, 실제 부도율은 {pc(CAL['mean_obs'], 2)}이었다. 10분위별 최대 차이는 {CAL['max_abs_decile_gap'] * 100:.2f}%p이고, 기대 보정오차(ECE)는 {CAL['ECE'] * 100:.2f}%p이다."))
    H.append(P.p(f"Hosmer–Lemeshow 카이제곱 통계량은 {CAL['HL_chi2']:.1f}(자유도 {CAL['HL_df']})로 보정 귀무가설이 기각된다. 표본이 {num(CAL['n'])}건으로 커서 0.8%p 이하의 차이도 유의하게 잡히므로, 통계적 유의성보다 차이의 크기를 기준으로 보면 보정은 양호하다. [[f:ru_calib]]은 10분위별 예측과 실제를 보여준다."))
    H.append(P.fig("ru_calib", "fig11_보정.png", "부도확률 10분위별 예측과 실제 부도율(시험 자료)."))
    return "".join(H)
