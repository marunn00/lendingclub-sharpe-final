"""130 최종 보고서(통계청 보도자료 형식) HTML 생성. 수치는 모두 산출물 표·json 에서 읽는다(손으로 옮긴 값은 표/팀보고서_인쇄값.csv 의 팀 보고서 인쇄값뿐).
입력: 121·124·126·127·128 표(산출물/최종보고서/표), 116·119(산출물/우수사례평가), 111(산출물/테스트평가/표), 129 그림. 출력: 산출물/최종보고서/최종보고서.html (PDF는 visualize export)."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import base64, importlib, json, numpy as np, pandas as pd
from pathlib import Path
from statistics import NormalDist
ROOT = Path(__file__).resolve().parents[1]; O = ROOT / "산출물" / "최종보고서"; T = O / "표"; F = O / "fig"; BP = ROOT / "산출물" / "우수사례평가"; TP = ROOT / "산출물" / "테스트평가"
MAIN = pd.read_csv(T / "주요결과_T1_T2.csv"); CI = pd.read_csv(T / "구간.csv"); ROB = pd.read_csv(T / "강건성.csv").set_index("strategy"); REP = json.load(open(O / "결과_T1_T2.json"))
V3J = pd.read_csv(T / "v3_판정.csv").set_index("model"); PR = pd.read_csv(T / "팀보고서_인쇄값.csv"); O15 = pd.read_csv(T / "OOT_2015.csv"); O16 = pd.read_csv(T / "OOT_2016.csv")
LP = pd.read_csv(BP / "이수용/표/정책표.csv").set_index("정책"); LC = pd.read_csv(BP / "이수용/표/신뢰구간.csv"); LB = pd.read_csv(BP / "이수용/표/만기_연도별.csv").set_index("구분")
U3 = pd.read_csv(BP / "표/우3_요약.csv"); U3W = pd.read_csv(BP / "표/우3_seed승률.csv").set_index("rule"); U1 = pd.read_csv(BP / "표/우1_혼동행렬_금액위험.csv")
KP = pd.read_csv(TP / "표/운영지점_요약.csv"); PAIR = pd.read_csv(TP / "표/쌍체차이.csv")
REPR = pd.read_csv(T / "대표성_연도별.csv"); CZ = pd.read_csv(T / "중도절단_연도별.csv").set_index("발급연도"); CIF = pd.read_csv(T / "누적부도_CIF.csv"); CAL = json.load(open(T / "보정검정.json"))
DUR = pd.read_csv(T / "교수_A_듀레이션.csv"); DT1 = pd.read_csv(T / "교수_A_듀레이션_T1.csv").set_index("strategy"); OLS = pd.read_csv(T / "교수_B_OLS.csv").set_index("모형"); LS = pd.read_csv(T / "교수_C_롱숏.csv").set_index("순위")
MO = pd.read_csv(T / "교수_D_모먼트.csv"); RG = pd.read_csv(T / "교수_E_레짐_성과.csv"); FL = pd.read_csv(T / "교수_F_림보.csv").set_index("방식"); GG = pd.read_csv(T / "교수_G_직함군.csv").set_index("직함군")
OC = pd.read_csv(T / "교수_G_희소직함_직군잔차.csv"); PS = json.load(open(T / "교수_요약.json")); GR = pd.read_csv(T / "등급구성_final.csv"); CUR = pd.read_csv(T / "곡선_T1_T2.csv")
BPL = pd.read_csv(T / "채권_풀.csv"); HRT = pd.read_csv(T / "채권_HPR잣대.csv").set_index("strategy"); PST = pd.read_csv(T / "채권_PSA.csv").set_index("집단"); BGL = pd.read_csv(T / "채권_GLM.csv", index_col=0)   # 131
CMOT = pd.read_csv(T / "채권_CMO.csv"); DURB = pd.read_csv(T / "채권_듀레이션.csv"); LADT = pd.read_csv(T / "채권_사다리.csv"); BREG = pd.read_csv(T / "채권_시장회귀.csv"); BS = json.load(open(T / "채권_요약.json"))

# ---------- 값 꺼내기·표기
def mv(ru, k, c="SR_test"): return float(MAIN[(MAIN.ruler == ru) & (MAIN.strategy == k)][c].iloc[0])
def ci(ru, k, c): return float(CI[(CI.ruler == ru) & (CI.strategy == k)][c].iloc[0])
def pv(r, k): return float(PR[(PR.잣대 == r) & (PR.항목 == k)].값.iloc[0])
def oot(D, k, c="SR"): return float(D[D.strategy == k][c].iloc[0])
def u3(m, c="시험SR_중앙"): return float(U3[(U3.rule == "R1") & (U3.model == m)][c].iloc[0])
def u1(pt, m): return float(U1[U1.지점.str.startswith(pt) & U1.모델.str.startswith(m)].total_risk.iloc[0])
def kp(k, c, a=0.25): return float(KP[(KP.strategy == k) & np.isclose(KP.rate, a)][c].iloc[0])
def pair(n, c, a=0.25): return float(PAIR[(PAIR["compare"] == n) & np.isclose(PAIR.rate, a)][c].iloc[0])     # .compare 는 DataFrame 메서드
def lc(p, c, rf="국채"): return float(LC[(LC.rf == rf) & (LC.정책 == p)][c].iloc[0])
def cif(t, g, m): return float(CIF[(CIF.만기 == t) & (CIF.집단 == g) & (CIF.개월 == m)].누적부도.iloc[0])
f3 = lambda x: f"{x:.3f}".replace("-", "−"); f2 = lambda x: f"{x:.2f}".replace("-", "−")
def d3(x, d=3): return ("+" if x >= 0 else "△") + f"{abs(x):.{d}f}"                       # 증감 표기(통계청): 음수 △
pc = lambda x, d=1: f"{x * 100:.{d}f}%".replace("-", "−"); num = lambda x: f"{int(x):,}"; usd = lambda x: f"${x / 1e6:,.0f}M"
Q = float(V3J.loc["V3", "q_star"]); QP = f"{Q * 100:.0f}%"

RF = importlib.import_module("report_fmt"); NB = RF.Numbering(F); table, figure = NB.table, NB.figure; L1, L2, L3, note = RF.L1, RF.L2, RF.L3, RF.note   # 공통 형식(132와 공유)
SRC_T = "lending_club_2020_test.csv(결과 확정 744,608건), 본 연구 계산"

# ---------- 핵심 수치
SR, SR2 = mv("T1", "V3"), mv("T2", "V3"); LO, HI = ci("T1_대출부트스트랩", "V3", "SR_lo"), ci("T1_대출부트스트랩", "V3", "SR_hi"); BLO, BHI = ci("T1", "V3", "SR_lo"), ci("T1", "V3", "SR_hi")
DEFF = (BHI - BLO) / (HI - LO); S8 = pv("T1", "제안 모형(S8)"); M4 = pv("T2", "M4_TwoStage(주력)"); RAROC = pv("이수용_국채", "RAROC 모델(최종)")
APR, APD, ARET, AVOL = mv("T1", "V3", "승인율_test"), mv("T1", "V3", "승인부도율_test"), mv("T1", "V3", "연수익률_test"), mv("T1", "V3", "변동성_test")
LSY20, LSY20Q = LP.loc["우리 방식 20%", "국채"], LP.loc["우리 방식 20%", "승인율"]
R18 = REPR[REPR.yr <= 2018]; APP_N, REJ_N = R18.승인_대출.sum(), R18.거절_신청.sum(); PSHARE = APP_N / (APP_N + REJ_N)
FA, FR_ = REPR[REPR.yr <= 2013].승인_FICO_중앙, REPR[REPR.yr <= 2013].거절_FICO_중앙
MZ = OLS.loc["MZ 불편성 z = α + β·μ̂"]; M4o = OLS.loc["M4 +우리 예측"]; LSf = LS.loc["최종 모형"]; REG = PS["레짐"]; LIM = PS["림보"]
bad, good = REG["bad"], 1 - REG["bad"]; rgv = lambda g, s: float(RG[(RG.국면 == g) & (RG.전략 == s)].T1.iloc[0])
OOT15, OOT16 = oot(O15[O15.ruler == "T2"], "V3"), oot(O16[O16.ruler == "T1"], "V3", "SR_승인6_4")
MON = {k: float(ROB.loc[k, "T1_월별할당"]) for k in ("V3", "M6v2", "B0", "GRADE")}; FIX, FIXQ = float(ROB.loc["V3", "T1_고정점수선"]), float(ROB.loc["V3", "고정점수선_승인율"])   # 발급 시기 진단(121)
YA = lambda k, g="2017–20": float(ROB.loc[k, f"승인율_{g}"])
WVT = pd.read_csv(T / "누적부도_발급연도내.csv"); wvv = lambda y, g: float(WVT[(WVT.발급연도 == y) & (WVT.집단 == g)].누적부도_36개월.iloc[0])
U3R = {r: int(U3W.loc[r, "V3>B0_seed수"]) for r in ("R0", "R1", "R2")}
bp = lambda g, t, c: float(BPL[(BPL.풀 == g) & (BPL.만기 == t)][c].iloc[0]); cm = lambda g, tr, c: float(CMOT[(CMOT.풀 == g) & (CMOT.트랜치 == tr)][c].iloc[0])   # 채권·MBS(131)
dv = lambda g, case, sec, c: float(DURB[(DURB.풀 == g) & (DURB.경우 == case) & (DURB.증권 == sec)][c].iloc[0]); rgr = lambda kind, x="Baa 회사채": BREG[(BREG.회귀 == kind) & (BREG.설명변수 == x)].iloc[0]
GSP = {g: bp(f"등급{g}", 36, "실현스프레드_전") for g in "ABCDEFG"}; GPK = max(GSP, key=GSP.get); HV, HB, HM = (HRT.loc[k] for k in ("V3", "B0", "M6v2")); GB = BS["GLM"]
LAB = bool((LADT["LC 실현 승인(36개월)"] > LADT["Baa 회사채"]).all()); BONF = NormalDist().inv_cdf(1 - 0.05 / (2 * len(BREG)))   # 시장 회귀 4개 본페로니 임계값
BCPR = pd.read_csv(T / "채권_CPR.csv"); cpr_at = lambda a: float(BCPR[(BCPR.집단 == "36_전체") & (BCPR.나이 == a)].CPR.iloc[0])
CASEB = [("추정 반응(β̂)", "추정 β̂"), ("고정(β=0)", "고정"), ("가상 MBS형(β=+|β̂|)", "가상 MBS형")]; LBY = [str(int(y)) for y in LADT.발급연도[LADT["LC 실현 전체(36개월)"] < LADT["Baa 회사채"]]]
sp_ = lambda x: f"{x * 100:.2f}%p".replace("-", "−")

# ---------- 스코어카드(8개 잣대)
SCORE = [
    ("과제 정의 포트폴리오 Sharpe", "Sharpe 최종보고서 2장", "시험 확정 744,608건", f"Sharpe 보고서 제안 {f3(S8)}", f"<b>{f3(SR)}</b><br><small>[{f3(BLO)}, {f3(BHI)}]</small>", d3(SR - S8), "◐ 동등"),
    ("기회풀 Sharpe(만기 코호트)", "M4 최종보고서 2장", f"만기 코호트 {num(REP['n_ref'])}건", f"M4 주력 {f3(M4)}", f"<b>{f3(SR2)}</b><br><small>[{f3(ci('T2', 'V3', 'SR_lo'))}, {f3(ci('T2', 'V3', 'SR_hi'))}]</small>", d3(SR2 - M4), "● 우세"),
    ("대출 단위 Sharpe(국채 rf)", "이수용 보고서 5장", f"만기 지난 대출 {num(LP.loc['전부 승인', '승인건수'])}건", f"이수용 RAROC {f3(RAROC)}", f"<b>{f3(LSY20)}</b><br><small>[{f3(lc('우리 방식 20%', 'SR_lo'))}, {f3(lc('우리 방식 20%', 'SR_hi'))}]</small>", d3(LSY20 - RAROC), "● 우세"),
    ("대출 IRR Sharpe(100회 반복)", "우수사례 3", "무작위 20% 표본 ×100", f"부도확률 순위 {f3(u3('B0'))}<br><small>(본 연구 재채점)</small>", f"<b>{f3(u3('V3'))}</b><br><small>우세 반복 R0 {U3R['R0']}·R1 {U3R['R1']}·R2 {U3R['R2']}/100</small>", d3(u3("V3") - u3("B0")), "◑ 혼재"),
    ("금액 위험(Total Risk)", "우수사례 1", "시험 확정 744,608건", f"부도확률 순위 {usd(u1('가', '부도확률'))}<br><small>(본 연구 재채점)</small>", f"<b>{usd(u1('가', '최종'))}</b><br><small>거절=부도율 지점 {usd(u1('나', '최종'))}</small>", f"{usd(u1('가', '부도확률') - u1('가', '최종'))} 감소", "◑ 혼재"),
    ("발급월 현금흐름 Sharpe(36개월)", "본 연구 사전계획", "시험 확정 744,608건", f"부도확률 순위 {f2(kp('B0', 'SR36_test'))}", f"<b>{f2(kp('V3', 'SR36_test'))}</b><br><small>차이 [{f2(pair('V3 − B0', 'dSR36_lo'))}, {f2(pair('V3 − B0', 'dSR36_hi'))}]</small>", d3(kp("V3", "SR36_test") - kp("B0", "SR36_test"), 2), "● 우세"),
    ("보유기간수익률 Sharpe(국채 재투자)", "채권분석 강의안 4.3절", "시험 확정 744,608건", f"부도확률 순위 {f3(HB.HPR_SR)}<br><small>(본 연구 재채점)</small>", f"<b>{f3(HV.HPR_SR)}</b><br><small>[{f3(HV.SR_lo)}, {f3(HV.SR_hi)}]</small>", d3(HB.d_V3), "● 우세" if HB.d_lo > 0 else "◐ 동등"),
    ("시간 외: 기회풀 Sharpe 2016~17", "M4 최종보고서 OOT", f"{num(oot(O15[O15.ruler == 'T2'], 'V3', 'n'))}건(2015년까지 학습)", f"M4 보고서 최고(M2) {f3(pv('T2_OOT_2016-2017', 'M2_PD_Decile_Mix(승인 30%)'))}", f"<b>{f3(OOT15)}</b>", d3(OOT15 - pv("T2_OOT_2016-2017", "M2_PD_Decile_Mix(승인 30%)")), "● 우세"),
    ("시간 외: 과제 정의 Sharpe 2017~20", "Sharpe 최종보고서 6.4절", f"{num(oot(O16, 'V3', 'n'))}건(2016년까지 학습)", f"Sharpe 보고서 제안 {f3(pv('T1_OOT_2017-2020', '제안 모형'))}<br><small>(승인 6.4%)</small>", f"<b>{f3(OOT16)}</b><br><small>(같은 승인 6.4%)</small>", d3(OOT16 - pv("T1_OOT_2017-2020", "제안 모형")), "○ 열세")]
NWIN = sum(s[-1].startswith("●") for s in SCORE)
NTAG = "·".join(f"{sum(s[-1].startswith(m) for s in SCORE)}개 {w}" for m, w in (("●", "우세"), ("◐", "동등"), ("◑", "혼재"), ("○", "열세")) if any(s[-1].startswith(m) for s in SCORE))
# 기준선 재현 대조(인쇄값, 재계산): 같은 잣대인지 판정
rep_rows = [("과제 정의", "전부 승인 Sharpe", pv("T1", "전체 승인"), REP["T1_all_test"]), ("과제 정의", "완전 예측(Oracle) Sharpe", pv("T1", "Oracle(부도 완전 예측)"), REP["T1_oracle_test"]),
            ("과제 정의", "시험 부도율", pv("T1", "시험 부도율"), REP["default_test"]), ("과제 정의", "평균 국채금리", pv("T1", "평균 만기매칭 국채금리"), REP["mean_rf_test"]),
            ("과제 정의(시간 외)", "2017~20 전부 승인", pv("T1_OOT_2017-2020", "전체 승인"), oot(O16, "ALL")),
            ("기회풀", "만기 기준 코호트 전부 승인", pv("T2", "전부 승인"), REP["T2_all_ref"]), ("기회풀", "순수만기 코호트 전부 승인", pv("T2_2_순수만기", "전부 승인"), ROB.loc["ALL", "T2_2_순수만기(둘 다 ≤15-03)"]),
            ("기회풀", "완결 코호트 전부 승인", pv("T2_3_완결", "전부 승인"), ROB.loc["ALL", "T2_3_완결(둘 다 ≤14-12)"]), ("기회풀", "평탄 코호트 전부 승인", pv("T2_4_평탄", "전부 승인"), ROB.loc["ALL", "T2_4_평탄(둘 다 ≤17-12)"])]
rep_rows += [("기회풀", f"rf {x}% 전부 승인", pv(f"T2_rf{x}", "전부 승인"), ROB.loc["ALL", f"T2_rf{x}"]) for x in range(4)]
rep_rows += [("기회풀", "관리비 1% 전부 승인", pv("T2_관리비1pct", "전부 승인"), ROB.loc["ALL", "T2_관리비1pct"]), ("기회풀(시간 외)", "2016~17 전부 승인", pv("T2_OOT_2016-2017", "전부 승인"), oot(O15[O15.ruler == "T2"], "ALL")),
             ("이수용", "전부 승인 Sharpe(국채)", pv("이수용_국채", "전부 승인"), LP.loc["전부 승인", "국채"]), ("이수용", "A등급만 Sharpe(국채)", pv("이수용_국채", "A등급만"), LP.loc["A등급만", "국채"])]
MAXGAP = max(abs(a - b) for g, _, a, b in rep_rows if not g.startswith("이수용"))
# 본문 서술의 근거가 되는 집계(문장마다 데이터로 확인)
cv3, cb0, cgr = (CUR[(CUR.strategy == k) & (CUR.q < 1)].set_index("q").T1_test for k in ("V3", "B0", "GRADE")); NQ, NQW = len(cv3), int(((cv3 > cb0) & (cv3 > cgr)).sum())   # 100%는 모든 전략이 같음
TOPQ = cv3[cv3 >= cv3.max() - 0.005].index; GRR = GR.승인부도율 / GR.등급부도율
T2V = [("T2_" + c, r, k) for c, r, k in (("1_만기기준(36≤17-03, 60≤15-03)", "T2", "M4_TwoStage(주력)"), ("2_순수만기(둘 다 ≤15-03)", "T2_2_순수만기", "M4_TwoStage"), ("3_완결(둘 다 ≤14-12)", "T2_3_완결", "M4_TwoStage"), ("4_평탄(둘 다 ≤17-12)", "T2_4_평탄", "M4_TwoStage"))] \
      + [(f"T2_rf{x}", f"T2_rf{x}", "M4_TwoStage") for x in range(4)] + [("T2_관리비1pct", "T2_관리비1pct", "M4_TwoStage")]
T2WIN = sum(ROB.loc["V3", c] > pv(r, k) for c, r, k in T2V)
LSYRF = sum(LP.loc["우리 방식 20%", c] > LP.loc["A등급만", c] for c in ("rf 0%", "rf 1%", "rf 2%", "rf 3%", "국채")); LYRS = [i for i in LB.index if i.isdigit()]; LSYYR = sum(LB.loc[y, "우리 방식 25%"] > LB.loc[y, "A등급만"] for y in LYRS)

# ---------- 본문
CSS = RF.CSS

H = [f"<html><head><meta charset='utf-8'><title>LendingClub 대출 승인 전략 최종 보고서</title><style>{CSS}</style></head><body>"]

# ===== 표지
H.append(f"""<div class='cover'>
<table class='hdr'><tr><td class='lab'>과 제</td><td>서울대학교 ABS 과정 14기 · 통계 &amp; 데이터사이언스 python 실습 · LendingClub 팀 프로젝트</td><td rowspan='2' style='text-align:center;width:24%'><span class='badge'>최종보고서</span></td></tr>
<tr><td class='lab'>제 출</td><td>2026. 11. 24.(화)</td></tr></table>
<div class='title'>위험 대비 수익 순위 승인 전략,<br>과제 정의 포트폴리오 Sharpe {f3(SR)}</div>
<div class='sub'>- 전부 승인({f3(mv('T1', 'ALL'))})·부도확률 순위({f3(mv('T1', 'B0'))})를 크게 웃돌고, {len(SCORE)}개 평가 잣대 중 {NTAG} -</div>
<div class='keys'>
{L1(f"(주 결과) 시험 대출 744,608건 중 점수 상위 {QP}를 승인하고 나머지 자금을 국채로 운용하면 Sharpe {f3(SR)}[95% 구간 {f3(BLO)}~{f3(BHI)}]")}
{L2(f"승인 대출 부도율 {pc(APD)}(전체 {pc(REP['default_test'])}), 포트폴리오 연 수익률 {pc(ARET, 2)}(국채 {pc(REP['mean_rf_test'], 2)})")}
{L2(f"단, 매달 같은 비율을 승인하면 {f3(MON['V3'])}(부도확률 순위 {f3(MON['B0'])}) → 개선의 상당 부분은 대출을 내주는 시기의 선택에서 나옴(Ⅳ-4)")}
{L1("(같은 잣대 비교) 팀 공유 보고서 3종·우수사례 2종의 평가 정의를 그대로 재현하고 기준선 인쇄값과 일치함을 확인한 뒤 비교")}
{L2(f"기회풀 Sharpe {f3(SR2)}(M4 보고서 {f3(M4)}), 이수용 방식 {f3(LSY20)}(RAROC {f3(RAROC)}) 등 {NWIN}개 우세, 과제 정의 잣대는 팀 최고 모형과 동등({f3(SR)} 대 {f3(S8)})")}
{L1("(통계 품질) 표본 대표성·우측 중도절단·보정을 수치로 진단하고, 상환 중 대출까지 넣은 생존분석으로 보완")}
{L2(f"자료는 같은 기간 신청의 {pc(PSHARE)}만 담은 승인 대출이며, 2018년 발급분은 {pc(CZ.loc[2018, '결과확정비율'], 0)}만 결과 확정")}
{L1("(이론의 실증) 듀레이션·스프레드 회귀·10분위 롱-숏·레짐 전환으로 모형의 작동 원리를 검증")}
{L2(f"점수 1분위 − 10분위의 연환산 초과수익 차이 발급월 평균 {LSf.롱숏_평균:.1f}%p(Newey–West t = {LSf.NW_t:.1f}), 예측-실현 회귀 기울기 {MZ.b_mu:.3f}(표준오차 {MZ.se_mu:.3f})")}
{L1("(채권·MBS 형식) 대출 풀을 회사채·MBS처럼 수익률·스프레드·조기상환 속도·듀레이션으로 다시 평가")}
{L2(f"보유기간수익률 Sharpe {f3(HV.HPR_SR)}(부도확률 순위 {f3(HB.HPR_SR)}), 36개월 승인 풀은 약정 스프레드의 {pc(bp('V3', 36, '보존율'), 0)}를 실현(전부 승인 {pc(bp('ALL', 36, '보존율'), 0)}, 수수료 전 기준)")}
</div>
<table class='hdr foot'><tr><td class='lab'>작 성</td><td>○조 (조원 이름 기입)</td></tr><tr><td class='lab'>재 현</td><td>scripts/110·117·121~131 (부록 5) · 모든 표·그림 원자료: 산출물/최종보고서/표</td></tr></table>
</div>""")

# ===== 일러두기·목차
H.append("<div class='page'><h3>일 러 두 기</h3>")
H.append(L1("본 보고서는 LendingClub 대출 자료로 「승인한 대출 + 거절한 자금(국채)」 포트폴리오의 Sharpe ratio를 최대로 만드는 승인 전략을 만들고 평가한 결과임"))
H.append(L2(f"학습 자료 lending_club_2020_train.csv {num(1755295)}건(결과 확정 {num(REP['n_train_confirmed'])}건), 시험 자료 lending_club_2020_test.csv {num(1170198)}건(결과 확정 {num(REP['n_test'])}건). 두 파일은 같은 기간(2007.6~2020.9) 발급의 서로 다른 대출(겹침 0건)이며, 시험 자료는 원자료의 무작위 40% 표본으로 판정됨"))
H.append(L2("모형 선택(변수·모수·승인율)의 채택 판정은 모두 학습 자료의 5겹 교차검증으로 하였고, 시험 자료의 결과 열(상환 상태·상환액)은 채점에만 사용함. 다만 v2·v3 후보는 1차 시험 채점 이후에 착안됨(부록 4)"))
H.append(L2("표본 결과이므로 95% 신뢰구간을 함께 해석할 필요가 있음. 같은 달 발급된 대출은 경기 충격을 공유하므로 주된 구간은 발급월 단위 블록 부트스트랩(군집 재표본)으로 계산함"))
H.append(L1("유의사항"))
H.append(L2(f"자료는 LendingClub이 1차 심사로 승인한 대출뿐임(2007~2018년 신청의 {pc(PSHARE)}). 거절 신청자의 상환 결과는 관측되지 않으므로 결과는 승인 대출 모집단에 한정됨(Ⅱ-2)"))
H.append(L2("2017년 이후 발급분은 상환 중 대출이 많아 결과 확정 대출만 보면 조기 부도·조기 완납 쪽으로 치우침(우측 중도절단, Ⅱ-3)"))
H.append(L2("팀 공유 보고서의 수치는 해당 보고서 인쇄값을 옮긴 것이며(부록 6), 같은 정의로 다시 계산한 기준선과 대조해 정의 일치를 확인함(Ⅴ-1)"))
H.append(L2("Ⅶ장의 채권·MBS 형식(만기수익률·보유기간수익률·수익률 스프레드·CPR/PSA·자동이전증권·순차 지급 CMO·유효 듀레이션·볼록성)은 「채권분석」(2025년 2학기, 한림대 조규성) 강의안의 정의를 따르며, 시장 금리는 FRED 원자료(Moody's 회사채, Freddie Mac 모기지, 연준 개인대출 금리)임"))
H.append(L2("시험 채점 이후의 변경(최종 모형 v2·v3 채택, 운영 승인율 변경)은 모두 학습 자료 근거로 결정했으며 동기와 함께 부록 4에 기록함"))
H.append(L2("통계표의 숫자는 반올림하여 세목과 합계가 일치하지 않을 수 있음. 증감·차이 열의 △는 음(−)의 값임"))
H.append(L1("재현용 코드와 모든 표·그림의 원자료는 scripts/ 및 산출물/최종보고서/표에 수록됨(부록 5)"))
H.append("</div><div class='page'><h3>목 차</h3><div class='toc'>")         # 목차는 독립 쪽
for a, bs in [("□ 한눈에 보는 결과 · 요약", []), ("Ⅰ. 과제와 평가 기준", ["1. 과제 정의", "2. 주 지표: 과제 정의 포트폴리오 Sharpe", "3. 보조 잣대 8종"]),
              ("Ⅱ. 자료와 표본 대표성", ["1. 자료 구성", "2. 모집단 대비 표본(승인 대출만 관측)", "3. 우측 중도절단"]), ("Ⅲ. 모형과 승인 규칙", ["1. 구조", "2. 구성 요소와 근거", "3. 승인율 선택"]),
              ("Ⅳ. 주요 결과", ["1. 승인율별 Sharpe", "2. 단계별 기여 분해", "3. 승인 포트폴리오 특성과 구성", "4. 발급 시기 집중과 매달 같은 비율 승인"]), ("Ⅴ. 같은 잣대 비교(비교성·일관성)", ["1. 정의 재현 확인", f"2. {len(SCORE)}개 잣대 종합", "3. 잣대별 상세", "4. 해석: 목표 정렬의 효과"]),
              ("Ⅵ. 이론의 실증과 강건성", ["1. 잔존만기·듀레이션", "2. 스프레드 분해와 예측 불편성(OLS)", "3. 10분위 롱-숏", "4. 1·2차 모먼트", "5. 레짐 전환", "6. 무위험수익률·비용·가중·기간", "7. 시간 외 검증", "8. 생존분석(상환 중 대출 포함)", "9. 부도확률 보정 검정"]),
              ("Ⅶ. 채권·MBS 형식 평가", ["1. 약정·실현 수익률과 스프레드", "2. 새 잣대: 보유기간수익률 Sharpe", "3. 조기상환·부도 속도(CPR·CDR·PSA)", "4. 자동이전증권과 순차 지급 CMO", "5. 유효 듀레이션·볼록성", "6. 같은 시기 시장 상품과의 비교"]),
              ("Ⅷ. 운영 설계와 보완 방안", ["1. 림보 3분류와 사람 심사", "2. 비정형 자료(직함)", "3. LLM·에이전트 활용 구조", "4. 인과 추론과 예측의 구분"]), ("Ⅸ. 결론과 한계", []),
              ("□ 통계표", []), ("◇ 부록", ["1. 작성 개요", "2. 주요 용어 설명", "3. 통계 품질 자체 점검", "4. 계획 대비 변경 기록", "5. 재현 방법", "6. 팀 공유 보고서 인쇄값"])]:
    H.append(f"<p class='lv1'>{a}</p>" + "".join(f"<p class='lv2'>{b}</p>" for b in bs))
H.append("</div></div>")

# ===== 한눈에 보는 결과
H.append("<div class='page'><h2>한눈에 보는 결과</h2><div class='kpis'>"
         f"<div class='kpi'><div class='l'>과제 정의 Sharpe</div><div class='v'>{f3(SR)}</div><div class='c'>전부 승인 {f3(mv('T1', 'ALL'))} · 부도확률 순위 {f3(mv('T1', 'B0'))}</div></div>"
         f"<div class='kpi'><div class='l'>승인율</div><div class='v'>{QP}</div><div class='c'>학습 교차검증에서 Sharpe 최대 지점</div></div>"
         f"<div class='kpi'><div class='l'>승인 대출 부도율</div><div class='v'>{pc(APD)}</div><div class='c'>전체 {pc(REP['default_test'])}</div></div>"
         f"<div class='kpi'><div class='l'>포트폴리오 연 수익률</div><div class='v'>{pc(ARET, 2)}</div><div class='c'>국채 {pc(REP['mean_rf_test'], 2)} · 변동성 {pc(AVOL, 2)}</div></div></div>")
H.append(figure("fig01_요약.png", "전략별 과제 정의 포트폴리오 Sharpe(시험)", note=f"각 전략의 승인율은 학습 교차검증에서 같은 잣대를 최대로 하는 값(전체 풀 기준). 세부등급 순서는 최적이 '거의 전부 거절(0.5%)'. 매달 같은 비율을 승인하면 최종 모형 {f3(MON['V3'])}(Ⅳ-4).", src=SRC_T, w="86%", h="54mm"))
SCNOTE = "비교 대상: 팀 공유 보고서 3종은 인쇄값, 우수사례 1·3, 발급월 Sharpe, 보유기간수익률 Sharpe는 같은 잣대로 본 연구가 다시 채점한 부도확률 순위(인쇄값 아님). 대괄호는 95% 구간(발급월 군집). ● 우세 ◐ 동등(구간 중첩) ◑ 혼재(지점·규칙에 따라 우열이 바뀜) ○ 열세."
H.append(table(f"{len(SCORE)}개 평가 잣대 종합(시험 자료, 각 잣대 정의 그대로)", ["잣대", "정의 출처", "표본", "비교 대상", "최종 모형", "차이", "판정"],
               [list(s) for s in SCORE], note=SCNOTE,
               src="팀 공유 보고서 3종(부록 6), 우수사례 1·3, 본 연구 계산", widths=["19%", "13%", "15%", "15%", "14%", "11%", "8%"]))
H.append("</div>")

# ===== 요약
H.append("<div class='page summ'><h2>요 약</h2>")
H.append("<h4>Ⅰ. 과제와 평가 기준(Ⅰ장)</h4>")
H.append(L2("주 지표는 과제 안내의 정의(승인 대출 + 거절 자금 국채 포트폴리오의 Sharpe)를 그대로 옮긴 「과제 정의 포트폴리오 Sharpe」이며, 팀 공유 보고서·우수사례·채권 형식의 잣대 8종을 보조로 함께 보고함"))
H.append("<h4>Ⅱ. 자료와 표본 대표성(Ⅱ장)</h4>")
H.append(L2(f"(대표성) 2007~2018년 신청 {num(APP_N + REJ_N)}건 중 승인 대출은 {num(APP_N)}건({pc(PSHARE)}). 거절 신청자 FICO 중앙값 {FR_.min():.0f}~{FR_.max():.0f}점으로 승인 대출({FA.min():.0f}~{FA.max():.0f}점)보다 낮음 → 결과는 승인 대출 모집단의 「2차 선별」 성과"))
H.append(L2(f"(중도절단) 36개월 대출의 결과 확정 비율은 2016년 이전 100% → 2017년 {pc(CZ.loc[2017, '결과확정비율'], 0)}, 2018년 {pc(CZ.loc[2018, '결과확정비율'], 0)}, 2019년 {pc(CZ.loc[2019, '결과확정비율'], 0)}로 낮아져, 최근 연도 결과 확정 표본은 조기 부도 쪽으로 치우침"))
H.append("<h4>Ⅲ. 모형과 승인 규칙(Ⅲ장)</h4>")
H.append(L2(f"부도확률(XGBoost·LightGBM·CatBoost 평균)과 완납·부도 시 조건부 초과수익을 따로 예측해 기대 초과수익(1차 모먼트)과 위험(2차 모먼트)을 만들고, 그 비율 순으로 상위 {QP}를 승인(승인율은 학습 교차검증으로 선택)"))
H.append("<h4>Ⅳ. 주요 결과(Ⅳ장)</h4>")
H.append(L2(f"과제 정의 Sharpe {f3(SR)}: 전부 승인 대비 {d3(SR - mv('T1', 'ALL'))}, 부도확률 순위 대비 {d3(SR - mv('T1', 'B0'))}[{d3(ci('T1', 'B0', 'd_lo'))}~{d3(ci('T1', 'B0', 'd_hi'))}], 학습 교차검증 {f3(mv('T1', 'V3', 'SR_train'))}과 비슷(무작위 분할 기준. 시간 순 분할은 Ⅵ-7)"))
H.append(L2(f"(발급 시기) 승인이 2016년 이전 발급분에 몰림(2017~20년 발급분 승인율 {pc(YA('V3'))}, 부도확률 순위 {pc(YA('B0'))}). 매달 같은 비율을 승인하면 {f3(MON['V3'])}로 낮아지지만 v2({f3(MON['M6v2'])})·부도확률 순위({f3(MON['B0'])})보다 높음"))
H.append(L2(f"같은 달 대출의 상관을 반영한 군집 구간 폭은 대출 단위 구간의 {DEFF:.1f}배 → 독립 가정은 불확실성을 크게 과소평가"))
H.append("<h4>Ⅴ. 같은 잣대 비교(Ⅴ장)</h4>")
H.append(L2(f"팀 공유 보고서 기준선 인쇄값과 재계산 값이 일치(과제 정의·기회풀 잣대 차이 {MAXGAP:.3f} 이하) → 같은 잣대 확인 후 비교: {len(SCORE)}개 중 {NTAG}(열세 = 2017년 이후 시간 외). 이수용 잣대는 기준선 차이 0.012~0.013으로 근사 재현"))
H.append(L2("이득 모형의 학습 목표를 과제 잣대 수익으로 맞춘 v3는 과제 정의 Sharpe를 높였으나 현금흐름 ROE는 v2보다 낮고, 그 우위 일부는 최근 발급분 회피에서 나옴 → 「무엇을 맞히도록 학습했는가」가 순위를 결정"))
H.append("<h4>Ⅵ. 이론의 실증과 강건성(Ⅵ장)</h4>")
H.append(L2(f"(듀레이션) 36개월 완납 대출의 실현 가중평균만기는 {DUR[(DUR['T'] == 36) & (DUR.상태 == '완납')].실현WAL_년.iloc[0]:.2f}년. 과제 잣대처럼 계약만기(3년)로 연율화하면 수익률이 {pc(DUR[(DUR['T'] == 36) & (DUR.상태 == '완납')].수익_계약만기환산.iloc[0])}로 약정금리({pc(DUR[(DUR['T'] == 36) & (DUR.상태 == '완납')].약정금리.iloc[0])})보다 크게 낮게 잡힘. 듀레이션 일관 수익률로 바꿔도 최종 모형 1위 유지({f3(DT1.loc['V3', 'T1_듀레이션일관'])})"))
H.append(L2(f"(회귀) 스프레드·세부등급·부도확률을 통제해도 우리 예측 초과수익의 계수 {M4o.b_mu:.2f}(t = {M4o.t_mu:.1f}). 예측-실현 기울기 {MZ.b_mu:.3f}(1과 차이 없음). 다만 절편 {MZ.alpha:.2f}%p로 결합 불편성 검정은 기각(실현이 예측보다 평균 0.3%p 높은 보수적 편향)"))
H.append(L2(f"(롱-숏·레짐) 1분위 − 10분위 연환산 초과수익 차이 발급월 평균 {LSf.롱숏_평균:.1f}%p, {pc(LSf.양수_월비율, 0)}의 달에서 양(+). 2국면 마르코프 전환 모형에서 악화 국면의 선별 효과(+{rgv('악화 국면', '최종 모형') - rgv('악화 국면', '전부 승인'):.3f})가 정상 국면(+{rgv('정상 국면', '최종 모형') - rgv('정상 국면', '전부 승인'):.3f})보다 큼"))
H.append(L2(f"(생존분석) 상환 중 대출을 중도절단으로 포함한 36개월 누적부도율: 승인 {pc(cif(36, '최종 모형 승인', 36))}, 거절 {pc(cif(36, '최종 모형 거절', 36))}. 같은 발급연도 안에서도 2016년 {pc(wvv(2016, '승인'))} 대 {pc(wvv(2016, '거절'))}, 2017년 {pc(wvv(2017, '승인'))} 대 {pc(wvv(2017, '거절'))}"))
H.append(L2(f"(시간 외) 2015년까지 학습 → 2016~17년 기회풀 Sharpe {f3(OOT15)}(M4 보고서 최고 값 {f3(pv('T2_OOT_2016-2017', 'M2_PD_Decile_Mix(승인 30%)'))}). 2016년까지 학습 → 2017~20년 과제 정의 Sharpe는 같은 승인율 6.4%에서 {f3(OOT16)}로 Sharpe 보고서({f3(pv('T1_OOT_2017-2020', '제안 모형'))})보다 낮음"))
H.append("<h4>Ⅶ. 채권·MBS 형식 평가(Ⅶ장)</h4>")
H.append(L2(f"(수익률·스프레드) 36개월 최종 모형 승인 풀의 실현 스프레드(수수료 전) {sp_(bp('V3', 36, '실현스프레드_전'))}(약정의 {pc(bp('V3', 36, '보존율'), 0)}), 전부 승인 {sp_(bp('ALL', 36, '실현스프레드_전'))}({pc(bp('ALL', 36, '보존율'), 0)}). 등급별 실현 스프레드는 {GPK}등급에서 가장 크고 F·G등급은 음(−)"))
H.append(L2(f"(새 잣대) 중간 현금흐름을 국채로 재투자하는 보유기간수익률 Sharpe: 최종 {f3(HV.HPR_SR)}, 부도확률 순위 {f3(HB.HPR_SR)}(차이 {d3(HB.d_V3)}[{d3(HB.d_lo)}, {d3(HB.d_hi)}]), v2({f3(HM.HPR_SR)})와는 {'차이 구간이 0을 포함해 동등' if HM.d_lo <= 0 <= HM.d_hi else '차이가 유의'}"))
H.append(L2(f"(MBS) 36개월 대출의 평균 CPR {pc(PST.loc['36_전체', '평균CPR'])}(약 {PST.loc['36_전체', 'PSA속도'] * 100:.0f}% PSA). 승인 대출의 CDR은 거절의 {PST.loc['36_승인', '평균CDR'] / PST.loc['36_거절', '평균CDR'] * 100:.0f}%. 「금리 하락 시 조기상환 증가」 가설은 기각(금리 하락 폭 계수 {GB['beta_dtsy']:.3f}, 달력 연월 군집 z = {GB['z_cl_dtsy']:.1f}). 자동이전증권 가격은 조기상환 가정에 거의 둔감(±50bp 가격 차 {pc(BS['가격차_50bp']['V3'], 2)})하고, IO·PO의 금리 위험만 그 부호에 따라 달라짐"))
H.append(L2(f"(CMO·시장) A 70%·B 20%·C 10% 순차 구조의 C 손실률: 승인 {pc(cm('V3', 'C', '손실률'), 0)}, 전체 {pc(cm('ALL', 'C', '손실률'), 0)}. 승인 풀 실현 스프레드는 {'매년' if LAB else '대부분의 해에'} Baa 회사채 약정 스프레드보다 컸으나 만기·기준 국채가 달라 배수로는 비교하지 않음. Baa 스프레드는 대출 실현 스프레드를 예측하지 못함"))
H.append("<h4>Ⅷ. 운영 설계와 보완 방안(Ⅷ장)</h4>")
H.append(L2(f"(림보) 승인 경계·모형 불일치·희소 구간 {pc(LIM['비중'])}을 사람 심사로 넘기면 Sharpe {f3(FL.loc['자동 규칙(그대로)', 'T1'])} → {f3(FL.loc['림보 심사(정확도 70%)', 'T1'])}(심사 정확도 70% 가정), 상한 {f3(FL.loc['림보 완전 심사(상한)', 'T1'])}"))
H.append(L2(f"(비정형) 학습에 30건 미만인 희소 직함이 시험 대출의 {pc(PS['직함']['희소비중'], 0)}. 이들에서 직군별 예측 잔차가 체계적(발급월 군집 기준 {PS['직함']['유의직군']}/{PS['직함']['직군수']}개 직군 1% 유의, 규칙에 안 걸린 '기타' 포함) → 직함 텍스트의 LLM 분류로 보완 여지"))
H.append("</div>")

# ===== Ⅰ
H.append("<div class='page'><h2>Ⅰ. 과제와 평가 기준</h2><h3>1. 과제 정의</h3>")
H.append(L1("투자자 입장에서 대출 신청마다 승인·거절을 정하고, 거절한 자금은 같은 만기의 미국 국채에 투자함. 목표는 이 포트폴리오의 Sharpe ratio를 최대로 만드는 승인 전략(과제 소개 2쪽)"))
H.append(L2("과제 안내는 「신청 정보 → 부도확률 → 승인·거절 → 포트폴리오 Sharpe」 흐름과 Sharpe 정의만 정하고, 각 단계의 방법은 팀 재량으로 둠"))
H.append("<h3>2. 주 지표: 과제 정의 포트폴리오 Sharpe</h3>")
H.append("<div class='formula'>대출 i의 연 수익률 r<sub>i</sub> = (총 상환액<sub>i</sub> ÷ 투자원금<sub>i</sub>)<sup>12/T<sub>i</sub></sup> − 1 &nbsp;(T<sub>i</sub> = 계약만기 36·60개월)<br>"
         "포트폴리오 단위 수익 x<sub>i</sub> = r<sub>i</sub> (승인) &nbsp;|&nbsp; rf<sub>i</sub> (거절: 발급월·만기 맞춤 미국 국채 DGS3·DGS5 월평균)<br>"
         "<b>Sharpe = (평균 x − 평균 rf) ÷ 표준편차 x</b> &nbsp;(대출 단위 동일 가중, 결과 확정 대출 전체)</div>")
H.append(L2("이 정의는 팀 공유 「LendingClub_Sharpe_최종보고서」 2장과 같아 결과를 1:1로 비교할 수 있음. 전부 거절하면 0, 전부 승인하면 −0.114 → 0보다 크면 「국채보다 나은 선별」"))
H.append(L2("계약만기 연율화는 원금이 매달 상환되는 분할상환 구조와 조기상환을 반영하지 못하는 한계가 있음 → Ⅵ-1에서 듀레이션 일관 수익률로 바꿔 결론이 유지되는지 확인"))
H.append("<h3>3. 보조 잣대 8종</h3>")
H.append(table("평가 잣대 정의", ["잣대", "핵심 정의", "평가 표본", "출처"], [
    ["기회풀 Sharpe", "평균<sub>N</sub>(승인×(r − rf)) ÷ 표준편차<sub>N</sub>, r = 상환액 기준 계약만기 연율화", "만기 코호트(36개월 ≤2017.3, 60개월 ≤2015.3 발급)", "M4 최종보고서 2장"],
    ["대출 단위 Sharpe", "대출별 (실현 총수익률 − rf × 잔액·년)의 평균 ÷ 표준편차(승인 대출)", "만기 지난 대출(상태 무관)", "이수용 보고서 5장"],
    ["대출 IRR Sharpe", "대출별 월 현금흐름 IRR − rf, 검증 표본에서 기준선 선택 후 시험, 100회 반복", "무작위 20% 표본", "우수사례 3"],
    ["금액 위험(Total Risk)", "놓친 부도 손실(승인했는데 부도) + 놓친 수익(거절했는데 완납)", "결과 확정 대출", "우수사례 1"],
    ["발급월 현금흐름 Sharpe", "발급월 묶음별 초과수익(재구성 월별 현금흐름)의 평균 ÷ 표준편차", "결과 확정 대출", "본 연구 사전계획"],
    ["보유기간수익률 Sharpe", "중간 현금흐름을 국채로 계약만기까지 재투자한 보유기간수익률(HPR)로 계산한 과제 정의형 Sharpe(Ⅶ-2)", "결과 확정 대출", "채권분석 강의안 4.3절"],
    ["시간 외 검증(2종)", "과거만으로 다시 학습 → 이후 발급분 채점", "2016~17년 / 2017~20년 발급", "M4·Sharpe 보고서"]],
    src="각 보고서 원문", widths=["17%", "45%", "23%", "15%"]))
H.append("</div>")

# ===== Ⅱ
H.append("<div class='page'><h2>Ⅱ. 자료와 표본 대표성</h2><h3>1. 자료 구성</h3>")
H.append(table("학습·시험 자료 구성", ["구분", "원본 행", "결과 확정", "부도율", "평균 국채금리", "전부 승인 Sharpe"], [
    ["학습(lending_club_2020_train.csv)", num(1755295), num(REP["n_train_confirmed"]), pc(REP["default_train"], 2), "—", f3(REP["T1_all_train"])],
    ["시험(lending_club_2020_test.csv)", num(1170198), num(REP["n_test"]), pc(REP["default_test"], 2), pc(REP["mean_rf_test"], 2), f3(REP["T1_all_test"])]],
    unit="건, %", note="결과 확정 = Fully Paid·Charged Off·Default(정책 미충족 상태 제외). 학습·시험 id 겹침 0건, 두 파일 모두 2007.6~2020.9 발급.", src="원자료, 본 연구 계산"))
H.append(L2("시험 자료는 원자료에서 무작위로 뽑은 40% 표본으로 판정함(발급연도·등급·상태 분포가 학습 자료와 비례, scripts/20_test_split_sim.py) → 학습 자료 안의 무작위 교차검증이 시험 상황과 대응"))
H.append("<h3>2. 모집단 대비 표본: 승인 대출만 관측</h3>")
H.append(L1(f"2007~2018년 LendingClub 신청 {num(APP_N + REJ_N)}건 중 본 자료(승인 대출)는 {num(APP_N)}건으로 {pc(PSHARE)}에 불과함"))
H.append(L2(f"플랫폼 승인율은 연 {pc(R18.플랫폼_승인율.min(), 0)}~{pc(R18.플랫폼_승인율.max(), 0)}이며 2016년 이후 하락(2018년 {pc(REPR.set_index('yr').loc[2018, '플랫폼_승인율'])})"))
H.append(L2(f"거절 신청자의 FICO 중앙값은 {FR_.min():.0f}~{FR_.max():.0f}점(2007~2013년)으로 승인 대출({FA.min():.0f}~{FA.max():.0f}점)보다 뚜렷이 낮고, 다수가 660점 미만"))
H.append(L2("시사점: 본 모형의 성과는 「LendingClub이 이미 승인한 대출 중에서 다시 고르는」 2차 선별 성과임. 전체 신청자로 일반화하려면 거절 추론이 필요(Ⅷ-4)"))
H.append(figure("fig03_대표성.png", "연도별 승인 대출·거절 신청 건수와 플랫폼 승인율(왼쪽), 신용점수 중앙값(오른쪽)", note="거절 자료의 Risk_Score는 2013.11.5 이후 FICO가 아닌 Vantage 점수이므로 신용점수 비교는 2007~2013년만 표시. 2019년 이후 거절 자료 없음.", src="rejected_2007_to_2018Q4.csv(27,648,741건), 학습·시험 자료"))
H.append("<h3>3. 우측 중도절단</h3>")
H.append(L1("최근 발급분일수록 아직 상환 중인 대출이 많아, 결과 확정 대출만 보면 「일찍 끝난 대출」(조기 부도·조기 완납)만 남음"))
H.append(L2(f"36개월 대출의 결과 확정 비율: 2016년까지 100% → 2017년 {pc(CZ.loc[2017, '결과확정비율'], 0)}, 2018년 {pc(CZ.loc[2018, '결과확정비율'], 0)}, 2019년 {pc(CZ.loc[2019, '결과확정비율'], 0)}"))
H.append(L2(f"2018년 발급분의 결과 확정 대출 부도율은 {pc(CZ.loc[2018, '확정대출_부도율'])}이지만, 상환 중 대출을 중도절단으로 넣은 경쟁위험 누적부도율(관측 기간 내)은 {pc(CZ.loc[2018, '누적부도_36개월'])}"))
H.append(L2("대응: 주 지표는 과제 정의대로 결과 확정 대출 전체로 계산하되, 만기 코호트 잣대(Ⅴ-3)와 상환 중 대출을 포함한 생존분석(Ⅵ-8)으로 결론이 유지되는지 확인함"))
H.append(figure("fig04_중도절단.png", "발급연도별 결과 확정 비율과 부도율 추정 방법 비교(시험, 36개월 대출)", note="경쟁위험 누적부도율은 Aalen–Johansen 추정(부도 = 사건, 완납 = 경쟁 사건, 상환 중 = 중도절단). 2018년 이후는 36개월을 다 관측하지 못해 관측 기간까지의 값.", src="lending_club_2020_test.csv 전체 1,170,198건, 본 연구 계산", w="92%"))
H.append("</div>")

# ===== Ⅲ
GD = V3J.loc["V3"]
H.append("<div class='page'><h2>Ⅲ. 모형과 승인 규칙</h2><h3>1. 구조</h3>")
H.append("<div class='diag'><div class='n'><b>① 신청 정보</b>신청 시점 변수 71개<br>(사후 정보 31개 제외, 데이터 관문 통과)</div><div class='a'>→</div>"
         "<div class='n'><b>② 부도확률 p̂</b>XGBoost(3시드)·LightGBM·CatBoost 평균</div><div class='a'>→</div>"
         "<div class='n'><b>③ 조건부 초과수익</b>완납 시 â, 부도 시 d̂<br>(XGBoost 회귀, 목표 = 과제 잣대 수익 − 국채)</div><div class='a'>→</div>"
         "<div class='n'><b>④ 1·2차 모먼트</b>μ̂ = (1−p̂)â + p̂d̂<br>σ̂² = p̂(1−p̂)(â−d̂)² + (1−p̂)s<sub>a</sub>² + p̂s<sub>d</sub>²</div><div class='a'>→</div>"
         f"<div class='n'><b>⑤ 승인 규칙</b>점수 μ̂/σ̂ 상위 {QP} 승인,<br>나머지 자금은 국채</div></div>")
H.append(L2("부도확률만으로 고르면 「안전하지만 금리가 낮은」 대출만 남아 국채 대비 초과수익이 작음 → 부도확률(무엇이 잘못될 확률)과 조건부 수익(잘못되면·잘되면 얼마)을 따로 예측해 기대 초과수익과 위험을 함께 봄"))
H.append("<h3>2. 구성 요소와 근거</h3>")
H.append(table("모형 구성 요소와 채택 근거", ["구성 요소", "방법", "채택 근거(학습 자료만)"], [
    ["입력 변수", "신청 시점 정보 71개(금리·등급·FICO·DTI·소득·이력 등), 고차원 범주(우편번호·직함)는 교차적합 목표 인코딩", "사후 정보(상환액·최근 FICO 등) 31개 제외, 누수 관문 통과"],
    ["부도확률", "XGBoost 3시드 평균 + LightGBM + CatBoost 단순 평균", f"교차검증 연 수익률(승인 30%) +0.035%p[0.014, 0.049] (v2 채택)"],
    ["조건부 초과수익", "완납·부도 대출 각각 XGBoost 회귀, 목표 z = 과제 잣대 수익 − 국채, 대출 동일 가중", f"교차검증 과제 정의 Sharpe {f3(V3J.loc['M6v2', 'T1_train'])} → {f3(GD.T1_train)}, 차이 {d3(GD.d_T1_v3_minus_v2, 4)}[{d3(GD.d_lo, 4)}, {d3(GD.d_hi, 4)}] (v3 채택)"],
    ["위험", "혼합분포 분산(부도·완납 상태 전이 + 상태별 잔차)", "1차 모먼트만 쓴 순위와 비교(Ⅳ-2, Ⅵ-4)"],
    ["승인 규칙", f"점수 μ̂/σ̂ 상위 q 승인, q = {QP}", f"학습 교차검증에서 과제 정의 Sharpe 최대({f3(GD.T1_train)}, 0.5% 격자)"]],
    src="scripts/110·117·124·125, 학습 교차검증(튜닝 보류 10% 제외 1,004,516건)", widths=["14%", "48%", "38%"]))
H.append("<h3>3. 승인율 선택</h3>")
H.append(L2(f"승인율은 학습 교차검증 곡선의 최대점({QP})으로 정했고 시험 결과를 보고 바꾸지 않음. 학습 {f3(mv('T1', 'V3', 'SR_train'))} → 시험 {f3(SR)}으로 차이 {d3(SR - mv('T1', 'V3', 'SR_train'))} — 우수사례에서 지적된 「검증→시험 성능 붕괴」가 없음(무작위 분할 기준. 과거로 학습해 미래에 적용하는 시간 순 검증은 Ⅵ-7)"))
H.append(L2("사전 계획의 대표 승인율은 25%(현금흐름 ROE 최대)였으나, 과제 목표(포트폴리오 Sharpe)에 맞춰 같은 학습 자료의 과제 정의 Sharpe 최대점으로 바꿈(부록 4)"))
H.append("</div>")

# ===== Ⅳ
LAD = [("ALL", "전부 승인"), ("GRADE", "세부등급 순서"), ("B0", "부도확률 순위"), ("M5", "+ 조건부 수익(1차 모먼트)"), ("M6", "+ 위험(2차 모먼트) = v1"), ("M6v2", "+ 부도확률 3모형 평균 = v2"), ("V3", "+ 학습 목표 정렬 = 최종 v3")]
lr, prev = [], None
for k, lab in LAD:
    v = mv("T1", k); row = [lab, f"{mv('T1', k, 'q') * 100:.1f}", f3(v), d3(v - prev) if prev is not None else "—", "—" if k == "V3" else f"{d3(ci('T1', k, 'd_final'))} [{d3(ci('T1', k, 'd_lo'))}, {d3(ci('T1', k, 'd_hi'))}]"]
    lr.append((row, "hl" if k == "V3" else "")); prev = v
H.append("<div class='page'><h2>Ⅳ. 주요 결과</h2><h3>1. 승인율별 Sharpe</h3>")
H.append(L1(f"최종 모형은 승인율 100% 미만 격자 {NQ}개 지점 중 {NQW}개에서 부도확률 순위·세부등급 순서보다 높고, 최댓값과 0.005 이내인 구간은 승인 {TOPQ.min() * 100:.1f}~{TOPQ.max() * 100:.1f}%"))
H.append(L2("세부등급 순서는 어떤 승인율에서도 0을 넘지 못함(최적 = 거의 전부 거절) → LendingClub 등급만으로는 국채를 이길 수 없음(팀 공유 Sharpe 보고서와 같은 결론)"))
H.append(figure("fig02_승인율곡선.png", "승인율별 과제 정의 포트폴리오 Sharpe", note="빨간 빈 원은 팀 공유 Sharpe 보고서 인쇄값(같은 시험 자료·같은 정의). 점선은 최종 모형의 학습 교차검증 곡선.", src=SRC_T))
H.append("<h3>2. 단계별 기여 분해</h3>")
H.append(table("단계별 과제 정의 Sharpe(시험)", ["단계", "승인율(%)", "Sharpe", "직전 단계 대비", "최종 모형 − 해당 단계 [95% 구간]"], lr,
               note="각 단계의 승인율은 학습 교차검증에서 따로 선택. 구간은 발급월 블록 부트스트랩 1,000회 쌍체 차이.", src=SRC_T, widths=["30%", "11%", "11%", "15%", "33%"]))
H.append(L2(f"가장 큰 개선은 「부도확률 → 조건부 수익」({d3(mv('T1', 'M5') - mv('T1', 'B0'))})에서 나옴. 위험(2차 모먼트) 추가는 이 잣대에서 차이가 작고({d3(mv('T1', 'M6') - mv('T1', 'M5'))}), 3모형 평균({d3(mv('T1', 'M6v2') - mv('T1', 'M6'))})과 목표 정렬({d3(mv('T1', 'V3') - mv('T1', 'M6v2'))})이 그다음"))
H.append("<h3>3. 승인 포트폴리오 특성과 구성</h3>")
PF = [("ALL", "전부 승인"), ("GRADE", "세부등급 순서"), ("B0", "부도확률 순위"), ("M6v2", "v2"), ("V3", "최종 모형 v3")]
H.append(table("전략별 포트폴리오 특성(시험)", ["전략", "승인율", "연 수익률", "변동성", "승인 대출 부도율", "Sharpe", "95% 구간(발급월 군집)"],
               [([lab, pc(mv("T1", k, "승인율_test")), pc(mv("T1", k, "연수익률_test"), 2), pc(mv("T1", k, "변동성_test"), 2), pc(mv("T1", k, "승인부도율_test")), f3(mv("T1", k)), f"[{f3(ci('T1', k, 'SR_lo'))}, {f3(ci('T1', k, 'SR_hi'))}]"], "hl" if k == "V3" else "") for k, lab in PF]
               + [(["(참고) 완전 예측(부도 아닌 대출만 승인)", pc(REP["oracle_share"]), "—", "—", "0.0%", f3(REP["T1_oracle_test"]), "이론 상한"], "sub")],
               unit="%", note=f"연 수익률·변동성은 포트폴리오 단위 수익 x의 평균·표준편차(국채 평균 {pc(REP['mean_rf_test'], 2)}).", src=SRC_T))
H.append(L2(f"최종 모형의 군집 구간 [{f3(BLO)}, {f3(BHI)}]은 대출 단위 구간 [{f3(LO)}, {f3(HI)}]보다 {DEFF:.1f}배 넓음 → 같은 달 대출은 금리·경기 충격을 공유하므로, 대출을 독립으로 보는 구간은 불확실성을 과소평가함"))
H.append(L2(f"등급별로는 A등급의 {pc(GR.set_index('등급').loc['A', '승인율'], 0)}, B등급의 {pc(GR.set_index('등급').loc['B', '승인율'], 0)}를 승인하면서도 C~F등급에서 일부를 고르며, 승인 대출 부도율은 등급 전체 부도율의 {GRR.min() * 100:.0f}~{GRR.max() * 100:.0f}% 수준(A~F등급)"))
H.append(figure("fig12_등급구성.png", "등급별 승인율과 부도율(최종 모형, 시험)", note="G등급은 승인 100건 미만이라 승인 대출 부도율을 표시하지 않음.", src=SRC_T, w="86%"))
H.append("<h3>4. 발급 시기 집중과 매달 같은 비율 승인</h3>")
H.append(L1(f"승인 규칙은 전체 풀에서 점수 상위 {QP}를 고르므로, 점수가 좋은 시기의 대출은 많이 승인되고 나쁜 시기의 대출은 거의 승인되지 않음(발급 시기 선택)"))
VK = [("V3", "최종 모형 v3"), ("M6v2", "v2"), ("B0", "부도확률 순위"), ("GRADE", "세부등급 순서")]
H.append(table("승인 규칙별 과제 정의 Sharpe와 발급연도군별 승인율(시험)", ["전략", "승인율 2007~11", "승인율 2012~16", "승인율 2017~20", "전체 풀 상위 q(기본)", "학습 기준 고정 점수선", "발급월·만기마다 상위 q"],
               [([lab, pc(YA(k, "2007–11")), pc(YA(k, "2012–16")), pc(YA(k)), f3(mv("T1", k)), f"{f3(ROB.loc[k, 'T1_고정점수선'])}<br><small>(승인 {pc(ROB.loc[k, '고정점수선_승인율'])})</small>", f3(MON[k])], "hl" if k == "V3" else "") for k, lab in VK]
               + [(["전부 승인", "100.0%", "100.0%", "100.0%", f3(mv("T1", "ALL")), f3(mv("T1", "ALL")), f3(mv("T1", "ALL"))], "sub")],
               note="승인율은 해당 발급연도군 결과 확정 대출 중 승인 비율(기본 규칙). 고정 점수선 = 학습 교차검증 점수의 상위 q 경계를 시험에 그대로 적용(발급 시점에 구현 가능). 발급월·만기마다 상위 q = 매달 금액 기준 같은 비율 승인. q는 각 전략의 학습 선택값.",
               src=SRC_T, widths=["17%", "12%", "12%", "12%", "14%", "17%", "16%"]))
H.append(L2(f"최종 모형은 2017~20년 결과 확정 대출의 {pc(YA('V3'))}만 승인(v2 {pc(YA('M6v2'))}, 부도확률 순위 {pc(YA('B0'))}). 학습 자료로 정한 점수선을 그대로 적용해도(승인 {pc(FIXQ)}) Sharpe {f3(FIX)}로 같아, 이 시기 선택은 발급 시점에 구현 가능한 규칙임"))
H.append(L2(f"매달 같은 비율을 승인하면 최종 모형 {f3(MON['V3'])}로 낮아지지만 v2({f3(MON['M6v2'])})·부도확률 순위({f3(MON['B0'])})·세부등급({f3(MON['GRADE'])})보다 높음 → 같은 달 안의 선별력은 유지되며, 기본 규칙과의 차이({d3(SR - MON['V3'])})는 발급 시기 선택 효과"))
H.append(note("추정: 과제 잣대 수익은 계약만기로 연율화하므로, 학습 자료에서 결과가 확정된 2017년 이후 발급분(조기 부도·조기 완납 위주)은 수익이 낮게 잡힘. 이 잣대 수익을 목표로 학습한 v3는 최근 발급분을 낮게 평가했을 수 있고, 시험 채점도 같은 잣대라 그 효과가 함께 보상됨. 실제 운용 성과는 매달 같은 비율 규칙의 값을 보수적 참고값으로 볼 것을 권함"))
H.append("</div>")

# ===== Ⅴ
maxgap = MAXGAP
fmt = lambda lab, x: pc(x, 2) if ("부도율" in lab or "금리" in lab) else f3(x)
H.append("<div class='page'><h2>Ⅴ. 같은 잣대 비교(비교성·일관성)</h2><h3>1. 정의 재현 확인</h3>")
H.append(L1(f"다른 보고서와 숫자를 비교하려면 먼저 「같은 잣대」인지 확인해야 함 → 각 보고서의 기준선(모형과 무관한 값)을 같은 정의로 다시 계산해 인쇄값과 대조"))
H.append(L2(f"과제 정의·기회풀 잣대의 기준선 {len([r for r in rep_rows if not r[0].startswith('이수용')])}개 모두 인쇄값과의 차이 {maxgap:.3f} 이하 → 같은 잣대로 판정"))
H.append(L2(f"이수용 잣대는 재계산 값이 인쇄값보다 0.012~0.013 높음. 평가 건수가 그 보고서(467,101건)보다 {num(LP.loc['전부 승인', '승인건수'] - 467101)}건 많지만(60개월 62,522건은 일치) 이 차이만으로는 값 차이를 설명하지 못함(원인 미확인) → 근사 재현으로 보고, 우열은 같은 표본 안의 비교(최종 모형 대 A등급만)로 판단"))
H.append(table("기준선 인쇄값과 재계산 값 대조", ["잣대", "항목", "인쇄값", "재계산", "차이"], [[g, lab, fmt(lab, a), fmt(lab, b), d3(b - a, 4 if ('부도율' in lab or '금리' in lab) else 3)] for g, lab, a, b in rep_rows],
               note="인쇄값은 부록 6. 재계산은 시험 자료에 같은 정의를 적용한 값.", src="팀 공유 보고서 3종, 본 연구 계산", widths=["18%", "34%", "16%", "16%", "16%"]))
H.append(f"<h3>2. {len(SCORE)}개 잣대 종합</h3>")
H.append(table(f"{len(SCORE)}개 평가 잣대 종합", ["잣대", "정의 출처", "표본", "비교 대상", "최종 모형", "차이", "판정"], [list(s) for s in SCORE],
               note=SCNOTE + " ◐: 최종 모형의 95% 구간이 비교 값을 포함. ◑ 우수사례 1: 승인 25%에서는 최소 위험, 거절 비율 = 기저 부도율 지점에서는 v2·부도확률 순위보다 큼. ◑ 우수사례 3: IRR 계산 불가 처리 규칙(R0·R1·R2)에 따라 우열이 바뀜.", src="부록 6, 본 연구 계산", widths=["19%", "13%", "15%", "15%", "14%", "11%", "8%"]))
H.append(L1(f"{len(SCORE)}개 중 {NTAG}"))
H.append(L2(f"과제 정의 잣대: 최종 {f3(SR)} 대 팀 공유 Sharpe 보고서 {f3(S8)} — 차이 {d3(SR - S8)}로 발급월 군집 95% 구간[{f3(BLO)}, {f3(BHI)}] 안(대출 단위 구간 [{f3(LO)}, {f3(HI)}]으로도 안) → 통계적으로 같은 수준"))
H.append(L2(f"혼재(우수사례 3): 대표 규칙 R1(IRR 계산 불가 = −100%)에서는 100회 중 {U3R['R1']}회 부도확률 순위보다 높지만, 원 발표 규칙 R0에서는 {U3R['R0']}회, 음수 IRR도 초과수익 0으로 두는 R2에서는 {U3R['R2']}회 → 손실 처리 방식에 따라 결론이 바뀌므로 혼재로 판정"))
H.append(L2("열세: 2016년까지 학습해 2017~20년에 적용한 시간 외 검증. 이 구간의 결과 확정 대출은 조기 종료 위주로 치우친 표본이지만(Ⅱ-3), 그것만으로 열세를 설명할 근거는 없어 한계로 남김"))
H.append("<h3>3. 잣대별 상세</h3><h4>가. 기회풀 Sharpe 변형 9종(M4 보고서와 같은 표 구성)</h4>")
coh = [("1_만기기준(36≤17-03, 60≤15-03)", "만기 기준 코호트", "T2", "M4_TwoStage(주력)"), ("2_순수만기(둘 다 ≤15-03)", "순수만기(≤2015.3)", "T2_2_순수만기", "M4_TwoStage"),
       ("3_완결(둘 다 ≤14-12)", "완결(≤2014.12)", "T2_3_완결", "M4_TwoStage"), ("4_평탄(둘 다 ≤17-12)", "평탄(≤2017.12)", "T2_4_평탄", "M4_TwoStage")]
rows = [[lab, f3(ROB.loc["ALL", "T2_" + c]), f3(pv(r, k)), f3(ROB.loc["V3", "T2_" + c]), d3(ROB.loc["V3", "T2_" + c] - pv(r, k))] for c, lab, r, k in coh]
rows += [[f"rf {x}% 고정", f3(ROB.loc["ALL", f"T2_rf{x}"]), f3(pv(f"T2_rf{x}", "M4_TwoStage")), f3(ROB.loc["V3", f"T2_rf{x}"]), d3(ROB.loc["V3", f"T2_rf{x}"] - pv(f"T2_rf{x}", "M4_TwoStage"))] for x in range(4)]
rows += [["연 1% 관리비 차감", f3(ROB.loc["ALL", "T2_관리비1pct"]), f3(pv("T2_관리비1pct", "M4_TwoStage")), f3(ROB.loc["V3", "T2_관리비1pct"]), d3(ROB.loc["V3", "T2_관리비1pct"] - pv("T2_관리비1pct", "M4_TwoStage"))]]
H.append(table("기회풀 Sharpe: 코호트·무위험수익률·비용 변형", ["변형", "전부 승인(재계산)", "M4 주력(인쇄값)", "최종 모형", "차이"], rows, note=f"최종 모형은 시험 전체 풀의 상위 {QP} 승인(학습에서 기회풀 Sharpe 최대). 만기 코호트 안 승인 비율은 {pc(mv('T2', 'V3', '코호트승인율_test'))}. 차이 = 최종 − M4.", src="M4 최종보고서 4·7장, 본 연구 계산"))
H.append(L2(f"{len(T2V)}개 변형 중 {T2WIN}개에서 최종 모형이 M4 주력 모형보다 높음. rf를 3%로 올리면 둘 다 음수가 되지만 최종 모형의 손실이 더 작음"))
H.append("<h4>나. 이수용 방식(만기 지난 대출, rf 수준별)</h4>")
pol = [("전부 승인", "전부 승인"), ("A등급만", "A등급만"), ("부도확률 낮은 순 20%", "부도확률 낮은 순 20%"), ("v2 20%", "v2(승인 20%)"), ("우리 방식 20%", "최종 모형(승인 20%)"), ("우리 방식 25%", "최종 모형(승인 25%)")]
rows = [([lab, pc(LP.loc[p, "승인율"]), pc(LP.loc[p, "부도율"]), pc(LP.loc[p, "연수익률"], 2)] + [f3(LP.loc[p, c]) for c in ("rf 0%", "rf 1%", "rf 2%", "rf 3%", "국채")], "hl" if p == "우리 방식 20%" else "") for p, lab in pol]
rows += [(["이수용 RAROC(인쇄값)", "17.0%", "7.6%", "6.42%"] + [f3(pv(f"이수용_rf{x}", "RAROC 모델(최종)")) for x in range(4)] + [f3(RAROC)], "ref")]
H.append(table("이수용 방식 대출 단위 Sharpe", ["정책", "승인율", "부도율", "연수익률", "rf 0%", "rf 1%", "rf 2%", "rf 3%", "국채"], rows,
               note=f"평가 대상 = 36개월 ≤2017.3, 60개월 ≤2015.3 발급(상태 무관) {num(LP.loc['전부 승인', '승인건수'])}건. 최종 모형 − A등급만(국채) {d3(lc('우리 방식 20%', '−A등급만'))}[{d3(lc('우리 방식 20%', '−A등급만_lo'))}, {d3(lc('우리 방식 20%', '−A등급만_hi'))}].",
               src="이수용 보고서 5장, 본 연구 계산"))
H.append(L2(f"이수용 보고서에서는 「A등급만」 규칙이 RAROC 모델을 이겼으나, 최종 모형(승인 20%)은 rf 5개 수준 중 {LSYRF}개에서 A등급만보다 높고, 발급연도 {LYRS[0]}~{LYRS[-1]}년 {len(LYRS)}개 해 중 {LSYYR}개 해에서 우세(승인 25% 기준)"))
H.append("<h4>다. 우수사례 1·3 방식과 현금흐름 지표</h4>")
mm = [("V3", "최종", "최종 모형 v3"), ("M6v2", "우리 방식 v2", "v2"), ("B0", "부도확률", "부도확률 순위"), ("GRADE", "세부등급", "세부등급 순서")]
rows = [([lab, f3(u3(k)), f3(u3(k, "승인25_시험SR_중앙")), usd(u1("가", m)), usd(u1("나", m)), f2(kp(k, "SR36_test")), f2(kp(k, "SR60_test")), f"{kp(k, 'ROE_test'):.2f}%"], "hl" if k == "V3" else "") for k, m, lab in mm]
H.append(table("우수사례 1·3 방식과 현금흐름 지표", ["모형", "우3 IRR Sharpe(검증 선택)", "우3 승인 25%", "우1 Total Risk(승인 25%)", "우1 Total Risk(거절=부도율)", "발급월 Sharpe 36개월", "발급월 Sharpe 60개월", "연 수익률(ROE)"], rows,
               note="우3: 100회 반복 중앙값(IRR 계산 불가 = −100% 규칙 R1, R0·R2는 Ⅴ-2). 우1: Total Risk = 놓친 부도 손실 + 놓친 수익(작을수록 좋음). 발급월 Sharpe·ROE는 승인 25%(사전계획 지점).", src="우수사례 1·3 방식 재현, 본 연구 계산",
               widths=["15%", "12%", "10%", "13%", "13%", "12%", "12%", "13%"]))
H.append("<h3>4. 해석: 목표 정렬의 효과</h3>")
H.append(L1("v3(이득 목표 = 과제 잣대 수익)는 과제 정의 Sharpe에서, v2(이득 목표 = 실제 보유기간 연환산 수익)는 현금흐름 ROE·이수용 잣대에서 근소하게 앞섬"))
H.append(L2(f"과제 정의 Sharpe: v3 − v2 = {d3(ci('T1', 'M6v2', 'd_final'), 4)}[{d3(ci('T1', 'M6v2', 'd_lo'), 4)}, {d3(ci('T1', 'M6v2', 'd_hi'), 4)}], ROE: v3 − v2 = {d3(pair('V3 − M6v2', 'dROE'))}%p[{d3(pair('V3 − M6v2', 'dROE_lo'))}, {d3(pair('V3 − M6v2', 'dROE_hi'))}]"))
H.append(L2(f"단, v3의 과제 정의 Sharpe 우위 일부는 2017~20년 발급분 회피에서 나옴(이 기간 승인율 v3 {pc(YA('V3'))} 대 v2 {pc(YA('M6v2'))}). 매달 같은 비율 규칙에서도 v3 {f3(MON['V3'])} 대 v2 {f3(MON['M6v2'])}로 앞서지만 차이는 작음(Ⅳ-4)"))
H.append(L2("모형이 「무엇을 맞히도록」 학습했는지가 어떤 잣대에서 이기는지를 결정함 → 최종 모형은 과제가 정한 목표(포트폴리오 Sharpe)에 맞춘 v3로 정하고, 다른 잣대의 손실 크기를 함께 공개"))
H.append(note("평가 잣대 자체도 가정을 담고 있음: 과제 정의 Sharpe는 계약만기 연율화로 분할상환·조기상환을 반영하지 못함(Ⅵ-1). 잣대가 바뀌면 최적 모형도 바뀔 수 있으므로 여러 잣대의 결과를 함께 보는 것이 안전함"))
H.append("</div>")

# ===== Ⅵ
d36 = DUR[(DUR["T"] == 36) & (DUR.상태 == "완납")].iloc[0]
rows = [[f"{r['T']}개월", r.상태, num(r.n), f"{r.계약WAL_년:.2f}", f"{r.계약듀레이션_년:.2f}", f"{r.실현WAL_년:.2f}", pc(r.약정금리), pc(r.수익_계약만기환산), pc(r.수익_실보유기간환산), pc(r.수익_듀레이션일관)] for _, r in DUR.iterrows()]
H.append("<div class='page'><h2>Ⅵ. 이론의 실증과 강건성</h2>")
H.append(note("이 장은 계량경제·채권 이론(잔존만기·듀레이션, 스프레드, 분위 포트폴리오, OLS 불편성, 레짐 전환, 생존분석)을 실제 자료로 구현해 모형의 작동 원리와 한계를 확인함. 최종 모형·승인율은 바꾸지 않는 진단 분석임"))
H.append("<h3>1. 잔존만기·듀레이션: 수익률을 무엇으로 나누는가</h3>")
H.append(L1("분할상환 대출은 원금이 매달 돌아오므로 실제 자금이 묶인 기간(가중평균만기, WAL)은 계약만기보다 훨씬 짧음"))
H.append(table("만기·상태별 기간 지표와 수익률 환산 방식 비교(만기 기준 코호트)", ["만기", "상태", "대출", "계약 WAL(년)", "계약 듀레이션(년)", "실현 WAL(년)", "약정금리", "계약만기 환산", "실보유기간 환산", "듀레이션 일관"], rows,
               note="계약 WAL·맥컬리 듀레이션은 약정금리 원리금균등 상환표로 계산. 실현 WAL = 받은 이자 ÷ 금리 ÷ 원금(잔액·년). 듀레이션 일관 수익률 = (총수령/원금)^(1/실현 WAL) − 1.", src=SRC_T))
H.append(L2(f"36개월 완납 대출: 계약 듀레이션 {d36.계약듀레이션_년:.2f}년, 조기상환으로 실현 WAL은 {d36.실현WAL_년:.2f}년. 듀레이션 일관 수익률({pc(d36.수익_듀레이션일관)})은 이론대로 약정금리({pc(d36.약정금리)})와 거의 같지만, 계약만기 환산은 {pc(d36.수익_계약만기환산)}로 과소평가"))
H.append(L2(f"과제 정의 Sharpe를 듀레이션 일관 수익률로 다시 계산해도 순위 유지: 최종 {f3(DT1.loc['V3', 'T1_듀레이션일관'])}, 부도확률 순위 {f3(DT1.loc['B0', 'T1_듀레이션일관'])}, 세부등급 {f3(DT1.loc['GRADE', 'T1_듀레이션일관'])}, 전부 승인 {f3(DT1.loc['ALL', 'T1_듀레이션일관'])}"))
H.append("<h3>2. 스프레드 분해와 예측 불편성(OLS, 발급월 군집 표준오차)</h3>")
ols_rows = []
for m in ("M1 스프레드", "M2 +세부등급", "M3 +부도확률", "M4 +우리 예측"):
    r = OLS.loc[m]; cell = lambda v: "—" if pd.isna(r.get(f"b_{v}")) else f"{r[f'b_{v}']:.3f}<br><small>({r[f'se_{v}']:.3f})</small>"
    ols_rows.append(([m, cell("spread"), "포함" if "등급" in m or m.startswith("M3") or m.startswith("M4") else "—", cell("p"), cell("mu"), f"{r.R2:.4f}", f"{r.AIC:,.0f}"], "hl" if m.startswith("M4") else ""))
ols_rows.append(([f"MZ: z = α + β·μ̂", "—", "—", "—", f"{MZ.b_mu:.3f}<br><small>({MZ.se_mu:.3f})</small>", f"{MZ.R2:.4f}", f"α = {MZ.alpha:.3f}({MZ.se_alpha:.3f})"], ""))
H.append(table("실현 초과수익 회귀(만기 기준 코호트 467,484건)", ["모형", "스프레드(%p)", "세부등급 더미", "부도확률(%p)", "예측 초과수익 μ̂", "R²", "AIC"], ols_rows,
               note=f"종속변수 z = 실현 초과수익(과제 잣대 수익 − 국채, %p). 괄호는 발급월 군집 표준오차. μ̂ 계수의 군집 표준오차는 단순 OLS 표준오차의 {M4o.se_mu / M4o.se_naive_mu:.1f}배. MZ 결합검정(α=0, β=1) F = {MZ.wald_F:.1f}(기각).",
               src="statsmodels OLS, 본 연구 계산", widths=["20%", "14%", "11%", "14%", "15%", "10%", "16%"]))
H.append(L2(f"스프레드만 넣으면 계수가 음(−)({OLS.loc['M1 스프레드', 'b_spread']:.3f}): 높은 금리는 높은 부도와 함께 오므로 「금리가 높을수록 더 번다」는 성립하지 않음"))
H.append(L2(f"우리 예측 μ̂을 넣으면 계수 {M4o.b_mu:.2f}(t = {M4o.t_mu:.1f})로 스프레드·등급·부도확률이 설명하지 못하는 부분을 설명하고, 스프레드 계수는 {M4o.b_spread:.3f}(유의하지 않음)로 줄어듦 → μ̂가 스프레드 정보를 흡수"))
H.append(L2(f"예측 불편성: 기울기 β = {MZ.b_mu:.3f}(표준오차 {MZ.se_mu:.3f})로 1과 구별되지 않아 크기 비례는 맞지만, 절편 α = {MZ.alpha:.2f}%p(표준오차 {MZ.se_alpha:.3f})로 결합검정(α=0, β=1)에서 불편성은 기각 → 실현 수익이 예측보다 평균 0.3%p 높은 보수적 편향"))
H.append("<h3>3. 10분위 롱-숏: 1등급 매수·10등급 매도 가상 실험</h3>")
H.append(L1("발급월·만기 안에서 점수로 10개 분위를 만들고, 1분위(가장 좋은 점수) 매수·10분위 매도 포트폴리오의 월별 초과수익을 계산"))
H.append(table("분위 포트폴리오 성과(만기 기준 코호트)", ["순위 기준", "1분위", "10분위", "롱-숏(발급월 평균)", "Newey–West t", "양(+)인 달", "롱-숏 Sharpe", "분위 단조성"],
               [([k, f"{LS.loc[k, 'D1_평균']:.2f}", f"{LS.loc[k, 'D10_평균']:.2f}", f"{LS.loc[k, '롱숏_평균']:.2f}", f"{LS.loc[k, 'NW_t']:.1f}", pc(LS.loc[k, '양수_월비율'], 0), f"{LS.loc[k, '롱숏_Sharpe']:.2f}", f"{-LS.loc[k, '단조성_Spearman']:.2f}"], "hl" if k == "최종 모형" else "") for k in LS.index],
               unit="%p, 연환산 초과수익", note=f"각 값은 연환산 초과수익의 발급월 평균. 롱-숏 Sharpe = 발급월 롱-숏 값의 평균 ÷ 표준편차(연환산·거래 가능 Sharpe 아님). Newey–West 시차 6. 분위 단조성 = 분위 번호와 분위 평균 초과수익의 순위상관(부호 반전, 1 = 완전 단조). 발급월 {int(LSf.개월)}개.", src=SRC_T))
H.append(L2(f"최종 모형의 롱-숏 스프레드 {LSf.롱숏_평균:.1f}%p는 부도확률 순위({LS.loc['부도확률 순위', '롱숏_평균']:.1f}%p)·세부등급({LS.loc['세부등급 순위', '롱숏_평균']:.1f}%p)보다 크고, 1~10분위가 완전 단조"))
H.append(note("P2P 대출은 공매도가 불가능하므로 롱-숏은 순위의 판별력 검증용 가상 실험이며, 실제 운영은 상위 분위만 사는 롱 온리(승인) 전략임"))
H.append(figure("fig06_분위롱숏.png", "분위별 평균 초과수익(왼쪽)과 최종 모형 1분위 − 10분위 월별 추이(오른쪽)", note="2007~2009년 초기에는 월별 대출 수가 적어 변동이 큼.", src=SRC_T))
H.append("<h3>4. 1·2차 모먼트: 평균과 위험을 함께 예측</h3>")
H.append(L2(f"예측 평균(1차)과 실현 평균의 10분위 상관 {PS['모먼트_상관']['평균']:.3f}, 예측 표준편차(2차)와 실현 표준편차의 상관 {PS['모먼트_상관']['표준편차']:.3f} → 두 모먼트 모두 크기 순서를 잘 맞춤"))
H.append(L2("위험 예측은 하위 분위에서 실제보다 약간 크게 잡혀(보수적), 점수 μ̂/σ̂에서 위험이 낮은 대출을 과도하게 우대하지 않음"))
H.append(figure("fig07_모먼트.png", "예측 대 실현: 기대 초과수익(왼쪽)과 초과수익 표준편차(오른쪽), 10분위", src=SRC_T, w="82%"))
H.append("<h3>5. 레짐 전환: 신용 국면에 따라 선별의 가치가 달라지는가</h3>")
H.append(L1(f"학습 자료 36개월 대출의 발급월 평균 초과수익({REG['first'] // 100}.{REG['first'] % 100:02d}~{REG['last'] // 100}.{REG['last'] % 100:02d}, {REG['months']}개월)에 평균·분산이 바뀌는 2국면 마르코프 전환 모형을 최우추정(가우스 오차)"))
H.append(table("마르코프 전환 모형 추정 결과", ["국면", "평균 초과수익(%p)", "분산", "유지 확률", "기대 지속(개월)", "해당 월 수"],
               [[("악화 국면" if i == bad else "정상 국면"), f"{REG['mean'][i]:.2f}", f"{REG['sigma2'][i]:.3f}", f"{REG['p_stay'][i]:.3f}", f"{REG['dur'][i]:.0f}", str(REG['bad_months'] if i == bad else REG['months'] - REG['bad_months'])] for i in (good, bad)],
               note=f"AIC: 2국면 {REG['aic']:.1f}, 1국면 {REG['aic_1국면']:.1f} → 2국면 모형 선택. 국면 분류는 평활 확률 0.5 기준.", src="statsmodels MarkovRegression, 학습 자료"))
H.append(L2(f"악화 국면(2009년 말, 2015년 이후)에서 전부 승인 Sharpe는 {f3(rgv('악화 국면', '전부 승인'))}로 낮지만 최종 모형은 {f3(rgv('악화 국면', '최종 모형'))} → 선별 효과 +{rgv('악화 국면', '최종 모형') - rgv('악화 국면', '전부 승인'):.3f}(정상 국면 +{rgv('정상 국면', '최종 모형') - rgv('정상 국면', '전부 승인'):.3f})"))
H.append(note("국면은 사후 실현 수익으로 식별한 것이므로 그대로 운영에 쓸 수 없음. 운영에는 발급 시점에 관측 가능한 선행 지표(초기 연체율, 금리·스프레드)로 국면 확률을 추정해야 함(Ⅷ-3)"))
H.append(figure("fig08_레짐.png", "발급월 평균 초과수익과 악화 국면 확률(학습 자료, 36개월 대출)", src="학습 자료, statsmodels MarkovRegression", w="92%"))
H.append("<h3>6. 무위험수익률·비용·가중·기간</h3>")
kk = [("V3", "최종 모형"), ("B0", "부도확률 순위"), ("ALL", "전부 승인")]
cols = [("T1", "기본"), ("T1_금액가중", "금액 가중"), ("T1_관리비1pct", "관리비 1%"), ("T1_rf0", "rf 0%"), ("T1_rf3", "rf 3%"), ("T1_36m", "36개월"), ("T1_60m", "60개월"), ("T1_2007–11", "2007~11"), ("T1_2012–16", "2012~16"), ("T1_2017–20", "2017~20")]
H.append(table("과제 정의 Sharpe 강건성", ["전략"] + [b for _, b in cols], [([lab] + [f3(ROB.loc[k, c]) for c, _ in cols], "hl" if k == "V3" else "") for k, lab in kk],
               note="승인 집합은 기본과 같게 고정하고 계산 방식·평가 범위만 바꿈. 만기·연도는 해당 범위의 대출로 계산(승인은 전체 풀에서 판정). 팀 공유 Sharpe 보고서의 금액 가중 값은 0.148.", src=SRC_T))
H.append(L2("모든 변형에서 최종 모형 > 부도확률 순위 > 전부 승인. rf가 3%로 오르면 모든 전략이 0 근처로 내려감. 이 표는 승인 집합을 고정한 계산이므로, 금리에 따라 승인율을 조정하는 규칙의 효과는 따로 검증해야 함(Ⅷ-3)"))
H.append(figure("fig09_rf민감도.png", "무위험수익률 수준별 Sharpe: 과제 정의(왼쪽)·기회풀(오른쪽)", src="본 연구 계산, M4 최종보고서 7장", w="88%"))
H.append("<h3>7. 시간 외 검증</h3>")
o15 = O15[O15.ruler == "T2"]; o16 = O16[O16.ruler == "T1"]
H.append(table("시간 외 검증(과거만으로 다시 학습)", ["구간", "학습 기간", "대출", "전부 승인", "세부등급", "부도확률 순위", "최종 모형", "비교(인쇄값)"], [
    (["2016~17 발급, 기회풀", "2015년까지", num(oot(o15, "V3", "n")), f3(oot(o15, "ALL")), f3(oot(o15, "GRADE")), f3(oot(o15, "B0")), f3(OOT15), f"M4 MeanVar {f3(pv('T2_OOT_2016-2017', 'M4_MeanVar_Util(자유 임계값)'))} · M2 {f3(pv('T2_OOT_2016-2017', 'M2_PD_Decile_Mix(승인 30%)'))}"], "hl"),
    (["2017~20 발급, 과제 정의(승인 26%)", "2016년까지", num(oot(o16, "V3", "n")), f3(oot(o16, "ALL")), f3(oot(o16, "GRADE")), f3(oot(o16, "B0")), f3(oot(o16, "V3")), "—"], ""),
    (["2017~20 발급, 과제 정의(승인 6.4%)", "2016년까지", num(oot(o16, "V3", "n")), "—", f3(oot(o16, "GRADE", "SR_승인6_4")), f3(oot(o16, "B0", "SR_승인6_4")), f3(OOT16), f"Sharpe 보고서 {f3(pv('T1_OOT_2017-2020', '제안 모형'))}"], "")],
    note=f"승인율은 전체 학습 자료 교차검증 값({QP} 등, 각 전략 각자 값)을 그대로 씀 — 시간 외 학습 구간만으로 다시 고르지 않음. 승인 6.4% 행은 Sharpe 보고서 OOT 승인율에 맞춘 사후 비교. 전부 승인 2017~20 {f3(oot(o16, 'ALL'))}은 Sharpe 보고서 인쇄값 {f3(pv('T1_OOT_2017-2020', '전체 승인'))}, 2016~17 {f3(oot(o15, 'ALL'))}은 M4 보고서 인쇄값 {f3(pv('T2_OOT_2016-2017', '전부 승인'))}과 대조.", src="scripts/127, 팀 공유 보고서 2종", widths=["22%", "10%", "9%", "9%", "9%", "11%", "10%", "20%"]))
H.append(L2(f"2016~17년(2015년까지 학습): 최종 모형 {f3(OOT15)}로 M4 보고서의 최고 값보다 높고, 부도확률 순위({f3(oot(o15, 'B0'))})보다도 높음"))
H.append(L2(f"2017~20년(2016년까지 학습): 결과 확정 대출 기준으로는 모든 전략·승인율에서 음(−)이며, 같은 승인율 6.4%에서 최종 {f3(OOT16)}로 Sharpe 보고서({f3(pv('T1_OOT_2017-2020', '제안 모형'))})보다 낮음 → 이 구간에서는 열세로 판정"))
H.append(L2("원인 추정: (1) 이 구간의 결과 확정 대출은 2018년 42%, 2019년 16%만 확정된 조기 종료 표본이고, (2) 국채금리 상승(평균 2.0%)으로 대출의 초과수익이 구조적으로 낮아진 국면임. 운영상으로는 「고정 승인율」보다 「국면에 따라 승인율을 낮추는 규칙」이 필요(Ⅷ-3)"))
H.append(figure("fig10_시간외검증.png", "시간 외 검증 승인율별 Sharpe", note="빨간 빈 원은 팀 공유 보고서 인쇄값. 세로 점선은 운영 승인율.", src="scripts/127, 팀 공유 보고서 2종"))
H.append("<h3>8. 생존분석: 상환 중 대출을 버리지 않는 평가</h3>")
H.append(L1("결과 확정 대출만 쓰는 평가는 상환 중 대출 41만여 건을 버림 → 이를 중도절단 관측으로 포함한 경쟁위험 누적부도율(Aalen–Johansen)로 승인·거절 집단을 비교"))
H.append(table("최종 모형 승인·거절 집단의 누적부도율(시험 전체 1,170,198건)", ["만기", "경과", "전체 신청", "최종 모형 승인", "최종 모형 거절", "거절 − 승인"],
               [[f"{t}개월", f"{m}개월", pc(cif(t, '전체 신청', m)), pc(cif(t, '최종 모형 승인', m)), pc(cif(t, '최종 모형 거절', m)), d3((cif(t, '최종 모형 거절', m) - cif(t, '최종 모형 승인', m)) * 100, 1) + "%p"] for t, ms in ((36, (12, 24, 36)), (60, (12, 36, 60))) for m in ms],
               unit="%", note=f"부도 = 사건, 완납 = 경쟁 사건, 상환 중·연체·유예 = 마지막 납입 월에서 중도절단. 승인 = 시험 전체 신청의 점수 상위 {QP}.", src="lending_club_2020_test.csv 전체, 본 연구 계산"))
H.append(L2(f"36개월 대출의 36개월 누적부도율: 승인 {pc(cif(36, '최종 모형 승인', 36))}, 거절 {pc(cif(36, '최종 모형 거절', 36))}, 60개월 대출의 60개월: 승인 {pc(cif(60, '최종 모형 승인', 60))}, 거절 {pc(cif(60, '최종 모형 거절', 60))}"))
H.append(L2(f"승인 집단은 2016년 이전 발급분에 몰려 있어(Ⅳ-4) 위 비교에는 발급 시기 차이가 섞임 → 같은 발급연도(36개월 대출) 안에서 다시 비교해도 2015년 {pc(wvv(2015, '승인'))} 대 {pc(wvv(2015, '거절'))}, 2016년 {pc(wvv(2016, '승인'))} 대 {pc(wvv(2016, '거절'))}, 2017년 {pc(wvv(2017, '승인'))} 대 {pc(wvv(2017, '거절'))}로 상환 중 대출을 포함해도 선별력 유지"))
H.append(figure("fig05_누적부도.png", "발급 후 경과 개월별 누적부도율(경쟁위험)", src="lending_club_2020_test.csv 전체, 본 연구 계산", w="94%"))
H.append("<h3>9. 부도확률 보정 검정</h3>")
H.append(L2(f"Cox 보정 회귀 logit P(부도) = a + b·logit(p̂): 기울기 b = {CAL['slope']:.3f}(표준오차 {CAL['slope_se']:.3f}, 1보다 약간 커 예측 확률이 다소 완만), 절편 a = {CAL['intercept']:.3f}({CAL['intercept_se']:.3f}). 평균 예측 {pc(CAL['mean_pred'], 2)}, 실제 {pc(CAL['mean_obs'], 2)}, 10분위 최대 차이 {CAL['max_abs_decile_gap'] * 100:.2f}%p, ECE {CAL['ECE'] * 100:.2f}%p"))
H.append(L2(f"Hosmer–Lemeshow 카이제곱 통계량 = {CAL['HL_chi2']:.0f}(자유도 8)로 기각되지만, 표본 {num(CAL['n'])}건에서는 0.8%p 이하의 차이도 유의하게 잡힘 → 통계적 유의성보다 차이의 크기(실질적 유의성)로 판단할 때 보정은 양호"))
H.append(figure("fig11_보정.png", "부도확률 10분위별 예측 대 실제(시험)", src=SRC_T, w="46%"))
H.append("</div>")

# ===== Ⅶ 채권·MBS 형식(131)
H.append("<div class='page'><h2>Ⅶ. 채권·MBS 형식 평가</h2>")
H.append(note("「채권분석」 강의안의 수익률(4장)·듀레이션(5장)·볼록성(6장)·수익률 스프레드(7.5절)·주택저당증권(10장) 형식을 대출에 그대로 대입해, 대출 풀을 같은 시기의 회사채·MBS처럼 가격과 수익률로 다시 평가함. 최종 모형·승인율은 바꾸지 않는 평가 분석임"))
H.append("<h3>1. 대출을 채권으로 보기: 약정·실현 수익률과 스프레드</h3>")
H.append("<div class='formula'>약정 스프레드 = 약정금리(만기수익률) − 만기 맞춤 국채 &nbsp;|&nbsp; 실현 스프레드(실현 유효 가산금리) = 풀 현금흐름 수익률 − 국채<br>"
         "풀 현금흐름 수익률 = 12 × IRR(−Σ대출액, 나이별 순현금 합계)<br>실현 스프레드(수수료 전) = 약정금리와 같은 기준(납입액 + 순회수), 수수료 후 = 투자자 순현금(서비스 수수료 1% 차감) &nbsp;|&nbsp; 보존율 = 실현(수수료 전) ÷ 약정 &nbsp;|&nbsp; CNL(누적 순손실) = (상각 원금 − 순회수) ÷ 대출액</div>")
PLB = [("V3", "최종 모형 승인"), ("V3_거절", "최종 모형 거절"), ("M6v2", "v2 승인"), ("B0", "부도확률 순위 승인"), ("ALL", "전부 승인")]
H.append(table("풀별 약정·실현 수익률과 스프레드(만기 기준 코호트)", ["풀", "만기", "대출", "약정금리", "약정 스프레드", "실현 스프레드(수수료 전)", "실현 스프레드(수수료 후)", "보존율", "실현 WAL(년)", "CNL"],
               [([lab, f"{t}개월", num(bp(g, t, "n")), pc(bp(g, t, "약정금리"), 2), sp_(bp(g, t, "약정스프레드")), sp_(bp(g, t, "실현스프레드_전")), sp_(bp(g, t, "실현스프레드_후")), pc(bp(g, t, "보존율"), 0), f"{bp(g, t, 'WAL'):.2f}", pc(bp(g, t, "CNL"))], "hl" if g == "V3" else "") for t in (36, 60) for g, lab in PLB],
               note="풀 = 해당 대출을 대출 나이 기준으로 합친 자동이전증권형 정태 풀(static pool). 상각 = 마지막 납입 + 6개월, 회수금은 회수 수수료 차감(설계상 허용 가정). 보존율은 약정금리와 같은 수수료 전 기준이며, 수수료 전·후의 차이는 서비스 수수료로 신용 손실이 아님.", src=SRC_T,
               widths=["16%", "7%", "9%", "8%", "9%", "11%", "11%", "8%", "10%", "11%"]))
H.append(L2(f"36개월 최종 모형 승인 풀의 실현 스프레드는 수수료 전 {sp_(bp('V3', 36, '실현스프레드_전'))}로 약정 스프레드({sp_(bp('V3', 36, '약정스프레드'))})의 {pc(bp('V3', 36, '보존율'), 0)}를 지킴(수수료 1%를 빼면 {sp_(bp('V3', 36, '실현스프레드_후'))}). 전부 승인 풀은 {sp_(bp('ALL', 36, '실현스프레드_전'))}({pc(bp('ALL', 36, '보존율'), 0)}), 거절 풀은 {sp_(bp('V3_거절', 36, '실현스프레드_전'))}({pc(bp('V3_거절', 36, '보존율'), 0)})"))
H.append(L2(f"승인 풀은 약정 스프레드가 전체보다 낮지만({sp_(bp('V3', 36, '약정스프레드'))} 대 {sp_(bp('ALL', 36, '약정스프레드'))}) 실현 스프레드는 높음 → 선별의 효과는 「금리가 높은 대출」이 아니라 「약정 스프레드를 지키는 대출」을 고르는 데서 나옴"))
H.append(table("등급별 신용 스프레드 곡선(36개월, 만기 기준 코호트)", ["구분"] + [f"{g}등급" for g in "ABCDEFG"],
               [["약정 스프레드"] + [sp_(bp(f"등급{g}", 36, "약정스프레드")) for g in "ABCDEFG"], ["실현 스프레드"] + [sp_(GSP[g]) for g in "ABCDEFG"],
                ["보존율(수수료 전)"] + [pc(bp(f"등급{g}", 36, "보존율"), 1) for g in "ABCDEFG"], ["CNL"] + [pc(bp(f"등급{g}", 36, "CNL")) for g in "ABCDEFG"]], src=SRC_T))
H.append(L2(f"등급이 낮을수록 약정 스프레드는 커지지만(강의안 7.5절: 신용등급이 낮을수록 스프레드 증가), 실현 스프레드는 {GPK}등급({sp_(GSP[GPK])})에서 가장 크고 그 아래로 줄어 F·G등급은 음(−)({sp_(GSP['F'])}, {sp_(GSP['G'])}) → 하위 등급의 약정 스프레드는 손실을 보상하지 못함"))
H.append("<h3>2. 새 잣대: 보유기간수익률 Sharpe</h3>")
H.append("<div class='formula'>보유기간수익률 HPR = (Σ<sub>j</sub> 순현금<sub>j</sub> × (1 + rf/12)<sup>T−j</sup> ÷ 대출액)<sup>12/T</sup> − 1 (중간 현금흐름을 국채로 계약만기 T까지 재투자, 강의안 4.3절)<br>"
         "승인 x = HPR, 거절 x = rf, Sharpe = (평균 x − 평균 rf) ÷ 표준편차 x &nbsp;|&nbsp; 판정(사전 고정): 최종 − 부도확률 순위의 95% 구간 하한 > 0 이면 우세</div>")
HK = [("V3", "최종 모형 v3"), ("M6v2", "v2"), ("B0", "부도확률 순위"), ("GRADE", "세부등급 순서"), ("ALL", "전부 승인")]
H.append(table("보유기간수익률 Sharpe(시험 결과 확정 744,608건)", ["전략", "승인율", "과제 정의 Sharpe", "보유기간수익률 Sharpe [95% 구간]", "승인 대출 초과 HPR", "최종 − 해당 [95% 구간]"],
               [([lab, pc(HRT.loc[k, "q"]), f3(HRT.loc[k, "T1"]), f"{f3(HRT.loc[k, 'HPR_SR'])} [{f3(HRT.loc[k, 'SR_lo'])}, {f3(HRT.loc[k, 'SR_hi'])}]", pc(HRT.loc[k, "승인_초과HPR"], 2),
                  "—" if k == "V3" else f"{d3(HRT.loc[k, 'd_V3'], 4)} [{d3(HRT.loc[k, 'd_lo'], 4)}, {d3(HRT.loc[k, 'd_hi'], 4)}]"], "hl" if k == "V3" else "") for k, lab in HK],
               note="승인 집합은 각 전략의 과제 정의 잣대 승인 집합 그대로(다시 고르지 않음). 구간은 발급월 블록 부트스트랩 1,000회(쌍체). HPR은 서비스 수수료 1%를 뺀 순현금 기준. 계약만기 뒤에 들어오는 회수금은 만기 시점으로 할인하고, 만기 가치가 음수면 0으로 둠(영향 작음).", src=SRC_T))
H.append(L2("과제 정의 잣대는 조기상환으로 받은 돈을 계약만기까지 수익 없이 두는 셈이지만(Ⅵ-1), 채권의 보유기간수익률은 이를 국채로 재투자 → 조기상환이 잦은 대출의 수익을 채권과 같은 방식으로 셈"))
H.append(L2(f"최종 모형 {f3(HV.HPR_SR)}는 부도확률 순위({f3(HB.HPR_SR)})보다 {d3(HB.d_V3)}[{d3(HB.d_lo)}, {d3(HB.d_hi)}] 높아 사전 판정 기준으로 {'우세' if HB.d_lo > 0 else '동등'}. v2({f3(HM.HPR_SR)})와의 차이는 {d3(HM.d_V3, 4)}[{d3(HM.d_lo, 5)}, {d3(HM.d_hi, 5)}]로 {'구간이 0을 포함해 동등(점추정은 v2가 높음)' if HM.d_lo <= 0 <= HM.d_hi else '유의'} → 과제 잣대에 맞춘 v3의 목표 정렬 이점은 재투자를 반영한 잣대에서는 나타나지 않음(Ⅴ-4와 같은 결론)"))
H.append("<h3>3. 조기상환·부도 속도: CPR·CDR·PSA</h3>")
H.append("<div class='formula'>SMM<sub>t</sub> = Σ(t월 만기 전 완납 대출의 예정잔액) ÷ Σ(t월 위험집합의 예정잔액), CPR = 1 − (1 − SMM)<sup>12</sup> (강의안 10.1절, 예정잔액 = p.88 원금잔액 식)<br>"
         "100% PSA: CPR<sub>t</sub> = min(0.2% × t, 6%) &nbsp;|&nbsp; CDR: 같은 방식의 부도(첫 미납월) 비율 &nbsp;|&nbsp; 상환 중 대출은 마지막 납입월에서 중도절단</div>")
PSR = [("36_전체", "36개월 전체"), ("36_승인", "36개월 최종 모형 승인"), ("36_거절", "36개월 최종 모형 거절"), ("60_전체", "60개월 전체"), ("60_승인", "60개월 최종 모형 승인"), ("60_거절", "60개월 최종 모형 거절")]
H.append(table("조기상환·부도 속도(시험 전체 1,170,198건, 상환 중 포함)", ["집단", "PSA 속도", "평균 CPR", "평균 CDR", "PSA 속도(30개월 이하)", "평균 CPR(30개월 이하)"],
               [([lab, f"{PST.loc[g, 'PSA속도'] * 100:.0f}%", pc(PST.loc[g, "평균CPR"]), pc(PST.loc[g, "평균CDR"]), f"{PST.loc[g, 'PSA속도_30'] * 100:.0f}%", pc(PST.loc[g, "평균CPR_30"])], "hl" if "승인" in g else "") for g, lab in PSR],
               note=f"PSA 속도 = 위험집합 잔액 가중 최소제곱으로 CPR<sub>t</sub> ≈ k × 100% PSA를 맞춘 k. 평균은 위험집합 잔액 가중. 30개월 이하 = 만기 직전 최종 납입 기록 차이를 뺀 점검. 부도 달은 조기상환 위험집합에서 제외하고, 연체·유예 대출은 마지막 납입월에서 중도절단(자료 끝 무렵 CDR이 약간 낮게 잡힐 수 있음). 승인 = 시험 전체 신청의 점수 상위 {QP}(Ⅵ-8과 같음).", src="lending_club_2020_test.csv 전체, 본 연구 계산"))
H.append(figure("fig13_CPR.png", "대출 나이별 조기상환율(CPR, 왼쪽)과 부도율(CDR, 오른쪽), 36개월 대출", note="주황 점선은 주택담보대출 기준 100% PSA와 전체에 맞춘 PSA 곡선. 만기 직전(34~35개월)의 급등은 잔액이 작은 구간의 완납으로, 최종 납입 시점 기록 차이가 섞였을 수 있음.", src="lending_club_2020_test.csv 전체, 본 연구 계산"))
H.append(L2(f"36개월 대출의 평균 CPR은 {pc(PST.loc['36_전체', '평균CPR'])}로 주택담보대출 기준 속도(100% PSA, 정체 구간 CPR 6%)의 {PST.loc['36_전체', '평균CPR'] / 0.06:.1f}배, 60개월은 {pc(PST.loc['60_전체', '평균CPR'])}. 만기 직전 달을 빼도(30개월 이하 {pc(PST.loc['36_전체', '평균CPR_30'])}, {PST.loc['36_전체', 'PSA속도_30'] * 100:.0f}% PSA) CPR은 PSA형 정체 없이 나이와 함께 계속 올라(19개월 {pc(cpr_at(19), 0)} → 30개월 {pc(cpr_at(30), 0)}), 대출은 MBS보다 훨씬 「짧은」 자산"))
H.append(L2(f"최종 모형 승인 대출은 CPR {pc(PST.loc['36_승인', '평균CPR'])}(거절 {pc(PST.loc['36_거절', '평균CPR'])})로 더 빨리 상환되고, CDR {pc(PST.loc['36_승인', '평균CDR'])}로 거절({pc(PST.loc['36_거절', '평균CDR'])})의 {PST.loc['36_승인', '평균CDR'] / PST.loc['36_거절', '평균CDR'] * 100:.0f}% → 실현 WAL이 짧고 신용 손실이 작음(1절)"))
H.append(L2(f"부도 대출의 손실률(36개월, ABS 관례 지표): 승인 {pc(BS['손실률_36']['V3'])}, 전체 {pc(BS['손실률_36']['ALL'])} → 무담보 대출이라 회수가 적어 손실은 부도 빈도로 거의 결정됨"))
AGB = ["(3, 6]", "(6, 12]", "(12, 18]", "(18, 24]", "(24, 36]", "(36, 60]"]
agl = lambda b: "나이 {}~{}개월".format(int(b.strip("(]").split(", ")[0]) + 1, b.strip("(]").split(", ")[1])
H.append(table("조기상환 결정요인: 셀 집계 이항 GLM(logit)", ["설명변수", "계수", "군집 표준오차", "z(군집)"],
               [(["금리 하락 폭 ΔTsy(%p)", f"{BGL.loc['dtsy', '계수']:.3f}", f"{BGL.loc['dtsy', '군집표준오차']:.3f}", f"{BGL.loc['dtsy', '군집z']:.1f}"], "hl")]
               + [[agl(b), f"{BGL.loc[f'C(agebin)[T.{b}]', '계수']:.3f}", f"{BGL.loc[f'C(agebin)[T.{b}]', '군집표준오차']:.3f}", f"{BGL.loc[f'C(agebin)[T.{b}]', '군집z']:.1f}"] for b in AGB],
               note=f"(발급월 × 나이 × 만기 × 등급) {num(GB['셀'])}칸, 위험 대출-월 {num(GB['위험월'])}, 만기 전 완납 {num(GB['조기상환'])}건. 기준 = 나이 1~3개월. 계절(1~12월)·등급·만기·발급연도 더미 포함(표시 생략), 계절 결합 Wald 카이제곱 {GB['계절_chi2_cl']:.0f}(자유도 {GB['계절_df']}, {'p < 0.001' if GB['계절_p_cl'] < 0.001 else 'p ≥ 0.001'}). 표준오차는 달력 연월 {GB['군집']}개 군집 샌드위치: ΔTsy·계절은 같은 연월의 모든 칸에 공통이라, 칸 독립 가정(Pearson 척도 {GB['scale']:.2f})으로는 ΔTsy z = {GB['z_dtsy']:.1f}로 과대.",
               src="statsmodels GLM, lending_club_2020_test.csv 전체", widths=["40%", "20%", "20%", "20%"]))
H.append(L2(f"사전 가설 판정: 금리 하락 폭 계수 {GB['beta_dtsy']:.3f}(달력 연월 군집 z = {GB['z_cl_dtsy']:.1f}) → 「금리 하락 시 조기상환 증가」(강의안 10.1절 ①)는 이 자료에서 기각(부호 반대). 나이가 들수록 조기상환이 늘고(⑤ 경과 기간), 계절(1~12월) 효과도 군집 기준으로 결합 유의(④ 계절 요인)"))
H.append(note("추정: 무담보 소비자 대출은 신용 스프레드가 커서 국채금리 변화로 생기는 재융자 유인이 작고, 국채금리 하락기는 경기 둔화·신용 긴축기와 겹쳐 오히려 재융자가 어려웠을 수 있음. 셀 집계 회귀의 연관이며, 같은 시기의 경기·신용 여건 변화와 구분되지 않으므로 인과 효과가 아님"))
H.append("<h3>4. 자동이전증권과 순차 지급 CMO</h3>")
H.append(L1("36개월 풀의 원금 회수(예정·조기상환·순회수)를 A(70%) → B(20%) → C(10%) 순서로 지급하고, 순손실은 C → B → A 순서로 차감하는 순차 지급 CMO(강의안 10.2절)를 구성"))
H.append(table("순차 지급 CMO 트랜치별 손실률과 WAL(36개월 풀, 예시 구조)", ["풀", "CNL", "A(70%) 손실률", "A WAL(년)", "B(20%) 손실률", "B WAL(년)", "C(10%) 손실률", "C WAL(년)"],
               [([lab, pc(cm(g, "A", "CNL")), *[x for tr in "ABC" for x in (pc(cm(g, tr, "손실률")), f"{cm(g, tr, 'WAL'):.2f}")]], "hl" if g == "V3" else "") for g, lab in (("V3", "최종 모형 승인"), ("B0", "부도확률 순위 승인"), ("ALL", "전부 승인"))],
               note="손실 배분(후순위부터 차감)은 민간 ABS 관례(강의안의 CMO는 손실이 없는 기관 MBS). 초과 이자(excess spread)는 손실 보전에 쓰지 않은 보수적 구조.", src=SRC_T))
ABZ = bool((CMOT[CMOT.트랜치.isin(["A", "B"])].손실률 == 0).all())
H.append(L2(f"{'세 풀 모두 A·B 트랜치는 손실이 없고 손실은 C 트랜치가 흡수' if ABZ else '손실은 주로 C 트랜치가 흡수'}: C 손실률 최종 모형 승인 {pc(cm('V3', 'C', '손실률'), 0)}, 부도확률 순위 {pc(cm('B0', 'C', '손실률'), 0)}, 전부 승인 {pc(cm('ALL', 'C', '손실률'), 0)} → 승인 풀은 같은 구조에서 후순위 손실이 훨씬 작음"))
H.append(L2(f"부도확률 순위 풀의 손실이 가장 작지만(CNL {pc(cm('B0', 'A', 'CNL'))}) 실현 스프레드는 {sp_(bp('B0', 36, '실현스프레드_전'))}로 최종 모형({sp_(bp('V3', 36, '실현스프레드_전'))})보다 낮음 → 기준은 손실 최소가 아니라 손실 대비 스프레드"))
H.append(L2(f"원금을 순서대로 갚아 승인 풀의 A 트랜치 WAL은 {cm('V3', 'A', 'WAL'):.2f}년, C는 {cm('V3', 'C', 'WAL'):.2f}년으로 만기 구간이 나뉨 → 강의안 10.2절 순차 지급 CMO의 목적(조기상환 위험의 재배분)이 대출 풀에서도 그대로 작동"))
H.append("<h3>5. 유효 듀레이션·볼록성: 금리가 움직이면 대출 풀 가격은?</h3>")
H.append("<div class='formula'>ED = (P<sub>−</sub> − P<sub>+</sub>) ÷ ((y<sub>+</sub> − y<sub>−</sub>) × P) &nbsp;|&nbsp; ECV = (P<sub>+</sub> + P<sub>−</sub> − 2P) ÷ ((0.5(y<sub>+</sub> − y<sub>−</sub>))² × P) (강의안 5.6·6.4절, ±50bp)<br>"
         "조기상환 SMM<sub>t</sub>(Δ) = SMM<sub>t</sub> × exp(β × Δ), Δ = −수익률 변화(%p), β = β̂(3절 ΔTsy 계수) · 0(고정) · +|β̂|(가상 MBS형: 금리 하락 시 조기상환 증가), 부도는 고정</div>")
H.append(table("36개월 대표 풀의 유효 듀레이션(ED)·볼록성(ECV): 조기상환 금리 반응 가정별", ["풀", "증권"] + [f"{b} {m}" for _, b in CASEB for m in ("ED", "ECV")],
               [([lab, sec] + [f"{dv(g, case, sec, m):.2f}" for case, _ in CASEB for m in ("ED", "ECV")], "hl" if (g == "V3" and sec == "자동이전증권") else "") for g, lab in (("V3", "최종 모형 승인"), ("ALL", "전부 승인")) for sec in ("자동이전증권", "IO", "PO")],
               note=f"기대 현금흐름 모형: 풀 금액가중 약정금리, 나이별 SMM·MDR(3절), 손실률(회수 6개월 후). 기준 수익률 = 1절의 수수료 전 풀 수익률(승인 {pc(dv('V3', '고정(β=0)', '자동이전증권', '기준수익률'), 2)}, 전체 {pc(dv('ALL', '고정(β=0)', '자동이전증권', '기준수익률'), 2)}). IO = 이자, PO = 원금·회수. β̂ = {GB['beta_dtsy']:.3f}.", src="본 연구 계산",
               widths=["17%", "13%", "11.6%", "11.6%", "11.6%", "11.6%", "11.6%", "12%"]))
H.append(figure("fig15_가격곡선.png", "수익률 변화에 따른 가격(최종 모형 승인 풀, 기준 = 100)", note="실선 = 추정 반응(β̂), 점선 = 고정(β = 0), 점-선 = 가상 MBS형(β = +|β̂|).", src="본 연구 계산"))
H.append(L2(f"승인 풀 자동이전증권의 유효 듀레이션은 세 가정에서 {min(dv('V3', c, '자동이전증권', 'ED') for c, _ in CASEB):.2f}~{max(dv('V3', c, '자동이전증권', 'ED') for c, _ in CASEB):.2f}로, 계약 상환표 기준 듀레이션({d36.계약듀레이션_년:.2f}년, Ⅵ-1)보다 짧음 → 빠른 조기상환 때문"))
H.append(L2(f"자동이전증권은 ±50bp에서 β̂와 고정의 가격 차이가 {pc(BS['가격차_50bp']['V3'], 2)} 이하 → 3년 안에 원금을 상각하는 짧은 대출이라 조기상환 옵션의 가격 효과가 작음. 볼록성 값(β̂ {dv('V3', CASEB[0][0], '자동이전증권', 'ECV'):.2f}, 고정 {dv('V3', CASEB[1][0], '자동이전증권', 'ECV'):.2f}, 가상 MBS형 {dv('V3', CASEB[2][0], '자동이전증권', 'ECV'):.2f})의 차이는 이 작은 가격 차를 0.005²로 나눠 커진 것이라, MBS와 볼록성을 비교하는 근거로 쓰지 않음"))
H.append(L2(f"조기상환 위험은 이자(IO)와 원금(PO)으로 나눌 때 드러남: IO의 유효 듀레이션은 β̂ {dv('V3', CASEB[0][0], 'IO', 'ED'):.2f}, 고정 {dv('V3', CASEB[1][0], 'IO', 'ED'):.2f}, 가상 MBS형 {dv('V3', CASEB[2][0], 'IO', 'ED'):.2f} → 금리 반응의 부호가 IO 금리 위험의 방향을 정하므로, 대출 풀로 IO·PO형 상품을 만든다면 조기상환 금리 반응의 인과 추정이 먼저 필요"))
H.append(note("β̂는 3절의 연관 추정치(국채금리 변화에 대한 logit 계수)인데, 여기서는 풀 수익률(신용 스프레드 포함) 변화에 그대로 쓰고 SMM에 exp(βΔ)를 곱하는 근사를 씀. 금리 변화는 할인율과 조기상환에만 반영하고 부도율·손실률은 고정한 모형 계산임"))
H.append("<h3>6. 같은 시기 시장 상품과의 비교: 위험 사다리</h3>")
H.append(figure("fig14_스프레드사다리.png", "발급연도별 국채 대비 스프레드: 시장 상품 대 LendingClub", note="시장 상품은 약정(사전) 스프레드, LendingClub 실현은 36개월 만기 기준 코호트 풀(2017년은 1~3월 발급분).", src="FRED(AAA·BAA·MORTGAGE30US·TERMCBPER24NS·DGS2·DGS10), 본 연구 계산"))
LCOL = [("Aaa 회사채", "Aaa"), ("Baa 회사채", "Baa"), ("30년 모기지", "모기지 30년"), ("24개월 개인대출", "개인대출 24개월"), ("LC 약정 A등급", "LC 약정 A"), ("LC 약정 전체", "LC 약정 전체"), ("LC 실현 전체(36개월)", "LC 실현 전체"), ("LC 실현 승인(36개월)", "LC 실현 승인")]
H.append(table("발급연도별 스프레드 위험 사다리", ["발급연도"] + [b for _, b in LCOL], [[str(int(r.발급연도))] + [f"{r[a] * 100:.2f}" for a, _ in LCOL] for _, r in LADT.iterrows()], unit="%p",
               note="Aaa·Baa·모기지 = 10년 국채 대비, 개인대출 = 2년 국채 대비(상업은행, 분기), LC = 발급월 만기 맞춤 국채 대비. LC = LendingClub, 승인 = 최종 모형.", src="FRED, 본 연구 계산"))
H.append(table("LendingClub 스프레드와 시장 스프레드의 관계(Newey–West 6)", ["회귀", "설명변수", "관측(월)", "계수", "표준오차", "t", "R²"],
               [[r.회귀, r.설명변수, str(int(r.관측)), f"{r.계수:.3f}", f"{r.표준오차:.3f}", f"{r.t:.2f}", f"{r.R2:.3f}"] for _, r in BREG.iterrows()],
               note="동조성 = 월별 1차 차분 회귀(연속된 달끼리만, 2008~2018, 월 300건 이상), 예측 = 발급월 실현 스프레드(36개월 풀, 월 300건 이상)를 발급월 Baa 스프레드에 회귀. 단위 %p.", src="본 연구 계산", widths=["34%", "14%", "10%", "10%", "11%", "10%", "11%"]))
H.append(L2(f"최종 모형 승인 풀의 실현 스프레드(수수료 전)는 {int(LADT.발급연도.min())}~{int(LADT.발급연도.max())}년 {'모든 해에' if LAB else '대부분의 해에'} 같은 해 Baa 회사채 약정 스프레드보다 큼. " + (f"전부 승인 풀은 {'·'.join(LBY)}년에 Baa보다 낮음" if LBY else "전부 승인 풀도 모든 해에 Baa보다 높음")
            + ". 다만 LendingClub은 사후 실현·3년 국채 대비, Baa는 사전 약정·10년 국채 대비라 크기(배수)는 비교하지 않음(3년 국채 대비로 바꾸면 Baa 스프레드는 기간 프리미엄만큼 커짐)"))
H.append(L2(f"LendingClub 약정 스프레드의 월별 변화는 모기지 스프레드 변화와 같은 방향(계수 {rgr('동조성: Δ LC 약정 스프레드', '30년 모기지').계수:.2f}, t = {rgr('동조성: Δ LC 약정 스프레드', '30년 모기지').t:.2f}, 회귀 {len(BREG)}개 본페로니 보정 임계 {BONF:.2f} {'미만이라 보정 후 유의하지 않음' if abs(rgr('동조성: Δ LC 약정 스프레드', '30년 모기지').t) < BONF else '초과'})이나 Baa 회사채와는 유의하지 않음(t = {rgr('동조성: Δ LC 약정 스프레드').t:.2f}). 발급월 Baa 스프레드는 대출 실현 스프레드를 예측하지 못함(전체 t = {rgr('예측: 발급월 실현 스프레드(전체 풀)').t:.2f}, 승인 t = {rgr('예측: 발급월 실현 스프레드(승인 풀)').t:.2f}) → 대출의 위험 프리미엄은 회사채 시장과 다른 소비자 신용 고유의 위험을 반영한 것으로 보임(추정)"))
H.append(note("시장 스프레드는 약정(사전) 값이고 만기가 다름(Baa·모기지는 장기물, 개인대출은 24개월). LendingClub 등급과 회사채 신용등급의 대응표가 없어 「같은 위험」은 등급 매칭 대신 위험 사다리로 비교함. 동조성 회귀의 두 스프레드는 서로 다른 만기의 국채를 빼므로 공통 국채 변동이 섞일 수 있음"))
H.append("</div>")

# ===== Ⅷ
H.append("<div class='page'><h2>Ⅷ. 운영 설계와 보완 방안</h2><h3>1. 림보 3분류와 사람 심사</h3>")
H.append(L1("애매한 신청은 자동으로 정하지 않고 「림보」로 분류해 사람이 심사: 승인·거절·림보 3분류"))
H.append(L2(f"림보 기준(사전 고정): ① 승인 경계 ±2.5%p, ② 승인 예정 중 세 부도확률 모형의 불일치 상위 5%, ③ 승인 예정 중 희소 구간(학습 자료의 등급×목적×만기 1,000건 미만) → 신청의 {pc(LIM['비중'])}"))
H.append(L2(f"림보 승인 대출의 부도율 {pc(LIM['부도율_림보승인'])}, 초과수익 표준편차 {LIM['z_sd_림보']:.2f}%p로 비림보 승인({pc(LIM['부도율_비림보승인'])}, {LIM['z_sd_비림보승인']:.2f}%p)보다 불확실"))
H.append(table("림보 처리 방식별 과제 정의 Sharpe(시험)", ["처리 방식", "승인율", "Sharpe", "자동 규칙 대비"],
               [([k, pc(FL.loc[k, "승인율"]), f3(FL.loc[k, "T1"]), "—" if k.startswith("자동") else d3(FL.loc[k, "T1"] - FL.loc["자동 규칙(그대로)", "T1"])], "hl" if "70%" in k else "") for k in FL.index],
               note="정확도 70% = 심사자가 실제 결과를 70% 확률로 맞힌다고 가정한 모의실험(난수 시드 0). 완전 심사는 이론 상한.", src=SRC_T))
H.append(L2("시사점: 심사 정확도 70%를 가정한 모의실험에서는 소수(약 7%)의 애매한 사례에 사람을 투입하는 효과가 컸음(실제 심사 정확도는 운영에서 확인 필요). 심사 결과와 사유를 기록해 「희귀 사례의 암묵지」를 학습 자료로 축적하는 구조가 필요(3절)"))
H.append("<h3>2. 비정형 자료: 직함 텍스트</h3>")
H.append(table("직함 유형별 부도율과 모형 성능(시험)", ["직함 유형", "대출", "실제 부도율", "예측 부도확률", "AUC"],
               [[k, num(GG.loc[k, "n"]), pc(GG.loc[k, "부도율"]), pc(GG.loc[k, "예측부도확률"]), f"{GG.loc[k, 'AUC']:.3f}"] for k in ("일반", "희소(학습 30건 미만)", "미기재")], src=SRC_T))
oc_rows = [[r.직군, num(r.n), pc(r.실제부도율), pc(r.예측부도확률), d3(r.잔차 * 100, 1) + "%p", f"{r.t_군집:.1f}"] for _, r in OC.iterrows()]
H.append(table("희소 직함 대출의 직군별 예측 잔차(실제 − 예측)", ["직군(키워드 규칙)", "대출", "실제 부도율", "예측 부도확률", "잔차", "t(발급월 군집)"], oc_rows,
               note=f"직군은 공개된 키워드 규칙으로 분류(scripts/128 RULES), '기타'는 규칙에 걸리지 않은 나머지. 1,000건 이상 직군만 표시. t는 발급월 군집 표준오차 기준, |t| > 2.58이면 1% 유의(단순 표준오차 기준으로는 {PS['직함']['유의직군_단순']}개).", src=SRC_T))
H.append(L2(f"학습 자료에 30건 미만인 희소 직함이 시험 대출의 {pc(PS['직함']['희소비중'], 0)}. 모형은 이들의 직함 정보를 거의 쓰지 못해 직군별 잔차가 체계적({PS['직함']['유의직군']}/{PS['직함']['직군수']}개 직군 1% 유의, 기술·IT·관리직·사무직은 부도를 과대예측). 희소 직함 전체가 평균 {abs(PS['직함']['희소_평균잔차']) * 100:.1f}%p 과대예측되므로, 직군 간 차이는 이 공통 편차와 함께 봐야 함"))
H.append(L2("방안: 직함·대출 제목 등 비정형 텍스트를 LLM으로 표준 직군·산업 코드로 분류해 정형 변수로 추가. LLM은 분류만 하고 수치는 만들지 않으며, 분류표는 사람이 검수"))
H.append("<h3>3. LLM·에이전트 활용 구조(보완 방안)</h3>")
H.append("<div class='diag'><div class='n'><b>비정형·내부 자료</b>직함·대출 제목·상담 기록·심사 메모 등</div><div class='a'>→</div>"
         "<div class='n'><b>LLM 분류</b>직군·목적·위험 신호를 정해진 코드로 분류<br>(값 생성 금지, 근거 문장 함께 저장)</div><div class='a'>→</div>"
         "<div class='n'><b>정형 변수화</b>코드표 검수 → 모형 입력 추가<br>(학습 교차검증으로만 채택)</div><div class='a'>→</div>"
         "<div class='n'><b>림보 심사</b>사람 판단·사유 기록<br>= 희귀 사례 암묵지</div><div class='a'>→</div>"
         "<div class='n'><b>재학습·감시 에이전트</b>월별 데이터 관문·드리프트·보정 점검 자동화</div></div>")
H.append(L2("오픈 루프(모형이 정하고 끝) → 닫힌 루프(사람 판단과 결과가 다시 학습 자료로 돌아옴)로 전환하는 것이 핵심. 림보 사례는 결과가 나오면 「모형 대 심사자」 정확도를 비교해 심사 범위를 조정"))
H.append(L2("국면 적응: 초기 연체율·국채금리·신규 대출 스프레드처럼 발급 시점에 관측되는 지표로 악화 국면 확률을 추정하고, 확률이 높으면 승인율을 낮추는 규칙을 학습 자료의 시간 외 검증으로 설계(Ⅵ-5·7의 한계 대응)"))
H.append(L2("에이전트화: 데이터 수집 → 관문 검사 → 점수 산출 → 림보 배정 → 사후 성과 집계를 단계별 에이전트로 나누고, 각 단계의 판단 근거를 기록해 감사 가능하게 함"))
H.append("<h3>4. 인과 추론과 예측의 구분</h3>")
H.append(L2("본 모형은 예측 모형임. 회귀 계수(Ⅵ-2)는 「조건부 연관」이며, 예를 들어 「금리를 올리면 수익이 준다」는 인과 효과로 해석할 수 없음"))
H.append(L2(f"제약적 상황 → 일반적 상황: 결과는 LendingClub이 승인한 {pc(PSHARE)}의 신청에서만 검증됨. 전체 신청자로 확장하려면 거절 추론(승인 확률 가중, 선택 모형, 일부 무작위 승인 실험)이 필요하며, 거절 자료에는 상환 결과가 없어 가정에 의존함"))
H.append(L2("정책 효과(새 승인 규칙을 썼을 때 실제로 얼마나 달라지는가)를 인과적으로 확인하려면 일부 신청에 무작위로 다른 규칙을 적용하는 실험 설계가 가장 확실함"))

# ===== Ⅸ
H.append("<h2 style='margin-top:14px'>Ⅸ. 결론과 한계</h2>")
H.append(L1(f"결론: 부도확률만이 아니라 「잘되면·잘못되면 얼마」를 함께 예측해 위험 대비 기대 초과수익 순으로 상위 {QP}를 승인하면, 과제 정의 포트폴리오 Sharpe {f3(SR)}로 전부 승인·부도확률 순위·등급 규칙을 크게 앞섬"))
H.append(L2(f"팀 공유 보고서·우수사례·채권 형식의 평가 정의로 비교한 {len(SCORE)}개 잣대 중 {NTAG}, 과제 정의 잣대는 팀 최고 모형과 동등"))
H.append(L2(f"개선의 상당 부분은 발급 시기 선택에서 나오며, 매달 같은 비율을 승인하면 {f3(MON['V3'])}(부도확률 순위 {f3(MON['B0'])}) → 같은 달 안의 선별력은 유지되나 실제 운용 성과는 이 값에 더 가까울 수 있음"))
H.append(L2("롱-숏·OLS·보정 검정에서 순위와 크기 비례가 실현 결과와 맞아(수준은 0.3%p 보수적), 성과가 우연보다 모형 구조에서 나온다는 근거가 일관됨"))
H.append(L2(f"채권으로 보면 승인 풀은 약정 스프레드의 {pc(bp('V3', 36, '보존율'), 0)}를 지켰고(수수료 전, 전체 {pc(bp('ALL', 36, '보존율'), 0)}), 조기상환이 빠른 짧은 자산이라 금리 가격 위험이 작음(유효 듀레이션 약 {dv('V3', CASEB[1][0], '자동이전증권', 'ED'):.1f}년, 조기상환 가정에 거의 둔감)"))
H.append(table("한계와 남은 과제", ["한계", "영향", "대응·남은 과제"], [
    [f"승인 대출만 관측(신청의 {pc(PSHARE)})", "결과는 승인 대출 모집단의 2차 선별에 한정", "거절 추론·무작위 승인 실험(Ⅷ-4)"],
    ["우측 중도절단(2017년 이후)", "최근 연도 결과 확정 표본은 조기 종료 쪽으로 치우침", "만기 코호트·생존분석 병행(Ⅵ-8)"],
    ["2017년 이후 시간 외 성과 열세", "고정 승인율은 금리 상승 국면에 취약", "국면 적응 승인율(Ⅷ-3)"],
    ["발급 시기 집중(2017~20년 승인 거의 없음)", f"Sharpe {f3(SR)} 중 일부는 시기 선택 효과, 잣대의 중도절단 편향을 학습했을 가능성", f"매달 같은 비율 규칙 병기({f3(MON['V3'])}, Ⅳ-4), 시기 선택 근거의 검증은 과제"],
    ["채권·MBS 비교의 근사(Ⅶ)", "시장 스프레드는 약정 값·만기 불일치, 등급 대응표 없음, 조기상환 금리 반응은 연관 추정(IO·PO 금리 위험의 방향이 그 부호에 좌우됨)", "회사채 등급별 실현 수익률·대출별 금리 경로 자료로 재검증"],
    ["과제 잣대의 계약만기 연율화", "분할상환·조기상환 수익을 과소평가", "듀레이션 일관 수익률로 결론 확인(Ⅵ-1)"],
    ["시험 채점 후 모형 변경(v2·v3)", "사후 선택 편향 가능성", "학습 자료로만 결정·전 과정 공개(부록 4), v1·v2 결과 병기"],
    ["대출 간 상관(공분산) 미반영 점수", "동시 부도 위험 과소평가 가능", "군집 구간으로 불확실성 보고, 포트폴리오 단위 위험 모형은 과제"]],
    widths=["28%", "36%", "36%"]))
H.append("</div>")

# ===== 통계표
cq = [0.05, 0.10, 0.15, 0.20, 0.26, 0.30, 0.40, 0.50, 0.70, 1.0]
def cq_val(k, q, c="T1_test"):
    s = CUR[(CUR.strategy == k) & np.isclose(CUR.q, q)]; return f3(float(s[c].iloc[0])) if len(s) else (f3(mv("T1", "ALL")) if q == 1.0 else "—")
H.append("<div class='page'><h2>통 계 표</h2>")
H.append(table("승인율별 과제 정의 포트폴리오 Sharpe(시험)", ["전략"] + [f"{q * 100:.0f}%" for q in cq],
               [([SCN, *[cq_val(k, q) for q in cq]], "hl" if k == "V3" else "") for k, SCN in (("V3", "최종 모형 v3"), ("M6v2", "v2"), ("M6", "v1"), ("M5", "기대 초과수익 순위"), ("B0", "부도확률 순위"), ("GRADE", "세부등급 순서"))],
               note="승인 100%는 전부 승인과 같음.", src=SRC_T))
H.append(table("승인율별 기회풀 Sharpe(시험, 만기 기준 코호트)", ["전략"] + [f"{q * 100:.0f}%" for q in cq],
               [([SCN, *[cq_val(k, q, "T2_test") if q < 1 else f3(mv("T2", "ALL")) for q in cq]], "hl" if k == "V3" else "") for k, SCN in (("V3", "최종 모형 v3"), ("M6v2", "v2"), ("B0", "부도확률 순위"), ("GRADE", "세부등급 순서"))], src=SRC_T))
G2 = GR.copy()
H.append(table("등급별 승인 구성(최종 모형, 시험)", ["등급", "대출", "승인", "승인율", "등급 전체 부도율", "승인 대출 부도율"],
               [[r.등급, num(r.대출), num(r.승인), pc(r.승인율), pc(r.등급부도율), "—" if pd.isna(r.승인부도율) else pc(r.승인부도율)] for _, r in G2.iterrows()], unit="건, %", src=SRC_T))
H.append(table("발급연도별 결과 확정 비율과 부도율(시험, 36개월 대출)", ["발급연도", "대출", "결과 확정 비율", "결과 확정 대출 부도율", "누적부도율(경쟁위험)", "최종 모형 승인율"],
               [[str(y), num(r.대출_36개월), pc(r.결과확정비율), pc(r.확정대출_부도율), pc(r.누적부도_36개월), pc(r.승인율_최종모형)] for y, r in CZ.iterrows() if y >= 2009], unit="건, %",
               note=f"최종 모형 승인율은 시험 전체 신청(상환 중 포함)의 점수 상위 {QP} 기준. 2018~19년 발급분은 거의 승인되지 않음 — 점수에 발급 시점 국채금리가 들어가 있고, 학습 자료의 같은 연도 결과 확정 대출은 조기 종료 위주라 이 연도를 낮게 평가했을 가능성이 있음(Ⅳ-4).", src="lending_club_2020_test.csv 전체, 본 연구 계산"))
H.append("</div>")

# ===== 부록
H.append("<div class='page'><h2>부 록</h2><h3>1. 작성 개요</h3>")
H.append(table("통계 작성 개요", ["항목", "내용"], [
    ["작성 목적", "승인 대출 + 거절 자금(국채) 포트폴리오의 Sharpe ratio를 최대로 만드는 대출 승인 전략 수립·평가"],
    ["작성 대상", f"LendingClub 2007.6~2020.9 발급 대출 {num(REPR.승인_대출.sum())}건(학습 1,755,295건, 시험 1,170,198건) 및 2007~2018년 거절 신청 {num(REJ_N)}건(대표성 진단용)"],
    ["평가 기준", "과제 정의 포트폴리오 Sharpe(주), 팀 공유 보고서·우수사례·채권 강의안·본 연구 잣대 8종(보조)"],
    ["작성 방법", "학습 자료 5겹 교차검증으로 모형·승인율 선택 → 시험 자료 1회 채점, 발급월 블록 부트스트랩 구간"],
    ["무위험수익률", "FRED DGS3(36개월)·DGS5(60개월) 발급월 월평균(주 지표·팀 잣대). 현금흐름 지표는 발급 전월 값(사전계획)"],
    ["주요 가정", "현금흐름 지표: 상각월 = 마지막 납입 + 6개월, 회수 수수료 1%, 원리금균등 복원(설계상 허용 가정)"],
    ["작성 주체", "서울대학교 ABS 과정 14기 ○조"]], widths=["18%", "82%"]))
H.append("<h3>2. 주요 용어 설명</h3>")
H.append(table("주요 용어", ["용어", "설명"], [
    ["Sharpe ratio", "(평균 수익률 − 무위험수익률) ÷ 수익률 표준편차. 위험 한 단위당 국채를 넘는 수익"],
    ["초과수익", "대출 수익률 − 같은 발급월·만기의 국채 수익률"], ["승인율 q", "전체 신청 중 승인하는 비율. 점수 상위 q를 승인"],
    ["교차검증(OOF)", "학습 자료를 5묶음으로 나눠, 각 묶음을 나머지로 학습한 모형으로 예측. 모형이 보지 않은 자료의 성능 추정"],
    ["발급월 블록 부트스트랩", "발급월 단위로 복원추출해 구간을 계산. 같은 달 대출의 상관(군집)을 반영"],
    ["기회풀 Sharpe", "모든 신청을 분모로 두고 승인 대출의 초과수익만 반영(거절 = 국채 = 초과수익 0)한 Sharpe(M4 보고서)"],
    ["가중평균만기(WAL)·듀레이션", "원금이 실제로 묶여 있는 평균 기간. 분할상환·조기상환이면 계약만기보다 짧음"],
    ["경쟁위험 누적부도율", "부도와 완납을 서로 경쟁하는 사건으로, 상환 중 대출을 중도절단으로 처리해 추정한 누적부도 확률(Aalen–Johansen)"],
    ["Mincer–Zarnowitz 회귀", "실현값을 예측값에 회귀(z = α + β·예측). α = 0, β = 1이면 예측이 불편"],
    ["Newey–West t", "시계열 자기상관·이분산을 고려한 표준오차로 계산한 t 통계량"],
    ["마르코프 전환 모형", "관측되지 않는 국면(정상·악화)이 확률적으로 바뀌며 국면마다 평균·분산이 다른 시계열 모형"],
    ["림보", "자동 승인·거절이 애매해 사람 심사로 넘기는 신청"], ["ECE·HL 검정", "예측 확률과 실제 비율의 차이(ECE: 평균 절대 차이, HL: 카이제곱 검정)"],
    ["실현 스프레드(유효 가산금리)", "실현 현금흐름 수익률 − 만기 맞춤 국채 수익률. 약정 스프레드 중 신용 손실을 뺀 뒤 남은 위험 프리미엄"],
    ["보유기간수익률(HPR)", "중간 현금흐름을 재투자해 보유기간 끝에 얻는 연평균 수익률(강의안 4.3절)"], ["SMM·CPR·PSA", "월·연 조기상환율, 주택담보대출 기준 속도(100% PSA = 30개월까지 매월 0.2%p 증가 후 CPR 6%)"],
    ["순차 지급 CMO·IO·PO", "풀 원금을 트랜치 순서대로 지급하는 MBS, 이자만(IO)·원금만(PO) 받는 분리형 MBS(강의안 10.2절)"], ["CNL", "누적 순손실 = (상각 원금 − 순회수) ÷ 대출액"],
    ["유효 듀레이션·볼록성", "금리를 위아래로 움직였을 때의 실제 가격 변화로 계산한 금리 민감도와 그 곡률(강의안 5.6·6.4절)"]], widths=["24%", "76%"]))
H.append("<h3>3. 통계 품질 자체 점검(국가통계 품질 5개 차원)</h3>")
H.append(table("통계 품질 자체 점검", ["차원", "점검 내용", "근거", "자체 평가(5점)"], [
    ["관련성", "과제 목표를 그대로 주 지표로 채택했는가, 이용자(팀·평가자)의 잣대를 함께 제공하는가", "과제 정의 Sharpe 주 지표, 팀·우수사례·채권 형식 잣대 8종 병기(Ⅰ·Ⅴ·Ⅶ)", "5"],
    ["정확성", "표본오차·비표본오차를 진단·공개했는가", f"군집 구간, 보정 검정, 대표성(승인 대출 {pc(PSHARE)})·중도절단 진단, 시험 1회 채점(Ⅱ·Ⅵ)", "4"],
    ["시의성/정시성", "최근 상황에서도 성립하는가", "2017년 이후 시간 외 열세를 공개, 2020.9 이후 자료 없음(Ⅵ-7)", "3"],
    ["비교성/일관성", "다른 보고서와 같은 정의로 비교 가능한가, 잣대 간 결론이 일관한가", f"기준선 재현 차이 ≤ {maxgap:.3f}(이수용 잣대는 근사 재현), {len(SCORE)}개 잣대 중 {NTAG}(Ⅴ)", "4"],
    ["접근성/명확성", "작성 방법·코드·원자료를 제공하는가", "용어 설명·계산식·재현 순서·인쇄값 원자료 공개(부록)", "5"]],
    note="통계청(현 국가데이터처) 국가통계 품질진단의 5개 차원 구성을 따른 작성자 자체 평가임.", widths=["14%", "34%", "42%", "10%"]))
H.append("<h3>4. 계획 대비 변경 기록</h3>")
H.append(table("시험 채점 이후 변경 사항", ["순서", "변경", "동기", "결정 근거(학습 자료만)", "시험 영향(과제 정의 Sharpe)"], [
    ["0", "사전계획 고정: v1(위험 대비 수익), 대표 승인 25%", "—", "교차검증 ROE 최대(1.962%)", f"v1 {f3(mv('T1', 'M6'))}"],
    ["1", "v2: 부도확률 3모형 평균", "검수에서 이전 채택 요소 누락 발견", "교차검증 ROE@30 +0.035%p[0.014, 0.049]", f"{f3(mv('T1', 'M6'))} → {f3(mv('T1', 'M6v2'))}"],
    ["2", "이수용 방식 표본 경계 수정(＜ → ≤)", "그 보고서 건수와 불일치(454,462 대 467,101)", "60개월 62,522건 일치 확인", "이수용 잣대 값 소폭 변동"],
    ["3", "v3: 이득 모형 목표 = 과제 잣대 수익", "팀 Sharpe 보고서와 같은 잣대 비교(0.140 대 0.147)", f"교차검증 {d3(GD.d_T1_v3_minus_v2, 4)}[{d3(GD.d_lo, 4)}, {d3(GD.d_hi, 4)}], 하한 > 0 사전 규칙", f"{f3(mv('T1', 'M6v2'))} → {f3(SR)}"],
    ["4", f"운영 승인율 25% → {QP}", "과제 목표(포트폴리오 Sharpe)와 사전 지표(ROE) 불일치", "교차검증 과제 정의 Sharpe 최대점", "—"],
    ["5", "시간 외 검증·생존분석·품질 점검 추가", "팀 보고서 비교·대표성 검토", "모형 선택에 쓰지 않는 진단", "—"]],
    note="v1·v2 결과는 모든 표에 함께 실었음. 시험 자료의 결과 열은 학습과 채택 판정에 쓰지 않았으나, v2·v3 후보의 착안은 1차 시험 채점 이후임(사후 선택 위험은 Ⅸ 한계 참고).", widths=["6%", "24%", "24%", "26%", "20%"]))
H.append("<h3>5. 재현 방법</h3>")
H.append(table("재현 순서", ["순서", "스크립트", "내용"], [
    ["1", "98_*·110_test_fit_apply.py", "전처리·데이터 관문·부도확률(XGBoost)·이득 모형(v1) 학습, 시험 점수"], ["2", "117_test_blend.py", "LightGBM·CatBoost 학습, 3모형 평균(v2)"],
    ["3", "124_v3_cv.py → 125_v3_fit_apply.py", "v3 교차검증 판정 → 최종 학습·시험 점수"], ["4", "121_team_metrics.py", "과제 정의·기회풀 Sharpe, 구간, 강건성"],
    ["5", "119·116·111", "이수용·우수사례 1·3·현금흐름 지표"], ["6", "126_quality_checks.py", "대표성·중도절단·경쟁위험 누적부도·보정 검정"],
    ["7", "127_oot.py 2015 / 2016", "시간 외 재학습·채점"], ["8", "128_prof_methods.py", "듀레이션·OLS·롱-숏·모먼트·레짐·림보·직함"],
    ["9", "131_bond_mbs.py", "채권·MBS 형식 평가(수익률·스프레드·HPR 잣대·CPR/PSA·CMO·듀레이션·시장 스프레드, FRED 수집)"], ["10", "129_final_figs.py → 130_final_report.py", "그림 → 본 보고서 HTML(PDF 변환)"]], note="실행 환경: /opt/anaconda3/bin/python3(pandas·polars·xgboost·lightgbm·catboost·statsmodels).", widths=["8%", "34%", "58%"]))
H.append("<h3>6. 팀 공유 보고서 인쇄값</h3>")
H.append(table("비교에 사용한 인쇄값(원문 전사)", ["보고서", "쪽", "잣대", "항목", "값", "승인율", "비고"],
               [[r.보고서.replace("_", " "), str(r.쪽), r.잣대, r.항목, f"{r.값:g}", "" if pd.isna(r.승인율) else f"{r.승인율 * 100:.1f}%", "" if pd.isna(r.비고) else r.비고] for _, r in PR.iterrows()],
               src="산출물/최종보고서/표/팀보고서_인쇄값.csv", widths=["22%", "5%", "15%", "22%", "8%", "8%", "20%"]))
H.append("</div></body></html>")
(O / "최종보고서.html").write_text("".join(H), encoding="utf-8"); RF.dump_tables(NB, O / "보고서표"); print("saved", len("".join(H)), "tables", NB.tn, "figures", NB.fn)
