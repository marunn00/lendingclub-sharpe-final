"""135 최종 보고서 v4 HTML 생성: 「Sharpe 1계조건(FOC)에서 출발한 대출 승인 규칙의 설계·구현·검증」.
순서(요청 반영): 아이디어·설계 → 구현 단계 → 상세 검증(기존 최종 보고서 Ⅰ~Ⅷ장 그대로) → 결론(마지막 본문 장) → 통계표·부록. 결론·결과 요약 쪽은 맨 뒤.
새 표·그림은 접두어 F(표 F1…, 그림 F1…)로 번호를 매겨 기존 표 1~42와 섞이지 않게 한다. 새 수치는 모두 표/ 의 CSV·JSON에서 읽는다(134·132·136 산출).
기존 본문은 130이 만든 산출물/최종보고서/최종보고서.html 에서 장 단위로 가져온다(문구·표 번호 불변). 출력: 산출물/최종보고서/최종보고서_v4.html (PDF는 visualize export), 보고서표/표F*.csv."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, json, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RF = importlib.import_module("report_fmt")
ROOT = Path(__file__).resolve().parents[1]; O = ROOT / "산출물" / "최종보고서"; T = O / "표"; F = O / "fig"
NB = RF.Numbering(F, "F"); table, figure, L1, L2, L3, note = NB.table, NB.figure, RF.L1, RF.L2, RF.L3, RF.note
f4 = lambda x: f"{x:.4f}".replace("-", "−"); f3 = lambda x: f"{x:.3f}".replace("-", "−"); f2 = lambda x: f"{x:.2f}".replace("-", "−"); pc = lambda x, d=1: f"{x * 100:.{d}f}%".replace("-", "−")
d4 = lambda x: ("+" if x >= 0 else "△") + f"{abs(x):.4f}"; d3 = lambda x: ("+" if x >= 0 else "△") + f"{abs(x):.3f}"
num = lambda x: f"{int(x):,}"
# ---------- 입력
LAD = pd.read_csv(T / "랩_회귀사다리.csv"); RD = pd.read_csv(T / "랩_회귀반복.csv"); CLS = pd.read_csv(T / "랩_분류.csv"); CLR = pd.read_csv(T / "랩_분류반복.csv"); FCR = pd.read_csv(T / "랩_FOC_반복.csv")
MB = pd.read_csv(T / "랩_마진구간.csv"); RDG = pd.read_csv(T / "랩_잔차진단_시드0.csv"); PCV = pd.read_csv(T / "랩_p값_선택변수.csv"); LJ = json.load(open(T / "랩_요약.json")); FJ = json.load(open(T / "FOC_요약.json"))
FTR = pd.read_csv(T / "FOC_학습곡선.csv"); FTE = pd.read_csv(T / "FOC_시험곡선_사후참고.csv"); SSJ = json.load(open(T / "교수_H_상태공간_요약.json")); SSM = pd.read_csv(T / "교수_H_상태공간_월별.csv"); SS = pd.read_csv(T / "교수_H_상태공간.csv")
MAIN = pd.read_csv(T / "주요결과_T1_T2.csv"); mv = lambda ru, k, c="SR_test": float(MAIN[(MAIN.ruler == ru) & (MAIN.strategy == k)][c].iloc[0])
REP = json.load(open(O / "결과_T1_T2.json")); V3J = pd.read_csv(T / "v3_판정.csv").set_index("model"); QS = float(V3J.loc["V3", "q_star"])
PS = json.load(open(T / "교수_요약.json")); LS = pd.read_csv(T / "교수_C_롱숏.csv").set_index("순위"); OLS = pd.read_csv(T / "교수_B_OLS.csv").set_index("모형"); LSf = LS.loc["최종 모형"]; MZ = OLS.loc["MZ 불편성 z = α + β·μ̂"]
CI = pd.read_csv(T / "구간.csv"); ROB = pd.read_csv(T / "강건성.csv").set_index("strategy"); PR = pd.read_csv(T / "팀보고서_인쇄값.csv"); O16 = pd.read_csv(T / "OOT_2016.csv"); REPR = pd.read_csv(T / "대표성_연도별.csv")
DUR = pd.read_csv(T / "교수_A_듀레이션.csv"); T1TAB = pd.read_csv(O / "보고서표" / [p.name for p in sorted((O / "보고서표").glob("표01_*.csv"))][0]); NLIM = len(pd.read_csv([p for p in sorted((O / "보고서표").glob("표32_*.csv"))][0]))
ALT = pd.read_csv(ROOT / "산출물" / "대안모델" / "_work" / "results.csv").set_index("id")
R18 = REPR[REPR.yr <= 2018]; PSHARE = R18.승인_대출.sum() / (R18.승인_대출.sum() + R18.거절_신청.sum()); OOT16 = float(O16[O16.strategy == "V3"]["SR_승인6_4"].iloc[0]); OOT16_TEAM = float(PR[(PR.잣대 == "T1_OOT_2017-2020") & (PR.항목 == "제안 모형")].값.iloc[0])
NJ = {m: int((T1TAB["판정"].str[0] == m).sum()) for m in "●◐◑○"}; WAL36 = float(DUR[(DUR["T"] == 36) & (DUR.상태 == "완납")].실현WAL_년.iloc[0]); MON = float(ROB.loc["V3", "T1_월별할당"]); NFOLD = int(ALT.loc["B0_full", "rows"])

# ---------- 파생 값(판별 규칙은 아래 문구와 같은 코드)
P = RD.pivot(index="seed", columns="step", values="rmse_hold"); PS_ = RD.pivot(index="seed", columns="step", values="SR26")
def verdict(a, b):
    """단계 a 를 단계 b 와 같은 표본 쌍으로 비교: 점검 RMSE 중앙값 차이·더 낮은 반복 비율·SR26 차이. 빠른 판별 규칙(탐색적, 실험 후 문서화)."""
    dr = P[a] - P[b]; ds = PS_[a] - PS_[b]; lo = float((dr < 0).mean()); m = float(dr.median())
    tag = "설명력↑" if (m <= -0.0005 and lo >= 0.75) else ("악화" if (m >= 0.0005 and lo <= 0.25) else "차이 없음")
    return tag, m, lo, float(ds.median()), float((ds > 0).mean())
STEP = {int(r.step): r for r in LAD.itertuples()}; PCOL = RD.groupby("step").p.median()
IDEA = {0: "평균만(기준선)", 1: "가격·약정 변수만(OLS)", 2: "+신용이력", 3: "+재무", 4: "+시장·시기 = 전체 변수 OLS", 5: "Ridge(L2 규제)", 6: "Lasso(L1 규제)", 7: "2단계: 부도확률 × 조건부 이득", 8: "비선형: HistGB 1단계", 9: "v3-lite: HistGB 2단계 + 비율 순위"}
BASE = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4}; VD = {s: verdict(s, BASE[s]) for s in BASE}
GAP = {s: STEP[s].과적합간격 for s in STEP}; SH = LJ["특성군"]; NFEAT = LJ["특성수"]
pv_ = PCV.median(numeric_only=True); NP = int(pv_.p_total); N05 = int(pv_.p05); NBF = int(pv_.p_bonf); NLZ = float(pv_.lasso_nz); AR, AL = float(pv_.ridge_alpha), float(pv_.lasso_alpha)
CL = CLS.set_index("model"); CK = {k: k for k in CL.index}; M_NONE, M_L2, M_L1, M_SV, M_SW, M_LZ = (CL.index[i] for i in range(6))
def cpair(a, b, col):
    x = CLR[CLR.model == a].set_index("seed")[col]; y = CLR[CLR.model == b].set_index("seed")[col]; dd = x - y; return float(dd.median()), float((dd > 0).mean())
fo = FCR; FOC_D = float((fo.SR_foc - fo.SR_rank26).median()); FOC_W = float((fo.SR_foc > fo.SR_rank26).mean()); FOC_DQ = float((fo.SR_foc - fo.SR_rank_qmatch).median()); FOC_WQ = float((fo.SR_foc > fo.SR_rank_qmatch).mean())
mgR, mgA = MB[MB.측.str.startswith("거절")].set_index("|d|"), MB[MB.측.str.startswith("승인")].set_index("|d|")
BAND = float(MB[MB["|d|"].isin(["0~0.25", "0.25~0.5"])].비중.sum()); wacc = lambda df: float((df.대출 * df.맞음비율).sum() / df.대출.sum()); BAND_ACC = wacc(MB[MB["|d|"].isin(["0~0.25", "0.25~0.5"])])
FAR_ACC = wacc(MB[MB["|d|"].isin(["1~2", "2 이상"])])
NEAR, FARB = ["0~0.25", "0.25~0.5"], ["1~2", "2 이상"]; sd_ = lambda pre: MB[MB.측.str.startswith(pre)]
NR, FR_, NA, FA_ = wacc(sd_("거절")[sd_("거절")["|d|"].isin(NEAR)]), wacc(sd_("거절")[sd_("거절")["|d|"].isin(FARB)]), wacc(sd_("승인")[sd_("승인")["|d|"].isin(NEAR)]), wacc(sd_("승인")[sd_("승인")["|d|"].isin(FARB)])

# ---------- 그림(제목은 캡션에): FOC 곡선, 상태공간
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "legend.frameon": False})
NAVY, BLUE, ORANGE, GRAY, RED = "#1F4E79", "#5B9BD5", "#ED7D31", "#A5A5A5", "#C00000"
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9))
axs[0].semilogx(FTR["λ"], FTR["학습OOF_Sharpe"], color=NAVY, label="학습 OOF Sharpe(λ 선택 근거)"); axs[0].axvline(FJ["lam_star"], color=RED, ls=":"); axs[0].axvline(FJ["학습_이론_lam"], color=ORANGE, ls="--")
axs[0].text(FJ["lam_star"] * 0.93, 0.07, f"λ*={FJ['lam_star']:.2f}", color=RED, ha="right"); axs[0].text(FJ["학습_이론_lam"] * 1.06, 0.07, f"SR/(2σ)={FJ['학습_이론_lam']:.2f}", color=ORANGE)
axs[0].axhline(FJ["학습_SR_v3_q26"], color=GRAY, ls="-", lw=1); axs[0].text(FTR["λ"].iloc[3], FJ["학습_SR_v3_q26"] + 0.002, "비율 순위 26%", color=GRAY); axs[0].set_xlabel("위험회피 계수 λ(로그 축)"); axs[0].set_ylabel("Sharpe"); axs[0].legend(loc="lower left", fontsize=8.5, bbox_to_anchor=(0.0, 0.0))
axs[1].semilogx(FTE["λ"], FTE["승인율"], color=BLUE); axs[1].axvline(FJ["lam_star"], color=RED, ls=":"); axs[1].set_xlabel("위험회피 계수 λ(로그 축)"); axs[1].set_ylabel("시험 승인율")
fig.tight_layout(); fig.savefig(F / "lab4_FOC곡선.png", dpi=200); plt.close(fig)
fig, ax = plt.subplots(figsize=(10.5, 3.6)); ym = pd.to_datetime(SSM.ym.astype(int).astype(str), format="%Y%m")
ax.plot(ym, SSM.z, ".", color=GRAY, label="월별 평균 초과수익 z(%p)"); ax.plot(ym, SSM["평활_수준"], color=NAVY, lw=2, label="평활 상태 μ_t(지역 수준 모형)"); ax.set_ylabel("%p"); ax2 = ax.twinx(); ax2.plot(ym, SSM.p_bad, color=ORANGE, lw=1.2, ls="--", label="악화 국면 확률(마르코프)")
ax2.set_ylim(0, 1.05); ax2.set_ylabel("확률"); ax2.grid(False); h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9)
fig.tight_layout(); fig.savefig(F / "lab5_상태공간.png", dpi=200); plt.close(fig)

st = LAD[LAD.step >= 1]; fig, ax = plt.subplots(figsize=(6.2, 4.2)); x = np.arange(len(st)); ax.bar(x - 0.2, st.rmse_train, 0.4, color=BLUE, label="학습(in-sample)"); ax.bar(x + 0.2, st.rmse_hold, 0.4, color=NAVY, label="점검(out-of-sample)")
ax.set_xticks(x); ax.set_xticklabels([f"{int(s)}" for s in st.step]); ax.set_xlabel("아이디어 사다리 단계(표 F6)"); ax.set_ylabel("RMSE (z), 20회 중앙값"); ax.set_ylim(st.rmse_train.min() * 0.9, st.rmse_hold.max() * 1.02); ax.legend()
fig.tight_layout(); fig.savefig(F / "lab2_잔차사다리.png", dpi=200); plt.close(fig)
# ---------- 기존 본문 장 가져오기(130 산출 HTML)
OLD = (O / "최종보고서.html").read_text(encoding="utf-8"); PIECES = re.split(r"(?=<div class='page)", OLD)[1:]
def old(key):
    for p in PIECES:
        m = re.search(r"<h[23]>(.*?)</h[23]>", p, flags=re.S)
        if m and key in re.sub(r"<[^>]+>", "", m.group(1)): return p.replace("</body></html>", "")
    raise KeyError(key)
OLD_MAIN = [old(k) for k in ("Ⅰ. 과제와 평가 기준", "Ⅱ. 자료와 표본 대표성", "Ⅲ. 모형과 승인 규칙", "Ⅳ. 주요 결과", "Ⅴ. 같은 잣대 비교", "Ⅵ. 이론의 실증과 강건성", "Ⅶ. 채권·MBS 형식 평가")]
_p8 = old("Ⅷ. 운영 설계와 보완 방안"); _i = _p8.index("<h2 style='margin-top:14px'>Ⅸ. 결론과 한계</h2>")      # 130은 Ⅸ장을 Ⅷ장 쪽 안에 이어 붙임 → 여기서 나눔
OLD_MAIN.append(_p8[:_i] + "</div>"); OLD_CONC = "<div class='page'>" + _p8[_i:].replace("<h2 style='margin-top:14px'>", "<h2>")
OLD_STAT, OLD_APP = old("통 계 표"), old("부 록")

# ====================================================== 본문
H = [f"<html><head><meta charset='utf-8'><title>Sharpe 1계조건(FOC) 기반 대출 승인 규칙: 설계·구현·검증</title><style>{RF.CSS} .step td:first-child {{ white-space: nowrap; }} .exp td {{ text-align: left; vertical-align: top; }}</style></head><body>"]
H.append(f"""<div class='cover'>
<table class='hdr'><tr><td class='lab'>과 제</td><td>서울대학교 ABS 과정 14기 · 통계 &amp; 데이터사이언스 python 실습 · LendingClub 팀 프로젝트</td><td rowspan='2' style='text-align:center;width:24%'><span class='badge'>최종보고서</span></td></tr>
<tr><td class='lab'>제 출</td><td>2026. 11. 24.(화)</td></tr></table>
<div class='title'>Sharpe 1계조건(FOC)에서 출발한<br>대출 승인 규칙의 설계 · 구현 · 검증</div>
<div class='sub'>- First-Order Condition 승인 규칙 + 대출별 조건부 수익 예측, 아이디어 → 가설 → 데이터 분할 → 잔차 줄이기 → 분류·임계값 → 규칙 순서 -</div>
<div class='keys'>
{L1("(핵심 아이디어) 포트폴리오 Sharpe의 1계조건(First-Order Condition, FOC)이 승인 규칙이다")}
{L2("대출 하나를 거절 → 승인으로 바꿨을 때 Sharpe가 늘어날 조건: <b>E[r − r<sub>f</sub>] − λ·ΔV &gt; 0</b>, λ = SR ÷ (2σ<sub>p</sub>)")}
{L2("ΔV = 그 대출이 포트폴리오 분산을 늘리는 몫(위험 한계기여) → 기대 초과수익이 위험의 대가보다 크면 승인")}
{L1("(우리가 한 것) FOC 규칙에 넣을 E[r]와 위험 ΔV를 대출별로 예측하는 모형과, 그 모형을 믿을 수 있는지 확인하는 절차 전체")}
{L2("예측 오차(잔차)를 줄이는 과정: OLS → Ridge·Lasso → 2단계 → 비선형, 각 단계를 in-sample · validation · hold-out으로 판별")}
{L2("분류 관점: ROC · 임계값 거리(margin) · 소프트마진, 그리고 교수님 강조 항목(스프레드·듀레이션·롱숏·OLS·레짐·상태공간·림보·LLM)")}
{L1("(읽는 순서) Part A 설계 → B 잔차 → C 분류 → D FOC 규칙 → E 교수님 항목 → F 상세 검증(기존 Ⅰ~Ⅷ장) → <b>G 결론(마지막)</b>")}
{L2("결과 숫자와 불리한 결과(시간 외 열세, 발급 시기 집중, 사후 변경)는 G장과 부록 4에 모아 공개함")}
</div>
<table class='hdr foot'><tr><td class='lab'>작 성</td><td>○조 (조원 이름 기입)</td></tr><tr><td class='lab'>재 현</td><td>scripts/121·124·125 + 132·134·135·136 · 모든 표·그림 원자료: 산출물/최종보고서/표</td></tr></table>
</div>""")
# 초록
H.append("<div class='page'><h2>초 록 (Abstract)</h2>")
H.append(L1("<b>문제(Problem)</b>: 승인한 대출 + 거절한 자금(국채)으로 이루어진 포트폴리오의 Sharpe ratio를 최대로 만드는 승인 규칙을 찾는다"))
H.append(L2("부도확률이 낮은 대출을 고르는 것과 Sharpe를 키우는 것은 다른 문제임 → 대출마다 「얼마 벌고 얼마 잃는가」와 「포트폴리오 위험에 얼마나 보태는가」가 필요함"))
H.append(L1("<b>아이디어(Idea)</b>: Sharpe를 대출 하나의 승인 여부로 미분한 1계조건(FOC)을 승인 규칙으로 쓴다"))
H.append(L2("규칙: <b>승인 ⇔ E[r<sub>i</sub> − r<sub>f,i</sub>] − λ·{Var(r<sub>i</sub>) + (E r<sub>i</sub> − R<sub>p</sub>)² − (r<sub>f,i</sub> − R<sub>p</sub>)²} &gt; 0</b>, λ = SR ÷ (2σ<sub>p</sub>)  (유도: A-2)"))
H.append(L2("이 규칙이 필요로 하는 두 재료 = 대출별 기대 초과수익(1차 모먼트) + 위험(2차 모먼트) → 예측 모형이 규칙의 품질을 정함"))
H.append(L1("<b>방법(Method)</b>: 규칙의 재료를 만드는 모형을 단계별로 쌓고, 단계마다 예측 오차(잔차)가 줄어드는지 확인한다"))
H.append(L2("In-sample(학습 구간) 잔차를 줄이는 것은 쉬우므로, 같은 단계의 validation · hold-out 잔차와의 간격(generalization gap)으로 과적합을 판별함(B장)"))
H.append(L2("정규화 회귀(Ridge · Lasso), 로지스틱·소프트마진(SVM), ROC·임계값 거리를 소표본(20,000건 × 20회)에서 구현해 쓰는 이유 · 쓰지 않는 이유를 가름(B·C장)"))
H.append(L1("<b>검증(Validation)</b>: 하이퍼파라미터(α·C·λ)와 승인율 q는 학습 자료에서만 고르고 시험 자료는 채점에만 쓴다(A-4). 다만 모형·규칙의 선택(v2·v3, FOC를 대표 규칙으로 올린 것)은 첫 시험 채점 이후에 정했음(G-5, 부록 4)"))
H.append(L2("채점은 과제 정의 Sharpe, 팀 보고서·우수사례·채권 형식의 잣대 9개로 같은 기준 비교, 구간은 같은 달 대출의 상관을 반영(발급월 군집 부트스트랩)"))
H.append(L1("<b>이 보고서가 말하는 FOC 규칙의 위치</b>: 같은 예측에 적용한 비율 순위 규칙과 통계적으로 구분되지 않는다(D장, G장). 규칙의 가치는 성능이 아니라 「왜 이 경계에서 자르는가」를 식으로 설명한다는 데 있음"))
H.append(L1("<b>결과와 한계</b>는 마지막 장(G)에 모음. 교수님 강조 항목의 구현 위치는 E장의 표 F16"))
H.append("</div>")
# 일러두기·목차
H.append("<div class='page'><h3>일 러 두 기</h3>")
H.append(L1("이 보고서는 <b>아이디어와 구현 순서대로</b> 읽도록 구성함: 결론은 맨 마지막 장(G)이며, 앞 장의 표는 판단을 위한 중간 과정임"))
H.append(L2("새로 쓴 장(Part A~E, G)의 표·그림 번호는 F를 붙임(표 F1, 그림 F1). Part F는 기존 최종 보고서 Ⅰ~Ⅷ장을 문구·번호 그대로 가져온 상세 검증이며 표 1~42를 씀"))
H.append(L2("통계표·부록은 기존 보고서와 같음(부록 4 = 계획 대비 변경 기록, 이번 추가분은 G-5)"))
H.append(L1("자료와 구간"))
H.append(L2(f"학습 lending_club_2020_train.csv(결과 확정 {num(REP['n_train_confirmed'])}건)과 시험 lending_club_2020_test.csv(결과 확정 {num(REP['n_test'])}건)는 서로 다른 대출. 시험 결과 열은 채점에만 사용"))
H.append(L2("소표본 실험(Part B·C)은 학습 파일에서만 20,000건을 뽑아 학습 60% / 검증 20% / 점검 20%로 나눔(20회 반복). 시험 파일은 쓰지 않음"))
H.append(L2("전체 표본 결과의 구간은 발급월 단위 블록 부트스트랩(군집 재표본). 소표본 결과는 20회 반복의 중앙값이며 방향 확인용"))
H.append(L1("표기: 영어 용어는 통계·데이터사이언스 강의에서 쓰는 이름을 그대로 쓰고, 처음 나올 때 한국어 뜻을 붙임(in-sample = 학습 구간, out-of-sample = 학습에 쓰지 않은 구간, residual = forecast error = 실제 − 예측)"))
H.append(L2("증감·차이의 △는 음(−)의 값. 반올림하여 합계가 맞지 않을 수 있음"))
H.append(L1("<b>빠른 판별 규칙(탐색적)</b>: B장에서 아이디어를 하나씩 넣을 때 「설명력↑」는 점검(hold-out) RMSE의 중앙값 차이가 0.0005 이상 줄고 20회 중 15회 이상 더 낮은 경우로 정함. 이 규칙은 실험을 본 뒤 문서화한 탐색적 기준임(사전 고정 아님)"))
H.append("</div>")
H.append("<div class='page'><h3>목 차</h3><div class='toc'>")
for a, bs in [("Part A. 아이디어와 설계", ["A-1. 한 장 지도: 아이디어 · 근거 · 판정 위치", "A-2. 핵심 아이디어: Sharpe 1계조건(FOC) 유도", "A-3. 가설과 비교(차이)를 보는 법", "A-4. 데이터와 분할: training · validation · hold-out", "A-5. 구현 과정 도식"]),
              ("Part B. 구현 1: 예측 오차(잔차) 줄이기", ["B-1. 정의: 잔차, in-sample, out-of-sample", "B-2. 아이디어 사다리와 판별", "B-3. Ridge · Lasso: 쓸까 말까", "B-4. 잔차 진단: 완납 · 부도 혼합", "B-5. 소표본 결과의 전체 표본 연결"]),
              ("Part C. 구현 2: 분류 · ROC · 임계값 거리 · 소프트마진", ["C-1. 정의", "C-2. 정규화 로지스틱 · 소프트마진 비교", "C-3. 임계값 거리와 결정의 크기(magnitude)", "C-4. 판정"]),
              ("Part D. 구현 3: FOC 승인 규칙", ["D-1. 규칙과 하이퍼파라미터 λ", "D-2. 소표본 20회", "D-3. 전체 표본: 학습 CV로 λ*, 시험 1회 채점", "D-4. 이론 λ와 선택 λ의 차이"]),
              ("Part E. 교수님 강조 항목 구현", ["E-1. 항목별 구현 위치와 핵심 결과", "E-2. 상태공간(state-space) 모형과 가우스 추정", "E-3. 림보 · 휴먼 루프 · LLM 설계"]),
              ("Part F. 상세 검증(기존 Ⅰ~Ⅷ장)", ["Ⅰ 과제 · Ⅱ 자료 · Ⅲ 모형 · Ⅳ 결과 · Ⅴ 같은 잣대 · Ⅵ 이론 실증 · Ⅶ 채권·MBS · Ⅷ 운영·보완"]),
              ("Part G. 결론(마지막)", ["G-1. 가설 판정", "G-2. 설명력이 높아진 지점", "G-3. 최종 규칙과 성과", "G-4. 한계와 남은 과제(기존 Ⅸ장)", "G-5. 이번 추가분의 사후 선택 공개와 재현"]), ("□ 통계표 · ◇ 부록", [])]:
    H.append(f"<p class='lv1'>{a}</p>" + "".join(f"<p class='lv2'>{b}</p>" for b in bs))
H.append("</div></div>")

# ---------------- Part A
H.append("<div class='page'><h2>Part A. 아이디어와 설계</h2><h3>A-1. 한 장 지도: 아이디어 · 근거 · 판정 위치</h3>")
H.append(L1("각 아이디어를 「왜 그렇게 생각했나(논리) → 어떤 데이터로 보나 → 무엇과 비교하나」 순서로 적고, 판정은 G-1에 모았음"))
H.append(table("아이디어 지도(Idea map)", ["ID", "아이디어", "논리(왜)", "근거 데이터 · 비교 대상", "구현 위치"], [
    ["I1", "목표는 부도 회피가 아니라 Sharpe", "부도가 안 나는 대출 ≠ 국채보다 많이 버는 대출(저위험 = 저금리)", "시험 744,608건: 부도확률 순위 대 기대 초과수익 순위", "Part F Ⅳ-2"],
    ["I2", "수익은 완납·부도의 혼합분포 → 2단계 예측", "완납은 +수 %, 부도는 −십수 % → 한 식으로 맞추면 두 봉우리를 평균으로 뭉갬", "잔차의 집단별 표준편차(완납 대 부도)", "B-4"],
    ["I3", "변수를 군별로 쌓아 설명력이 느는 지점을 찾음", "가격 → 신용이력 → 재무 → 시장·시기 순으로 정보를 늘림", "학습 12,000 · 검증 4,000 · 점검 4,000, 20회", "B-2"],
    ["I4", "Ridge(L2): 계수를 눌러 과적합 간격을 줄임", "변수 수 p가 표본 n에 비해 크면 in-sample 잔차만 줄고 out-of-sample은 안 줄어듦", "α를 validation으로 선택, 점검에서 확인", "B-3"],
    ["I5", "Lasso(L1): 쓸모없는 변수를 0으로", "변수 선택이 되면 해석이 쉬움", "비영 계수 수, OLS 유의 변수 수(p-value)", "B-3"],
    ["I6", "비선형(트리)", "금리·등급·시기의 상호작용을 선형식이 못 잡음", "소표본 점검 RMSE, 전체 표본 AUC(점수표 대 XGBoost)", "B-2, B-5"],
    ["I7", "ROC · 임계값 거리 → 림보 구간", "점수가 승인 경계에 가까운 대출은 결정이 불안정함", "|d| 구간별 결정이 맞은 비율", "C-3"],
    ["I8", "소프트마진(SVM) · |z| 가중", "경계 근처 위반(slack)을 허용하고, 크게 잃는 대출에 더 벌점", "순위 품질(AUC, Sharpe)을 로지스틱과 비교", "C-2"],
    ["I9", "FOC 승인 규칙", "Sharpe를 직접 미분한 조건이라 경계 설정에 이론적 근거가 있음", "같은 예측에서 비율 순위 규칙과 쌍체 비교", "D"],
    ["I10", "상태공간(local level) · 레짐 전환", "신용 환경은 서서히 또는 갑자기 바뀜 → 국면 진단", "월별 평균 초과수익 90개월, AIC·한 걸음 예측", "E-2"]],
    widths=["5%", "21%", "30%", "28%", "16%"]))
H.append("<h3>A-2. 핵심 아이디어: Sharpe 1계조건(FOC) 유도</h3>")
H.append(L1("<b>표기</b>: N개 대출 단위, 대출 i의 수익 r<sub>i</sub>, 같은 발급월·만기의 국채 r<sub>f,i</sub>, 포트폴리오 단위 수익 x<sub>i</sub> = r<sub>i</sub>(승인) 또는 r<sub>f,i</sub>(거절)"))
H.append("<div class='formula'>R<sub>p</sub> = (1/N) Σ x<sub>i</sub> &nbsp;&nbsp; R<sub>f</sub> = (1/N) Σ r<sub>f,i</sub> &nbsp;&nbsp; σ<sub>p</sub>² = (1/N) Σ (x<sub>i</sub> − R<sub>p</sub>)² &nbsp;&nbsp; <b>SR = (R<sub>p</sub> − R<sub>f</sub>) ÷ σ<sub>p</sub></b></div>")
H.append(L1("<b>유도</b>: 대출 하나(i)를 거절에서 승인으로 바꿨을 때 Sharpe가 어떻게 변하는가"))
H.append(L2("① 평균의 변화: ΔR<sub>p</sub> = (r<sub>i</sub> − r<sub>f,i</sub>) ÷ N"))
H.append(L2("② 분산의 변화(1차 근사): Δσ<sub>p</sub>² = [ (r<sub>i</sub> − R<sub>p</sub>)² − (r<sub>f,i</sub> − R<sub>p</sub>)² ] ÷ N   ← E[x²]의 변화에서 R<sub>p</sub>²의 변화(2R<sub>p</sub>ΔR<sub>p</sub>)를 뺀 값"))
H.append(L2("③ Sharpe의 미분: dSR = (1/σ<sub>p</sub>) · [ dR<sub>p</sub> − (SR ÷ 2σ<sub>p</sub>) · dσ<sub>p</sub>² ]   (R<sub>f</sub>는 고정, dσ<sub>p</sub> = dσ<sub>p</sub>² ÷ 2σ<sub>p</sub>)"))
H.append(L2("④ 따라서 dSR &gt; 0 ⇔ (r<sub>i</sub> − r<sub>f,i</sub>) − λ·[ (r<sub>i</sub> − R<sub>p</sub>)² − (r<sub>f,i</sub> − R<sub>p</sub>)² ] &gt; 0, <b>λ = SR ÷ (2σ<sub>p</sub>)</b>"))
H.append(L2("⑤ r<sub>i</sub>는 미래 값이므로 기댓값으로 바꿈: E[(r<sub>i</sub> − R<sub>p</sub>)²] = Var(r<sub>i</sub>) + (E r<sub>i</sub> − R<sub>p</sub>)²"))
H.append("<div class='formula'><b>FOC 승인 규칙</b>: 승인 ⇔ E[r<sub>i</sub> − r<sub>f,i</sub>] − λ · { Var(r<sub>i</sub>) + (E r<sub>i</sub> − R<sub>p</sub>)² − (r<sub>f,i</sub> − R<sub>p</sub>)² } &gt; 0<br>"
         "혼합분포의 재료: E r<sub>i</sub> = (1 − p<sub>i</sub>)a<sub>i</sub> + p<sub>i</sub>d<sub>i</sub>, &nbsp; Var(r<sub>i</sub>) = (1 − p<sub>i</sub>)σ<sub>a</sub>² + p<sub>i</sub>σ<sub>d</sub>² + p<sub>i</sub>(1 − p<sub>i</sub>)(a<sub>i</sub> − d<sub>i</sub>)²  &nbsp;(p = 부도확률, a = 완납 시 수익, d = 부도 시 수익)</div>")
H.append(L1("<b>이 규칙을 어떻게 읽나</b>"))
H.append(L2("왼쪽 첫 항 = 이 대출이 주는 기대 초과수익(이득). 중괄호 = 이 대출이 포트폴리오 분산에 보태는 몫(비용). λ = 위험 한 단위의 값어치"))
H.append(L2("비율 규칙 (E[r] − r<sub>f</sub>) ÷ σ<sub>i</sub>와의 차이: 비율 규칙은 위험을 선형(σ<sub>i</sub>)으로 벌주지만 포트폴리오 분산에 대한 실제 기여는 이차(σ<sub>i</sub>²)이며 포트폴리오 평균과의 거리에도 의존함"))
H.append(L2("출처: 「LendingClub_Sharpe_최종보고서」 5.4절의 승인 규칙. 우리는 같은 식을 우리 예측(v3)에 적용해 구현·검증함. 대출 간 공분산은 0으로 가정한 근사"))
H.append("</div>")

H.append("<div><h3>A-3. 가설과 비교(차이)를 보는 법</h3>")
H.append(L1("가설마다 「무엇과 무엇의 차이를, 어느 데이터로, 어떤 통계량으로, 어떤 기준으로」 볼지 먼저 정함. 판정 결과는 G-1에서만 말함"))
H.append(table("가설과 판정 규칙(결과는 G-1)", ["ID", "가설(H)", "근거(왜 이렇게 가정했나)", "비교(차이)를 보는 대상", "판정 규칙"], [
    ["H1", "부도확률 순위보다 기대 초과수익 기반 순위(최종 모형)가 Sharpe가 높다", "부도 회피와 수익은 다른 문제(I1)", "시험 Sharpe의 쌍체 차이(발급월 군집 구간)", "구간 하한 &gt; 0이면 지지"],
    ["H2", "완납과 부도의 잔차 크기가 다르다(혼합분포)", "수익의 두 봉우리(Part F Ⅳ)", "집단별 잔차 표준편차, 1단계 대 2단계 잔차", "표준편차 비 ≥ 2이면 지지"],
    ["H3", "정규화(Ridge · Lasso)가 out-of-sample 오차를 줄인다", "p/n이 크면 과적합 간격이 큼", "OLS 대비 점검 RMSE의 쌍체 차이(20회)", "빠른 판별 규칙(탐색적, 일러두기)"],
    ["H4", "비선형 모형이 선형보다 설명력이 높다", "상호작용·비선형 효과", "소표본 점검 RMSE, 전체 표본 AUC", "소표본: 빠른 판별 규칙(탐색적) / 전체: AUC 차이 ≥ 0.005"],
    ["H5", "FOC 규칙이 비율 순위 규칙보다 Sharpe가 높다", "FOC는 Sharpe의 직접 미분(A-2)", "같은 예측에서 시험 Sharpe의 쌍체 차이(발급월 군집 구간)", "구간 하한 &gt; 0이면 지지, 0 포함이면 구분되지 않음(동등으로 표기)"],
    ["H6", "임계값에 가까운 대출일수록 결정이 불안정하다", "점수가 경계 근처이면 기대 이득이 0에 가까움", "|d| 구간별 결정이 맞은 비율과 평균 |z|", "같은 쪽(승인·거절) 안에서 먼 구간 − 가까운 구간 ≥ 10%p이면 그 쪽에서 지지(탐색적)"],
    ["H7", "소프트마진·|z| 가중이 순위 품질을 올린다", "경계 위반 허용 + 큰 손실에 큰 벌점", "같은 목표(z &gt; 0)로 학습한 로지스틱 대비 SR26·AUC의 쌍체 차이(20회)", "SR26 중앙값 차이 ≥ +0.005이고 15회 이상 우세(탐색적)"],
    ["H8", "신용 환경에는 천천히 움직이는 상태가 있다", "월별 초과수익이 국면별로 다름(I10)", "AIC, 한 걸음 앞 예측 RMSE", "AIC 최소 + 평균 예측보다 RMSE 작음"]],
    widths=["5%", "24%", "20%", "29%", "22%"]))
H.append("<h3>A-4. 데이터와 분할: training · validation · hold-out</h3>")
H.append(L1("<b>두 층의 분할</b>: 본 분석(전체 표본)과 소표본 실험(Part B·C)이 서로 다른 분할을 씀. 어느 데이터가 무엇에 쓰였는지 표로 고정"))
H.append(table("본 분석(전체 표본)의 분할", ["구간", "자료 · 건수", "무엇을 하나", "무엇을 하지 않나"], [
    ["Training (CV 풀 5겹, 겹마다 4/5로 적합)", f"학습 파일 결과 확정 {num(REP['n_train_confirmed'])}건 중 튜닝 보류 10%를 뺀 CV 풀 {num(NFOLD)}건", "부도확률 · 완납/부도 조건부 이득 모형 적합(in-sample 잔차)", "시험 자료 접근"],
    ["Validation (남은 1겹 = OOF)", "같은 자료의 각 겹이 한 번씩", "승인율 q, 위험회피 λ, 모형 채택 판정(124)을 모두 여기서", "시험 결과를 보고 고르기"],
    ["Early-stopping 보류 10%", "학습 겹 안의 무작위 10%", "부스팅 라운드 수 선택", "—"],
    ["Test (시험 1회)", f"시험 파일 결과 확정 {num(REP['n_test'])}건", "고정된 규칙으로 한 번 채점, 발급월 군집 구간", "모형·규칙 선택(단, v2·v3·FOC 채택은 사후 변경: G-5)"]],
    widths=["20%", "28%", "30%", "22%"]))
H.append(table("소표본 실험(Part B·C)의 분할", ["구간", "건수 · 비율", "무엇을 하나", "비고"], [
    ["Training(in-sample)", f"{num(LJ['학습'])}건(60%)", "계수 추정, 결측 대체값·표준화 평균(학습 구간으로만)", "여기서 잔차를 줄이는 것은 쉬움"],
    ["Validation", f"{num(LJ['검증'])}건(20%)", "하이퍼파라미터 선택: Ridge α(28개 격자 10<sup>−2</sup>~10<sup>4.5</sup>), Lasso α(28개 10<sup>−5</sup>~10<sup>−1</sup>), 로지스틱 C(9개), SVM C(9개), FOC λ(120개)", "validation RMSE/AUC가 가장 좋은 값 선택"],
    ["Hold-out(점검)", f"{num(LJ['점검'])}건(20%)", "고른 값으로 한 번만 채점: RMSE, R², Sharpe(승인율 26%)", "out-of-sample 성능으로 보고"],
    ["반복", f"{LJ['반복']}회(표본·분할을 매번 새로 뽑음)", "중앙값과 쌍체 비교(몇 번 이겼나)", "시험 파일은 쓰지 않음"]],
    note="재무 변수 중 희소 범주 빈도(hc__zip3·emp_title 빈도, 결과 열 미사용)는 학습 파일 전체의 범주 빈도로 만든 값이라 각 반복의 검증·점검 대출의 범주도 세어짐(결과 정보는 없는 약한 transductive 사용).",
    widths=["20%", "20%", "40%", "20%"]))
H.append("</div>")

H.append("<div class='page'><h3>A-5. 구현 과정 도식</h3>")
H.append(L1("아래 흐름으로 만들고, 같은 흐름이 B~D장의 순서임. 중간의 「아이디어 대입」에서 설명력이 높아지는 지점을 빠르게 판별함"))
nodes = [("S1 가설", "H1~H8을 근거와 판정 규칙과 함께 고정(A-3)"), ("S2 분할", "training · validation · hold-out 고정(A-4)"), ("S3 In-sample 적합", "모형 학습, 잔차 e = z − ẑ 계산"), ("S4 잔차 진단", "집단별 잔차, MZ 기울기, 간격"), ("S5 아이디어 대입", "변수 군 · 규제 · 2단계 · 비선형을 하나씩 추가")]
nodes2 = [("S6 Validation", "α · C · λ · q 선택"), ("S7 분류 · 임계값", "ROC, margin d, 림보 구간"), ("S8 FOC 규칙", "E[r], ΔV로 승인 결정"), ("S9 시험 1회 채점", "9개 잣대, 군집 구간"), ("S10 보완", "레짐 · 상태공간 · 림보 · LLM")]
def diag(ns): return "<div class='diag'>" + "<div class='a'>→</div>".join(f"<div class='n'><b>{a}</b>{b}</div>" for a, b in ns) + "</div>"
H.append(diag(nodes) + diag(nodes2))
H.append(table("구현 단계별 질문 · 입력 · 산출 · 판정", ["단계", "묻는 것", "하는 일 · 쓰는 데이터", "산출", "판정 기준", "스크립트"], [
    ["S1", "무엇을 어떻게 비교할까", "근거 데이터로 가설 H1~H8 작성", "표 F2", "H1·H2·H5·H8 사전, H3·H4·H6·H7 탐색적", "—"],
    ["S2", "무엇을 어디에 쓸까", "5겹 CV + 시험 1회 / 소표본 60·20·20 × 20회", "표 F3·F4", "시험은 채점에만", "124·134"],
    ["S3", "in-sample 잔차는 얼마인가", "학습 구간 적합, RMSE·R²", "표 F6", "—", "134"],
    ["S4", "잔차에 구조가 남았나", "완납·부도별 잔차, 예측 십분위 편향, MZ 기울기", "표 F9", "표준편차 비, β", "134·128"],
    ["S5", "어느 아이디어에서 설명력이 느나", "사다리 10단계, 쌍체 비교", "표 F6, 그림 F1", "빠른 판별 규칙", "134"],
    ["S6", "하이퍼파라미터는", "validation에서 α·C·λ·q 선택", "그림 F2, 표 F7", "validation 최소/최대", "134·124"],
    ["S7", "경계 근처는 믿을 만한가", "ROC, margin d, 소프트마진", "표 F11·F12, 그림 F3", "맞은 비율 차", "134"],
    ["S8", "규칙은 무엇인가", "FOC: 이득 − λ·위험 &gt; 0", "표 F14·F15, 그림 F4", "쌍체 구간", "132·134"],
    ["S9", "시험에서는", "고정 규칙 1회 채점, 같은 잣대 비교", "Part F 표 1~", "구간 하한", "121"],
    ["S10", "시장·구조 변화에는", "레짐, 상태공간, 롱숏, 림보, LLM", "표 F16~F19, 그림 F5", "AIC · RMSE", "128·136"]],
    widths=["6%", "17%", "31%", "14%", "17%", "15%"], cls="step"))
H.append("</div>")

# ---------------- Part B
H.append("<div class='page'><h2>Part B. 구현 1: 예측 오차(잔차) 줄이기</h2><h3>B-1. 정의: 잔차, in-sample, out-of-sample</h3>")
H.append(L1("<b>예측할 것</b>: 대출 i의 초과수익 z<sub>i</sub> = r<sub>i</sub> − r<sub>f,i</sub> (과제 정의 수익 − 같은 발급월·만기 국채). FOC 규칙의 E[r − r<sub>f</sub>]가 바로 E[z]"))
H.append("<div class='formula'>예측 ẑ<sub>i</sub> = f(x<sub>i</sub>; θ̂) &nbsp;&nbsp; <b>잔차(residual, forecast error) e<sub>i</sub> = z<sub>i</sub> − ẑ<sub>i</sub></b><br>"
         "In-sample MSE = (1/n<sub>tr</sub>) Σ<sub>tr</sub> e<sub>i</sub>² &nbsp;&nbsp; Out-of-sample MSE = (1/n<sub>ho</sub>) Σ<sub>ho</sub> e<sub>i</sub>² &nbsp;&nbsp; <b>일반화 간격(gap) = validation RMSE − 학습 RMSE</b><br>"
         "Bias–variance 분해: E[(z − ẑ)²] = Bias(ẑ)² + Var(ẑ) + σ<sub>ε</sub>² &nbsp;&nbsp; R²<sub>ho</sub> = 1 − Σ e² ÷ Σ (z − z̄<sub>tr</sub>)²<br>"
         "Mincer–Zarnowitz(MZ): z = α + β·ẑ + u, 불편이면 α = 0, β = 1 &nbsp;&nbsp; 판정 지표 SR26 = 점수 상위 26% 승인 시 점검 구간 Sharpe</div>")
H.append(L2("In-sample 잔차는 모형이 복잡해지면 항상 줄어듦 → 줄었다는 사실만으로는 좋아졌다고 못 함. 같은 단계에서 <b>validation · hold-out 잔차가 함께 줄고 gap이 늘지 않아야</b> 설명력이 높아진 것"))
H.append(L2("그래서 단계마다 세 값(학습 · 검증 · 점검 RMSE)과 gap, MZ 기울기, 결정 가치(SR26)를 함께 적음"))
H.append("<h3>B-2. 아이디어 사다리와 판별</h3>")
H.append(L1(f"소표본 {num(LJ['N'])}건(학습 {num(LJ['학습'])} · 검증 {num(LJ['검증'])} · 점검 {num(LJ['점검'])}) × {LJ['반복']}회. 변수 {NFEAT}개 = 가격·약정 {SH['가격·약정']} + 신용이력 {SH['신용이력']} + 재무 {SH['재무']} + 시장·시기 {SH['시장·시기']}(결측 표지 추가)"))
rows = []
for s in range(10):
    r = STEP[s]; v = VD.get(s); rows.append(([f"{s}", IDEA[s], "—" if np.isnan(PCOL.get(s, np.nan)) else f"{PCOL[s]:.0f}", f4(r.rmse_train), f4(r.rmse_valid), f4(r.rmse_hold), f4(r.과적합간격), f3(r.r2_hold), ("—" if s == 0 else f2(r.mz_hold)),
                                 f"{f3(r.SR26)} [{f3(r.SR26_q25)}, {f3(r.SR26_q75)}]", "기준선" if v is None else f"<b>{v[0]}</b><br><small>기준 {BASE[s]}단계 대비 {d4(v[1])}, {v[2] * 100:.0f}% 우세</small>"], "hl" if (v and v[0] == "설명력↑") else ""))
H.append(table("아이디어 사다리: in-sample · validation · hold-out 잔차(20회 중앙값)", ["단계", "아이디어", "p", "학습 RMSE", "검증 RMSE", "점검 RMSE", "간격", "점검 R²", "MZ β", "SR26 [IQR]", "판별"], rows,
               note="RMSE는 z(수익률 단위) 기준. p = 변수 수(Lasso는 비영 계수 수). 판별 = 점검 RMSE의 같은 표본 쌍 차이(기준 단계는 표에 표기, 1~4단계는 직전 단계, 5~9단계는 4단계 전체 OLS)와 그 차이가 더 낮았던 반복 비율. SR26의 [ ]는 20회의 사분위.",
               src="학습 파일 소표본(20,000건 × 20회), 본 연구 계산(scripts/134)", widths=["5%", "20%", "4%", "7%", "7%", "7%", "6%", "6%", "5%", "14%", "19%"]))
H.append(figure("lab2_잔차사다리.png", "단계별 in-sample · out-of-sample RMSE(20회 중앙값)", note="막대 사이의 간격이 일반화 gap. 단계 번호는 표 F6.", src="본 연구 계산", w="55%", h="70mm"))
H.append(L1("<b>읽는 법과 판별 결과</b>"))
imp = [s for s in VD if VD[s][0] == "설명력↑"]; non = [s for s in VD if VD[s][0] != "설명력↑"]
H.append(L2(f"설명력이 높아진 단계: {', '.join(f'{s}단계({IDEA[s]})' for s in imp)}. 점검 RMSE가 {f4(STEP[0].rmse_hold)}(평균만) → {f4(STEP[4].rmse_hold)}(전체 OLS)로 줄고 점검 R²가 {f3(STEP[4].r2_hold)}"))
H.append(L2(f"차이가 없는 단계: {', '.join(f'{s}단계({IDEA[s]})' for s in non)}. 특히 재무 변수(3단계)는 gap만 {f4(GAP[2])} → {f4(GAP[3])}로 키우고 점검 RMSE는 줄이지 못함 → 같은 정보가 신용이력에 이미 들어 있다고 해석(추정)"))
H.append(L2(f"결정 가치(SR26)는 가격만 {f3(STEP[1].SR26)} → +신용이력 {f3(STEP[2].SR26)} → 전체 {f3(STEP[4].SR26)}로 단계적으로 오르며, RMSE 개선보다 더 크게 보임 → 평균 오차가 조금만 줄어도 순위 품질은 크게 달라질 수 있음(추정)"))
H.append(L2(f"비선형(HistGB 1단계)은 학습 RMSE {f4(STEP[8].rmse_train)}로 가장 낮지만 gap이 {f4(GAP[8])}(OLS의 약 {GAP[8] / GAP[4]:.0f}배)이고 점검 RMSE는 OLS와 같음 → in-sample 잔차 감소가 곧 설명력 향상은 아님을 보여 주는 사례"))
H.append("</div>")

H.append("<div class='page'><h3>B-3. Ridge · Lasso: 쓸까 말까</h3>")
H.append("<div class='formula'>OLS: min<sub>β</sub> Σ (z<sub>i</sub> − x<sub>i</sub>′β)² &nbsp;&nbsp; <b>Ridge</b>: + α‖β‖<sub>2</sub>² (계수를 0 쪽으로 균일하게 수축) &nbsp;&nbsp; <b>Lasso</b>: + α‖β‖<sub>1</sub> (일부 계수를 정확히 0으로)<br>"
         "α는 hyperparameter: 클수록 단순(bias↑, variance↓). validation RMSE가 최소인 α*를 고르고 hold-out으로 확인</div>")
H.append(figure("lab1_알파곡선.png", "규제 강도 α에 따른 in-sample · validation RMSE (Ridge, Lasso)와 Lasso 비영 계수 수", note=f"seed 0 한 번의 곡선. 점선 = validation 최소 α*. 20회 중앙값 α*: Ridge {AR:.2f}, Lasso {AL:.1e}.", src="본 연구 계산", h="62mm"))
H.append(table("OLS 유의성(p-value)과 Lasso 선택 변수(20회 중앙값)", ["항목", "값", "해석"], [
    ["변수 수 p(결측 표지 포함)", f"{NP}", "표본 n = 12,000 → p/n = " + pc(NP / LJ["학습"], 2)],
    ["OLS에서 p-value &lt; 0.05 변수 수", f"{N05}", "대부분 변수는 개별로는 유의하지 않음(다중 공선성 포함)"],
    ["OLS에서 Bonferroni(0.05/p) 통과 변수 수", f"{NBF}", "다중비교를 보정하면 거의 남지 않음"],
    ["Lasso 비영 계수 수", f"{NLZ:.1f}", "검증으로 고른 α*가 작아 변수를 거의 안 지움"],
    ["Ridge α* · Lasso α*", f"{AR:.2f} · {AL:.1e}", "규제가 약하게만 필요하다는 신호"]], widths=["34%", "16%", "50%"]))
rr = VD[5]; ll_ = VD[6]
H.append(L1("<b>사용 · 비사용 판정과 이유</b>"))
H.append(table("정규화 회귀를 쓰는가", ["대상", "판정", "이유(수치)", "다시 쓰는 조건"], [
    ["Ridge", f"본 모형에는 쓰지 않음({rr[0]})", f"p/n = {pc(NP / LJ['학습'], 2)}로 작아 OLS의 gap이 {f4(GAP[4])}에 불과, Ridge 후 gap {f4(GAP[5])}. 점검 RMSE 차이 {d4(rr[1])}(OLS 대비 {rr[2] * 100:.0f}% 우세)", "변수 수가 표본의 수 %를 넘거나, 서로 거의 같은 변수가 많은 소표본"],
    ["Lasso", f"본 모형에는 쓰지 않음({ll_[0]})", f"비영 계수 {NLZ:.1f}/{NP}: 지울 변수가 거의 없음. 점검 RMSE 차이 {d4(ll_[1])}", "변수 해석·보고가 목적일 때(변수 선택표 용도)"],
    ["OLS(전체)", "해석·기준선으로 사용", f"점검 R² {f3(STEP[4].r2_hold)}, MZ β {f2(STEP[4].mz_hold)}. 계수 부호를 말할 수 있어 표 14 회귀에 사용", "—"],
    ["트리 계열(전체 표본)", "본 모형에 사용", "소표본에서는 이득이 없으나 전체 표본에서는 AUC가 오름(B-5)", "표본이 작아지면 gap 확인 후 재검토"]], widths=["14%", "22%", "42%", "22%"]))
H.append("<h3>B-4. 잔차 진단: 완납 · 부도 혼합</h3>")
rd0 = RDG.set_index(["집단", "모형"]); sdr = float(rd0.loc[("부도", "1단계 Ridge"), "학습_잔차SD"] / rd0.loc[("완납", "1단계 Ridge"), "학습_잔차SD"])
H.append(table("잔차의 집단별 평균과 표준편차(seed 0)", ["집단", "모형", "학습 평균 잔차", "학습 잔차 SD", "점검 평균 잔차", "점검 잔차 SD", "점검 대출"],
               [[r.집단, r.모형, f4(r.학습_평균잔차), f4(r.학습_잔차SD), f4(r.점검_평균잔차), f4(r.점검_잔차SD), num(r.대출_점검)] for r in RDG.itertuples()], note="잔차 = z − ẑ. 평균 잔차가 0이 아닌 것은 집단을 나누지 않고 평균 쪽으로 예측한 결과.", src="본 연구 계산", widths=["8%", "26%", "14%", "14%", "14%", "14%", "10%"]))
H.append(L2(f"<b>부도 집단 잔차 SD가 완납의 {sdr:.1f}배</b>({f4(rd0.loc[('부도', '1단계 Ridge'), '학습_잔차SD'])} 대 {f4(rd0.loc[('완납', '1단계 Ridge'), '학습_잔차SD'])}) → 한 식으로 맞추면 잔차 분산이 이분산(heteroskedastic)이고, 평균 잔차가 완납에서 +, 부도에서 −로 갈림 = 혼합분포 증거(H2)"))
H.append(L2(f"2단계(부도확률 × 조건부 이득)는 집단별 평균 잔차를 거의 바꾸지 못함: 조건부 모형이 집단 안의 변동을 거의 못 맞히기 때문(점검 R² 사다리의 7단계 {f3(STEP[7].r2_hold)} 대 4단계 {f3(STEP[4].r2_hold)}). 2단계의 값은 평균이 아니라 <b>위험(Var) 계산</b>에 있음: FOC 규칙의 Var(r)에 p(1−p)(a−d)² 항이 들어가기 때문"))
H.append("<h3>B-5. 소표본 결과의 전체 표본 연결</h3>")
alt = lambda i: float(ALT.loc[i, "AUC"])
H.append(table("전체 표본에서의 부도 분류 AUC(학습 파일 300,000행, 5겹)", ["모형", "계열", "AUC", "XGBoost 대비"], [
    ["WOE 점수표(로지스틱형)", "선형", f4(alt("M1b_sub")), d4(alt("M1b_sub") - alt("B0_sub"))], ["EBM(설명 가능한 부스팅)", "가법·비선형", f4(alt("M1a_sub")), d4(alt("M1a_sub") - alt("B0_sub"))],
    ["신경망(MLP)", "비선형", f4(alt("M4a_sub")), d4(alt("M4a_sub") - alt("B0_sub"))], ["XGBoost", "트리 앙상블", f4(alt("B0_sub")), "기준"]], note="102번대 대안 모형 탐색 결과(results.csv)에서 같은 30만 행·같은 겹.", src="산출물/대안모델/_work/results.csv", widths=["34%", "22%", "22%", "22%"]))
H.append(L2(f"전체 표본에서는 선형 점수표가 XGBoost보다 AUC가 {abs(alt('M1b_sub') - alt('B0_sub')):.3f} 낮음 → 표본이 클 때(n ≈ 30만~100만)는 트리의 유연성이 값을 함(H4 전체 표본 판정은 G-1). 소표본에서 이득이 없다는 B-2와 같은 일반 원칙 「표본이 작으면 유연한 모형의 gap이 커진다」로 설명됨"))
H.append("</div>")

# ---------------- Part C
H.append("<div class='page'><h2>Part C. 구현 2: 분류 · ROC · 임계값 거리 · 소프트마진</h2><h3>C-1. 정의</h3>")
H.append(L1("<b>왜 분류를 보나</b>: 승인은 결국 0/1 결정. 점수가 경계(임계값 τ)에서 얼마나 떨어져 있는지(거리)가 결정의 믿음직함을 정함"))
H.append("<div class='formula'>점수 s<sub>i</sub>(클수록 부도 위험), 임계값 τ: s<sub>i</sub> ≥ τ이면 거절 &nbsp;&nbsp; <b>ROC</b>: TPR(τ) = P(s ≥ τ | 부도), FPR(τ) = P(s ≥ τ | 완납) &nbsp;&nbsp; <b>AUC</b> = P(s<sub>부도</sub> &gt; s<sub>완납</sub>)<br>"
         "임계값 거리(margin) d<sub>i</sub> = (a<sub>i</sub> − τ) ÷ sd(a), a<sub>i</sub> = 승인 점수(클수록 승인), τ = 상위 26% 경계 &nbsp;&nbsp; 결정이 맞음 = (승인 &amp; z &gt; 0) 또는 (거절 &amp; z ≤ 0)<br>"
         "<b>소프트마진(Soft margin) SVM</b>: min<sub>w,b</sub> ½‖w‖² + C Σ ξ<sub>i</sub> &nbsp; s.t. t<sub>i</sub>(w′x<sub>i</sub> + b) ≥ 1 − ξ<sub>i</sub>, ξ<sub>i</sub> ≥ 0 &nbsp;(hinge loss max(0, 1 − t<sub>i</sub>f(x<sub>i</sub>)), ξ = slack 변수)<br>"
         "<b>|z| 가중 소프트마진</b>: C Σ |z<sub>i</sub>|·ξ<sub>i</sub> → 크게 잃거나 크게 버는 대출의 경계 위반에 더 큰 벌점(손익의 크기 = magnitude 반영)</div>")
H.append(L2("C가 작을수록 마진을 넓게 잡고 위반(slack)을 많이 허용(soft), 클수록 위반을 엄격히 벌점(hard에 가까움). C는 validation AUC로 선택"))
H.append("<h3>C-2. 정규화 로지스틱 · 소프트마진 비교</h3>")
cr = []
for m_ in CL.index:
    r = CL.loc[m_]; cr.append(([m_, f"{r.C:.3g}", f4(r.auc_valid), f4(r.auc_hold), f4(r.auc_sign_hold), f3(r.SR26), f"{r.nonzero:.0f}"], "hl" if m_ in (M_SV, M_SW) else ""))
H.append(table("분류 모형 비교(점검 구간, 20회 중앙값)", ["모형", "C(검증 선택)", "검증 AUC(부도)", "점검 AUC(부도)", "점검 AUC(z&gt;0)", "SR26", "비영 계수"], cr,
               note="AUC(부도) = 부도 점수의 순위 정확도, AUC(z&gt;0) = 승인 점수로 초과수익 &gt; 0 대출을 가려내는 정확도. SR26 = 점수 상위 26% 승인 시 점검 Sharpe. 하이퍼파라미터 선택 기준은 모형마다 달라(L2 로지스틱 = 검증 logloss, L1 = 검증 부도 AUC, SVM·대조군 = 검증 z&gt;0 AUC) 같은 기준의 비교가 아님(모두 검증 구간만 사용).", src="본 연구 계산(scripts/134)", widths=["30%", "10%", "12%", "12%", "13%", "11%", "12%"]))
dl2 = cpair(M_L2, M_NONE, "auc_hold"); dl1 = cpair(M_L1, M_L2, "auc_hold"); dlz = cpair(M_LZ, M_L2, "SR26"); dsv = cpair(M_SV, M_LZ, "SR26"); dsw = cpair(M_SW, M_LZ, "SR26"); dsw2 = cpair(M_SW, M_SV, "SR26"); dsa = cpair(M_SV, M_LZ, "auc_sign_hold")
H.append(L2(f"L2 규제는 정규화 없음 대비 점검 AUC {d4(dl2[0])}({dl2[1] * 100:.0f}% 우세), L1은 L2 대비 {d4(dl1[0])}({dl1[1] * 100:.0f}% 우세) → 변수 {NP}개 · 표본 12,000에서는 규제 종류의 차이가 작음"))
H.append(L1("<b>효과 분리</b>: 모형 간 차이에는 「무엇을 맞히도록 학습했나(목표)」와 「어떤 손실을 쓰나(소프트마진)」가 섞임 → 같은 목표(z &gt; 0)로 학습한 로지스틱을 대조군으로 추가"))
H.append(L2(f"<b>목표 정렬 효과</b>(로지스틱 z &gt; 0 목표 − 로지스틱 부도 목표): SR26 {d3(dlz[0])}({dlz[1] * 100:.0f}% 우세) → 부도를 맞히는 모형보다 이득의 부호를 맞히는 모형이 Sharpe가 높음. 같은 원리가 Part F 표 5의 v3(목표 정렬) 단계에서도 나타남"))
H.append(L2(f"<b>소프트마진 효과</b>(SVM 무가중 − 로지스틱 z &gt; 0 목표): SR26 {d3(dsv[0])}({dsv[1] * 100:.0f}% 우세), 부호 AUC {d4(dsa[0])}. |z| 가중은 {d3(dsw[0])}({dsw[1] * 100:.0f}% 우세), 가중 − 무가중 {d3(dsw2[0])}({dsw2[1] * 100:.0f}% 우세)"))
H.append("<h3>C-3. 임계값 거리와 결정의 크기(magnitude)</h3>")
mr = []
for side, df in (("거절측(τ 미만)", mgR), ("승인측(τ 이상)", mgA)):
    for b in df.index: r = df.loc[b]; mr.append([side, b, num(r.대출), pc(r.비중), pc(r.맞음비율), f3(r.평균z), f3(r.평균절대z), pc(r.부도율)])
H.append(table("임계값 거리 |d| 구간별 결정 정확도와 크기(점검 구간 풀링 80,000건)", ["측", "|d|", "대출", "비중", "결정이 맞은 비율", "평균 z", "평균 |z|", "부도율"], mr,
               note="점수 = v3-lite(HistGB 2단계) 비율 E/위험, τ = 상위 26% 경계, d = (점수 − τ) ÷ 점수 표준편차. 점검 구간만 사용.", src="본 연구 계산(scripts/134)", widths=["16%", "11%", "10%", "9%", "17%", "13%", "12%", "12%"]))
H.append(figure("lab3_ROC_마진.png", "ROC 곡선(점검 구간, 로지스틱 3종)과 임계값 거리별 결정이 맞은 비율", note="왼쪽: 같은 20,000건 표본의 seed 0. 오른쪽: 20회 점검 구간 풀링.", src="본 연구 계산", h="62mm"))
H.append(L1("<b>읽는 법</b>"))
H.append(L2(f"거절측: 경계에서 멀수록 맞은 비율이 {pc(mgR.loc['0~0.25', '맞음비율'])} → {pc(mgR.loc['2 이상', '맞음비율'])}로 오르고, 평균 z는 {f3(mgR.loc['0~0.25', '평균z'])} → {f3(mgR.loc['2 이상', '평균z'])}로 내려감 = 멀리 거절한 대출은 실제로 국채보다 못한 대출이 많음. 가까운 거절의 {pc(1 - NR, 0)}는 사실 z &gt; 0인 대출(놓친 대출)이지만 평균 z가 작아 놓친 이득은 작음"))
H.append(L2(f"승인측: 맞은 비율이 {pc(mgA.loc['0~0.25', '맞음비율'])} → {pc(mgA.loc['1~2', '맞음비율'])}로 완만하게 오르고 평균 z도 {f3(mgA.loc['0~0.25', '평균z'])} → {f3(mgA.loc['1~2', '평균z'])}(작은 양수). 승인 쪽 수익의 크기는 작아 경계 근처 승인도 큰 손실은 아님"))
H.append(L2(f"같은 쪽 안에서 비교: 거절측 가까운 구간(|d| &lt; 0.5) {pc(NR)} → 먼 구간(|d| ≥ 1) {pc(FR_)}(+{(FR_ - NR) * 100:.0f}%p), 승인측 {pc(NA)} → {pc(FA_)}(+{(FA_ - NA) * 100:.0f}%p). |d| &lt; 0.5 구간은 전체의 {pc(BAND)}이고 평균 |z|가 작아 틀려도 손실 크기가 작음(magnitude 관점)"))
H.append(note("「맞음」은 승인측에서 z &gt; 0, 거절측에서 z ≤ 0이라 두 쪽의 수준을 직접 비교할 수 없음. 경계에서 승인측 약 86% 대 거절측 약 17%의 차이는 대부분 대출이 z &gt; 0이라는 기준 비율 때문(정의에 따른 차이)이며 거리 효과가 아님. 거리 효과는 같은 쪽 안의 변화로만 읽음. 승인측 |d| ≥ 2는 64건뿐"))
H.append("<h3>C-4. 판정</h3>")
H.append(table("분류·소프트마진 아이디어의 사용 판정", ["아이디어", "판정", "이유", "보고서 반영"], [
    ["정규화 로지스틱(L1·L2)", "본 모형 비채택", f"L2 대비 규제 종류 차이가 작음(AUC {d4(dl1[0])})", "부도확률은 트리 앙상블 3종 평균"],
    ["ROC 임계값 거리(margin)", "운영 설계에 채택", f"거절측: 가까운(|d|&lt;0.5) {pc(NR)} 대 먼(|d|≥1) {pc(FR_)}, 승인측 {pc(NA)} 대 {pc(FA_)}. 가까운 구간(전체의 {pc(BAND)})은 평균 |z|가 작음", "림보 구간 정의의 근거(Part F Ⅷ-1)"],
    ["소프트마진 SVM", ("보조 비교용" if dsv[0] < 0.005 else "추가 검토"), f"같은 목표의 로지스틱 대비 SR26 {d3(dsv[0])}(무가중), {d3(dsw[0])}(|z| 가중)", "큰 이득은 목표 정렬 효과에서 나옴(위 효과 분리)"],
    ["|z| 가중", ("차이 있음" if (dsw2[0] >= 0.005 and dsw2[1] >= 0.75) else "차이 불분명"), f"가중 − 무가중 SR26 {d3(dsw2[0])}({dsw2[1] * 100:.0f}% 우세)", "크기 정보는 이미 v3의 이득 회귀 목표에 들어 있음"]], widths=["20%", "18%", "38%", "24%"]))
H.append("</div>")

# ---------------- Part D
H.append("<div><h2 style='margin-top:14px'>Part D. 구현 3: FOC 승인 규칙</h2><h3>D-1. 규칙과 하이퍼파라미터 λ</h3>")
H.append(L1("<b>구현 순서</b>: ① 대출별 p, a, d를 예측(v3) → ② E r과 Var(r) 계산 → ③ λ를 정하고 R<sub>p</sub>를 고정점으로 구함 → ④ 승인 집합 결정"))
H.append("<div class='formula'>E<sub>i</sub> = E[z<sub>i</sub>] = (1 − p<sub>i</sub>)a<sub>i</sub> + p<sub>i</sub>d<sub>i</sub> &nbsp;&nbsp; V<sub>i</sub> = Var(r<sub>i</sub>)<br>"
         "승인 집합 A(λ, R<sub>p</sub>) = { i : E<sub>i</sub> − λ[ V<sub>i</sub> + (E<sub>i</sub> + r<sub>f,i</sub> − R<sub>p</sub>)² − (r<sub>f,i</sub> − R<sub>p</sub>)² ] &gt; 0 } &nbsp;&nbsp; 고정점: R<sub>p</sub> = 평균(승인 대출은 r̂<sub>i</sub>, 거절은 r<sub>f,i</sub>) 반복 계산<br>"
         "<b>하이퍼파라미터 λ</b>: 학습 OOF에서 Sharpe를 최대로 하는 λ*(격자 160개, 0.05~500). 이론은 λ = SR ÷ (2σ<sub>p</sub>)가 최적에서 성립한다고 말함</div>")
H.append(L2("λ가 크면 위험에 민감해져 승인이 줄고(그림 F4 오른쪽), 작으면 이득만 보고 승인이 늘어남. 비율 순위 규칙은 이 곡선 위의 한 점(승인율 26%)을 순위로 정하는 방식"))
H.append("<h3>D-2. 소표본 20회</h3>")
H.append(table("소표본에서 FOC 대 비율 순위(v3-lite 예측, 점검 구간)", ["항목", "FOC(λ 검증 선택)", "비율 순위 26%", "비율 순위(같은 승인율)"], [
    ["Sharpe 중앙값", f3(fo.SR_foc.median()), f3(fo.SR_rank26.median()), f3(fo.SR_rank_qmatch.median())],
    ["승인율 중앙값", pc(fo.q_foc.median()), "26.0%", pc(fo.q_foc.median())],
    ["FOC − 해당 규칙(같은 반복의 차이 중앙값)", "—", d3(FOC_D), d3(FOC_DQ)],
    ["FOC가 더 높았던 반복 비율", "—", pc(FOC_W, 0), pc(FOC_WQ, 0)],
    ["λ 선택값 · 이론값 SR/(2σ) 중앙값", f"{fo.lam.median():.2f} · {fo.lam_theory.median():.2f}", "—", "—"]], note="v3-lite = 소표본에서 구현한 HistGB 2단계. 같은 승인율 비교는 FOC가 고른 승인율만큼 비율 순위로 승인한 값.", src="본 연구 계산(scripts/134)", widths=["38%", "20%", "20%", "22%"]))
H.append(L2(f"소표본에서는 FOC가 같은 승인율의 비율 순위보다 {pc(FOC_WQ, 0)} 반복에서 높고 중앙값 차이 {d3(FOC_DQ)}. 26% 비율 순위와는 {pc(FOC_W, 0)}(차이 {d3(FOC_D)})로 우열이 갈리지 않음 → 점검 구간이 4,000건이라 이 차이는 구분하기 어려운 수준(방향 확인용)"))
H.append("<h3>D-3. 전체 표본: 학습 CV로 λ*, 시험 1회 채점</h3>")
H.append(figure("lab4_FOC곡선.png", "위험회피 계수 λ에 따른 Sharpe(학습 OOF)와 승인율", note="왼쪽: 학습 5겹 OOF의 Sharpe 곡선이 λ* 선택의 근거(시험 결과 사용 안 함). 점선 주황 = 이론값 SR/(2σ). 오른쪽: 시험 자료의 승인율(점수만 사용).", src="본 연구 계산(scripts/132)", w="100%", h="64mm"))
H.append(table("FOC 규칙의 학습 선택과 시험 1회 채점(v3 예측 그대로)", ["항목", "FOC(λ*)", "비율 순위 26%(v3)", "비율 순위(같은 승인율)"], [
    ["λ*(학습 OOF에서 선택)", f"{FJ['lam_star']:.2f}", "—", "—"],
    ["학습 OOF Sharpe · 승인율", f"{f3(FJ['학습_SR_foc'])} · {pc(FJ['학습_승인율_foc'])}", f"{f3(FJ['학습_SR_v3_q26'])} · 26.0%", "—"],
    ["시험 Sharpe", f"<b>{f4(FJ['시험_SR_foc'])}</b>", f4(FJ["시험_SR_v3_q26"]), f4(FJ["시험_SR_순위_같은승인율"])],
    ["시험 승인율 · 승인 대출 부도율", f"{pc(FJ['시험_승인율_foc'])} · {pc(FJ['시험_부도율_foc'])}", f"26.0% · {pc(FJ['시험_부도율_v3_q26'])}", f"{pc(FJ['시험_승인율_foc'])} · —"],
    ["FOC − 해당 규칙 [95% 구간, 발급월 군집]", "—", f"{d4(FJ['차이_foc_v3'])} [{d4(FJ['차이_구간'][0])}, {d4(FJ['차이_구간'][1])}]", f"{d4(FJ['차이_foc_순위_같은승인율'])} [{d4(FJ['차이_같은승인율_구간'][0])}, {d4(FJ['차이_같은승인율_구간'][1])}]"],
    ["승인 집합 겹침(Jaccard)", f"{FJ['자카드']:.2f}", "—", "—"]], note="시험 결과 열은 채점에만 사용. λ*와 비율 순위의 26%는 모두 학습 자료로 선택. FOC를 대표 규칙으로 올린 결정은 시험 채점 이후(G-5).", src="본 연구 계산(scripts/132)", widths=["34%", "20%", "23%", "23%"]))
H.append(L2(f"시험 Sharpe는 FOC {f4(FJ['시험_SR_foc'])}, 비율 순위 {f4(FJ['시험_SR_v3_q26'])}: 차이 {d4(FJ['차이_foc_v3'])}의 구간이 0을 포함 → <b>동등</b>(H5). 같은 승인율로 맞춰도 {d4(FJ['차이_foc_순위_같은승인율'])}[{d4(FJ['차이_같은승인율_구간'][0])}, {d4(FJ['차이_같은승인율_구간'][1])}]로 같은 결론"))
H.append(L2(f"차이는 구성에 있음: FOC는 승인을 {pc(FJ['시험_승인율_foc'])}로 줄이고 승인 대출 부도율을 {pc(FJ['시험_부도율_foc'])}(비율 순위 {pc(FJ['시험_부도율_v3_q26'])})로 낮추면서 같은 Sharpe를 냄 → 변동성 σ = {pc(FJ['시험_σ_foc'], 2)}"))
H.append("<h3>D-4. 이론 λ와 선택 λ의 차이</h3>")
H.append(L2(f"학습에서 고른 λ* = {FJ['lam_star']:.2f}, 같은 지점의 이론값 SR/(2σ<sub>p</sub>) = {FJ['학습_이론_lam']:.2f}(시험 {FJ['시험_이론_lam']:.2f}): 이론값이 {abs(FJ['학습_이론_lam'] / FJ['lam_star'] - 1) * 100:.0f}% 큼"))
H.append(L2("이론은 E[r]와 Var(r)를 정확히 안다고 가정함. 예측에 오차가 있으면 이득 추정치가 과장되어(선택 편향, winner's curse) 같은 승인 집합을 얻으려면 더 작은 λ가 필요해질 수 있음(추정, 직접 검증하지 않음)"))
H.append(L2("대출 간 공분산을 0으로 둔 근사도 σ<sub>p</sub> 추정에 영향을 줌. 따라서 이론 λ를 고정값으로 쓰지 않고 학습 CV로 고르는 것이 안전하며, 이론값은 점검용 참고치로만 사용함"))
H.append("</div>")

# ---------------- Part E
LSV = lambda c: float(LSf[c])
H.append("<div class='page'><h2>Part E. 교수님 강조 항목 구현</h2><h3>E-1. 항목별 구현 위치와 핵심 결과</h3>")
H.append(L1("교수님이 강조하신 항목마다 「무엇을, 어떻게, 어디에」 구현했는지 모았음. 새로 추가한 것은 E-2(상태공간)와 E-3(림보 · LLM 설계)"))
H.append(table("교수님 강조 항목 점검표", ["강조 항목", "구현한 것(방법)", "핵심 결과", "위치"], [
    ["스프레드 ‘예측’ (상대 스프레드)", "시장 수준 스프레드는 예측이 어려움 → 같은 시점 안의 상대 순위(점수 10분위)를 예측. 스프레드·등급·부도확률을 통제한 OLS", f"예측 초과수익 계수 {OLS.loc['M4 +우리 예측', 'b_mu']:.2f}(t = {OLS.loc['M4 +우리 예측', 't_mu']:.1f}). Baa 스프레드는 대출 실현 스프레드를 예측하지 못함", "Ⅵ-2, Ⅶ-6 (표 14, 28)"],
    ["잔존만기 · 듀레이션", "계약 WAL · 맥컬리 듀레이션 · 실현 WAL 비교, 수익률 환산 3종, 유효 듀레이션 · 볼록성", f"36개월 완납의 실현 가중평균만기 {WAL36:.2f}년 → 계약만기 연율화는 수익률을 낮게 잡음", "Ⅵ-1, Ⅶ-5 (표 13, 26)"],
    ["1등급 롱 · 10등급 숏 시뮬레이션", "발급월 안에서 점수 10분위, 1분위 − 10분위 초과수익(Newey–West t)", f"롱숏 {LSV('롱숏_평균'):.1f}%p, t = {LSV('NW_t'):.1f}, {pc(LSV('양수_월비율'), 0)}의 달에서 양(+)", "Ⅵ-3 (표 15)"],
    ["선형회귀: 모형 구현 · 해석, 불편추정", "statsmodels OLS(발급월 군집 표준오차), Mincer–Zarnowitz로 α = 0 · β = 1 검정", f"MZ 기울기 {MZ.b_mu:.3f}(표준오차 {MZ.se_mu:.3f}), 절편은 0이 아님(보수적 편향)", "Ⅵ-2, B장 (표 14, F6)"],
    ["더 나은 모형을 찾는 과정 · 패키지 활용", "사다리 10단계(scikit-learn · statsmodels · xgboost · lightgbm · catboost), 빠른 판별 규칙", f"설명력↑ 단계: {', '.join(str(s) for s in imp)}단계", "B장 (표 F6)"],
    ["레짐 스위칭 · 상태 모델 · 가우스 추정", "마르코프 전환(statsmodels) + 상태공간 local level(가우스 최대우도, Kalman 평활)", f"AIC: 상태공간 {SSJ['aic']['(b) 지역 수준 상태공간']:.0f} &lt; 마르코프 {SSJ['aic']['(c) 2국면 마르코프 전환']:.0f} &lt; 상수 {SSJ['aic']['(a) 상수 평균']:.0f}", "Ⅵ-5(표 16), E-2(표 F17)"],
    ["계량경제 인과 추론 / 예측 기법: 제약적 → 일반적 상황", "「금리를 올리면 수익이 준다」를 인과로 읽지 않음, 거절 추론(승인 확률 가중)으로 일반화 방안", f"승인 {pc(PSHARE)} 표본에서만 검증됨을 한계로 명시", "Ⅷ-4"],
    ["보완점 언급과 방안", "한계 표(영향 · 대응), 우측 중도절단 · 대표성 진단", f"한계 {NLIM}항목", "G-4 (표 32)"],
    ["정성(Qualitative): LLM 활용", "직함 · 사유 텍스트의 LLM 분류와 검증 설계", f"희소 직함 대출 {pc(PS['직함']['희소비중'], 0)}, 직군별 잔차 체계적", "Ⅷ-2·3, E-3"],
    ["1차 모먼트 · 2차 모먼트(volatility) 보완", "혼합분포로 E[r]와 Var(r)를 함께 예측, 십분위별 예측 대 실현", "예측 평균 · 표준편차와 실현의 대응", "Ⅵ-4, A-2"],
    ["애매한 것은 림보 · 휴먼 루프 · 암묵지", "승인 경계 ±2.5%p ∪ 모형 불일치 ∪ 희소 구간 = 림보, 사람 심사 결과를 기록해 학습 자료로 환류", f"림보 {pc(PS['림보']['비중'])}, 심사 정확도 70% 가정 시 Sharpe 상승", "Ⅷ-1, C-3, E-3"],
    ["구조화되지 않은 데이터 · 고유(proprietary) 데이터", "직함 텍스트 희소성 · 직군 키워드 · 사유 텍스트", "희소 직함의 AUC가 일반과 비슷하나 직군별 잔차가 한쪽으로 쏠림", "Ⅷ-2 (표 30, 31)"]],
    widths=["20%", "34%", "30%", "16%"], cls="exp"))
H.append("<h3>E-2. 상태공간(state-space) 모형과 가우스 추정</h3>")
H.append("<div class='formula'>관측식 z<sub>t</sub> = μ<sub>t</sub> + ε<sub>t</sub>, ε<sub>t</sub> ~ N(0, σ<sub>ε</sub>²) &nbsp;&nbsp; 상태식(local level) μ<sub>t+1</sub> = μ<sub>t</sub> + η<sub>t</sub>, η<sub>t</sub> ~ N(0, σ<sub>η</sub>²)<br>"
         "모수 (σ<sub>ε</sub>², σ<sub>η</sub>²)는 Kalman filter의 예측오차 분해 가우스 우도를 최대화(MLE)해 추정, 상태 μ<sub>t</sub>는 Kalman smoother로 도출 &nbsp;|&nbsp; 비교: 상수 평균, 2국면 마르코프 전환(평균·분산 전환)</div>")
H.append(L2(f"자료: 학습 파일 36개월 대출의 발급월 평균 초과수익 z, 월 300건 이상, {SSJ['월수']}개월(Part F Ⅵ-5와 같은 시계열이나 표 16의 AIC는 90개월 기준이고 여기서는 상태공간 모형과 맞춰 89개 관측으로 계산해 값이 다름). 모형·판정 기준(AIC 최소 + 한 걸음 예측)은 실행 전에 고정"))
ssr = [[SS.모형[i], f"{SS.모수[i]}", f"{SS.로그우도[i]:.1f}", (f"<b>{SS.AIC[i]:.1f}</b>" if SS["AIC 최소"][i] else f"{SS.AIC[i]:.1f}"), ("—" if pd.isna(SS['한걸음예측_RMSE(최근30개월)'][i]) else f3(SS['한걸음예측_RMSE(최근30개월)'][i])), SS.설명[i]] for i in range(len(SS))]
H.append(table("상태공간 대 마르코프 전환 대 상수 평균", ["모형", "모수", "로그우도", "AIC", "한 걸음 예측 RMSE(최근 30개월)", "추정 결과"], ssr,
               note=f"AIC는 −2·로그우도 + 2k를 같은 89개 관측으로 계산(상태공간은 첫 관측을 확산 초기값에 씀). 한 걸음 앞 예측 RMSE: 지역 수준 {f3(SSJ['rmse']['지역 수준'])}, 임의보행 {f3(SSJ['rmse']['임의보행'])}, 확대 창 평균 {f3(SSJ['rmse']['전체 평균'])}(표의 (a) 칸은 이 확대 창 평균 예측값).", src="statsmodels UnobservedComponents · MarkovRegression, 본 연구 계산(scripts/136)", widths=["20%", "6%", "10%", "8%", "16%", "40%"]))
H.append(figure("lab5_상태공간.png", "월별 평균 초과수익과 평활 상태(상태공간), 악화 국면 확률(마르코프)", note="두 방법이 같은 시기를 가리키는지 비교.", src="본 연구 계산", h="58mm"))
H.append(L2(f"상태공간(local level)이 AIC 최소({SSJ['aic']['(b) 지역 수준 상태공간']:.0f}), 표준화 잔차의 자기상관은 유의하지 않음(Ljung–Box p = {SSJ['ljungbox_p']:.2f}, 관측이 89개라 검정력은 낮음). 신호 대 잡음비 σ<sub>η</sub>²/σ<sub>ε</sub>² = {SSJ['signal_noise']:.2f} → 월 변동의 절반 이상(약 55%)이 서서히 움직이는 「수준」 변화"))
H.append(L2(f"평활 수준과 마르코프의 악화 확률의 상관 {SSJ['상관_수준_대_악화확률']:.2f}: 두 방법이 같은 시기를 악화(수준이 낮음)로 봄. 한 걸음 예측 RMSE는 임의보행({f3(SSJ['rmse']['임의보행'])})과 거의 같아 <b>수준은 지속적이지만 앞으로의 값을 더 잘 맞히지는 못함</b>(H8: A-3 규칙은 AIC 최소 · 평균 예측보다 작은 RMSE로 지지. 임의보행보다 나아지지 않는다는 점은 사전 기준이 아닌 추가 관찰)"))
H.append(L2("활용: 승인 규칙을 바꾸는 신호가 아니라 최근 수준(μ̂<sub>t</sub>)을 모니터링해 수익표와 λ를 재추정할 시점을 알리는 진단 도구(추정)"))
H.append("</div>")
H.append("<div class='page'><h3>E-3. 림보 · 휴먼 루프 · LLM 설계</h3>")
H.append(L1("<b>설계안</b>이며 이번 보고서에서 LLM을 호출해 성능을 평가하지는 않았음. 어떤 일을 맡기고 어떻게 검증할지의 구조를 정함"))
H.append(table("LLM 작업 설계(Qualitative 활용)", ["작업", "입력", "출력 스키마", "검증 방법", "가드레일"], [
    ["직함 → 직군 분류", "emp_title 원문(희소 직함 포함)", "{직군 코드, 신뢰도, 근거 한 줄}", "키워드 규칙 분류와 일치율, 직군별 잔차(표 31)가 줄어드는지", "신뢰도 낮으면 「기타」, 분류 결과는 특성으로만 사용(결정 직접 입력 금지)"],
    ["림보 사례 사유 기록 정리", "심사자 메모, 모형 점수 · 불일치 사유", "{사유 코드, 요약, 재현 가능한 조건}", "코드별 사례 수 · 사후 부도율(결과 확정 후)", "개인정보 제거, 요약만 저장"],
    ["월별 감시 보고 초안", "관문 · 드리프트 · 보정 표", "수치 인용이 있는 문장 초안", "인용 수치를 표와 코드로 대조", "표에 없는 수치 생성 금지"]], widths=["16%", "18%", "20%", "26%", "20%"], cls="exp"))
H.append(table("림보 휴먼 루프: 암묵지가 쌓이는 닫힌 루프", ["단계", "기록하는 것", "되먹임", "주기"], [
    ["1 림보 진입", "진입 이유(경계 ±2.5%p / 모형 불일치 / 희소 구간), 점수, margin d", "d가 작은 사례 비율 모니터링(C-3)", "건별"],
    ["2 사람 판정", "승인 · 거절 · 보류, 사유 코드, 판정 근거", "사유 코드별 사례 축적(희귀 사례의 암묵지)", "건별"],
    ["3 결과 확정", "만기 · 부도 결과, 수익 z", "판정 정확도 = 결과로 확인(가정 70%를 실측으로 대체)", "분기"],
    ["4 규칙·모형 갱신", "림보 구간 폭, 희소 구간 정의, 학습 자료", "정확도가 낮은 구간을 사람 심사로 유지, 높은 구간은 자동화", "분기"]], widths=["14%", "34%", "38%", "14%"], cls="exp"))
H.append(L2("열린 루프(모형이 결정하고 끝) → 닫힌 루프(사람의 판단과 결과가 다시 학습 자료로 돌아옴). 림보 구간의 사람 판정 정확도를 실측하면 Part F Ⅷ-1의 「정확도 70% 가정」을 대체할 수 있음"))
H.append("</div>")

# ---------------- Part F (기존 장)
H.append("<div class='page'><h2>Part F. 상세 검증 (기존 최종 보고서 Ⅰ~Ⅷ장)</h2>")
H.append(L1("이 Part는 기존 최종 보고서의 Ⅰ~Ⅷ장을 문구와 표 번호(표 1~42) 그대로 가져옴. 앞의 아이디어를 전체 표본 · 시험 자료에서 검증한 결과임"))
H.append(L2("Part A~E의 표(표 F)는 소표본 실험과 FOC 규칙이고, 이 Part의 표(표 1~42)는 전체 표본의 같은 잣대 비교 · 이론 실증 · 채권 형식 평가 · 운영 설계임"))
H.append(L2("Ⅳ-4(발급 시기 집중), Ⅵ-7(시간 외 검증), Ⅴ-2(9개 잣대 종합)가 불리한 결과를 포함함 → G장에서 다시 모아 공개"))
H.append("</div>")
H.extend(OLD_MAIN)

# ---------------- Part G 결론(마지막)
H.append("<div><h2 style='margin-top:14px'>Part G. 결론</h2><h3>G-1. 가설 판정</h3>")
H.append(L1("A-3의 판정 규칙을 그대로 적용한 결과(H3·H4·H6·H7의 규칙은 결과를 본 뒤 문서화한 탐색적 기준)"))
H.append(table("가설별 판정", ["ID", "가설", "근거 수치", "판정", "위치"], [
    ["H1", "기대 초과수익 기반 순위(최종 모형) &gt; 부도확률 순위", f"시험 Sharpe {f3(mv('T1', 'V3'))} 대 {f3(mv('T1', 'B0'))}, 차이 구간 [{d3(float(CI[(CI.ruler == 'T1') & (CI.strategy == 'B0')].d_lo.iloc[0]))}, {d3(float(CI[(CI.ruler == 'T1') & (CI.strategy == 'B0')].d_hi.iloc[0]))}]", "<b>지지</b>(구간 하한 &gt; 0)" if float(CI[(CI.ruler == "T1") & (CI.strategy == "B0")].d_lo.iloc[0]) > 0 else "미지지", "Part F 표 5"],
    ["H2", "완납 · 부도 잔차가 다르다(혼합분포)", f"부도 잔차 SD가 완납의 {sdr:.1f}배", "<b>지지</b>" if sdr >= 2 else "기각", "표 F9"],
    ["H3", "정규화가 out-of-sample 오차를 줄인다", f"Ridge {d4(VD[5][1])}, Lasso {d4(VD[6][1])}(OLS 대비)", f"<b>기각</b>({VD[5][0]}·{VD[6][0]})", "표 F6, F8"],
    ["H4", "비선형이 선형보다 설명력이 높다", f"소표본 점검 RMSE {d4(VD[8][1])}; 전체 표본 AUC {d4(alt('B0_sub') - alt('M1b_sub'))}", ("소표본 <b>기각</b> · 전체 표본 <b>지지</b>" if (VD[8][0] != "설명력↑" and alt('B0_sub') - alt('M1b_sub') >= 0.005) else "혼재"), "표 F6, F10"],
    ["H5", "FOC 규칙 &gt; 비율 순위 규칙", f"시험 {d4(FJ['차이_foc_v3'])} [{d4(FJ['차이_구간'][0])}, {d4(FJ['차이_구간'][1])}]", "<b>동등</b>(구간이 0 포함: 구분되지 않음)" if FJ["차이_구간"][0] <= 0 <= FJ["차이_구간"][1] else "차이 있음", "표 F15"],
    ["H6", "임계값 근처일수록 결정이 불안정", f"거절측 {pc(NR)} → {pc(FR_)}, 승인측 {pc(NA)} → {pc(FA_)}(가까움 → 멂)", ("<b>거절측에서 지지</b>, 승인측은 차이 작음" if (FR_ - NR >= 0.10 and FA_ - NA < 0.10) else ("<b>지지</b>" if min(FR_ - NR, FA_ - NA) >= 0.10 else "미지지")), "표 F12"],
    ["H7", "소프트마진 · |z| 가중이 순위 품질을 올린다", f"같은 목표(z &gt; 0) 로지스틱 대비 SR26 {d3(dsv[0])}(무가중), {d3(dsw[0])}(가중)", "<b>지지</b>" if (dsw[0] >= 0.005 and dsw[1] >= 0.75) else "<b>미지지</b>(차이가 작거나 우세 반복이 적음)", "표 F11"],
    ["H8", "신용 환경에는 천천히 움직이는 상태가 있다", f"AIC 상태공간 {SSJ['aic']['(b) 지역 수준 상태공간']:.0f} 최소, 한 걸음 예측 RMSE {f3(SSJ['rmse']['지역 수준'])} 대 평균 예측 {f3(SSJ['rmse']['전체 평균'])}", "<b>A-3 규칙으로는 지지</b>(AIC 최소, RMSE 0.218 &lt; 평균 예측 1.015). 임의보행(0.217)보다는 나아지지 않음(사전 기준 아님)", "표 F17"]], widths=["5%", "25%", "30%", "26%", "14%"]))
H.append("<h3>G-2. 설명력이 높아진 지점</h3>")
H.append(L1(f"예측 오차를 줄이고 순위 품질을 올린 지점은 <b>정보를 추가한 곳</b>(2단계 신용이력 {d4(VD[2][1])}, 4단계 시장·시기 {d4(VD[4][1])}, 가격·조건 변수 단계도 판정을 통과).  규제 · 비선형 · 2단계는 소표본에서 점검 RMSE를 더 줄이지 못함"))
H.append(L2(f"결정 가치(SR26)는 평균만 {f3(STEP[0].SR26)} → 가격 {f3(STEP[1].SR26)} → +신용이력 {f3(STEP[2].SR26)} → 전체 {f3(STEP[4].SR26)}로 오르고 그 뒤로는 {f3(STEP[9].SR26)}(v3-lite)까지 정체"))
H.append(L2("전체 표본에서 트리의 이득은 AUC에서 보임(B-5). 즉 「어떤 정보를 쓰는가」가 「어떤 모형 형태를 쓰는가」보다 먼저이고, 모형 형태의 이득은 표본 크기에 달림"))
H.append("<h3>G-3. 최종 규칙과 성과</h3>")
H.append(L1(f"<b>승인 규칙 = FOC</b> (λ* = {FJ['lam_star']:.2f}, 학습 CV로 선택): E[r − r<sub>f</sub>] − λ·ΔV &gt; 0. 재료 = v3(완납 · 부도 조건부 이득 + 3모형 평균 부도확률)"))
H.append(L2(f"시험 Sharpe <b>{f3(FJ['시험_SR_foc'])}</b>(FOC) 대 {f3(FJ['시험_SR_v3_q26'])}(비율 순위 26%): 통계적으로 같은 수준. 전부 승인 {f3(mv('T1', 'ALL'))}, 부도확률 순위 {f3(mv('T1', 'B0'))}를 크게 웃돎(발급월 군집 구간은 Part F 표 6)"))
H.append(L2(f"FOC 규칙의 가치: 같은 Sharpe를 승인 {pc(FJ['시험_승인율_foc'])} · 승인 대출 부도율 {pc(FJ['시험_부도율_foc'])}로 달성(비율 순위 26% · {pc(FJ['시험_부도율_v3_q26'])})하고, 경계를 「이득 − λ × 위험의 대가」로 설명함"))
H.append(L2(f"9개 잣대 종합은 Part F 표 9: {NJ['●']}개 우세 · {NJ['◐']}개 동등 · {NJ['◑']}개 혼재 · {NJ['○']}개 열세(열세 = 2017년 이후 시간 외). 발급월마다 같은 비율을 승인하면 Sharpe가 {f3(MON)}로 낮아짐(표 7)"))
H.append(L1("<b>불리한 결과(공개)</b>"))
H.append(L2(f"시간 외 검증(2017~20년 발급, 같은 승인율 6.4%): {f3(OOT16)}로 팀 Sharpe 보고서({f3(OOT16_TEAM)})보다 낮음(Part F 표 18). 이 결과는 FOC 규칙에도 해당하는지 다시 계산하지 않았음"))
H.append(L2("발급 시기 집중: 개선의 상당 부분이 대출을 내주는 시기의 선택에서 나옴(Ⅳ-4). 이론 λ ≠ 선택 λ(D-4)"))
H.append("</div>")
_c = OLD_CONC.replace("<div class='page'>", "<div>", 1).replace("Ⅸ. 결론과 한계</h2>", "G-4. 한계와 남은 과제(기존 Ⅸ장: 결론과 한계)</h2>" + L2("주의: 아래 기존 Ⅸ장의 결론은 비율 순위 규칙(상위 26%)을 기준으로 쓴 것임. FOC 규칙은 같은 예측에서 이 규칙과 구분되지 않으며(차이 +0.0006, 구간이 0 포함) FOC를 대표 규칙으로 올린 것은 시험 채점 이후이고, FOC의 시간 외 검증은 다시 계산하지 않았음(G-5)"), 1)
_c = _c.replace("</tbody>", "<tr class=''><td>FOC 규칙을 시험 채점 이후 대표 규칙으로 채택</td><td>사후 선택 가능성, FOC의 시간 외 검증 미실시</td><td>G-5 공개, 시간 외 재계산은 과제</td></tr></tbody>", 1)
H.append(_c)
H.append("<div><h3>G-5. 이번 추가분의 사후 선택 공개와 재현</h3>")
H.append(table("추가분 기록(시험 채점 이후)", ["순서", "변경 · 추가", "동기", "결정 근거", "시험 영향"], [
    ["6", "FOC 규칙을 대표 승인 규칙으로 올림", "팀 공유 Sharpe 보고서의 1계조건을 중심에 두기로 한 논의", "λ*는 학습 5겹 OOF Sharpe 최대(시험 미사용). 승인 규칙 변경 자체는 시험 채점 이후", f"{f4(FJ['시험_SR_v3_q26'])} → {f4(FJ['시험_SR_foc'])}(구간이 0 포함: 개선으로 주장하지 않음)"],
    ["7", "소표본 실험(Ridge · Lasso · 분류 · 소프트마진)", "교수님 항목 · 방법 비교", "학습 파일만 사용, 시험 파일 미사용", "없음"],
    ["8", "상태공간 모형 추가", "교수님 강조 항목 중 빠진 부분", "학습 36개월 대출 월별 시계열", "없음"]], widths=["6%", "26%", "22%", "28%", "18%"]))
H.append(table("추가 재현 순서", ["순서", "스크립트", "내용"], [
    ["11", "132_lam_rule_experiment.py", "FOC 규칙: λ 학습 CV, 시험 1회, 표/FOC_*.csv"], ["12", "134_small_sample_lab.py", "소표본 사다리 · Ridge · Lasso · 분류 · 임계값 · FOC 20회(약 30분), 표/랩_*.csv, fig/lab*.png"],
    ["13", "136_state_space.py", "상태공간(local level) 대 마르코프, 표/교수_H_*.csv"], ["14", "135_report_v4.py", "이 보고서 HTML(PDF 변환)과 표/보고서표/표F*.csv"]], widths=["8%", "34%", "58%"]))
H.append("</div>")
H.append(OLD_STAT)
H.append(OLD_APP)
H.append("</body></html>")
GH = "".join(H); (O / "최종보고서_v4.html").write_text(GH, encoding="utf-8"); RF.dump_tables(NB, O / "보고서표", "표_색인_F.csv"); print("saved v4", len(GH), "새 표", NB.tn, "새 그림", NB.fn)
