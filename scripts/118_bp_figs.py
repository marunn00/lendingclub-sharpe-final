"""118 우수사례 1·3 평가 그림(116 표만 읽음). 출력: 산출물/우수사례평가/fig/"""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path
O = Path(__file__).resolve().parents[1] / "산출물" / "우수사례평가"; T = O / "표"; FG = O / "fig"; FG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False, "font.size": 16, "axes.titlesize": 19, "axes.labelsize": 16})
COL = {"M6v2": "#d62728", "B0": "#1f77b4", "GRADE": "#7f7f7f"}; LAB = {"M6v2": "우리 방식", "B0": "부도확률 XGBoost", "GRADE": "등급 순서"}
def save(fig, name, note): fig.text(.01, .005, note, fontsize=10.5, color="#555"); fig.tight_layout(rect=(0, .035, 1, 1)); fig.savefig(FG / name, dpi=150); plt.close(fig)
N3 = "우수사례 3 방식: 대출별 IRR - 발행월 국채(3년/5년), 승인 대출의 평균 ÷ 표준편차. 검증 = 학습 파일 교차검증 예측, 시험 = 시험 파일. 표본 각 148,922건 × 100회. IRR 계산 불가 = -100%."
R3 = pd.read_csv(T / "우3_seed별.csv"); R3 = R3[R3.rule == "R1"]; CV = pd.read_csv(T / "우3_곡선_seed0.csv"); CV = CV[CV.rule == "R1"]

fig, axs = plt.subplots(1, 2, figsize=(16, 6.8))
for m in ("GRADE", "B0", "M6v2"):
    for ax, col, ttl in ((axs[0], "test_SR", "검증에서 고른 최적 기준선"), (axs[1], "test_SR_at25", "승인 25% (우리가 미리 정한 지점)")):
        v = R3[R3.model == m][col].to_numpy(); xs = np.linspace(v.min() - .05, v.max() + .05, 300); ax.fill_between(xs, gaussian_kde(v)(xs), alpha=.35, color=COL[m]); ax.plot(xs, gaussian_kde(v)(xs), color=COL[m], lw=2.5, label=f"{LAB[m]} (중앙 {np.median(v):.2f})"); ax.set_title(ttl)
for ax in axs: ax.axvline(R3.test_SR_all.median(), color="k", ls=":", lw=1.5); ax.set_xlabel("시험 샤프 비율"); ax.legend(fontsize=13, loc="upper left")
axs[0].set_ylabel("밀도 (100회 반복)"); axs[0].text(R3.test_SR_all.median() + .01, axs[0].get_ylim()[1] * .55, "전부 승인\n(기존 LC 정책)", fontsize=12)
fig.suptitle("우수사례 3 방식으로 100번 반복 평가해도 우리 방식이 가장 높습니다", fontsize=20); save(fig, "fig6_우3_샤프분포.png", N3)

fig, ax = plt.subplots(figsize=(13, 6.8))
for m in ("GRADE", "B0", "M6v2"):
    d = CV[CV.model == m]; ax.plot(d.frac * 100, d.test_SR, color=COL[m], lw=3 if m == "M6v2" else 2, label=LAB[m])
ax.axhline(0, color="k", lw=1); ax.set_xlabel("승인 비율 (%, 건수 기준)"); ax.set_ylabel("시험 샤프 비율 (대출 단위)"); ax.legend(fontsize=14)
ax.set_title("우수사례 3 방식: 승인을 넓힐수록 샤프가 줄고, 전부 승인하면 0 근처입니다"); save(fig, "fig7_우3_승인비율별.png", N3 + " (seed 0)")

U = pd.read_csv(T / "우1_혼동행렬_금액위험.csv"); u = U[U.지점.str.startswith("나")].set_index("모델"); CU = pd.read_csv(T / "우1_금액위험_곡선.csv")
order = ["LendingClub(전부 승인)", "세부등급 규칙", "부도확률 XGBoost", "우리 방식 v2(3모형 부도확률)"]; nm = ["LendingClub\n전부 승인", "등급 순서", "부도확률\nXGBoost", "우리 방식"]
fig, axs = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw=dict(width_ratios=[1, 1.15])); x = np.arange(4)
la = u.loc[order, "loss_amnt"].to_numpy() / 1e6; li = u.loc[order, "loss_amnt_implicit"].to_numpy() / 1e6
axs[0].bar(x, la, color="#c0392b", label="놓친 부도 손실 (loss_amnt)"); axs[0].bar(x, li, bottom=la, color="#f5b7b1", label="잘못 거절한 대출의 기대수익 (implicit)")
for i in range(4): axs[0].text(i, la[i] + li[i] + 25, f"{la[i] + li[i]:,.0f}", ha="center", fontsize=14)
axs[0].set_xticks(x, nm, fontsize=13); axs[0].set_ylabel("Total Risk (백만 달러, 낮을수록 좋음)"); axs[0].set_ylim(0, 2250); axs[0].legend(fontsize=12, loc="upper right"); axs[0].set_title("거절 비율 = 부도율(19.5%)일 때")
for m in ("GRADE", "B0", "M6v2"):
    d = CU[CU.model == m]; axs[1].plot(d.frac * 100, d.total_risk / 1e6, "-o", ms=4, color=COL[m], lw=3 if m == "M6v2" else 2, label=LAB[m])
axs[1].axhline(U[U.모델.str.startswith("LendingClub")].total_risk.iloc[0] / 1e6, color="k", ls=":", lw=1.5); axs[1].text(8, 1870, "전부 승인", fontsize=12)
axs[1].set_xlabel("승인 비율 (%, 건수 기준)"); axs[1].set_ylabel("Total Risk (백만 달러)"); axs[1].legend(fontsize=13); axs[1].set_title("승인 비율별 Total Risk")
fig.suptitle("우수사례 1 방식: 우리 방식이 놓친 손실 + 놓친 수익의 합이 가장 작습니다", fontsize=20)
save(fig, "fig8_우1_금액위험.png", "우수사례 1 방식: loss_amnt = installment×기간 - total_pymnt(부도인데 승인), implicit = total_pymnt - loan_amnt(상환인데 거절). 시험 결과 확정 744,608건.")
print(sorted(p.name for p in FG.iterdir()))
