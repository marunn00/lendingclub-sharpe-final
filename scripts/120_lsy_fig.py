"""120 이수용 방식 평가 그림(119 표만 읽음). 이수용 RAROC 값은 그 보고서 표(5장)에 인쇄된 수치를 참고선으로만 그림(평가 대상 467,101건, 우리 454,462건)."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
O = Path(__file__).resolve().parents[1] / "산출물" / "우수사례평가" / "이수용"; T = O / "표"
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False, "font.size": 15, "axes.titlesize": 18, "axes.labelsize": 15})
P = pd.read_csv(T / "정책표.csv").set_index("정책"); CU = pd.read_csv(T / "승인율_곡선.csv"); rfx = [0, 1, 2, 3]; rfc = ["rf 0%", "rf 1%", "rf 2%", "rf 3%"]
LSY = {"rf 0%": .485, "rf 1%": .415, "rf 2%": .344, "rf 3%": .269, "국채": .414}                     # 이수용 보고서 5장 표, RAROC 모델(최종)
fig, axs = plt.subplots(1, 2, figsize=(16, 7))
sty = [("우리 방식 20%", "#d62728", "-", 3), ("우리 방식 25%", "#e8636a", "-", 2.5), ("우리 방식 50%", "#f2a5a8", "-", 2.5), ("A등급만", "#e67e22", "-", 2), ("전부 승인", "#7f7f7f", "-", 2)]
for nm, c, ls, lw in sty: axs[0].plot(rfx, P.loc[nm, rfc], "o" + ls, color=c, lw=lw, label=nm)
axs[0].plot(rfx, [LSY[k] for k in rfc], "s--", color="#1f77b4", lw=2, label="이수용 RAROC (보고서 값, 참고)")
axs[0].set_xticks(rfx, ["0%", "1%", "2%", "3%"]); axs[0].set_xlabel("무위험수익률 rf"); axs[0].set_ylabel("Sharpe (대출 단위)"); axs[0].set_ylim(0, 0.85); axs[0].legend(fontsize=12, loc="upper right"); axs[0].set_title("rf를 바꿔도 순서가 바뀌지 않습니다")
axs[1].plot(CU.실제승인율 * 100, CU.국채, "-o", color="#d62728", lw=3, label="우리 방식")
for nm, c, mk in (("A등급만", "#e67e22", "D"), ("A~B등급만", "#b9770e", "D"), ("전부 승인", "#7f7f7f", "D"), ("부도확률 낮은 순 20%", "#1f77b4", "^")):
    axs[1].scatter(P.loc[nm, "승인율"] * 100, P.loc[nm, "국채"], s=110, color=c, marker=mk, zorder=5, label=nm)
axs[1].scatter([17.0], [LSY["국채"]], s=110, marker="s", facecolors="none", edgecolors="#1f77b4", lw=2, label="이수용 RAROC (보고서 값, 참고)")
axs[1].axvline(25, color="#d62728", ls=":", lw=1.3); axs[1].axvline(50, color="#d62728", ls=":", lw=1.3)
axs[1].text(25.5, 0.82, "25%", color="#d62728", fontsize=12); axs[1].text(50.5, 0.82, "50%", color="#d62728", fontsize=12)
axs[1].set_xlabel("승인율 (%, 평가 대상 건수 기준)"); axs[1].set_ylabel("Sharpe (국채 rf)"); axs[1].set_ylim(0, 0.88); axs[1].legend(fontsize=11.5, loc="upper right"); axs[1].set_title("같은 승인율이면 우리 방식이 A등급 규칙보다 높습니다")
fig.suptitle("이수용 보고서의 평가 방식으로 다시 재도 우리 방식이 가장 높습니다", fontsize=20)
fig.text(.01, .005, "이수용 방식: 만기가 지난 시험 대출 454,462건(상태로 거르지 않음), 대출별 (실현 총수익률 - rf × 잔액·년)의 평균 ÷ 표준편차. 국채 = 발급월·만기 맞춤 3년/5년.", fontsize=10.5, color="#555")
fig.tight_layout(rect=(0, .035, 1, 1)); fig.savefig(O / "fig" / "fig9_이수용방식.png", dpi=150); print("saved")
