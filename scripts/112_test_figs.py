"""112 시험 결과 그림(111 표만 읽음). 출력: 산출물/테스트평가/fig/"""
import json, pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
O = Path(__file__).resolve().parents[1] / "산출물" / "테스트평가"; T = O / "표"; FG = O / "fig"; FG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False, "font.size": 17, "axes.titlesize": 21, "axes.labelsize": 18})
R = pd.read_csv(T / "ROE_곡선.csv"); S = pd.read_csv(T / "샤프_곡선.csv"); K = pd.read_csv(T / "운영지점_요약.csv"); res = json.load(open(O / "결과.json")); ref = res["ROE_ref"]["all"]
if (K.strategy == "M6v2").any():                                      # 최종 모델 = v2(117). 그림의 '우리 방식'은 v2, v1은 그리지 않음
    ren = lambda d: d.assign(strategy=d.strategy.replace({"M6": "M6v1", "M6v2": "M6"})); R, S, K = ren(R), ren(S), ren(K)
COL = {"M6": "#d62728", "B0": "#1f77b4", "GRADE": "#7f7f7f"}; LAB = {"M6": "우리 방식 (위험 대비 수익 순위)", "B0": "부도확률 모델 (XGBoost)", "GRADE": "등급 순서대로 승인"}
NOTE = "시험 데이터 중 결과가 확정된 대출 744,608건. 모델·승인율은 학습 데이터만으로 정했고, 시험 결과는 채점에만 썼습니다."
def save(fig, name): fig.text(.01, .005, NOTE, fontsize=11, color="#555"); fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(FG / name, dpi=150); plt.close(fig)

# fig1 ROE 곡선
fig, ax = plt.subplots(figsize=(13, 7.5)); g = R[R.group == "all"]
for k in ("GRADE", "B0", "M6"): d = g[g.strategy == k].sort_values("rate"); ax.plot(d.rate * 100, d.ROE, "-o", ms=4, lw=3 if k == "M6" else 2, color=COL[k], label=LAB[k])
ax.axhline(ref["treasury"], color="green", ls="--", lw=2); ax.text(70, ref["treasury"] + .25, f"전부 국채 {ref['treasury']:.2f}%", color="green", fontsize=15)
ax.axhline(0, color="k", lw=1); ax.axvline(25, color=COL["M6"], ls=":", lw=1.5); ax.text(26, -6.5, "미리 정한 승인율 25%", color=COL["M6"], fontsize=14)
ax.set_ylim(-9, 3); ax.set_xlabel("승인 비율 (%, 금액 기준)"); ax.set_ylabel("연 수익률 ROE (%)"); ax.legend(loc="lower left", fontsize=14)
ax.set_title(f"얼마나 승인하든 우리 방식이 가장 많이 법니다 (전부 승인하면 {ref['all_in']:.1f}%)"); save(fig, "fig1_승인율별_수익률.png")

# fig2 샤프 곡선(36개월)
fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharey=False)
for ax, t in zip(axs, (36, 60)):
    g = S[S.term == t]
    for k in ("GRADE", "B0", "M6"): d = g[g.strategy == k]; ax.plot(d.rate * 100, d.SR, lw=3 if k == "M6" else 2, color=COL[k], label=LAB[k])
    ax.axvline(25, color=COL["M6"], ls=":", lw=1.5); ax.set_xlabel("승인 비율 (%)"); ax.set_title(f"{t}개월 대출"); ax.set_ylim(0, None)
axs[0].set_ylabel("샤프 비율 (위험 1단위당 초과수익)"); axs[0].legend(fontsize=13, loc="upper right")
k25 = K[np.isclose(K.rate, .25)].set_index("strategy")
fig.suptitle(f"같은 위험에서 더 많이 법니다: 승인 25% 샤프 {k25.SR36_test['M6']:.2f} vs 등급 순서 {k25.SR36_test['GRADE']:.2f} (36개월)", fontsize=20); save(fig, "fig2_승인율별_샤프.png")

# fig3 1억 원 예시 막대
items = [("전부 승인", ref["all_in"], "#444"), ("등급 순서\n25% 승인", k25.ROE_test["GRADE"], COL["GRADE"]), ("전부 국채", ref["treasury"], "green"), ("부도확률 모델\n25% 승인", k25.ROE_test["B0"], COL["B0"]), ("우리 방식\n25% 승인", k25.ROE_test["M6"], COL["M6"])]
fig, ax = plt.subplots(figsize=(15, 7.5)); x = np.arange(len(items)); v = [i[1] for i in items]
ax.bar(x, v, color=[i[2] for i in items], width=.6); ax.axhline(0, color="k", lw=1); ax.set_xticks(x, [i[0] for i in items])
for i, val in enumerate(v): ax.text(i, val + .25 if val >= 0 else val - .4, f"{val:+.2f}%\n1억 원당\n{val * 100:+,.0f}만 원/년", ha="center", va="bottom" if val >= 0 else "top", fontsize=13)
ax.set_ylim(-13.5, 4.5); ax.set_ylabel("연 수익률 ROE (%)"); ax.set_title("1억 원을 맡기면 1년에 얼마를 벌까"); save(fig, "fig3_1억원_예시.png")

# fig4 학습 점검 vs 시험
m6 = k25.loc["M6"]; lab = ["ROE (%)", "샤프 36개월", "샤프 60개월"]; cv = [m6.ROE_cv, m6.SR36_cv, m6.SR60_cv]; te = [m6.ROE_test, m6.SR36_test, m6.SR60_test]
fig, ax = plt.subplots(figsize=(12, 7)); x = np.arange(3)
ax.bar(x - .2, cv, .4, color="#f4a582", label="학습 데이터 안 점검 (교차검증)"); ax.bar(x + .2, te, .4, color=COL["M6"], label="처음 보는 시험 데이터")
for i in range(3): ax.text(i - .2, cv[i] + .03, f"{cv[i]:.2f}", ha="center", fontsize=16); ax.text(i + .2, te[i] + .03, f"{te[i]:.2f}", ha="center", fontsize=16)
ax.set_xticks(x, lab); ax.set_ylim(0, 2.4); ax.legend(fontsize=15); ax.set_title("연습 문제 성적과 실제 시험 성적이 거의 같습니다 (우리 방식, 승인 25%)"); save(fig, "fig4_연습_대_시험.png")

# fig5 무엇을 다르게 고르나
G = pd.read_csv(T / "승인_등급구성.csv").set_index("등급"); W = pd.read_csv(T / "다르게_고른_대출.csv").set_index("그룹")
fig, axs = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw=dict(width_ratios=[1.3, 1]))
sh = G / G.sum() * 100; cols = plt.cm.RdYlGn_r(np.linspace(.05, .95, len(sh))); bottom = np.zeros(3)
for j, (gr, row) in enumerate(sh.iterrows()):
    axs[0].bar(["우리 방식", "부도확률 모델", "등급 순서"], row[["M6", "B0", "등급규칙"]].to_numpy(), bottom=bottom, color=cols[j], label=f"{gr}등급"); bottom += row[["M6", "B0", "등급규칙"]].to_numpy()
axs[0].set_ylabel("승인한 대출의 등급 구성 (%)"); axs[0].legend(fontsize=11, ncol=4, loc="upper center", columnspacing=.8, handlelength=1.2); axs[0].set_ylim(0, 122); axs[0].set_title("승인 25%의 등급 구성")
rows = [("우리 방식만 승인", "M6만 승인", COL["M6"]), ("부도확률 모델만 승인", "B0만 승인", COL["B0"])]
for i, (nm, key, c) in enumerate(rows):
    r = W.loc[key]; axs[1].bar(i, r["초과수익_금액가중"], color=c, width=.6)
    axs[1].text(i, r["초과수익_금액가중"] + (.15 if r["초과수익_금액가중"] > 0 else -.55), f"{r['초과수익_금액가중']:+.2f}%p\n금리 {r['평균금리']:.1f}%, 부도 {r['부도율'] * 100:.1f}%\n{int(r['건수']):,}건", ha="center", fontsize=13)
axs[1].axhline(0, color="k", lw=1); axs[1].set_xticks([0, 1], [r[0] for r in rows]); axs[1].set_ylim(-4.5, 3.5); axs[1].set_ylabel("국채보다 더 번 연 수익 (%p)"); axs[1].set_title("두 방식이 서로 다르게 고른 대출")
fig.suptitle("이자가 너무 낮은 대출 대신, 부도를 감안해도 이자를 충분히 받는 대출을 고릅니다", fontsize=20); save(fig, "fig5_무엇을_다르게_고르나.png")
print("figs saved", sorted(p.name for p in FG.iterdir()))
