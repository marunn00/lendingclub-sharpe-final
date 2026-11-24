"""129 최종 보고서 그림(121·124·126·127·128·131 표만 읽음, 계산 없음). 제목은 보고서 캡션([그림 n])에 두고 그림에는 단위·범례만 둔다(통계청 보도자료 관행).
팀 공유 보고서 값(빨간 빈 표식)은 표/팀보고서_인쇄값.csv(인쇄값 전사)에서 읽는다. 출력: 산출물/최종보고서/fig/*.png"""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
O = Path(__file__).resolve().parents[1] / "산출물" / "최종보고서"; T = O / "표"; F = O / "fig"; F.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False, "font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8, "legend.frameon": False})
NAVY, BLUE, ORANGE, GRAY, DARK, RED, GREEN = "#1F4E79", "#5B9BD5", "#ED7D31", "#A5A5A5", "#595959", "#C00000", "#70AD47"
SC = {"V3": (NAVY, "최종 모형(v3)"), "M6v2": (BLUE, "v2"), "B0": (ORANGE, "부도확률 순위"), "GRADE": (GRAY, "세부등급 순서")}
PR = pd.read_csv(T / "팀보고서_인쇄값.csv"); pv = lambda r, k: float(PR[(PR.잣대 == r) & (PR.항목 == k)].값.iloc[0])
MAIN = pd.read_csv(T / "주요결과_T1_T2.csv"); CUR = pd.read_csv(T / "곡선_T1_T2.csv"); ROB = pd.read_csv(T / "강건성.csv").set_index("strategy")
def save(fig, n): fig.tight_layout(); fig.savefig(F / n, dpi=200); plt.close(fig)

# 1 요약 막대: 과제 정의 Sharpe(시험)
m = MAIN[MAIN.ruler == "T1"].set_index("strategy"); order = [("ALL", "전부 승인"), ("GRADE", "세부등급 순서"), ("B0", "부도확률 순위"), ("M5", "기대 초과수익 순위"), ("M6", "v1 위험 대비 수익"), ("M6v2", "v2 (+3모형 평균)"), ("V3", "최종 v3 (+목표 정렬)")]
fig, ax = plt.subplots(figsize=(9, 4.2)); y = np.arange(len(order)); v = [m.loc[k, "SR_test"] for k, _ in order]
ax.barh(y, v, color=[GRAY] * 3 + [BLUE] * 3 + [NAVY], height=0.62)
for i, x in enumerate(v): ax.text(x + (0.004 if x >= 0 else -0.004), i, f"{x:.3f}", va="center", ha="left" if x >= 0 else "right", fontsize=11, fontweight="bold" if i == len(v) - 1 else None)
ax.axvline(0, color=DARK, lw=0.8); ax.axvline(pv("T1", "제안 모형(S8)"), color=RED, ls="--", lw=1.2); ax.text(pv("T1", "제안 모형(S8)") - 0.003, 1.0, "팀 공유 Sharpe 보고서\n제안 모형 0.147", color=RED, fontsize=9.5, ha="right")
ax.set_yticks(y, [b for _, b in order]); ax.set_xlabel("과제 정의 포트폴리오 Sharpe (시험 744,608건)"); ax.set_xlim(-0.16, 0.2); ax.grid(axis="y", visible=False); save(fig, "fig01_요약.png")

# 2 승인율별 곡선
fig, ax = plt.subplots(figsize=(10, 4.6))
for k in ("GRADE", "B0", "M6v2", "V3"):
    c = CUR[CUR.strategy == k]; ax.plot(c.q * 100, c.T1_test, color=SC[k][0], lw=2.6 if k == "V3" else 1.6, label=SC[k][1] + " (시험)")
c = CUR[CUR.strategy == "V3"]; ax.plot(c.q * 100, c.T1_train, color=NAVY, lw=1.2, ls="--", label="최종 모형 (학습 교차검증)")
q = float(MAIN[(MAIN.ruler == "T1") & (MAIN.strategy == "V3")].q.iloc[0]); ax.axvline(q * 100, color=NAVY, ls=":", lw=1.2); ax.text(q * 100 + 0.8, 0.175, f"운영 지점 {q * 100:.0f}% (학습에서 선택)", color=NAVY, fontsize=10)
for k, lab in (("제안 모형(S8)", "Sharpe 보고서 제안"), ("선행연구(JDQS) 재현", "JDQS 재현"), ("과제 기본 흐름(부도확률 컷오프)", "부도확률 컷오프")):
    r = PR[(PR.잣대 == "T1") & (PR.항목 == k)].iloc[0]; ax.scatter(r.승인율 * 100, r.값, s=70, facecolors="none", edgecolors=RED, lw=1.6, zorder=5); ax.annotate(lab, (r.승인율 * 100, r.값), xytext=(6, -12 if "컷오프" in k else 6), textcoords="offset points", color=RED, fontsize=9.5)
ax.scatter([100], [m.loc["ALL", "SR_test"]], color=DARK, zorder=5); ax.annotate(f"전부 승인 {m.loc['ALL', 'SR_test']:.3f}", (100, m.loc["ALL", "SR_test"]), xytext=(-110, 8), textcoords="offset points", fontsize=10)
ax.axhline(0, color=DARK, lw=0.8); ax.set_xlabel("승인율 (%, 점수 상위부터 승인, 나머지는 국채)"); ax.set_ylabel("Sharpe"); ax.set_xlim(0, 101); ax.set_ylim(-0.13, 0.2); ax.legend(fontsize=9.5, loc="upper right"); save(fig, "fig02_승인율곡선.png")

# 3 대표성
R = pd.read_csv(T / "대표성_연도별.csv"); r = R[R.yr <= 2018]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
a1.bar(r.yr - 0.2, r.거절_신청 / 1e6, width=0.4, color=GRAY, label="거절 신청"); a1.bar(r.yr + 0.2, r.승인_대출 / 1e6, width=0.4, color=NAVY, label="승인 대출(본 자료)")
b = a1.twinx(); b.plot(r.yr, r.플랫폼_승인율 * 100, color=ORANGE, marker="o", lw=2, label="플랫폼 승인율(오른쪽)"); b.set_ylim(0, 20); b.set_ylabel("승인율 (%)"); b.grid(False); b.spines["top"].set_visible(False)
a1.set_ylabel("건수 (백만)"); a1.legend(loc="upper left", fontsize=9.5); b.legend(loc="upper center", fontsize=9.5)
r2 = R[R.yr <= 2013]; a2.plot(r2.yr, r2.승인_FICO_중앙, color=NAVY, marker="o", lw=2, label="승인 대출 FICO 중앙값"); a2.plot(r2.yr, r2.거절_FICO_중앙, color=GRAY, marker="s", lw=2, label="거절 신청 FICO 중앙값")
a2.axhline(660, color=RED, ls="--", lw=1); a2.text(2007.1, 663, "660", color=RED, fontsize=9); a2.set_ylabel("FICO"); a2.set_ylim(540, 740); a2.legend(fontsize=9.5, loc="lower right"); a2.set_xticks(r2.yr)
save(fig, "fig03_대표성.png")

# 4 중도절단
Z = pd.read_csv(T / "중도절단_연도별.csv"); Z = Z[Z.발급연도 >= 2009]
fig, ax = plt.subplots(figsize=(10, 4)); ax.bar(Z.발급연도, Z.결과확정비율 * 100, color="#DDEBF7", edgecolor=BLUE, label="결과 확정 비율(%)")
ax.plot(Z.발급연도, Z.확정대출_부도율 * 100, color=RED, marker="o", lw=2, label="결과 확정 대출만의 부도율(%)"); ax.plot(Z.발급연도, Z.누적부도_36개월 * 100, color=NAVY, marker="s", lw=2, label="경쟁위험 누적부도율 36개월(%, 상환 중 포함)")
ax.set_ylabel("%"); ax.set_xticks(Z.발급연도); ax.legend(fontsize=9.5, loc="upper center", ncol=3); ax.set_ylim(0, 122); ax.set_yticks(range(0, 101, 20)); save(fig, "fig04_중도절단.png")

# 5 누적부도(Aalen–Johansen)
CF = pd.read_csv(T / "누적부도_CIF.csv"); fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for ax, t in zip(axs, (36, 60)):
    for g, col, ls in (("전체 신청", DARK, "-"), ("최종 모형 승인", NAVY, "-"), ("최종 모형 거절", GRAY, "--")):
        c = CF[(CF.만기 == t) & (CF.집단 == g)]; ax.plot(c.개월, c.누적부도 * 100, color=col, ls=ls, lw=2.4 if "승인" in g else 1.8, label=f"{g} (n={c.n.iloc[0]:,})")
        ax.annotate(f"{c.누적부도.iloc[-1] * 100:.1f}%", (c.개월.iloc[-1], c.누적부도.iloc[-1] * 100), xytext=(4, 0), textcoords="offset points", fontsize=9.5, color=col, va="center")
    ax.set_xlabel(f"발급 후 개월 ({t}개월 만기)"); ax.set_ylabel("누적부도율 (%)"); ax.legend(fontsize=9, loc="upper left"); ax.set_xlim(0, t + 7)
save(fig, "fig05_누적부도.png")

# 6 10분위·롱숏
DC = pd.read_csv(T / "교수_C_10분위.csv"); LSm = pd.read_csv(T / "교수_C_롱숏_월별.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
for i, (k, col) in enumerate((("최종 모형", NAVY), ("부도확률 순위", ORANGE), ("세부등급 순위", GRAY))):
    c = DC[DC.순위 == k]; a1.bar(c.분위 + (i - 1) * 0.27, c.평균초과수익, width=0.27, color=col, label=k)
a1.axhline(0, color=DARK, lw=0.8); a1.set_xticks(range(1, 11)); a1.set_xlabel("분위 (1 = 가장 좋은 점수, 발급월·만기 안)"); a1.set_ylabel("평균 초과수익 (%p, 연)"); a1.legend(fontsize=9.5)
ym = pd.to_datetime(LSm.ym.astype(int).astype(str), format="%Y%m"); a2.bar(ym, LSm.롱숏, width=25, color=np.where(LSm.롱숏 > 0, NAVY, RED)); a2.axhline(LSm.롱숏.mean(), color=ORANGE, ls="--", lw=1.5)
a2.text(0.98, 0.9, f"월평균 {LSm.롱숏.mean():.1f}%p", transform=a2.transAxes, ha="right", color=ORANGE, fontsize=10.5); a2.axhline(0, color=DARK, lw=0.8); a2.set_ylabel("1분위 - 10분위 (%p)"); a2.set_xlabel("발급연도")
a2.xaxis.set_major_locator(matplotlib.dates.YearLocator(2)); a2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
save(fig, "fig06_분위롱숏.png")

# 7 모먼트 보정
MO = pd.read_csv(T / "교수_D_모먼트.csv"); fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for ax, (lab, unit) in zip(axs, (("평균", "기대 초과수익 (%p)"), ("표준편차", "초과수익 표준편차 (%p)"))):
    c = MO[MO.모먼트 == lab]; lo, hi = min(c.예측.min(), c.실현.min()), max(c.예측.max(), c.실현.max()); ax.plot([lo, hi], [lo, hi], color=GRAY, ls="--", lw=1)
    ax.plot(c.예측, c.실현, color=NAVY, marker="o", lw=2); ax.set_xlabel(f"예측 {unit}"); ax.set_ylabel(f"실현 {unit}")
    ax.text(0.04, 0.92, f"{'1차' if lab == '평균' else '2차'} 모먼트 · 상관 {np.corrcoef(c.예측, c.실현)[0, 1]:.3f}", transform=ax.transAxes, fontsize=10.5, color=NAVY)
save(fig, "fig07_모먼트.png")

# 8 레짐
G = pd.read_csv(T / "교수_E_레짐_월별.csv"); ym = pd.to_datetime(G.ym.astype(int).astype(str), format="%Y%m")
fig, ax = plt.subplots(figsize=(10, 3.8)); ax.fill_between(ym, -1, 5, where=G.p_bad > 0.5, color="#FBE5D6", step="mid", label="악화 국면(평활 확률 > 0.5)")
ax.plot(ym, G.z, color=NAVY, lw=1.8, label="발급월 평균 초과수익 (%p)"); b = ax.twinx(); b.plot(ym, G.p_bad, color=ORANGE, lw=1.2, ls="--", label="악화 국면 확률(오른쪽)"); b.set_ylim(0, 1.05); b.grid(False); b.spines["top"].set_visible(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=9.5, loc="upper center", ncol=3, framealpha=0.95)
ax.set_ylim(-1, 4.2); ax.set_yticks(np.arange(-1, 3.6, 0.5)); b.set_ylim(0, 1.26); b.set_yticks(np.arange(0, 1.01, 0.2)); ax.set_ylabel("%p"); save(fig, "fig08_레짐.png")

# 9 rf 민감도(T1·T2)
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4)); rf = [0, 1, 2, 3]
for ax, ru in zip(axs, ("T1", "T2")):
    for k, col, lab in (("V3", NAVY, "최종 모형"), ("B0", ORANGE, "부도확률 순위"), ("ALL", DARK, "전부 승인")): ax.plot(rf, [ROB.loc[k, f"{ru}_rf{x}"] for x in rf], color=col, marker="o", lw=2.2 if k == "V3" else 1.6, label=lab)
    if ru == "T2":
        ax.plot(rf, [pv(f"T2_rf{x}", "M4_TwoStage") for x in rf], color=RED, marker="o", mfc="none", ls="--", lw=1.4, label="M4 보고서 인쇄값")
    ax.axhline(0, color=DARK, lw=0.8); ax.set_xticks(rf, [f"{x}%" for x in rf]); ax.set_xlabel("무위험수익률 rf (고정)"); ax.set_ylabel("과제 정의 Sharpe" if ru == "T1" else "기회풀 Sharpe"); ax.legend(fontsize=9.5)
save(fig, "fig09_rf민감도.png")

# 10 시간 외 검증
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (fn, gi, ru, ref) in zip(axs, (("OOT_곡선_2015.csv", "2016–2017 발급(T2)", "T2", "T2_OOT_2016-2017"), ("OOT_곡선_2016.csv", "2017–2020 발급(T1)", "T1", "T1_OOT_2017-2020"))):
    c = pd.read_csv(T / fn); c = c[c.구간 == gi]
    for k in ("V3", "B0", "GRADE"): s = c[c.strategy == k]; ax.plot(s.q * 100, s.SR, color=SC[k][0], lw=2.4 if k == "V3" else 1.5, label=SC[k][1])
    for _, r in PR[PR.잣대 == ref].iterrows():
        if pd.notna(r.승인율) and r.승인율 < 1: ax.scatter(r.승인율 * 100, r.값, s=70, facecolors="none", edgecolors=RED, lw=1.6, zorder=5); ax.annotate(r.항목.split("(")[0], (r.승인율 * 100, r.값), xytext=(5, 5), textcoords="offset points", color=RED, fontsize=9)
    ax.axhline(0, color=DARK, lw=0.8); ax.axvline(q * 100, color=NAVY, ls=":", lw=1.2); ax.set_xlim(0, 60); ax.set_xlabel("승인율 (%)"); ax.set_ylabel("기회풀 Sharpe" if ru == "T2" else "과제 정의 Sharpe"); ax.legend(fontsize=9.5, loc="lower right" if ru == "T2" else "lower left")
    ax.set_title("2015년까지 학습, 2016~2017년 발급분 채점" if ru == "T2" else "2016년까지 학습, 2017~2020년 발급분(결과 확정) 채점", fontsize=10.5, loc="left", color=DARK)
    if ru == "T1": ax.set_ylim(-0.2, 0.03)
save(fig, "fig10_시간외검증.png")

# 11 보정
CA = pd.read_csv(T / "보정_10분위.csv"); c = CA[CA.model == "3모형 평균"]; fig, ax = plt.subplots(figsize=(5.4, 4.2))
ax.plot([0, 50], [0, 50], color=GRAY, ls="--", lw=1); ax.plot(c.pred * 100, c.obs * 100, color=NAVY, marker="o", lw=2); ax.set_xlabel("예측 부도확률 (%, 10분위 평균)"); ax.set_ylabel("실제 부도율 (%)"); ax.set_xlim(0, 50); ax.set_ylim(0, 50)
save(fig, "fig11_보정.png")

# 12 등급 구성
GR = pd.read_csv(T / "등급구성_final.csv"); fig, ax = plt.subplots(figsize=(9, 3.8)); x = np.arange(len(GR))
ax.bar(x, GR.승인율 * 100, color="#DDEBF7", edgecolor=BLUE, label="등급 내 승인율(%)"); ax.plot(x, GR.등급부도율 * 100, color=GRAY, marker="s", lw=2, label="등급 전체 부도율(%)"); ax.plot(x, GR.승인부도율 * 100, color=NAVY, marker="o", lw=2, label="승인 대출 부도율(%)")
ax.set_xticks(x, GR.등급); ax.set_ylabel("%"); ax.legend(fontsize=9.5); save(fig, "fig12_등급구성.png")

# 13 CPR·CDR(131, 36개월)
CV = pd.read_csv(T / "채권_CPR.csv"); PSq = pd.read_csv(T / "채권_PSA.csv").set_index("집단"); fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
for g, col, lab in (("36_전체", DARK, "전체"), ("36_승인", NAVY, "최종 모형 승인"), ("36_거절", GRAY, "최종 모형 거절")):
    c = CV[CV.집단 == g]; lw = 2.4 if "승인" in g else 1.6
    a1.plot(c.나이[c.나이 < 36], c.CPR[c.나이 < 36] * 100, color=col, lw=lw, label=lab); a2.plot(c.나이, c.CDR * 100, color=col, lw=lw, label=lab)
age = np.arange(1, 36); psa = np.minimum(0.2 * age, 6.0); k = PSq.loc["36_전체", "PSA속도"]
a1.plot(age, psa, color=ORANGE, ls="--", lw=1.5, label="100% PSA(주택담보대출 기준)"); a1.plot(age, k * psa, color=ORANGE, ls=":", lw=1.5, label=f"{k * 100:.0f}% PSA(전체에 맞춤)")
a1.set_xlabel("대출 나이 (개월)"); a1.set_ylabel("CPR (연 조기상환율, %)"); a1.legend(fontsize=9); a2.set_xlabel("대출 나이 (개월)"); a2.set_ylabel("CDR (연 부도율, %)"); a2.legend(fontsize=9)
save(fig, "fig13_CPR.png")

# 14 스프레드 위험 사다리(131)
LD = pd.read_csv(T / "채권_사다리.csv"); fig, ax = plt.subplots(figsize=(10, 4.3))
for c, col, ls, lab in (("Aaa 회사채", GRAY, "-", "Aaa 회사채(약정)"), ("Baa 회사채", DARK, "-", "Baa 회사채(약정)"), ("30년 모기지", GREEN, "-", "30년 모기지(약정)"), ("24개월 개인대출", ORANGE, "-", "은행 24개월 개인대출(약정)"),
                        ("LC 약정 전체", BLUE, "--", "LendingClub 약정(전체)"), ("LC 실현 전체(36개월)", BLUE, "-", "LendingClub 실현(전체)"), ("LC 실현 승인(36개월)", NAVY, "-", "LendingClub 실현(최종 모형 승인)")):
    ax.plot(LD.발급연도, LD[c] * 100, color=col, ls=ls, marker="o", ms=4, lw=2.6 if "승인" in c else 1.6, label=lab)
ax.set_xlabel("발급연도"); ax.set_ylabel("국채 대비 스프레드 (%p)"); ax.set_ylim(0, 21.5); ax.set_yticks(range(0, 17, 2)); ax.legend(fontsize=8.8, ncol=3, loc="upper center"); save(fig, "fig14_스프레드사다리.png")

# 15 가격-수익률 곡선(131, 최종 모형 승인 풀)
PYC = pd.read_csv(T / "채권_가격곡선.csv"); fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, sec in zip(axs, ("자동이전증권", "IO", "PO")):
    for case, col, ls in (("추정 반응(β̂)", NAVY, "-"), ("고정(β=0)", GRAY, "--"), ("가상 MBS형(β=+|β̂|)", ORANGE, "-.")):
        c = PYC[PYC.경우 == case]; base = float(c[np.isclose(c.수익률변화, 0)][sec].iloc[0]); ax.plot(c.수익률변화 * 100, c[sec] / base * 100, color=col, ls=ls, lw=2, label=case.split("(")[0])   # 글꼴에 β 글자가 없어 범례는 이름만
    ax.text(0.5, 0.95, sec, transform=ax.transAxes, ha="center", va="top", fontsize=11, color=DARK); ax.set_xlabel("수익률 변화 (%p)"); ax.set_ylabel("가격 (기준 = 100)"); ax.legend(fontsize=8.5, loc="lower left")
save(fig, "fig15_가격곡선.png")
print("saved", sorted(p.name for p in F.glob("*.png")))
