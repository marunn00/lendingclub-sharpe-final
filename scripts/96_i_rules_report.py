"""96_i: 규칙 기준선 결과 → 그림 F20, 비교표 행 추가, 분석_결과.md §11 추가(재실행 시 교체). 입력: 96_h 결과(_work/rules_*.csv)."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; C = importlib.import_module("96_common")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False, "font.size": 15, "axes.titlesize": 18, "axes.labelsize": 16, "legend.fontsize": 13, "axes.grid": True, "grid.alpha": .25})
W = C.WORK; O = W.parent; T = pd.read_csv(W / "rules_summary.csv"); P = pd.read_csv(W / "rules_pairs.csv")
FOOT = "학습규칙: strict36 (36개월 발행≤2014 & 확정월<2015-01, n=32,120) → 시험 2015–16. 모형 = LightGBM raw141. 차이 = 모형 - 규칙, 6개월 블록 부트스트랩 95% CI. Sharpe = 월 빈티지 초과수익(FEDFUNDS 대비) 평균/표준편차, 연환산 아님"
sel = P[P.비교.isin(["초과수익 기준 최선 규칙", "규칙 자체 승인율"]) & ~P.승인율.str.contains("전량")].copy()
sel["lab"] = sel.승인율 + "\n(vs " + sel.최선_규칙.str.replace("규칙|", "", regex=False) + ")"
sel["lab"] = sel.lab.where(sel.비교 != "규칙 자체 승인율", sel.승인율)
fig, ax = plt.subplots(1, 2, figsize=(26, 10)); labs = list(dict.fromkeys(sel.lab)); x = np.arange(len(labs))
for j, (c, lo, hi, tt) in enumerate([("차_초과수익_pp", "lo_pp", "hi_pp", "초과수익 차이: 모형 - 최선 규칙 (%p/년)"), ("차_Sharpe", "Sharpe_lo", "Sharpe_hi", "월 빈티지 Sharpe 차이: 모형 - 최선 규칙")]):
    for off, mk, col in [(-.2, "모형|부도확률(raw141)", "#d62728"), (.2, "모형|NPV(raw141)", "#1f77b4")]:
        g = sel[sel.모형 == mk].set_index("lab").loc[labs]
        ax[j].bar(x + off, g[c], .4, color=col, label=mk.replace("모형|", ""), yerr=[g[c] - g[lo], g[hi] - g[c]], capsize=5)
    ax[j].axhline(0, color="k", lw=1.2); ax[j].set_xticks(x); ax[j].set_xticklabels(labs, fontsize=12); ax[j].set_title(tt); ax[j].legend()
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.text(0.5, 0.008, FOOT, ha="center", fontsize=11, color="#444"); fig.savefig(O / "fig" / "F20_모형_vs_규칙기준선.png", dpi=130); plt.close(fig)
# 비교표 행
cmp_ = pd.read_csv(O / "표_비교_이전vs새테이블.csv", encoding="utf-8-sig"); cmp_ = cmp_[~cmp_.분석.str.startswith("규칙 기준선")]
R = []; rule = "strict36(학습 발행≤2014 & 확정월<2015-01)"
for r in T.itertuples():
    R.append({"분석": "규칙 기준선 vs 모형: 초과수익(FEDFUNDS 대비, 헤드라인 2015–16, 월 안 순위)", "지표": f"{r.전략} / 승인율 {r.비교_승인율}", "이전값": np.nan, "이전_출처": "이전 보고 없음(신규 Step A)", "새테이블값": round(r.초과수익_pp, 3), "단위": "%p/년", "학습규칙": rule if r.구분 == "모형" else "학습 없음(규칙)", "새값_CI": f"[{r.초과수익_lo_pp:.2f}, {r.초과수익_hi_pp:.2f}]", "차이_이유": "신규 산출"})
    R.append({"분석": "규칙 기준선 vs 모형: 월 빈티지 Sharpe(mean/sd, 연환산 아님)", "지표": f"{r.전략} / 승인율 {r.비교_승인율}", "이전값": np.nan, "이전_출처": "이전 보고 없음(신규 Step A)", "새테이블값": round(r.Sharpe, 2), "단위": "무차원", "학습규칙": rule if r.구분 == "모형" else "학습 없음(규칙)", "새값_CI": f"[{r.Sharpe_lo:.2f}, {r.Sharpe_hi:.2f}]", "차이_이유": "신규 산출"})
for r in P.itertuples():
    R.append({"분석": f"규칙 기준선 vs 모형: 쌍 차이 모형-규칙 ({r.비교})", "지표": f"{r.모형} - {r.최선_규칙} / {r.승인율}", "이전값": np.nan, "이전_출처": "이전 보고 없음(신규 Step A)", "새테이블값": f"초과수익 {r.차_초과수익_pp:+.3f}%p; Sharpe {r.차_Sharpe:+.2f}", "단위": "%p/년; 무차원", "학습규칙": rule, "새값_CI": f"초과수익 [{r.lo_pp:.2f}, {r.hi_pp:.2f}]; Sharpe [{r.Sharpe_lo:.2f}, {r.Sharpe_hi:.2f}]", "차이_이유": "신규 산출"})
pd.concat([cmp_, pd.DataFrame(R)]).to_csv(O / "표_비교_이전vs새테이블.csv", index=False, encoding="utf-8-sig")
# md
md = (O / "분석_결과.md").read_text(encoding="utf-8"); md = md.split("\n## 11. ")[0].rstrip() + "\n"
g = lambda m, a: P[(P.비교 == "초과수익 기준 최선 규칙") & (P.모형 == m) & (P.승인율 == a)].iloc[0]
def f(r): return f"{r.차_초과수익_pp:+.3f}%p [{r.lo_pp:.3f}, {r.hi_pp:.3f}], Sharpe 차 {r.차_Sharpe:+.2f} [{r.Sharpe_lo:.2f}, {r.Sharpe_hi:.2f}]"
tab = "| 규칙/모형 | 승인율 | 초과수익 %p/년 [95% CI] | Sharpe [95% CI] |\n|---|---|---|---|\n" + "\n".join(f"| {r.전략} | {r.비교_승인율} | {r.초과수익_pp:.3f} [{r.초과수익_lo_pp:.3f}, {r.초과수익_hi_pp:.3f}] | {r.Sharpe:.2f} [{r.Sharpe_lo:.2f}, {r.Sharpe_hi:.2f}] |" for r in T.itertuples())
pt = "| 비교 | 승인율 | 모형 | 최선 규칙 | 초과수익 차 %p [CI] | Sharpe 차 [CI] |\n|---|---|---|---|---|---|\n" + "\n".join(f"| {r.비교} | {r.승인율} | {r.모형.replace('모형|','')} | {r.최선_규칙.replace('규칙|','')} | {r.차_초과수익_pp:+.3f} [{r.lo_pp:.3f}, {r.hi_pp:.3f}] | {r.차_Sharpe:+.2f} [{r.Sharpe_lo:.2f}, {r.Sharpe_hi:.2f}] |" for r in P.itertuples())
md += f"""
## 11. 규칙 기준선 vs 모형 (Step A, strict36, 시험 2015–16 36개월, 포트폴리오 수준)

**방법.** 같은 엔진(strict36 점수, 월 빈티지, 거절 자본 = FEDFUNDS, 6개월 순환 블록 부트스트랩 2,000회 seed 0)으로 규칙과 모형(LightGBM raw141)을 비교합니다. 규칙: 전량, 등급컷 A–B/A–C/A–D, 금리 낮은 순·높은 순(발행월 안 순위), 세부등급만(낮은 위험 순). 같은 셀 상위 30% 알파는 셀 상수 규칙에서 정의되지 않으므로 승인율을 맞춘 포트폴리오 비교(30%, 53%, 규칙 자체 승인율 = 시험 기간 금액 비중을 1% 격자로 반올림)로 대체했습니다. 모형과 규칙은 같은 재표집 인덱스를 써서 쌍 차이를 냅니다. **Sharpe = 월 빈티지 초과수익(FEDFUNDS 대비)의 평균/표준편차, 연환산 아님.** 구방식 횡단 Sharpe(DGS3/DGS5)는 쓰지 않았습니다.

{tab}

쌍 차이(모형 - 규칙):

{pt}

**결론 (시험 기간 기준).**
- 승인율 30%·53%에서 두 모형 모두 최선 규칙보다 초과수익이 **약간** 큽니다: 30%에서 부도 모형 {f(g('모형|부도확률(raw141)','30%'))}, NPV 모형 {f(g('모형|NPV(raw141)','30%'))}. 53%에서 부도 모형 {f(g('모형|부도확률(raw141)','53%'))}, NPV 모형 {f(g('모형|NPV(raw141)','53%'))}. 초과수익 차이의 CI는 0을 제외하지만 크기는 0.04~0.07%p/년으로 작습니다(규칙 자체의 초과수익 0.47%·0.85%의 약 8%).
- **Sharpe 기준으로는 모형 우위가 입증되지 않습니다.** 부도 모형은 차이의 CI가 0을 포함합니다(30%: +1.13 [-3.16, 2.78]). NPV 모형은 발행월 안 순위에서 30%·53% 모두 규칙보다 **낮습니다**(-1.92 [-7.76, -0.91], -0.61 [-3.63, -0.07]). 이 열세는 순위 단위에 의존합니다. 월×세부등급 셀 안 순위(§3)에서는 승인 30% Sharpe가 NPV raw141 6.64, NPV raw+nv 7.47로 세부등급 규칙 6.88과 비슷하거나 높습니다(CI는 크게 겹침). 최선 규칙은 시험 기간의 점추정으로 사후에 골랐으므로 규칙에 유리한(모형에 보수적인) 비교입니다.
- 규칙 자체 승인율(A–B 58%, A–C 84%, A–D 96%)에서는 부도 모형 차이의 CI가 모두 0을 포함하고, NPV 모형은 A–D에서 규칙보다 나쁩니다(-0.020%p [-0.032, -0.009]). 이 비교에서 모형은 매월 같은 전체 승인율(시험 기간 금액 비중을 1% 격자로 반올림)로 고르고 등급 규칙은 월마다 승인율이 달라 노출이 다릅니다.
- 금리 높은 순 규칙은 낮은 순보다 초과수익·Sharpe 모두 낮습니다(30%: 0.27 vs 0.47%p). 금리 낮은 순·세부등급만 규칙은 사실상 같은 선택입니다.
- 정리: 시험 기간에서 모형의 우위는 초과수익에서 통계적으로 존재하나 미미하고, 위험 조정(Sharpe)에서는 입증되지 않습니다. 시험 월 24개(독립 블록 약 4개)라 CI는 낙관적입니다. 위 차이는 학습 n=32,120(strict36) 모형에 한정됩니다.

그림: `fig/F20_모형_vs_규칙기준선.png`. 표: `표_규칙기준선.csv`, `표_규칙기준선_쌍차이.csv`. 스크립트: `scripts/96_h_rules.py`, `96_i_rules_report.py`.
"""
(O / "분석_결과.md").write_text(md, encoding="utf-8"); C.log("done", len(R))
