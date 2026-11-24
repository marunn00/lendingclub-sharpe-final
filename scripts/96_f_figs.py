"""96_f: 그림 → 산출물/기술IR_v5/fig/*.png (AppleGothic, 큰 글씨). 입력은 96_c·96_d·96_e 가 만든 _work/*.csv(모두 새 테이블에서 계산)."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; C = importlib.import_module("96_common")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False, "font.size": 15, "axes.titlesize": 18, "axes.labelsize": 16, "legend.fontsize": 13, "figure.dpi": 100, "axes.grid": True, "grid.alpha": .25})
W, FIG = C.WORK, C.FIG
cur = pd.read_csv(W / "curves.csv"); summ = pd.read_csv(W / "strategy_summary.csv"); hur = pd.read_csv(W / "hurdle.csv"); al = pd.read_csv(W / "alpha.csv"); al["model"] = al.model.str.replace(chr(0x2212), "-")
mk = pd.read_csv(W / "market.csv"); mci = pd.read_csv(W / "market_ci.csv"); vc = pd.read_csv(W / "vintage_curves.csv"); imp = pd.read_csv(W / "importance.csv"); imp = imp[imp.split == "strict36"]
cal, roc, lift, auc = (pd.read_csv(W / f"{n}.csv") for n in ("calibration", "roc", "lift", "auc")); cal, roc, lift = (x[x.split == "strict36"] for x in (cal, roc, lift)); auc_all = auc; auc = auc[auc.split == "strict36"]; sg = pd.read_csv(W / "stress_grid.csv"); sb = pd.read_csv(W / "stress_breakeven.csv")
br = pd.read_csv(W / "bridge.csv"); mm = pd.read_csv(W / "model_metrics.csv")
LAB = {"전량(무작위)": "전량 승인(무작위)", "등급컷": "등급 컷(저금리부터)", "logit|raw141": "로지스틱 부도(raw141)", "clf|raw141": "LightGBM 부도(raw141)", "reg|raw141": "LightGBM NPV(raw141)",
       "clf|raw+nv전체": "LightGBM 부도(raw+nv)", "reg|raw+nv전체": "LightGBM NPV(raw+nv)", "오라클(실현NPV)": "오라클(상한)"}
COL = {"전량(무작위)": "#888", "등급컷": "#8c564b", "logit|raw141": "#bcbd22", "clf|raw141": "#d62728", "reg|raw141": "#1f77b4", "clf|raw+nv전체": "#ff9896", "reg|raw+nv전체": "#17becf", "오라클(실현NPV)": "#2ca02c"}
KEY = [k for k in LAB if k != "오라클(실현NPV)"]
STRICT = "학습규칙: strict36 = 36개월 발행≤2014 & 확정월<2015-01 (n=32,120) → 시험 2015–16 (n=364,683). Sharpe = 월 빈티지 초과수익(FEDFUNDS 대비, 거절 자본=FEDFUNDS)의 평균/표준편차, 연환산 아님"
NONE_ = "학습 없음(기술통계). 입력: train_새테이블.parquet"
RULE = {n: NONE_ for n in ("F07", "F08", "F12", "F14", "F16", "F17")}
RULE["F15"] = "학습규칙: 막대 색 주황·파랑 = strict36(누수 없음); 빗금 = primary36(미래정보 포함 상한) 또는 m60(학습 확정월 제한 불가, 누수 포함)"
RULE["F18"] = "primary36 = 미래정보 포함 상한(누수 시연; 학습 82.5%가 2015-01 이후 확정) vs strict36(누수 없음)"
def save(fig, nm):
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.text(0.5, 0.006, RULE.get(nm[:3], STRICT), ha="center", fontsize=11, color="#444")
    fig.savefig(FIG / nm, dpi=130); plt.close(fig); C.log("fig", nm)
H = cur[(cur.split == "strict36") & (cur.subset == "헤드라인2015-16") & (cur.rank_unit == "월")]
# F01 승인율 곡선: 초과수익·Sharpe
fig, ax = plt.subplots(1, 2, figsize=(22, 9))
for k in KEY:
    g = H[H.strategy == k]; ax[0].plot(g.a * 100, g.excess_pooled * 100, color=COL[k], label=LAB[k], lw=2.5)
    if k in ("reg|raw141", "clf|raw141", "전량(무작위)"): ax[0].fill_between(g.a * 100, g.excess_lo * 100, g.excess_hi * 100, color=COL[k], alpha=.15)
    ax[1].plot(g.a * 100, g.sharpe, color=COL[k], label=LAB[k], lw=2.5)
    if k in ("reg|raw141", "clf|raw141"): ax[1].fill_between(g.a * 100, g.sharpe_lo, g.sharpe_hi, color=COL[k], alpha=.15)
ax[0].set(title="승인율별 초과수익 (거절 자본=실현 FEDFUNDS, 시험 2015–16 36개월)", xlabel="승인율(금액, %)", ylabel="전체 자본 기준 초과수익 (%p/년)"); ax[0].legend()
ax[1].set(title="승인율별 월 빈티지 Sharpe (평균/표준편차, 24개월)", xlabel="승인율(금액, %)", ylabel="Sharpe(무차원)")
for x in (30, 53): ax[0].axvline(x, color="k", ls=":", lw=1); ax[1].axvline(x, color="k", ls=":", lw=1)
save(fig, "F01_승인율_곡선_초과수익_Sharpe.png")
# F02 프런티어
fig, ax = plt.subplots(figsize=(14, 10))
for k in KEY:
    g = H[H.strategy == k]; ax.plot(g.e_sd * 100, g.e_mean * 100, color=COL[k], label=LAB[k], lw=2.5)
    for a in (30, 53): ax.scatter(g.e_sd.iloc[a - 1] * 100, g.e_mean.iloc[a - 1] * 100, color=COL[k], s=120, marker="o" if a == 30 else "s", edgecolor="k", zorder=5)
ax.set(title="위험–수익 프런티어 (월 빈티지; ●30% ■53%)", xlabel="월 빈티지 초과수익 표준편차 (%p)", ylabel="월 빈티지 초과수익 평균 (%p/년)"); ax.legend(); save(fig, "F02_프런티어.png")
# F03 구방식(횡단) Sharpe
fig, ax = plt.subplots(1, 2, figsize=(22, 9))
for k in KEY:
    g = H[H.strategy == k]; ax[0].plot(g.a * 100, g.old_SR, color=COL[k], label=LAB[k], lw=2.5); ax[1].plot(g.a * 100, g.old_excess * 100, color=COL[k], label=LAB[k], lw=2.5)
ax[0].axvline(30.3, color="k", ls="--"); ax[0].text(30.8, ax[0].get_ylim()[1] * .92, "구 보고 Sharpe 최대 30.3%", fontsize=13)
ax[1].axvline(55.8, color="k", ls="--"); ax[1].text(56.3, ax[1].get_ylim()[0] * .5 if ax[1].get_ylim()[0] < 0 else 0, "구 보고 수익금 최대 55.8%", fontsize=13)
ax[0].set(title="구방식 Sharpe = 초과수익/횡단 sd (22_frontier 정의)", xlabel="승인율(%)", ylabel="Sharpe"); ax[1].set(title="구방식 초과수익(대출별 연환산 - 국채)", xlabel="승인율(%)", ylabel="%p/년"); ax[0].legend(); save(fig, "F03_구방식_Sharpe_곡선.png")
# F04 전략 비교 막대
fig, ax = plt.subplots(figsize=(18, 9)); s = summ[(summ.split == "strict36") & (summ.subset == "헤드라인2015-16") & (summ.rank_unit == "월")].set_index("strategy"); ks = KEY + ["오라클(실현NPV)"]; x = np.arange(len(ks))
for i, (c, lb) in enumerate([("R30", "승인 30%"), ("R53", "승인 53%"), ("R100", "승인 100%")]): ax.bar(x + (i - 1) * .27, s.loc[ks, c] * 100, .27, label=lb)
ax.set_xticks(x); ax.set_xticklabels([LAB[k] for k in ks], rotation=20, ha="right"); ax.set(title="전략별 초과수익 (전체 자본, %p/년; 시험 2015–16)", ylabel="%p/년"); ax.legend(); save(fig, "F04_전략비교_30_53_100.png")
# F05 알파 forest
a = al[(al.split == "strict36") & (al.kind == "level") & (al.model != "오라클")]; ap = al[(al.split == "primary36") & (al.kind == "level") & (al.subset == "헤드라인2015-16")].set_index("model")
fig, ax = plt.subplots(1, 2, figsize=(22, 10), sharey=True); ms = ["logit|raw141", "clf|raw141", "reg|raw141", "clf|raw+채택", "reg|raw+채택", "clf|raw+채택+FLAG", "reg|raw+채택+FLAG", "clf|raw+nv전체", "reg|raw+nv전체", "clf|nv만", "reg|nv만"]
for j, (cl, lo, hi, tt) in enumerate([("alpha_invested_pp", "inv_lo", "inv_hi", "알파(투입 자본당)"), ("alpha_total_pp", "tot_lo", "tot_hi", "알파(전체 자본당)")]):
    for off, sn, c in [(-.25, "2015", "#1f77b4"), (0, "2016", "#d62728"), (.25, "헤드라인2015-16", "#2ca02c")]:
        g = a[a.subset == sn].set_index("model").loc[ms]; y = np.arange(len(ms)) + off
        ax[j].errorbar(g[cl], y, xerr=[g[cl] - g[lo], g[hi] - g[cl]], fmt="o", color=c, label=sn, capsize=4, ms=7)
    ax[j].plot(ap.loc[ms, cl], np.arange(len(ms)) + .42, "D", mfc="none", mec="#555", ms=9, label="primary36(누수 상한, 헤드라인)")
    ax[j].set_title(tt + " — 셀 안 상위 30%, 6개월 블록 95% CI [strict36]"); ax[j].set_xlabel("%p/년"); ax[j].axvline(0, color="k")
ax[0].set_yticks(range(len(ms))); ax[0].set_yticklabels(ms); ax[0].legend(); ax[0].invert_yaxis(); save(fig, "F05_알파_CI.png")
# F06 새 vs raw 특성 비교 (쌍 차이)
p = al[(al.split == "strict36") & (al.kind == "pair_diff")]
fig, ax = plt.subplots(figsize=(18, 10)); pm = list(p.model.unique()); 
for off, sn, c in [(-.25, "2015", "#1f77b4"), (0, "2016", "#d62728"), (.25, "헤드라인2015-16", "#2ca02c")]:
    g = p[p.subset == sn].set_index("model").loc[pm]; y = np.arange(len(pm)) + off
    ax.errorbar(g.alpha_invested_pp, y, xerr=[g.alpha_invested_pp - g.inv_lo, g.inv_hi - g.alpha_invested_pp], fmt="o", color=c, label=sn, capsize=4, ms=8)
ax.set_yticks(range(len(pm))); ax.set_yticklabels(pm); ax.axvline(0, color="k"); ax.invert_yaxis(); ax.legend(); ax.set(title="특성집합·모형 쌍 차이: 알파(투입 자본당) %p/년 [6개월 블록 95% CI]", xlabel="앞 - 뒤 (%p/년)"); save(fig, "F06_새변수_vs_raw_알파차이.png")
# F07 빈티지 곡선
fig, ax = plt.subplots(2, 2, figsize=(22, 14))
for i, tm in enumerate((36, 60)):
    for y in sorted(vc[vc.term == tm].year.unique()):
        g = vc[(vc.term == tm) & (vc.year == y) & (vc.age <= tm + 6)]
        ax[0, i].plot(g.age, g.cum_net_cash_per_A, label=str(y), lw=2.5); ax[1, i].plot(g.age, g.cum_writeoff_per_A * 100, label=str(y), lw=2.5)
    ax[0, i].axhline(1, color="k", ls=":"); ax[0, i].set(title=f"{tm}개월 누적 순현금 / 원금 (1.0 = 원금 회수)", xlabel="나이(월)"); ax[1, i].set(title=f"{tm}개월 누적 상각(원금, 상각월=K+6) / 원금 (%)", xlabel="나이(월)"); ax[0, i].legend()
save(fig, "F07_빈티지_곡선.png")
# F08 등급×연도 히트맵(36m)
g36 = mk[(mk.term == 36) & mk.grade.isin(list("ABCDEFG"))]; fig, ax = plt.subplots(1, 3, figsize=(26, 9))
for j, (c, tt, f, cm) in enumerate([("default_rate", "부도율(확정 대출, %)", 100, "Reds"), ("R_reinv", "R_reinv 초과수익(%p/년)", 100, "RdYlGn"), ("E_npv", "원금당 초과 NPV (%)", 100, "RdYlGn")]):
    pv = g36.pivot(index="grade", columns="year", values=c) * f; im = ax[j].imshow(pv.values, cmap=cm, aspect="auto"); ax[j].set_xticks(range(pv.shape[1])); ax[j].set_xticklabels(pv.columns); ax[j].set_yticks(range(7)); ax[j].set_yticklabels(pv.index); ax[j].set_title(tt); ax[j].grid(False)
    for (i, k), v in np.ndenumerate(pv.values): ax[j].text(k, i, f"{v:.1f}", ha="center", va="center", fontsize=14)
save(fig, "F08_등급x연도_부도_수익.png")
# F09 보정/ROC/리프트
fig, ax = plt.subplots(1, 3, figsize=(26, 9)); ms = ["clf|raw141", "clf|raw+nv전체", "logit|raw141"]
for k in ms:
    g = cal[cal.model == k]; ax[0].plot(g.mean_pred, g.obs_rate, "o-", color=COL[k], label=LAB[k], lw=2.5, ms=9)
    r = roc[roc.model == k]; ax[1].plot(r.fpr, r.tpr, color=COL[k], lw=2.5, label=f"{LAB[k]} AUC {auc[auc.model == k].auc.iloc[0]:.3f}")
    l = lift[lift.model == k]; ax[2].plot(l.decile, l.lift, "o-", color=COL[k], lw=2.5, label=LAB[k])
ax[0].plot([0, .5], [0, .5], "k--"); ax[0].set(title="보정(예측 vs 실현, 10분위; 시험 2015–16)", xlabel="평균 예측 부도확률", ylabel="실현 부도율"); ax[0].legend()
ax[1].plot([0, 1], [0, 1], "k--"); ax[1].set(title="ROC", xlabel="FPR", ylabel="TPR"); ax[1].legend(); ax[2].axhline(1, color="k", ls="--"); ax[2].set(title="리프트(예측 위험 10분위, 10=최고위험)", xlabel="분위", ylabel="리프트"); ax[2].legend(); save(fig, "F09_보정_ROC_리프트.png")
# F10 중요도
fig, ax = plt.subplots(1, 2, figsize=(24, 11))
for j, nm in enumerate(["clf", "reg"]):
    g = imp[(imp.model == nm) & (imp.fs == "raw+nv전체")].nlargest(20, "mean_abs_shap").iloc[::-1]
    ax[j].barh(g.feature, g.mean_abs_shap, color=["#ff7f0e" if f.startswith("nv_") else "#1f77b4" for f in g.feature]); ax[j].set_title(("부도 분류" if nm == "clf" else "NPV 회귀") + " mean|SHAP| 상위 20 (주황=nv_ 새 변수)")
save(fig, "F10_SHAP_중요도.png")
# F11 허들
hh = hur[(hur.split == "strict36") & (hur.subset == "헤드라인2015-16") & (hur.rank_unit == "월")]; fig, ax = plt.subplots(1, 2, figsize=(22, 9))
for k in KEY + ["오라클(실현NPV)"]:
    g = hh[hh.strategy == k]; ax[0].plot(g.h_pp, g.best_a * 100, "o-", color=COL[k], label=LAB[k], lw=2.5); ax[1].plot(g.h_pp, g.best_profit_pct, "o-", color=COL[k], lw=2.5)
ax[0].set(title="허들(FEDFUNDS+h)별 이익 최대 승인율", xlabel="h (%p)", ylabel="승인율(%)"); ax[0].legend(); ax[1].set(title="허들별 최대 이익 (원금당 %, 3년 누적, 전체 자본)", xlabel="h (%p)", ylabel="%"); save(fig, "F11_허들_민감도.png")
# F12 스트레스
fig, ax = plt.subplots(1, 3, figsize=(27, 9)); s0 = sg[(sg.group == "전체 2012–16")]
pv = s0.pivot(index="pd_mult", columns="lgd_mult", values="excess_pct"); im = ax[0].imshow(pv.values, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10); ax[0].grid(False)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(pv.columns); ax[0].set_yticks(range(len(pv))); ax[0].set_yticklabels(pv.index); ax[0].set(title="초과 NPV(%) 스트레스: PD배수 × LGD배수 (36m 전체)", xlabel="LGD 배수", ylabel="PD 배수")
for (i, k), v in np.ndenumerate(pv.values): ax[0].text(k, i, f"{v:.1f}", ha="center", va="center", fontsize=15)
b = sb[sb.group.str.startswith("등급")]; ax[1].bar(b.group, b.breakeven_pd_mult); ax[1].axhline(1, color="k", ls="--"); ax[1].set(title="손익분기 PD 배수(등급별, <1이면 이미 손실)")
b2 = sb[sb.group.isin(["2012", "2013", "2014", "2015", "2016", "전체 2012–16"])]; ax[2].bar(b2.group, b2.breakeven_pd_mult, color="#ff7f0e"); ax[2].axhline(1, color="k", ls="--"); ax[2].set(title="손익분기 PD 배수(연도별)"); save(fig, "F12_스트레스_격자.png")
# F13 30 vs 53
fig, ax = plt.subplots(1, 2, figsize=(22, 9)); 
for j, sn in enumerate(["헤드라인2015-16", "2016"]):
    s = summ[(summ.split == "strict36") & (summ.subset == sn) & (summ.rank_unit == "월")].set_index("strategy").loc[KEY]; y = np.arange(len(KEY))
    ax[j].errorbar(s.diff30_53 * 100, y, xerr=[(s.diff30_53 - s.diff30_53_lo) * 100, (s.diff30_53_hi - s.diff30_53) * 100], fmt="o", capsize=5, ms=9, color="#d62728")
    ax[j].set_yticks(y); ax[j].set_yticklabels([LAB[k] for k in KEY]); ax[j].axvline(0, color="k"); ax[j].set_title(f"초과수익 차 (승인 30% - 53%), {sn} [6개월 블록 95% CI]"); ax[j].set_xlabel("%p/년"); ax[j].invert_yaxis()
save(fig, "F13_30대53_차이.png")
# F14 다리표 + D–G vs A–C + 2016 등급
fig, ax = plt.subplots(1, 3, figsize=(27, 9)); xx = np.arange(3); cols = ["old_style_0p99TP_locked_tsy", "same_assump_cash_locked_tsy", "R_noreinv", "R_reinv"]; lbs = ["① 구 합산(0.99·TP, 국채 고정)", "② 같은 가정 재계산", "③ 기준=실현 FEDFUNDS", "④ +재투자(R_reinv)"]
for i, (c, l) in enumerate(zip(cols, lbs)): ax[0].bar(xx + (i - 1.5) * .2, br[c] * 100, .2, label=l)
ax[0].set_xticks(xx); ax[0].set_xticklabels(br.year); ax[0].set(title="다리표: 36개월 초과수익 %p/년", ylabel="%p/년"); ax[0].legend(fontsize=11)
for gname, c in [("A-C", "#1f77b4"), ("D-G", "#d62728")]:
    g = mk[(mk.term == 36) & (mk.grade == gname)]; ax[1].plot(g.year, g.R_reinv * 100, "o-", color=c, label=gname, lw=3, ms=10)
ax[1].axhline(0, color="k"); ax[1].set(title="D–G vs A–C (36m R_reinv, %p/년)"); ax[1].legend()
g = mk[(mk.term == 36) & (mk.year == 2016) & mk.grade.isin(list("ABCDEFG"))]; ax[2].bar(g.grade, g.R_reinv * 100, color=["#2ca02c" if v > 0 else "#d62728" for v in g.R_reinv]); ax[2].set(title="2016 발행 등급별 R_reinv (%p/년)"); save(fig, "F14_다리표_DG_AC_2016등급.png")
# F15 AUC 비교(특성집합)
m = mm[~mm.model.isin(["lgb_reg"])].copy(); m = pd.concat([m[m.split == "strict36"], m[m.split != "strict36"]]); m["lab"] = m.split + "\n" + m.fs + "\n" + m.model; fig, ax = plt.subplots(figsize=(26, 9)); ax.bar(range(len(m)), m.auc, color=["#1f77b4" if "nv" not in f else "#ff7f0e" for f in m.fs], hatch=None)
for b_, sp_ in zip(ax.patches, m.split):
    if sp_ != "strict36": b_.set_hatch("//"); b_.set_alpha(.6)
ax.set_ylim(.55, .72); ax.set_xticks(range(len(m))); ax.set_xticklabels(m.lab, fontsize=9); ax.set(title="시험 AUC: 분할×특성집합×모형 (주황=nv_ 포함)", ylabel="AUC"); save(fig, "F15_AUC_분할x특성집합.png")
# F16 시장 기준선 연도별 + CI
fig, ax = plt.subplots(1, 2, figsize=(22, 9)); t = mk[(mk.grade == "전체")].merge(mci, left_on=["term", "year"], right_on=["term", "year"])
for tm, c in ((36, "#1f77b4"), (60, "#d62728")):
    g = t[t.term == tm]; ax[0].errorbar(g.year, g.R_reinv * 100, yerr=[(g.R_reinv - g.R_reinv_lo) * 100, (g.R_reinv_hi - g.R_reinv) * 100], fmt="o-", color=c, capsize=5, label=f"{tm}개월 R_reinv", lw=2.5); ax[0].plot(g.year, g.R_noreinv * 100, "s--", color=c, label=f"{tm}개월 R_noreinv", alpha=.6)
    ax[1].errorbar(g.year, g.E_npv * 100, yerr=[(g.E_npv - g.E_npv_lo) * 100, (g.E_npv_hi - g.E_npv) * 100], fmt="o-", color=c, capsize=5, label=f"{tm}개월", lw=2.5)
ax[0].set(title="전량 매수 초과수익 (%p/년, 3개월 블록 95% CI)", xlabel="발행연도"); ax[0].legend(); ax[1].set(title="원금당 초과 NPV (%)", xlabel="발행연도"); ax[1].legend(); save(fig, "F16_시장기준선_연도별.png")
C.log("done")

# F17 열 분류 (fig01 대응): 원열 141 → 상장시점 특성 / 제외, nv_ 56 상태별
import polars as pl
sch = list(pl.read_parquet_schema(C.TABLE)); F_ = pd.read_parquet(W / "feats.parquet", columns=None) if False else None
meta_ = __import__("json").loads((OUT_ := C.OUT / "_work" / "prep.meta.json").read_text()); post = set(meta_["excluded_post"])
rawn = [c for c in sch[:141]]; used = sum(1 for c in rawn if c not in post); dic = pd.read_csv(C.ROOT / "산출물" / "최종테이블" / "새변수_사전.csv"); st = dic.status.value_counts()
fig, ax = plt.subplots(1, 2, figsize=(20, 9)); ax[0].bar(["상장시점 후보 열", "제외(사후·식별·텍스트)"], [used, 141 - used], color=["#1f77b4", "#d62728"]); ax[0].set_title("원열 141개 분류(96 규칙)")
for i, v in enumerate([used, 141 - used]): ax[0].text(i, v + 1, str(v), ha="center", fontsize=16)
ax[1].bar(st.index, st.values, color="#ff7f0e"); ax[1].set_title("nv_ 새 변수 56개 상태(새변수_사전.csv)")
for i, v in enumerate(st.values): ax[1].text(i, v + .5, str(v), ha="center", fontsize=16)
save(fig, "F17_열_분류.png")
# F18 누수 시연: primary36 vs strict36
ah = al[(al.kind == "level") & (al.subset == "헤드라인2015-16") & al.split.isin(["strict36", "primary36"])]
mds = ["clf|raw141", "reg|raw141", "clf|raw+nv전체", "reg|raw+nv전체"]; fig, ax = plt.subplots(1, 2, figsize=(22, 9)); x = np.arange(len(mds))
for off, sp_, c, lb in [(-.2, "strict36", "#1f77b4", "strict36(누수 없음)"), (.2, "primary36", "#d62728", "primary36(미래정보 포함 상한)")]:
    g = ah[ah.split == sp_].set_index("model").loc[mds]; ax[0].bar(x + off, g.alpha_invested_pp, .4, color=c, label=lb, yerr=[g.alpha_invested_pp - g.inv_lo, g.inv_hi - g.alpha_invested_pp], capsize=5)
    g2 = auc_all[(auc_all.split == sp_)].set_index("model").loc[[m_ for m_ in mds if m_.startswith("clf")]]; ax[1].bar(np.arange(len(g2)) + off, g2.auc, .4, color=c, label=lb, yerr=[g2.auc - g2.lo, g2.hi - g2.auc], capsize=5)
ax[0].set_xticks(x); ax[0].set_xticklabels(mds, rotation=15); ax[0].set(title="알파(투입 자본당, %p/년) 헤드라인 2015–16 [95% CI]"); ax[0].legend()
ax[1].set_xticks(range(2)); ax[1].set_xticklabels(["clf|raw141", "clf|raw+nv전체"]); ax[1].set_ylim(.55, .75); ax[1].set(title="부도 AUC(시험 2015–16) [95% CI]"); save(fig, "F18_누수시연_primary_vs_strict.png")
# F19 순위 단위 비교(월 vs 셀) + 선택 구성
fig, ax = plt.subplots(1, 3, figsize=(27, 9)); cs = cur[(cur.split == "strict36") & (cur.subset == "헤드라인2015-16")]
for k, c in (("clf|raw+nv전체", "#d62728"), ("reg|raw+nv전체", "#1f77b4")):
    for u, ls in (("월", "-"), ("셀", "--")):
        g = cs[(cs.strategy == k) & (cs.rank_unit == u)]; ax[0].plot(g.a * 100, g.sharpe, ls, color=c, lw=2.5, label=f"{LAB[k]} / {u} 안 순위"); ax[1].plot(g.a * 100, g.excess_pooled * 100, ls, color=c, lw=2.5, label=f"{LAB[k]} / {u} 안 순위")
ax[0].set(title="월 빈티지 Sharpe: 발행월 안 순위(실선) vs 월×세부등급 셀 안 순위(점선)", xlabel="승인율(%)", ylabel="Sharpe"); ax[0].legend(fontsize=11); ax[1].set(title="초과수익(%p/년)", xlabel="승인율(%)")
cp = pd.read_csv(W / "selection_composition.csv"); cp = cp[(cp.split == "strict36") & cp.strategy.isin(["clf|raw+nv전체", "reg|raw+nv전체"])]; lab = [f"{r.strategy.split('|')[0]}/{r.rank_unit}" for r in cp.itertuples()]
xx = np.arange(len(cp)); ax[2].bar(xx - .27, cp.int_rate_mean_pct, .27, label="평균 금리(%)"); ax[2].bar(xx, cp.default_rate * 100, .27, label="부도율(%)"); ax[2].bar(xx + .27, cp.share_CG_amt * 100, .27, label="C–G 금액 비중(%)")
ax[2].set_xticks(xx); ax[2].set_xticklabels(lab); ax[2].set_title("승인 30% 선택 구성"); ax[2].legend(); save(fig, "F19_순위단위_월vs셀_Sharpe.png")
C.log("done2")
