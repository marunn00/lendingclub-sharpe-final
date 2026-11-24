"""96_j: 덱 전용 그림 재작성 → 산출물/기술IR_v5/fig_deck/*.png. 슬라이드 폭(1160px)에 1:1로 놓이도록 11.6×4.7in·dpi200, 글자 ≥11.5pt, 패널 ≤2.
학습 규칙 문구는 그림 안에 넣지 않고 슬라이드 하단 줄(97_ir_v5.py)에 둠. 입력: 96_c~e 가 만든 _work/*.csv, 전처리_v5/부속/*.csv (모두 기존 계산 결과; 새 계산 없음)."""
import sys, json; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; C = importlib.import_module("96_common")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False, "font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12.5, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
                     "legend.fontsize": 11, "axes.grid": True, "grid.alpha": .25, "lines.linewidth": 2.4, "figure.dpi": 100})
W = C.WORK; FD = C.OUT / "fig_deck"; FD.mkdir(exist_ok=True); PPV = C.ROOT / "산출물" / "전처리_v5"; PB = PPV / "부속"
SZ = (11.6, 5.0)
def new(n=1, **kw): return plt.subplots(1, n, figsize=SZ, **kw)
def save(fig, nm, bottom=0.0):
    fig.tight_layout(rect=(0, bottom, 1, 1)); fig.savefig(FD / nm, dpi=200); plt.close(fig); C.log("fig", nm)
def legend_below(fig, ax, ncol=4): h, l = ax.get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False, fontsize=11)
cur = pd.read_csv(W / "curves.csv"); summ = pd.read_csv(W / "strategy_summary.csv"); hur = pd.read_csv(W / "hurdle.csv"); al = pd.read_csv(W / "alpha.csv"); al["model"] = al.model.str.replace(chr(0x2212), "-")
mk = pd.read_csv(W / "market.csv"); vc = pd.read_csv(W / "vintage_curves.csv"); imp = pd.read_csv(W / "importance.csv"); imp = imp[imp.split == "strict36"]
cal, roc, auc = (pd.read_csv(W / f"{n}.csv") for n in ("calibration", "roc", "auc")); cal, roc = (x[x.split == "strict36"] for x in (cal, roc)); auc_all = auc; auc = auc[auc.split == "strict36"]
sg = pd.read_csv(W / "stress_grid.csv"); sb = pd.read_csv(W / "stress_breakeven.csv"); mm = pd.read_csv(W / "model_metrics.csv"); RP = pd.read_csv(W / "rules_pairs.csv")
LAB = {"전량(무작위)": "전량 승인", "등급컷": "등급 컷", "logit|raw141": "로지스틱(부도)", "clf|raw141": "LightGBM 부도", "reg|raw141": "LightGBM NPV", "clf|raw+nv전체": "부도(+nv_ 56개)", "reg|raw+nv전체": "NPV(+nv_ 56개)", "오라클(실현NPV)": "오라클(상한)"}
COL = {"전량(무작위)": "#888", "등급컷": "#8c564b", "logit|raw141": "#bcbd22", "clf|raw141": "#d62728", "reg|raw141": "#1f77b4", "clf|raw+nv전체": "#ff9896", "reg|raw+nv전체": "#17becf", "오라클(실현NPV)": "#2ca02c"}
KEY = [k for k in LAB if k != "오라클(실현NPV)"]
H = cur[(cur.split == "strict36") & (cur.subset == "헤드라인2015-16") & (cur.rank_unit == "월")]
ALAB = {"A_내생보존": "A 내생 변수 보존", "B_내생제거": "B 내생 변수 제거"}
# ---------------- 전처리 5장
cfg = json.loads((PB / "run_config.json").read_text()); sp = pd.read_csv(PB / "분할_표본.csv")
ymf = lambda s: int(s.split("-")[0]) + (int(s.split("-")[1]) - 1) / 12
vs_, ts_, te_ = ymf(cfg["valid_start"]), ymf(cfg["test_start"]), ymf(cfg["test_end"]) + 1 / 12; t2s, t2e = ymf(cfg["t2_start"]), ymf(cfg["t2_end"]) + 1 / 12
fig, ax = new(2, gridspec_kw={"width_ratios": [1.25, 1]}); bars = [("학습", 2007.4, vs_, "tab:blue", cfg["window_sizes"]["train"]), ("검증", vs_, ts_, "tab:orange", cfg["window_sizes"]["valid"]),
    ("시험(헤드라인)", ts_, te_, "tab:green", cfg["window_sizes"]["test"]), ("보조 시험", t2s, t2e, "tab:red", cfg["window_sizes"]["test2"])]
for i, (lb, a_, b_, c_, n_) in enumerate(bars): ax[0].barh(i, b_ - a_, left=a_, color=c_, alpha=.85, height=.55); ax[0].text(a_ + .1, i - .42, f"{lb} n={n_:,}", fontsize=11.5, va="bottom")
for x_ in (vs_, ts_): ax[0].axvline(x_, color="k", ls="--", lw=1.4)
ax[0].set_yticks([]); ax[0].set_ylim(len(bars) - .3, -1); ax[0].set(xlabel="발행 시점(연)", title="분할 설계 (점선 = 해결 상한)")
nm_ = ["학습", "검증", "시험", "보조"][:len(sp)]; x = np.arange(len(nm_)); ax[1].bar(x - .2, sp.bad_rate, .4, label="창 부실률"); ax[1].bar(x + .2, sp.ref_bad_rate, .4, label="같은 발행월 해결 대출 전체")
for xi, (a_, b_, m1, m2) in enumerate(zip(sp.bad_rate, sp.ref_bad_rate, sp.mean_months_to_resolve, sp.ref_mean_months)): ax[1].text(xi, max(a_, b_) + .005, f"해결 {m1:.0f}개월\n(비교 {m2:.0f})", ha="center", fontsize=10.5)
ax[1].set_xticks(x); ax[1].set_xticklabels(nm_); ax[1].set(ylabel="부실률", title="일찍 끝난 행만 남는 정도 (생존 편향)", ylim=(0, .26)); ax[1].legend(loc="upper left", fontsize=10.5); save(fig, "P08_분할_설계.png")
cv = pd.read_csv(PB / "감소곡선.csv"); fig, ax = new(2)
for (v, ph), g in cv.groupby(["variant", "phase"], sort=False):
    g = g.sort_values("n_cols")
    for a_, k in zip(ax, ("auc", "logloss")):
        a_.plot(g.n_cols, g[k], "o-", label=ALAB.get(v, v), ms=8); bad = g[~g.accepted.astype(bool)]; a_.plot(bad.n_cols, bad[k], "rx", ms=13, mew=3)
for a_, t in zip(ax, ("검증 AUC", "검증 logloss")): a_.invert_xaxis(); a_.set(xlabel="남은 열 수 (오른쪽 → 왼쪽으로 축소)", ylabel=t, title=f"{t} (빨강 x = 중단)"); a_.legend()
save(fig, "P02_감소곡선.png")
fr = pd.read_csv(PB / "프론티어_측정.csv"); fig, ax = new(2); sh = {"기준": "기준(S6 직후)", "PSI/CSI 안정성": "PSI 안정성", "적대적 검증": "적대적 검증", "방향 일관성": "방향 일관성(최종)", "단조 제약": "단조 제약", "WoE/IV 스코어카드": "WoE 스코어카드"}
for a_, v in zip(ax, ("A_내생보존", "B_내생제거")):
    g = fr[(fr.variant == v) & fr.principle.isin(sh)].copy(); g = g[(g.principle != "WoE/IV 스코어카드") | g.config.str.contains("WoE 변환")]; g = g[(g.principle != "방향 일관성") | g.config.str.contains("반전")]
    lab = [f"{sh[p]} ({n}열)" for p, n in zip(g.principle, g.n_cols)]; y = np.arange(len(g)); a_.barh(y, g.te_auc, color=["#888"] + ["#1f77b4"] * (len(g) - 1), height=.6)
    a_.errorbar(g.te_auc, y, xerr=[g.te_auc - g.te_auc_lo, g.te_auc_hi - g.te_auc], fmt="none", ecolor="k", capsize=4)
    for yi, vv, hi_ in zip(y, g.te_auc, g.te_auc_hi): a_.text(hi_ + .002, yi, f"{vv:.4f}", va="center", fontsize=10.5)
    a_.set_yticks(y); a_.set_yticklabels(lab); a_.invert_yaxis(); a_.set_xlim(.57, .665); a_.set(title=f"{ALAB[v]}: 시험 AUC (95% 구간)", xlabel="AUC")
save(fig, "P05_원리별_효과.png")
adv = pd.read_csv(PB / "적대적_곡선.csv"); fig, ax = new()
for v, g in adv.groupby("variant", sort=False):
    ax.plot(g.n_cols, g.adv_auc, "o-", label=ALAB.get(v, v), ms=8); ax.annotate(f"{g.adv_auc.iloc[0]:.3f}", (g.n_cols.iloc[0], g.adv_auc.iloc[0]), textcoords="offset points", xytext=(4, 8), fontsize=11.5); ax.annotate(f"{g.adv_auc.iloc[-1]:.3f}", (g.n_cols.iloc[-1], g.adv_auc.iloc[-1]), textcoords="offset points", xytext=(4, -16), fontsize=11.5)
ax.invert_xaxis(); ax.set(xlabel="남은 열 수", ylabel="적대적 분류 AUC (0.5 = 구분 불가)", ylim=(.74, 1.05)); ax.legend(); save(fig, "P06_적대적_검증.png")
tg = fr[fr.principle == "타깃 선택"]; fig, ax = new(2); w = .38; short = {"부도 이진(기본)": "부도 이진", "원금 손실률 회귀": "원금 손실률", "실현 수익률 회귀": "실현 수익률"}
for a_, (k, t) in zip(ax, (("te_top30_ret", "시험 상위 30% 평균 수익률"), ("te_top30_default", "시험 상위 30% 부도율"))):
    for i, v in enumerate(("A_내생보존", "B_내생제거")):
        h = tg[tg.variant == v]; a_.bar(np.arange(len(h)) + i * w, h[k], w, label=ALAB[v])
        for xi, vv in enumerate(h[k]): a_.text(xi + i * w, vv + .002, f"{vv:.4f}" if k == "te_top30_ret" else f"{vv:.3f}", ha="center", fontsize=10)
    a_.set_xticks(np.arange(len(h)) + w / 2); a_.set_xticklabels([short.get(c, c) for c in h.config]); a_.set(title=t, ylim=(0, tg[k].max() * 1.3)); a_.legend(loc="upper left", fontsize=10.5)
save(fig, "P07_타깃_선택.png")
# ---------------- 결과 장
fig, ax = new(2)
for k in KEY:
    g = H[H.strategy == k]; ax[0].plot(g.a * 100, g.excess_pooled * 100, color=COL[k], label=LAB[k]); ax[1].plot(g.a * 100, g.sharpe, color=COL[k], label=LAB[k])
for a_ in ax:
    for x_ in (30, 53): a_.axvline(x_, color="k", ls=":", lw=1.2)
ax[0].set(title="초과수익 (%p/년)", xlabel="승인율 (금액, %)"); ax[1].set(title="월 빈티지 Sharpe", xlabel="승인율 (금액, %)"); legend_below(fig, ax[0], 4); save(fig, "R01_승인율_곡선.png", .13)
fig, ax = new()
for k in KEY:
    g = H[H.strategy == k]; ax.plot(g.e_sd * 100, g.e_mean * 100, color=COL[k], label=LAB[k])
    for a in (30, 53): ax.scatter(g.e_sd.iloc[a - 1] * 100, g.e_mean.iloc[a - 1] * 100, color=COL[k], s=110, marker="o" if a == 30 else "s", edgecolor="k", zorder=5)
ax.set(xlabel="월 빈티지 초과수익의 표준편차 (%p)", ylabel="월 빈티지 초과수익 평균 (%p/년)", title="위험–수익 곡선 (● 승인 30%, ■ 승인 53%)"); ax.legend(ncol=2, fontsize=10.5); save(fig, "R02_프런티어.png")
s = summ[(summ.split == "strict36") & (summ.subset == "헤드라인2015-16") & (summ.rank_unit == "월")].set_index("strategy"); ks = KEY + ["오라클(실현NPV)"]; y = np.arange(len(ks)); fig, ax = new()
for i, (c, lb) in enumerate([("R30", "승인 30%"), ("R53", "승인 53%"), ("R100", "승인 100%")]):
    ax.barh(y + (i - 1) * .27, s.loc[ks, c] * 100, .27, label=lb)
    for yi, vv in zip(y, s.loc[ks, c] * 100): ax.text(vv + .03, yi + (i - 1) * .27, f"{vv:.2f}", va="center", fontsize=9.5)
ax.set_yticks(y); ax.set_yticklabels([LAB[k] for k in ks]); ax.invert_yaxis(); ax.set(xlabel="초과수익 (전체 자본, %p/년)", xlim=(0, 3.3)); ax.legend(loc="center right"); save(fig, "R04_전략비교.png")
fig, ax = new(); yy = np.arange(len(KEY))
for off, sn, c in [(-.17, "헤드라인2015-16", "#d62728"), (.17, "2016", "#1f77b4")]:
    t = summ[(summ.split == "strict36") & (summ.subset == sn) & (summ.rank_unit == "월")].set_index("strategy").loc[KEY]
    ax.errorbar(t.diff30_53 * 100, yy + off, xerr=[(t.diff30_53 - t.diff30_53_lo) * 100, (t.diff30_53_hi - t.diff30_53) * 100], fmt="o", capsize=5, ms=8, color=c, label="시험 2015–16" if sn.startswith("헤드") else "2016만")
ax.set_yticks(yy); ax.set_yticklabels([LAB[k] for k in KEY]); ax.axvline(0, color="k"); ax.invert_yaxis(); ax.set(xlabel="초과수익 차: 승인 30% - 53% (%p/년, 95% 구간)"); ax.legend(loc="center right"); save(fig, "R13_30대53_차이.png")
a = al[(al.split == "strict36") & (al.kind == "level") & (al.model != "오라클")]; ap = al[(al.split == "primary36") & (al.kind == "level") & (al.subset == "헤드라인2015-16")].set_index("model")
ms = ["logit|raw141", "clf|raw141", "reg|raw141", "clf|raw+채택", "reg|raw+채택", "clf|raw+채택+FLAG", "reg|raw+채택+FLAG", "clf|raw+nv전체", "reg|raw+nv전체", "clf|nv만", "reg|nv만"]
ML = {"logit|raw141": "로지스틱 부도 (원 열만)", "clf|raw141": "부도 (원 열만)", "reg|raw141": "NPV (원 열만)", "clf|raw+채택": "부도 (+채택 변수)", "reg|raw+채택": "NPV (+채택 변수)", "clf|raw+채택+FLAG": "부도 (+채택·미확정)", "reg|raw+채택+FLAG": "NPV (+채택·미확정)", "clf|raw+nv전체": "부도 (+새 변수 56개)", "reg|raw+nv전체": "NPV (+새 변수 56개)", "clf|nv만": "부도 (새 변수만)", "reg|nv만": "NPV (새 변수만)"}
fig, ax = new(); g = a[a.subset == "헤드라인2015-16"].set_index("model").loc[ms]; yv = np.arange(len(ms))
ax.errorbar(g.alpha_invested_pp, yv, xerr=[g.alpha_invested_pp - g.inv_lo, g.inv_hi - g.alpha_invested_pp], fmt="o", color="#2ca02c", capsize=5, ms=8, label="누수 없음(strict36), 95% 구간")
ax.plot(ap.loc[ms, "alpha_invested_pp"], yv, "D", mfc="none", mec="#555", ms=9, label="누수 포함(primary36)"); ax.set_yticks(yv); ax.set_yticklabels([ML[m] for m in ms]); ax.invert_yaxis(); ax.axvline(0, color="k")
ax.set(xlabel="같은 셀 상위 30% 알파 (투입 자본당, %p/년)"); ax.legend(loc="lower right"); save(fig, "R05_알파_CI.png")
# 규칙 vs 모형 (2슬라이드)
RN = {"규칙|세부등급만(낮은 위험 순)": "세부등급 규칙", "규칙|금리 낮은 순": "금리 낮은 순 규칙"}
def rule_fig(sel, labs, nm, note_leg=True):
    fig, ax = new(2); x = np.arange(len(labs))
    for j, (c, lo, hi, tt) in enumerate([("차_초과수익_pp", "lo_pp", "hi_pp", "초과수익 차이 (%p/년)"), ("차_Sharpe", "Sharpe_lo", "Sharpe_hi", "월 빈티지 Sharpe 차이")]):
        for off, mk_, col, lb in [(-.2, "모형|부도확률(raw141)", "#d62728", "부도확률 모형"), (.2, "모형|NPV(raw141)", "#1f77b4", "NPV 모형")]:
            g_ = sel[sel.모형 == mk_]; ax[j].bar(x + off, g_[c], .4, color=col, label=lb, yerr=[g_[c] - g_[lo], g_[hi] - g_[c]], capsize=5)
        ax[j].axhline(0, color="k", lw=1.2); ax[j].set_xticks(x); ax[j].set_xticklabels(labs); ax[j].set_title(tt + "  (모형 - 규칙)")
    ax[0].legend(loc="upper right"); save(fig, nm)
f1 = RP[RP.비교 == "초과수익 기준 최선 규칙"]; rule_fig(f1, [f"{a_}\n(대 {RN[r]})" for a_, r in zip(f1[f1.모형 == "모형|부도확률(raw141)"].승인율, f1[f1.모형 == "모형|부도확률(raw141)"].최선_규칙)], "R20a_모형_vs_규칙_같은승인율.png")
f2 = RP[(RP.비교 == "규칙 자체 승인율") & ~RP.승인율.str.contains("전량")]; rule_fig(f2, [s_.replace("등급컷 ", "") + " 컷" for s_ in f2[f2.모형 == "모형|부도확률(raw141)"].승인율], "R20b_모형_vs_규칙_자체승인율.png")
fig, ax = new(2); ms3 = ["clf|raw141", "clf|raw+nv전체", "logit|raw141"]
for k in ms3:
    g = cal[cal.model == k]; ax[0].plot(g.mean_pred * 100, g.obs_rate * 100, "o-", color=COL[k], label=LAB[k], ms=8); r = roc[roc.model == k]; ax[1].plot(r.fpr, r.tpr, color=COL[k], label=f"{LAB[k]} (AUC {auc[auc.model == k].auc.iloc[0]:.3f})")
ax[0].plot([0, 50], [0, 50], "k--"); ax[0].set(title="보정: 예측 vs 실현 (10분위)", xlabel="평균 예측 부도확률 (%)", ylabel="실현 부도율 (%)"); ax[0].legend(loc="upper left", fontsize=10.5)
ax[1].plot([0, 1], [0, 1], "k--"); ax[1].set(title="ROC", xlabel="오경보율", ylabel="적중률"); ax[1].legend(loc="lower right", fontsize=10.5); save(fig, "R09_보정_ROC.png")
fig, ax = new(2)
for j, nm in enumerate(["clf", "reg"]):
    g = imp[(imp.model == nm) & (imp.fs == "raw+nv전체")].nlargest(14, "mean_abs_shap").iloc[::-1]
    ax[j].barh(g.feature, g.mean_abs_shap, color=["#ff7f0e" if f.startswith("nv_") else "#1f77b4" for f in g.feature]); ax[j].set_title(("부도 모형" if nm == "clf" else "NPV 모형") + " 상위 14 (주황 = 새 변수)"); ax[j].tick_params(axis="y", labelsize=10.5); ax[j].set_xlabel("평균 |SHAP|")
save(fig, "R10_SHAP.png")
pdm = [("clf|raw141", "부도 (원 열만)"), ("reg|raw141", "NPV (원 열만)"), ("clf|raw+nv全"[:-1] + "전체", "부도 (+새 변수)"), ("reg|raw+nv전체", "NPV (+새 변수)")]
pm = al[(al.split == "strict36") & (al.kind == "pair_diff") & (al.subset == "헤드라인2015-16")]; PL = {"reg|raw+nv전체 - reg|raw141": "NPV: 새 변수 56개 추가", "clf|raw+nv전체 - clf|raw141": "부도: 새 변수 56개 추가", "reg|raw+채택 - reg|raw141": "NPV: 채택 변수 3종 추가", "reg|raw+채택+FLAG - reg|raw141": "NPV: 채택+미확정 추가", "reg|raw141 - clf|raw141": "NPV 모형 - 부도 모형 (원 열만)", "reg|raw+nv전체 - clf|raw+nv전체": "NPV 모형 - 부도 모형 (+새 변수)", "reg|nv만 - reg|raw141": "NPV: 새 변수만 - 원 열만"}
pm = pm[pm.model.isin(PL)]; fig, ax = new(); yv = np.arange(len(pm)); ax.errorbar(pm.alpha_invested_pp, yv, xerr=[pm.alpha_invested_pp - pm.inv_lo, pm.inv_hi - pm.alpha_invested_pp], fmt="o", color="#2ca02c", capsize=5, ms=8)
ax.set_yticks(yv); ax.set_yticklabels([PL[m] for m in pm.model]); ax.invert_yaxis(); ax.axvline(0, color="k"); ax.set(xlabel="알파 차이 (투입 자본당, %p/년, 95% 구간)"); save(fig, "R06_새변수_알파차이.png")
fig, ax = new(); m = mm[mm.model != "lgb_reg"].copy(); m["fsl"] = m.fs.map({"raw141": "원 열만", "raw+채택": "+채택 변수", "raw+채택+FLAG": "+채택·미확정", "raw+nv전체": "+새 변수 56개", "nv만": "새 변수만"}); m["ml"] = m.model.map({"lgb_clf": "LightGBM", "logit_clf": "로지스틱"})
rows_ = m[m.split == "strict36"][["fsl", "ml", "fs", "model"]].drop_duplicates(); yv = np.arange(len(rows_)); cmap = {"strict36": "#1f77b4", "primary36": "#d62728", "m60": "#7f7f7f"}; lbs = {"strict36": "누수 없음(strict36)", "primary36": "누수 포함(primary36)", "m60": "60개월(누수 포함)"}
for i, sp_ in enumerate(("strict36", "primary36", "m60")):
    vals = [m[(m.split == sp_) & (m.fs == r.fs) & (m.model == r.model)].auc for r in rows_.itertuples()]; vals = [v.iloc[0] if len(v) else np.nan for v in vals]
    ax.barh(yv + (i - 1) * .27, vals, .27, color=cmap[sp_], label=lbs[sp_]); [ax.text(v + .002, yi + (i - 1) * .27, f"{v:.3f}", va="center", fontsize=9) for yi, v in zip(yv, vals) if not np.isnan(v)]
ax.set_yticks(yv); ax.set_yticklabels([f"{a_} · {b_}" for a_, b_ in zip(rows_.ml, rows_.fsl)]); ax.invert_yaxis(); ax.set_xlim(.5, .83); ax.set(xlabel="시험 AUC"); ax.legend(loc="lower right", fontsize=10.5); save(fig, "R15_AUC_분할별.png")
hh = hur[(hur.split == "strict36") & (hur.subset == "헤드라인2015-16") & (hur.rank_unit == "월")]; fig, ax = new(2)
for k in KEY + ["오라클(실현NPV)"]:
    g = hh[hh.strategy == k]; ax[0].plot(g.h_pp, g.best_a * 100, "o-", color=COL[k], label=LAB[k], ms=7); ax[1].plot(g.h_pp, g.best_profit_pct, "o-", color=COL[k], ms=7)
ax[0].set(title="이익이 최대인 승인율 (%)", xlabel="허들: FEDFUNDS + h (%p)"); ax[1].set(title="그때의 최대 이익 (원금당 %)", xlabel="허들: FEDFUNDS + h (%p)"); legend_below(fig, ax[0], 4); save(fig, "R11_허들.png", .13)
s0 = sg[sg.group == "전체 2012–16"]; pv = s0.pivot(index="pd_mult", columns="lgd_mult", values="excess_pct"); fig, ax = new(2, gridspec_kw={"width_ratios": [1, 1.2]})
ax[0].imshow(pv.values, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10); ax[0].grid(False); ax[0].set_xticks(range(pv.shape[1])); ax[0].set_xticklabels(pv.columns); ax[0].set_yticks(range(len(pv))); ax[0].set_yticklabels(pv.index); ax[0].set(title="초과 NPV (%) 스트레스: 부도 배수 × 손실 배수", xlabel="손실률(LGD) 배수", ylabel="부도(PD) 배수")
for (i, k), v in np.ndenumerate(pv.values): ax[0].text(k, i, f"{v:.1f}", ha="center", va="center", fontsize=12.5)
bb = sb[sb.group.isin(["2012", "2013", "2014", "2015", "2016", "전체 2012–16"]) | sb.group.str.startswith("등급")]; cols_ = ["#ff7f0e" if not g_.startswith("등급") else "#1f77b4" for g_ in bb.group]
ax[1].barh(range(len(bb)), bb.breakeven_pd_mult, color=cols_); ax[1].set_yticks(range(len(bb))); ax[1].set_yticklabels(bb.group); ax[1].invert_yaxis(); ax[1].axvline(1, color="k", ls="--"); ax[1].tick_params(axis="y", labelsize=10)
for yi, v in enumerate(bb.breakeven_pd_mult): ax[1].text(v + .03, yi, f"{v:.2f}", va="center", fontsize=9.5)
ax[1].set(title="손익분기 부도 배수 (1 미만이면 이미 손실)", xlabel="배수"); save(fig, "R12_스트레스.png")
fig, ax = new(2)
for y_ in sorted(vc[vc.term == 36].year.unique()):
    g = vc[(vc.term == 36) & (vc.year == y_) & (vc.age <= 42)]; ax[0].plot(g.age, g.cum_net_cash_per_A, label=str(y_)); ax[1].plot(g.age, g.cum_writeoff_per_A * 100, label=str(y_))
ax[0].axhline(1, color="k", ls=":"); ax[0].set(title="누적 순현금 / 원금 (1.0 = 원금 회수)", xlabel="경과 개월"); ax[1].set(title="누적 상각 / 원금 (%)", xlabel="경과 개월"); ax[0].legend(title="발행연도", fontsize=10.5); save(fig, "R07_빈티지_곡선.png")
g36 = mk[(mk.term == 36) & mk.grade.isin(list("ABCDEFG"))]; fig, ax = new(2)
for j, (c, tt, cm) in enumerate([("default_rate", "부도율 (%)", "Reds"), ("R_reinv", "초과수익 (%p/년)", "RdYlGn")]):
    pv = g36.pivot(index="grade", columns="year", values=c) * 100; ax[j].imshow(pv.values, cmap=cm, aspect="auto"); ax[j].grid(False); ax[j].set_xticks(range(pv.shape[1])); ax[j].set_xticklabels(pv.columns); ax[j].set_yticks(range(7)); ax[j].set_yticklabels(pv.index); ax[j].set(title=tt, xlabel="발행연도", ylabel="등급")
    for (i, k), v in np.ndenumerate(pv.values): ax[j].text(k, i, f"{v:.1f}", ha="center", va="center", fontsize=12, color="white" if (j == 0 and v > 33) or (j == 1 and v < -1.5) else "black")
save(fig, "R08_등급x연도.png")
ah = al[(al.kind == "level") & (al.subset == "헤드라인2015-16") & al.split.isin(["strict36", "primary36"])]; mds = ["clf|raw141", "reg|raw141", "clf|raw+nv전체", "reg|raw+nv전체"]; ML4 = ["부도\n(원 열만)", "NPV\n(원 열만)", "부도\n(+새 변수)", "NPV\n(+새 변수)"]; fig, ax = new(2); x = np.arange(4)
for off, sp_, c, lb in [(-.2, "strict36", "#1f77b4", "누수 없음 (strict36)"), (.2, "primary36", "#d62728", "누수 포함 (primary36)")]:
    g = ah[ah.split == sp_].set_index("model").loc[mds]; ax[0].bar(x + off, g.alpha_invested_pp, .4, color=c, label=lb, yerr=[g.alpha_invested_pp - g.inv_lo, g.inv_hi - g.alpha_invested_pp], capsize=5)
    g2 = auc_all[auc_all.split == sp_].set_index("model").loc[["clf|raw141", "clf|raw+nv전체"]]; ax[1].bar(np.arange(2) + off, g2.auc, .4, color=c, label=lb, yerr=[g2.auc - g2.lo, g2.hi - g2.auc], capsize=5)
ax[0].set_xticks(x); ax[0].set_xticklabels(ML4); ax[0].set(title="같은 셀 알파 (%p/년, 95% 구간)"); ax[0].legend(loc="upper left", fontsize=10.5); ax[1].set_xticks(range(2)); ax[1].set_xticklabels(["부도 (원 열만)", "부도 (+새 변수)"]); ax[1].set_ylim(.55, .75); ax[1].set(title="부도 AUC (95% 구간)"); save(fig, "R18_누수시연.png")
cs = cur[(cur.split == "strict36") & (cur.subset == "헤드라인2015-16")]; fig, ax = new(2, gridspec_kw={"width_ratios": [1.3, 1]})
for k, c in (("clf|raw+nv전체", "#d62728"), ("reg|raw+nv전체", "#1f77b4")):
    for u, ls in (("월", "-"), ("셀", "--")):
        g = cs[(cs.strategy == k) & (cs.rank_unit == u)]; ax[0].plot(g.a * 100, g.sharpe, ls, color=c, label=f"{'부도' if k.startswith('clf') else 'NPV'} 모형 · {'발행월 안' if u == '월' else '월×세부등급 안'} 순위")
ax[0].set(title="월 빈티지 Sharpe", xlabel="승인율 (%)"); ax[0].legend(fontsize=10, loc="upper right")
cp = pd.read_csv(W / "selection_composition.csv"); cp = cp[(cp.split == "strict36") & cp.strategy.isin(["clf|raw+nv전체", "reg|raw+nv전체"])]; lab = [f"{'부도' if r.strategy.startswith('clf') else 'NPV'}\n{'월 안' if r.rank_unit == '월' else '셀 안'}" for r in cp.itertuples()]
xx = np.arange(len(cp)); ax[1].bar(xx - .27, cp.int_rate_mean_pct, .27, label="평균 금리 (%)"); ax[1].bar(xx, cp.default_rate * 100, .27, label="부도율 (%)"); ax[1].bar(xx + .27, cp.share_CG_amt * 100, .27, label="C–G 등급 비중 (%)")
ax[1].set_xticks(xx); ax[1].set_xticklabels(lab); ax[1].set_title("승인 30%로 고른 대출의 구성"); ax[1].legend(fontsize=10, loc="upper left"); save(fig, "R19_순위단위.png")
C.log("done")
