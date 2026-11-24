"""98_d: results_raw.csv + results_ens.csv 병합 → 기준선(T0) 대비 델타(겹 쌍체 ΔAUC, 월 블록 부트스트랩 ΔSharpe) → 기법_결과.csv, 그림(fig/), _work/tables.md.
사용: python3 scripts/98_d_report.py"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; K = importlib.import_module("98_common"); log = K.log
import json, glob, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False, "font.size": 15})
W, FIG, OUT = K.WORK, K.FIG, K.OUT
R = pd.concat([pd.read_csv(W / f) for f in ("results_raw.csv", "results_ens.csv") if (W / f).exists()], ignore_index=True)
R = R[R.quick == 0].drop_duplicates("id", keep="last").reset_index(drop=True)
IDX = {"r30": 29, "r53": 52, "r70": 69, "r90": 89}; b = R[R.id == "T0"].iloc[0]; e_b = np.load(W / "e_T0.npy"); fb = np.array(json.loads(b.fold_auc))
rows = []
for _, r in R.iterrows():
    e_t = np.load(W / f"e_{r.id}.npy"); fa = np.array(json.loads(r.fold_auc)); dd = fa - fb
    d = dict(technique=r.id, group=r.grp, description=r["name"], protocol="무작위 층화 5-겹 CV(OOF)", AUC=r.AUC, logloss=r.LOGLOSS)
    for k in IDX: d["Sharpe_" + k] = r["SR_" + k]; d["excess_" + k] = r["XS_" + k]
    d.update(Sharpe_best=r.SR_best, excess_best=r.XS_best, a_best=r.a_best, dAUC=r.AUC - b.AUC, dAUC_fold_mean=dd.mean(), dAUC_fold_sd=dd.std(ddof=1), dAUC_folds_pos=f"{(dd > 0).sum()}/{len(dd)}", dLogloss=r.LOGLOSS - b.LOGLOSS)
    for k, i in IDX.items():
        m, lo, hi = (0, 0, 0) if r.id == "T0" else K.paired_sr_delta(e_t, e_b, i); d["dSharpe_" + k] = m; d["dSharpe_" + k + "_lo"] = lo; d["dSharpe_" + k + "_hi"] = hi
    d.update(dSharpe_best=r.SR_best - b.SR_best, dExcess_r53=r.XS_r53 - b.XS_r53, folds_auc=r.fold_auc); rows.append(d)
T = pd.DataFrame(rows)
ADD = [c for c in R.columns if c.startswith(("ROE_", "SR_", "XS_")) and "@" in c]                     # 연도군·만기별 Sharpe, 코스 ROE
for c in ADD: T[c] = R[c].to_numpy()
T["dAUC_sig"] = np.where(T.technique == "T0", "기준", np.where((T.dAUC_fold_mean > 2 * T.dAUC_fold_sd / np.sqrt(5)) & (T.dAUC_folds_pos.str[0] == "5"), "AUC 5/5겹 개선", "미확정"))   # verdict 열 폐기: 판정은 기법_탐색.md
T = T.sort_values(["group", "technique"])
T.to_csv(OUT / "기법_결과.csv", index=False, encoding="utf-8-sig", float_format="%.5f")
cols = ["technique", "AUC", "dAUC", "dAUC_fold_mean", "dAUC_fold_sd", "dAUC_folds_pos", "logloss", "Sharpe_r53", "dSharpe_r53", "dSharpe_r53_lo", "dSharpe_r53_hi", "Sharpe_best", "a_best", "dAUC_sig"]
(W / "tables.md").write_text(T[cols].to_markdown(index=False, floatfmt=".4f"))
a = T[T.technique != "T0"].iloc[::-1]; y = np.arange(len(a))
fig, ax = plt.subplots(1, 2, figsize=(26, max(10, 0.55 * len(a) + 3)), sharey=True)
ax[0].barh(y, a.dAUC_fold_mean, xerr=a.dAUC_fold_sd, color=np.where(a.dAUC_fold_mean > 0, "#1f77b4", "#d62728")); ax[0].set_title("ΔAUC (5겹 평균±SD, vs T0)")
ax[1].errorbar(a.dSharpe_r53, y, xerr=[a.dSharpe_r53 - a.dSharpe_r53_lo, a.dSharpe_r53_hi - a.dSharpe_r53], fmt="o", color="k", capsize=4); ax[1].set_title("ΔSharpe @승인율 53% (95% 월블록 부트스트랩, 쌍체)")
for x_ in ax: x_.axvline(0, color="gray", lw=1); x_.grid(axis="x", alpha=.3)
ax[0].set_yticks(y); ax[0].set_yticklabels(a.technique); plt.tight_layout(); plt.savefig(FIG / "fig1_기법별_델타.png", dpi=130); plt.close()
names = [t for t in ("T0", "T8_kept", "T8_tuned", "T8_avg", "T8_stack") if (W / f"e_{t}.npy").exists()]
plt.figure(figsize=(16, 10))
for t in names: e = np.load(W / f"e_{t}.npy"); plt.plot(K.AG * 100, e.mean(0) / e.std(0, ddof=1), lw=3, label=t)
plt.xlabel("승인율 (%)"); plt.ylabel("Sharpe (월 초과수익 mean/sd)"); plt.title("승인율-Sharpe 곡선 (무작위 5-겹 CV, 36m 전 발행연도, 무위험=만기일치 국채)"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "fig2_승인율_Sharpe곡선.png", dpi=130); plt.close()
fs = sorted(glob.glob(str(W / "prune_stability_*.json")))
if fs:
    s = json.load(open(fs[-1])); imp = pd.Series(s["imp"]).sort_values(ascending=False).head(30)[::-1]
    plt.figure(figsize=(16, 14)); plt.barh(imp.index, imp.values, color="#1f77b4"); plt.xlabel("mean |SHAP| (학습 겹 표본 2만 행, 겹 0)"); plt.title("특성 중요도 상위 30"); plt.tight_layout(); plt.savefig(FIG / "fig3_SHAP_중요도.png", dpi=130); plt.close()
log("done", T.shape)
