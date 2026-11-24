"""102 보고: 저장된 전체 모집단 예측(oof_<id>_full.npz)으로 발행연도군(2007–11·2012–16·2017–20) ROE@10/30/50/70 과 Sharpe@53%(36m) 를 표로 저장.
사용: python3 scripts/102_report.py id1 id2 ...  → 산출물/대안모델/연도군_결과.csv"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; M = importlib.import_module("102_common"); K, L = M.K, M.L
import numpy as np, pandas as pd
rows = []
for mid in sys.argv[1:]:
    s = np.full(M.n, np.nan); s[M.IX] = np.load(M.W2 / f"oof_{mid}.npz")["score"][M.IX]
    roe = K.course_roe(L, s, rates=(0.1, 0.3, 0.5, 0.7), groups=True)
    for g, r in roe.items():
        rows.append(dict(id=mid, group=g, n=r["n"], ROE10=r["r10"], ROE30=r["r30"], ROE50=r["r50"], ROE70=r["r70"]))
    gy = {"all": (2007, 2020), **K.YG}
    for g, (a, b) in gy.items():
        m = ~np.isnan(s) & (L.term.to_numpy() == 36) & L.issue_year.between(a, b).to_numpy()
        if m.sum() > 500:
            sh, _ = K.sharpe_eval(L[m], s[m], 36); [r.update(SR53_36=sh["SR_r53"]) for r in rows if r["id"] == mid and r["group"] == g]
    print(mid, "done", flush=True)
pd.DataFrame(rows).to_csv(M.OUT2 / "연도군_결과.csv", index=False, encoding="utf-8-sig"); print(pd.DataFrame(rows).round(3).to_string())
