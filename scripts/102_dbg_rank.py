import sys; sys.path.insert(0, "scripts")
import importlib; M = importlib.import_module("102_common"); import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score as A
tr, te, Xtr, Xte = M.fold_data(0, M.SUB, "sub")
mo = M.B.D.issue_month_code.to_numpy()[tr]; o = np.argsort(mo, kind="stable"); X_ = Xtr.iloc[o]; z = M.Z_TR[tr][o]; mo = mo[o]
lab = pd.Series(z).groupby(mo).rank(pct=True, method="first").mul(10).clip(upper=9.999).astype(int).to_numpy()
print("lab vs z corr", np.corrcoef(lab, z)[0, 1], "lab vs y", np.corrcoef(lab, M.y_all[tr][o])[0, 1], np.bincount(lab))
grp = np.bincount(mo - mo.min()); grp = grp[grp > 0]; print(grp.sum(), len(lab), grp[:5])


p = dict(objective="lambdarank", learning_rate=0.1, num_leaves=31, min_child_samples=100, num_threads=10, verbose=-1, label_gain=list(range(10)), max_bin=127)
b = lgb.train(p, lgb.Dataset(X_, lab, group=grp), 30)
pt = b.predict(X_); print("train corr(pred,lab)", np.corrcoef(pt, lab)[0,1], "train AUC(-pred) vs y", A(M.y_all[tr][o], -pt))
d = pd.DataFrame({"p": pt, "l": lab, "m": mo}); print("within-month spearman mean", d.groupby("m").apply(lambda g: g.p.corr(g.l, method="spearman") if len(g) > 50 else np.nan).mean())
print(Xte.int_rate.corr(pd.Series(M.y_all[te], index=Xte.index)), pd.Series(pt).corr(pd.Series(X_.int_rate.to_numpy())))
