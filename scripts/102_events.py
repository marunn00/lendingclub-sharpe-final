"""102 경쟁위험용 사건 라벨(학습 목표값, 특성 아님): 원본 CSV 의 last_pymnt_d·issue_d·term 에서 사건 종류·시점을 만든다.
클래스(0–10): 부도 0–4 = 마지막 납입 나이(개월) 구간 b=min((K−1)//12, 4) · 조기상환 5–9 · 10 = 만기 완납(K≥term). K = 마지막 납입월 − 발행월(98 와 같은 월 코드)."""
import sys; sys.path.insert(0, "scripts")
import importlib; M = importlib.import_module("102_common"); C = M.K.C
import numpy as np, pandas as pd, polars as pl
raw = pl.read_csv(C.ROOT / "lending_club_2020_train.csv", columns=["id", "issue_d", "last_pymnt_d", "term"], infer_schema_length=0, null_values=["", "NA"]).filter(pl.col("issue_d").is_not_null()).to_pandas().set_index("id")
r = raw.loc[M.B.D.id.to_numpy()]
K_ = (C.mcode_str(r.last_pymnt_d) - C.mcode_str(r.issue_d)).astype(float); K_[r.last_pymnt_d.isna().to_numpy()] = 0
term = M.B.D.term_m.to_numpy(); y = M.y_all; b = np.minimum((np.maximum(K_, 1) - 1) // 12, 4).astype(int)
ev = np.where(y == 1, b, np.where(K_ < term, 5 + b, 10)).astype(int)
print("K stats", pd.Series(K_).describe().round(1).to_dict()); print("class counts", np.bincount(ev).tolist(), "default share", y.mean())
print("check default cls == y:", ((ev <= 4) == (y == 1)).all(), "36m max b", b[term == 36].max())
np.save(M.W2 / "events.npy", ev)
