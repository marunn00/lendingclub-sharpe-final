"""98_a: 원본 lending_club_2020_train.csv → 간결 테이블 산출물/간결테이블/train_간결.parquet (nv_ 열 미사용).
구성: 식별·라벨 열 + 상장시점 원본 열(누수 31열 제거: 열_원장.csv '사후 결과') + 이름이 분명한 파생 변수. |ρ|>0.95 근사중복은 원본·일반 파생에서만 제거(안전자산 스프레드는 사용자 요청으로 보호).
시장금리 시점: 발행월 M−1 월평균(국채 DGS*, 일별→월평균) / FEDFUNDS는 M−1 월값. 열 선택은 라벨을 쓰지 않는 규칙(결측률·최빈값 비중·|ρ|)만 사용(행 200,000 무작위 표본, 시드 0).
사용: python3 scripts/98_a_clean_table.py   → train_간결.parquet, 변수설명.csv(98_a2에서 완성), _work/corr_pruned.csv"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; K = importlib.import_module("98_common"); C = K.C; log = K.log
import numpy as np, pandas as pd, polars as pl
import importlib; F = importlib.import_module("98_features"); CATS = F.CATS
CSV = C.ROOT / "lending_club_2020_train.csv"; OUTD = C.ROOT / "산출물" / "간결테이블"; OUTD.mkdir(exist_ok=True, parents=True)
D = F.build_raw_table(CSV)
# ---- 열 선택: 라벨 비사용 규칙. 보호: 스프레드·프리미엄·기준 시장열·핵심 원본/파생(변수설명 참조)
feat = [c for c in D.columns if c not in {"id", "issue_d", "issue_month_code", "loan_status", "default"} | set(CATS) | {"zip3", "emp_title_norm"}]
PROTECT = {"spread_10y", "spread_30y", "spread_matched", "spread_fedfunds", "rate_premium_subgrade", "tsy10_at_issue", "tsy_matched_at_issue", "int_rate", "installment", "loan_amnt", "term_m",
           "fico_mid", "credit_hist_years", "pti", "lti", "new_acc_share24", "open_acc_ratio", "bal_per_acc", "revol_util", "emp_length_yr", "total_acc", "revol_bal", "delinq_2yrs", "pub_rec",
           "pub_rec_bankruptcies", "mths_since_last_record", "annual_inc", "dti", "inq_last_6mths"}
samp = D.sample(200000, random_state=K.SEED); drop = {}; miss = D[feat].isna().mean()
mode_share = {c: (D[c].dropna().round(6) == D[c].dropna().round(6).mode().iloc[0]).mean() for c in feat if c not in PROTECT}
for c in feat:                                    # 규칙 1~3: 공동신청, 시대 결측(전 행 결측>50%, mths_since_* 제외), 희귀사건 열(최빈값 ≥98%)
    if c in PROTECT: continue
    if c.startswith("sec_app_") or c.endswith("_joint"): drop[c] = ("공동신청 열(결측≈93%)", np.nan)
    elif miss[c] > 0.25 and not c.startswith("mths_since_"): drop[c] = (f"시대 열(2015-12 이후 도입 등): 전 행 결측 {miss[c]:.0%}", np.nan)
    elif mode_share[c] >= 0.98: drop[c] = (f"희귀사건 열: 최빈값 비중 {mode_share[c]:.1%}", np.nan)
feat = [c for c in feat if c not in drop]
cm = samp[feat].corr(min_periods=1000).abs()
for i, a in enumerate(feat):
    for b in feat[i + 1:]:
        if a in drop or b in drop: continue
        if cm.loc[a, b] > 0.90:                   # 규칙 4: |ρ|>0.90 → 결측 적은 쪽(동률이면 이름순) 유지, 보호열 우선
            if a in PROTECT and b in PROTECT: continue
            if a in PROTECT: drop[b] = (a, cm.loc[a, b])
            elif b in PROTECT: drop[a] = (b, cm.loc[a, b])
            else: lo, hi = (a, b) if (miss[a], a) > (miss[b], b) else (b, a); drop[lo] = (hi, cm.loc[a, b])
pd.DataFrame([dict(dropped=k, reason_or_kept_instead=v[0], abs_corr=v[1]) for k, v in drop.items()]).to_csv(K.WORK / "corr_pruned.csv", index=False, encoding="utf-8-sig")
keep = [c for c in D.columns if c not in drop]; D = D[keep]
D.to_parquet(OUTD / "train_간결.parquet", index=False); log("saved", D.shape, "dropped", len(drop)); print(sorted(drop)); print([c for c in D.columns])
