"""98_a2: 간결 테이블 변수설명.csv(한글 의미·산식·단위) + 파생 변수 검증(상관, 단변량 AUC, 십분위 부도율·수익률, 전략 Sharpe, 스프레드 vs 원금리). 표 구성과 무관한 사후 기술 분석(라벨 사용, 표 열 선택에는 영향 없음). 대상: 확정 대출 전체(모든 발행연도).
출력: 산출물/간결테이블/변수설명.csv, 파생변수_분석.csv, 파생변수_십분위.csv, 스프레드_상관.csv, fig/fig0_*.png
수익률 = NPV_tsy/대출액 (만기일치 국채 할인). Sharpe = 36m 대출 전체 발행연도, 발행월 안에서 해당 변수로 정렬(최적 방향), 거절자금 만기일치 국채(98_common 정의)."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; K = importlib.import_module("98_common"); C = K.C; log = K.log
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False, "font.size": 16})
OUTD = C.ROOT / "산출물" / "간결테이블"
D = pd.read_parquet(OUTD / "train_간결.parquet")
DESC = {  # 열: (분류, 한글 의미, 산식/출처, 단위)
 "id": ("식별", "대출 고유번호", "원본 id", "-"), "issue_d": ("식별", "대출 발행월", "원본 issue_d (예: Dec-2015)", "월"),
 "issue_month_code": ("식별", "발행월 숫자코드(2005-01=0)", "96_common.mcode_str(issue_d)", "개월"), "term_m": ("상장시점", "만기(개월)", "원본 term에서 숫자만", "개월"),
 "loan_status": ("라벨", "대출 최종 상태(사후 결과; 모형 입력 금지)", "원본 loan_status", "-"), "default": ("라벨", "부도=1(Charged Off·Default), 완납=0, 진행중=결측", "loan_status에서 변환", "0/1"),
 "loan_amnt": ("상장시점", "신청 대출금액", "원본", "USD"), "int_rate": ("상장시점", "대출 금리(연)", "원본, % 기호 제거", "%"), "installment": ("상장시점", "월 상환액", "원본", "USD"),
 "annual_inc": ("상장시점", "연소득(자기신고)", "원본", "USD"), "dti": ("상장시점", "월 부채상환액/월소득(신규 대출 제외)", "원본", "%"), "delinq_2yrs": ("상장시점", "최근 2년 30일+ 연체 건수", "원본", "건"),
 "inq_last_6mths": ("상장시점", "최근 6개월 신용조회 수", "원본", "건"), "mths_since_last_record": ("상장시점", "마지막 공공기록 후 경과월(결측=기록 없음)", "원본", "개월"),
 "pub_rec": ("상장시점", "공공기록(파산·판결 등) 건수", "원본", "건"), "revol_bal": ("상장시점", "회전신용 잔액", "원본", "USD"), "revol_util": ("상장시점", "회전한도 대비 사용률", "원본, % 제거", "%"),
 "total_acc": ("상장시점", "전체 신용계좌 수", "원본", "건"), "mths_since_rcnt_il": ("상장시점", "최근 할부계좌 개설 후 경과월(2012+ 항목, 결측 많음)", "원본", "개월"),
 "total_rev_hi_lim": ("상장시점", "회전신용 총한도", "원본", "USD"), "acc_open_past_24mths": ("상장시점", "최근 24개월 신규 개설 계좌 수", "원본", "건"), "avg_cur_bal": ("상장시점", "계좌당 평균 잔액", "원본", "USD"),
 "bc_open_to_buy": ("상장시점", "은행카드 미사용 한도", "원본", "USD"), "bc_util": ("상장시점", "은행카드 한도 대비 사용률", "원본", "%"), "mo_sin_old_il_acct": ("상장시점", "가장 오래된 할부계좌 경과월", "원본", "개월"),
 "mo_sin_rcnt_rev_tl_op": ("상장시점", "가장 최근 회전계좌 개설 후 경과월", "원본", "개월"), "mo_sin_rcnt_tl": ("상장시점", "가장 최근 계좌 개설 후 경과월", "원본", "개월"),
 "mort_acc": ("상장시점", "주택담보대출 계좌 수", "원본", "건"), "mths_since_recent_bc": ("상장시점", "최근 은행카드 개설 후 경과월", "원본", "개월"), "mths_since_recent_inq": ("상장시점", "최근 신용조회 후 경과월", "원본", "개월"),
 "num_actv_bc_tl": ("상장시점", "활성 은행카드 계좌 수", "원본", "건"), "num_actv_rev_tl": ("상장시점", "활성 회전계좌 수", "원본", "건"), "num_op_rev_tl": ("상장시점", "개설 중 회전계좌 수", "원본", "건"),
 "num_tl_op_past_12m": ("상장시점", "최근 12개월 개설 계좌 수", "원본", "건"), "percent_bc_gt_75": ("상장시점", "한도 75% 초과 은행카드 비율", "원본", "%"), "pub_rec_bankruptcies": ("상장시점", "공공기록 파산 건수", "원본", "건"),
 "open_acc": ("상장시점", "개설 중 신용계좌 수", "원본", "건"), "mths_since_last_delinq": ("상장시점", "마지막 연체 후 경과월(결측=연체 없음)", "원본", "개월"), "mths_since_last_major_derog": ("상장시점", "마지막 90일+ 연체/상각 후 경과월", "원본", "개월"),
 "tot_coll_amt": ("상장시점", "추심 대상 총액", "원본", "USD"), "tot_cur_bal": ("상장시점", "전체 계좌 현재 잔액", "원본", "USD"), "mths_since_recent_bc_dlq": ("상장시점", "은행카드 마지막 연체 후 경과월", "원본", "개월"), "mths_since_recent_revol_delinq": ("상장시점", "회전계좌 마지막 연체 후 경과월", "원본", "개월"),
 "num_accts_ever_120_pd": ("상장시점", "120일+ 연체 경험 계좌 수", "원본", "건"), "num_bc_sats": ("상장시점", "만족(정상) 은행카드 계좌 수", "원본", "건"), "num_bc_tl": ("상장시점", "은행카드 계좌 수", "원본", "건"), "num_il_tl": ("상장시점", "할부 계좌 수", "원본", "건"),
 "num_rev_accts": ("상장시점", "회전계좌 수", "원본", "건"), "num_tl_90g_dpd_24m": ("상장시점", "최근 24개월 90일+ 연체 계좌 수", "원본", "건"), "pct_tl_nvr_dlq": ("상장시점", "연체 이력 없는 계좌 비율", "원본", "%"), "tax_liens": ("상장시점", "세금 유치권 건수", "원본", "건"),
 "total_bal_ex_mort": ("상장시점", "주택담보 제외 총잔액", "원본", "USD"), "total_il_high_credit_limit": ("상장시점", "할부 계좌 총한도", "원본", "USD"), "lti": ("파생·부담", "대출액/연소득", "loan_amnt/annual_inc", "비율"), "total_bc_limit": ("상장시점", "은행카드 총한도", "원본", "USD"), "emp_length_yr": ("상장시점", "근속연수(<1년=0, 10+년=10)", "원본 emp_length 숫자화", "년"),
 "sub_grade": ("상장시점", "세부등급 A1~G5(LendingClub 신용평가)", "원본", "범주"), "home_ownership": ("상장시점", "주거 형태", "원본", "범주"), "verification_status": ("상장시점", "소득 검증 여부", "원본", "범주"),
 "purpose": ("상장시점", "대출 목적", "원본", "범주"), "addr_state": ("상장시점", "거주 주", "원본", "범주"), "initial_list_status": ("상장시점", "최초 상장 구분 w/f", "원본", "범주"), "application_type": ("상장시점", "개인/공동 신청", "원본", "범주"),
 "zip3": ("상장시점", "우편번호 앞 3자리", "원본 zip_code 앞 3자", "범주(고유값 다수)"), "emp_title_norm": ("상장시점", "직함(소문자·특수문자 제거)", "원본 emp_title 정규화", "문자열(고유값 다수)"),
 "tsy10_at_issue": ("시장참고", "발행 전월 10년 국채금리(월평균)", "DGS10, 발행월 M−1 월평균", "%"), "tsy_matched_at_issue": ("시장참고", "만기 일치 국채금리: 36m→3년, 60m→5년", "DGS3/DGS5, 발행월 M−1 월평균", "%"),
 "spread_10y": ("파생·스프레드", "대출금리 − 10년 국채", "int_rate − tsy10_at_issue", "%p"), "spread_30y": ("파생·스프레드", "대출금리 − 30년 국채", "int_rate − DGS30(M−1 월평균)", "%p"),
 "spread_matched": ("파생·스프레드", "대출금리 − 만기일치 국채", "int_rate − tsy_matched_at_issue", "%p"), "spread_fedfunds": ("파생·스프레드", "대출금리 − 연방기금금리", "int_rate − FEDFUNDS(M−1 월값)", "%p"),
 "rate_premium_subgrade": ("파생·스프레드", "같은 발행월×만기×세부등급 금리 중앙값 대비 초과금리", "int_rate − median(int_rate | 발행월, 만기, sub_grade)", "%p"),
 "pti": ("파생·부담", "연 상환액/연소득", "installment×12/annual_inc", "비율"), "credit_hist_years": ("파생·신용", "신용이력 길이", "(발행월 − earliest_cr_line)/12", "년"), "fico_mid": ("파생·신용", "FICO 구간 중앙값", "(fico_range_low+fico_range_high)/2", "점"),
 "open_acc_ratio": ("파생·신용", "개설 중 계좌/전체 계좌", "open_acc/total_acc", "비율"), "bal_per_acc": ("파생·부담", "계좌당 총잔액", "tot_cur_bal/total_acc", "USD"), "new_acc_share24": ("파생·신용", "전체 계좌 중 최근 24개월 신규 비중", "acc_open_past_24mths/total_acc", "비율")}
if __name__ == "__main__":
    assert set(DESC) == set(D.columns), (set(D.columns) ^ set(DESC))
    L = K.load_loans(); W_ = D[D.default.notna().to_numpy()].reset_index(drop=True); assert len(W_) == len(L) and (W_.id.to_numpy() == L.id.to_numpy()).all()
    ret = (L.NPV_tsy / L.A).to_numpy(); m36 = (L.term == 36).to_numpy()
    miss = W_.isna().mean()
    pd.DataFrame([dict(column=c, 분류=DESC[c][0], 의미=DESC[c][1], 산식_출처=DESC[c][2], 단위=DESC[c][3], 확정창_결측률=round(miss[c], 4)) for c in D.columns]).to_csv(OUTD / "변수설명.csv", index=False, encoding="utf-8-sig")
    cand = [c for c in D.columns if DESC[c][0].startswith(("파생", "시장참고")) or c in ("int_rate", "installment", "loan_amnt", "annual_inc", "dti")]
    rows, dec = [], []
    for c in cand:
        x = W_[c].to_numpy(float); ok = ~np.isnan(x); a = roc_auc_score(W_.default[ok], x[ok])
        rho = spearmanr(x[ok], ret[ok]).statistic; sign = 1 if a >= 0.5 else -1                          # sign=+1: 값이 클수록 위험 → 낮은 값부터 승인
        pr = np.where(ok, sign * x, np.inf); pr = pr * 1.0
        sm, _ = K.sharpe_eval(L[m36], pr[m36]); sm0, _ = K.sharpe_eval(L[m36], np.random.RandomState(0).rand(m36.sum()))  # 36m 전체 발행연도
        rows.append(dict(feature=c, 분류=DESC[c][0], AUC_default=max(a, 1 - a), 부도와_관계="클수록 위험" if a >= 0.5 else "클수록 안전", spearman_수익률=rho, 결측률=1 - ok.mean(),
                         SR_r30=sm["SR_r30"], SR_r53=sm["SR_r53"], SR_r90=sm["SR_r90"], SR_best=sm["SR_best"], XS_r53=sm["XS_r53"], 무작위_SR_r53=sm0["SR_r53"]))
        q = pd.qcut(pd.Series(x[ok]).rank(method="first"), 10, labels=False)
        for k in range(10): dec.append(dict(feature=c, decile=k + 1, n=int((q == k).sum()), 평균값=float(x[ok][q.to_numpy() == k].mean()), 부도율=float(W_.default.to_numpy()[ok][q.to_numpy() == k].mean()), 평균수익률=float(ret[ok][q.to_numpy() == k].mean())))
    A = pd.DataFrame(rows).sort_values("AUC_default", ascending=False); A.to_csv(OUTD / "파생변수_분석.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    Dc = pd.DataFrame(dec); Dc.to_csv(OUTD / "파생변수_십분위.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    sp = ["int_rate", "spread_10y", "spread_30y", "spread_matched", "spread_fedfunds", "rate_premium_subgrade", "tsy10_at_issue", "tsy_matched_at_issue"]
    cr = W_[sp].corr(); cr.to_csv(OUTD / "스프레드_상관.csv", encoding="utf-8-sig", float_format="%.3f")
    # 그림: 상관 히트맵 + 십분위 부도율·수익률(스프레드류)
    fig, ax = plt.subplots(1, 3, figsize=(36, 11))
    im = ax[0].imshow(cr, vmin=-1, vmax=1, cmap="RdBu_r"); ax[0].set_xticks(range(len(sp))); ax[0].set_xticklabels(sp, rotation=60, ha="right"); ax[0].set_yticks(range(len(sp))); ax[0].set_yticklabels(sp)
    for i in range(len(sp)):
        for j in range(len(sp)): ax[0].text(j, i, f"{cr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=12)
    ax[0].set_title("금리·스프레드 상관(확정 대출 전체)")
    for c in ["int_rate", "spread_10y", "spread_matched", "spread_fedfunds", "rate_premium_subgrade"]:
        d = Dc[Dc.feature == c]; ax[1].plot(d.decile, d.부도율 * 100, "o-", lw=3, label=c); ax[2].plot(d.decile, d.평균수익률 * 100, "o-", lw=3, label=c)
    ax[1].set_title("십분위별 부도율(%)"); ax[2].set_title("십분위별 평균 수익률 NPV/대출액(%, 만기일치 국채 할인)")
    for x_ in ax[1:]: x_.set_xlabel("십분위(낮음→높음)"); x_.legend(); x_.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(K.FIG / "fig0_스프레드_분석.png", dpi=130); plt.close()
    print(A[["feature", "AUC_default", "부도와_관계", "spearman_수익률", "SR_r53", "무작위_SR_r53", "SR_best"]].round(3).to_string()); print(cr.round(3).to_string())
