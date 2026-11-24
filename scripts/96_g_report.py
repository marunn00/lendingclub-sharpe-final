"""96_g: 비교표(표_비교_이전vs새테이블.csv), 분석목록.csv, _work 표 → 결과 표 CSV 복사. 이전 값은 문서에서 인용(출처 열), 새 값은 _work/*.csv(새 테이블 계산)."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; C = importlib.import_module("96_common")
import numpy as np, pandas as pd
W, O = C.WORK, C.OUT
mk = pd.read_csv(W / "market.csv"); ci = pd.read_csv(W / "market_ci.csv"); br = pd.read_csv(W / "bridge.csv"); al = pd.read_csv(W / "alpha.csv"); sm = pd.read_csv(W / "strategy_summary.csv")
cu = pd.read_csv(W / "curves.csv"); hu = pd.read_csv(W / "hurdle.csv"); sg = pd.read_csv(W / "stress_grid.csv"); sb = pd.read_csv(W / "stress_breakeven.csv"); p60 = pd.read_csv(W / "pair60.csv"); vc = pd.read_csv(W / "vintage_curves.csv"); au = pd.read_csv(W / "auc.csv")
R = []
def add(a, m, old, src, new, unit, why, ci_txt="", prim=""):
    R.append(dict(분석=a, 지표=m, 이전값=old, 이전_출처=src, 새테이블값=new, 단위=unit, primary36_누수상한값=prim, 학습규칙=("strict36(학습 발행≤2014 & 확정월<2015-01)" if prim != "" or "strict" in a else "학습 없음(기술통계)"), 차이=(round(new - old, 4) if isinstance(old, (int, float)) and isinstance(new, (int, float)) and not (old != old) else ""), 새값_CI=ci_txt, 차이_이유=why))
f = lambda x: round(float(x), 4)
SN = "산출물/새지표_재분석/새지표_재분석.md"; V4 = "산출물/새테이블/새테이블_결과.md"
T = mk[(mk.grade == "전체")].merge(ci, on=["term", "year"]).set_index(["term", "year"])
OLD_E = {2012: 8.82, 2013: 9.73, 2014: 7.49, 2015: 4.82, 2016: 3.50}; OLD_R = {2012: 2.86, 2013: 3.15, 2014: 2.44, 2015: 1.59, 2016: 1.17}; OLD_N = {2012: 2.80, 2013: 3.03, 2014: 2.15, 2015: 0.95, 2016: 0.11}
for y in range(2012, 2017):
    r = T.loc[(36, y)]
    add("전량 매수 시장 기준선(36m)", f"{y} E_npv(원금당 초과 NPV)", OLD_E[y], SN + " §4", f(r.E_npv * 100), "%", "동일 정의·동일 현금흐름 재구성. 새 표 기준 확정 대출 n이 구 618,750보다 50건 적음(funded_amnt 결측/상태 필터 차)", f"[{r.E_npv_lo*100:.2f}, {r.E_npv_hi*100:.2f}]")
    add("전량 매수 시장 기준선(36m)", f"{y} R_reinv", OLD_R[y], SN + " §4", f(r.R_reinv * 100), "%p/년", "동일", f"[{r.R_reinv_lo*100:.2f}, {r.R_reinv_hi*100:.2f}] (3개월 블록)")
    add("전량 매수 시장 기준선(36m)", f"{y} R_noreinv", OLD_N[y], SN + " §4", f(r.R_noreinv * 100), "%p/년", "동일", f"[{r.R_noreinv_lo*100:.2f}, {r.R_noreinv_hi*100:.2f}]")
for y, (o1, o2, o3, o4) in {2014: (1.69, 1.63, 2.15, 2.44), 2015: (0.84, 0.75, 0.95, 1.59), 2016: (0.65, 0.54, 0.11, 1.17)}.items():
    b = br[br.year == y].iloc[0]
    for lab, o, c, why in [("① 구 합산(0.99·TP, 국채 발행시 고정, 재투자 없음)", o1, "old_style_0p99TP_locked_tsy", "구 값은 loan_amnt·policy_flag=0 기준, 새는 funded_amnt·전체 확정 대출로 재계산(0.00~0.03%p 차)"), ("② 같은 가정, 현금 정의 재계산", o2, "same_assump_cash_locked_tsy", "동일"), ("③ 기준을 실현 FEDFUNDS로 교체", o3, "R_noreinv", "동일"), ("④ 회수금 포함 재투자 R_reinv", o4, "R_reinv", "동일")]:
        add("다리표 구합산→R_reinv(36m)", f"{y} {lab}", o, SN + " §1.1", f(b[c] * 100), "%p/년", why)
OLDDG = {2012: (2.63, 3.92), 2013: (3.09, 3.47), 2014: (2.46, 2.34), 2015: (1.74, 0.76), 2016: (1.32, 0.37)}
for y, (ac, dg) in OLDDG.items():
    for g, o in (("A-C", ac), ("D-G", dg)):
        add("D–G vs A–C(36m R_reinv)", f"{y} {g}", o, SN + " §8.1", f(mk[(mk.term == 36) & (mk.year == y) & (mk.grade == g)].R_reinv.iloc[0] * 100), "%p/년", "동일")
for g, o in zip("ABCDEFG", [1.35, 1.46, 1.12, 0.76, -0.50, -1.62, -3.09]):
    add("2016 등급별 R_reinv(부호 반전 점검)", f"등급 {g}", o, SN + " §8.2", f(mk[(mk.term == 36) & (mk.year == 2016) & (mk.grade == g)].R_reinv.iloc[0] * 100), "%p/년", "동일")
add("2016 부호: 전체", "2016 R_reinv", 1.17, SN + " §8.2", f(T.loc[(36, 2016)].R_reinv * 100), "%p/년", "동일", f"[{T.loc[(36,2016)].R_reinv_lo*100:.2f}, {T.loc[(36,2016)].R_reinv_hi*100:.2f}]")
for i, r in p60.iterrows():
    old = 0.00 if i == 0 else 5.9
    add("60개월 vs 36개월(등급×연도 21짝, 36m 자본가중)", r.metric, old, SN + " §8.3", f(r.value), "%p", "짝지음 정의 동일, 월 블록 부트스트랩(구: 월 재표본, 신: 6개월 블록)", f"[{r.lo:.2f}, {r.hi:.2f}]")
# 순손실, 스트레스
for y, o in zip(range(2012, 2017), [7.19, 5.99, 6.70, 7.46, 8.73]): add("누적 순손실(발행원금 대비, 36m 평생)", str(y), o, V4 + " §2 NT01", f(T.loc[(36, y)].lifetime_net_loss * 100), "%", "순손실=원금−회수원금−회수금+회수수수료(사후 확정값). 구는 NT01 삼각표 K+6 평생값")
for y, o in zip(range(2012, 2015), [17.80, 15.29, 15.93]): add("누적 순손실(발행원금 대비, 60m 평생)", str(y), o, V4 + " §2 NT01", f(T.loc[(60, y)].lifetime_net_loss * 100), "%", "동일")
s0 = sg[sg.group == "전체 2012–16"].set_index(["pd_mult", "lgd_mult"]).excess_pct
add("스트레스(36m 전체 초과 NPV)", "기준 초과(PD×1)", 5.53, V4 + " NT08", f(s0[(1, 1)]), "%", "구는 정확식(rf_path), 새는 부도/비부도 평균 혼합식 — 닫힌 형태가 같아 거의 일치")
for m, o in [(1.25, 3.64), (1.5, 1.76), (2, -2.01), (3, -9.56)]: add("스트레스(36m 전체 초과 NPV)", f"PD×{m}", o, V4 + " NT08", f(s0[(m, 1)]), "%", "구 π'=min(mπ,1)·유형 혼합, 새는 부도 비중 금액가중 π 사용")
add("스트레스(36m 전체 초과 NPV)", "LGD×1.1(PD×1)", 4.8, V4 + " NT08", f(s0[(1, 1.1)]), "%", "LGD 정의 근사: 부도 대출 순손실(원금−회수)을 배수, 할인 생략")
bbe = sb.set_index("group")
add("스트레스 손익분기 PD 배수", "전체", 1.73, V4 + " NT08", f(bbe.loc["전체 2012–16"].breakeven_pd_mult), "배", "동일 구조")
for gname, o in zip("ABCDEFG", [3.16, 2.10, 1.53, 1.31, 1.10, 1.04, 0.88]): add("스트레스 손익분기 PD 배수", f"등급 {gname}", o, V4 + " NT08", f(bbe.loc[f"등급{gname}"].breakeven_pd_mult), "배", "동일 구조")
# 빈티지 곡선 타이밍
vv = vc[vc.term == 36]; sh = {}
for a in (12, 18, 24, 36):
    sh[a] = np.mean([g.cum_writeoff_per_A.iloc[a] / g.cum_writeoff_per_A.iloc[-1] for _, g in vv.groupby("year")])
for a, o in zip((12, 18, 24, 36), (20.2, 50.4, 75.0, 98.0)): add("손실 시점 곡선(36m, 평생 대비 누적 비중)", f"나이 {a}", o, V4 + " NT04", f(sh[a] * 100), "%", "구=순손실(상각−순회수), 새=원금 상각(상각월 K+6) 비중. 회수는 같은 달 유입이라 타이밍이 비슷")
# ---- ML 의존 항목: 새테이블값 = strict36, primary36 = 누수 상한 열
SM = lambda sp, sn, st, u="월": sm[(sm.split == sp) & (sm.subset == sn) & (sm.strategy == st) & (sm.rank_unit == u)].iloc[0]
HU = lambda sp, st, u="월": hu[(hu.split == sp) & (hu.subset == "헤드라인2015-16") & (hu.strategy == st) & (hu.rank_unit == u)].pivot(index="strategy", columns="h_pp", values="best_a").iloc[0]
for st_, tag, hs in [("reg|raw+nv전체", "NPV 직접", [(0, 100), (1, 87), (2, 2), (3, 1), (4, 1)]), ("clf|raw+nv전체", "부도확률", [(0, 100), (1, 96), (2, 1)])]:
    hs_, hp_ = HU("strict36", st_), HU("primary36", st_)
    for h, o in hs: add(f"허들 민감도: 이익 최대 승인율({tag}, 월 안 순위)", f"h=FEDFUNDS+{h}%p", o, SN + " §6.3", f(hs_[h] * 100), "%", "새: 학습 strict36·특성 raw+nv_ 144개·월별 순위. 구는 새변수 30개·셀 안 순위", prim=f(hp_[h] * 100))
ho, hop = HU("strict36", "오라클(실현NPV)"), HU("primary36", "오라클(실현NPV)")
for h, o in [(0, 79), (1, 76), (2, 71), (3, 63), (4, 52)]: add("허들 민감도: 오라클 최적 승인율", f"h={h}%p", o, SN + " §6.3", f(ho[h] * 100), "%", "구는 셀 안 순위, 새는 발행월 안 순위(금액 누적). 오라클은 학습 무관", prim=f(hop[h] * 100))
sub = pd.read_parquet(W / "loans.parquet"); s1516 = sub[(sub.term == 36) & sub.issue_year.between(2015, 2016)]; E1516 = s1516.NPV_short.sum() / s1516.A.sum()
add("손익분기 허들(전량 매수 평균 NPV = 허들)", "2015–16", 1.35, SN + " §6.3", f(((1 + E1516) ** (12 / 36) - 1) * 100), "%p/년", "새 = (1+E_npv)^(1/3)−1")
for y, o in ((2015, 1.58), (2016, 1.15)):
    s_ = sub[(sub.term == 36) & (sub.issue_year == y)]; add("손익분기 허들(전량 매수 평균 NPV = 허들)", str(y), o, SN + " §6.3", f(((1 + s_.NPV_short.sum() / s_.A.sum()) ** (1 / 3) - 1) * 100), "%p/년", "동일")
A = al[(al.split == "strict36") & (al.subset == "헤드라인2015-16")].set_index("model"); P = al[(al.split == "primary36") & (al.subset == "헤드라인2015-16")].set_index("model")
def alr(t, m): r = t.loc[m]; return f(r.alpha_invested_pp), f"[{r.inv_lo:.2f}, {r.inv_hi:.2f}]"
WHY_A = "새 모형은 raw 88열(+nv_)·LightGBM, 학습 n=32,120(구 7,986~32,120). primary36 열은 학습에 2015년 이후 확정 정보 포함(누수 상한)"
for lab, old, mdl in [("부도확률 (구:새 변수 / 신: nv만)", 0.14, "clf|nv만"), ("NPV 직접 (구:새 변수 / 신: nv만)", 0.37, "reg|nv만"), ("부도확률 (구:+원 PIT / 신: raw+nv전체)", 0.22, "clf|raw+nv전체"), ("NPV 직접 (구:+원 PIT / 신: raw+nv전체)", 0.44, "reg|raw+nv전체")]:
    v, c_ = alr(A, mdl); add("같은 셀 상위30% 알파(헤드라인 2015–16, 투입 자본당)", lab, old, SN + " §5.2 표S5", v, "%p/년", WHY_A, c_, prim=alr(P, mdl)[0])
for mdl in ("reg|raw+nv전체 - clf|raw+nv전체", "reg|raw141 - clf|raw141"):
    k = mdl.replace(" - ", " " + chr(0x2212) + " "); r = A.loc[k]; add("쌍 차이 NPV−부도 알파(헤드라인)", mdl, 0.22, SN + " §5.2(+원 PIT)", f(r.alpha_invested_pp), "%p/년", "구 0.22[0.05,0.34](+원 PIT)", f"[{r.inv_lo:.2f}, {r.inv_hi:.2f}]", prim=f(P.loc[k].alpha_invested_pp))
for k, lab in [("reg|raw+nv전체 " + chr(0x2212) + " reg|raw141", "NPV: raw+nv − raw141"), ("clf|raw+nv전체 " + chr(0x2212) + " clf|raw141", "부도: raw+nv − raw141"), ("reg|raw+채택 " + chr(0x2212) + " reg|raw141", "NPV: raw+채택 − raw141"), ("reg|raw+채택+FLAG " + chr(0x2212) + " reg|raw141", "NPV: raw+채택+FLAG − raw141")]:
    r = A.loc[k]; add("nv_ 증분 알파(raw 대비)", lab, 0.07 if "raw+nv" in lab and lab.startswith("NPV") else (0.01 if "채택 −" in lab else np.nan), SN + " §5.2/§7.2 (정의 상이: 구는 '원PIT−새변수' / C07 Δ)", f(r.alpha_invested_pp), "%p/년", "구는 원PIT의 가치, 새는 nv_의 가치(방향 반대)", f"[{r.inv_lo:.2f}, {r.inv_hi:.2f}]", prim=f(P.loc[k].alpha_invested_pp))
for sn_ in ("헤드라인2015-16", "2016"):
    for mdl, lab, oR30, oR53, od, odci in [("reg|raw+nv전체", "NPV 직접(+원PIT)", 0.66, 0.99, -0.33, "[-0.38,-0.28]"), ("clf|raw+nv전체", "부도확률", 0.43, 0.77, -0.34, "[-0.39,-0.29]")]:
        r = SM("strict36", sn_, mdl); pr_ = SM("primary36", "헤드라인2015-16", mdl) if sn_.startswith("헤드") else None; nn = lambda c: f(pr_[c] * 100) if pr_ is not None else ""
        og = lambda x: x if sn_ != "2016" else np.nan
        add(f"승인 30% vs 53% 이익({sn_}, 전체 자본 R_total, 월 안 순위)", f"{lab} R30", og(oR30), SN + " §6.5", f(r.R30 * 100), "%p/년", "순위 단위(월 vs 셀)·학습표본 차", prim=nn("R30"))
        add(f"승인 30% vs 53% 이익({sn_}, 전체 자본 R_total, 월 안 순위)", f"{lab} R53", og(oR53), SN + " §6.5", f(r.R53 * 100), "%p/년", "동일", prim=nn("R53"))
        add(f"승인 30% vs 53% 이익({sn_}, 전체 자본 R_total, 월 안 순위)", f"{lab} 차(30−53)", og(od), SN + " §6.5 " + odci, f(r.diff30_53 * 100), "%p/년", "부호 동일(53%가 더 큼)", f"[{r.diff30_53_lo*100:.2f}, {r.diff30_53_hi*100:.2f}]", prim=nn("diff30_53"))
        for u in ("월", "셀"):
            rr = SM("strict36", sn_, mdl, u); pq = SM("primary36", "헤드라인2015-16", mdl, u) if sn_.startswith("헤드") else None
            add(f"월 빈티지 Sharpe 30%/53% ({sn_}, mean/sd, 연환산 아님)", f"{lab} / {u} 안 순위", ("7.07/7.74" if mdl.startswith("reg") else "4.52/4.81") if sn_ != "2016" else "", SN + " §5.3(구: 셀 안 순위)", f"{rr.SR30:.2f}/{rr.SR53:.2f}", "무차원(월 빈티지 초과수익 평균/표준편차)", "구 정의는 셀 안 순위", f"SR30 [{rr.SR30_lo:.2f}, {rr.SR30_hi:.2f}]; SR53 [{rr.SR53_lo:.2f}, {rr.SR53_hi:.2f}]; 차 [{rr.SR30_minus_SR53_lo:.2f}, {rr.SR30_minus_SR53_hi:.2f}]", prim=(f"{pq.SR30:.2f}/{pq.SR53:.2f}" if pq is not None else ""))
for mdl, lab, old in [("reg|raw+nv전체", "NPV 모형", "15–82%"), ("clf|raw+nv전체", "부도 모형", "미보고")]:
    for u in ("월", "셀"):
        r = SM("strict36", "헤드라인2015-16", mdl, u); pq = SM("primary36", "헤드라인2015-16", mdl, u)
        add("월 빈티지 Sharpe 최대 승인율", f"{lab} / {u} 안 순위 argmax [95% CI]", old, SN + " §6.5", f"{r.argmax_sharpe*100:.0f}% [{r.argmax_sharpe_lo*100:.0f}, {r.argmax_sharpe_hi*100:.0f}]", "%", "구간이 넓으면 단일 최적 승인율 주장 불가", prim=f"{pq.argmax_sharpe*100:.0f}% [{pq.argmax_sharpe_lo*100:.0f}, {pq.argmax_sharpe_hi*100:.0f}]")
r = SM("strict36", "헤드라인2015-16", "clf|raw+nv전체"); pq = SM("primary36", "헤드라인2015-16", "clf|raw+nv전체")
for lab, old, c_, u, sc in [("Sharpe 최대 승인율", 30.3, "old_argmax_SR", "%", 100), ("Sharpe 최대 시 Sharpe", 0.138, "old_SR_max", "무차원", 1), ("Sharpe 최대 시 초과수익", 0.53, "old_excess_at_SRmax", "%p/년", 100), ("수익금 최대 승인율", 55.8, "old_argmax_excess", "%", 100)]:
    add("구방식 프런티어(횡단 Sharpe = 초과수익/횡단 sd, 국채 DGS3/DGS5 기준, 부도 모형 raw+nv)", lab, old, "산출물/운용점_탐색/요약_표본1_noyear_to2016.json", f(r[c_] * sc), u, "새: 시험 2015–16 36m·strict36; 구: 전 기간 무작위 분할·HGB. 안전자산=국채(월 빈티지 Sharpe의 FEDFUNDS와 다름)", prim=f(pq[c_] * sc))
a0 = au[au.split == "strict36"].set_index("model"); a1 = au[au.split == "primary36"].set_index("model")
for k, lab in [("clf|raw141", "LightGBM raw141"), ("clf|raw+채택", "LightGBM raw+채택"), ("clf|raw+채택+FLAG", "LightGBM raw+채택+FLAG"), ("clf|raw+nv전체", "LightGBM raw+nv"), ("clf|nv만", "LightGBM nv만"), ("logit|raw141", "로지스틱 raw141"), ("logit|raw+nv전체", "로지스틱 raw+nv")]:
    add("부도 판별력 AUC(시험 2015–16)", lab, np.nan, "이전 보고 수치 미확인", f(a0.loc[k].auc), "AUC", "비교할 이전 보고값 없음(신규 산출)", f"[{a0.loc[k].lo:.3f}, {a0.loc[k].hi:.3f}]", prim=f(a1.loc[k].auc))
cmp_ = pd.DataFrame(R); cmp_.to_csv(O / "표_비교_이전vs새테이블.csv", index=False, encoding="utf-8-sig")
# 분석목록
AL = [
 ("83_ir_v2 fig01 시점별 변수분류(원열 141 단계 배정)", "막대/표", "원열 141: t_initial 97 등", "scripts/83_ir_v2.py, fig_manifest.csv", "부분 재현(F17 열 분류 막대: 상장시점 사용 가능 vs 사후 열 제외)"),
 ("83_ir_v2 fig02 정보집합(사후정보 누수 시연)", "막대", "가짜 정확도 시연", "fig_manifest.csv", "미재현: 20k 표본 시연·사후열 누수 시연은 새 표 입력 범위 밖(별도 요청 시 가능)"),
 ("83_ir_v2 fig03~fig05 외부변수·수치변수 EDA·전처리 전후", "히스토그램/상관", "외부변수 상관행렬 등", "산출물/기술IR_v2/fig/*", "미재현(범위 제외): 외부변수·수치 EDA는 새 표의 nv_mac_* 로 대체 가능하나 이번 단계 비교 대상 아님"),
 ("83_ir_v2 fig06~fig10 거절 처리·승인율 추이·거부요인·승인/거절 프로필", "흐름도/추세/프로필", "거절 데이터 기반", "fig_manifest.csv", "재현 불가: 거절 파일은 허용 입력(새 표) 밖"),
 ("87_pooled_alpha 합산 방식 알파(0.60/0.53/0.52, 2014/15/16)", "표", "구 규칙(76번 P 사전부호), 비재투자·국채", "scripts/87_pooled_alpha.py", "재현(다른 규칙): 같은 셀 상위30% 합산 알파를 모형 5종×특성집합 5종으로 재계산(표_알파.csv, F05)"),
 ("89 새 지표 R_reinv/E_npv/R_noreinv (빈티지×만기, 등급)", "표/막대", "2012–16: 2.86/3.15/2.44/1.59/1.17", SN + " §4", "재현(수치 일치)"),
 ("89 다리표 구합산→R_reinv", "막대", "2014–16 ①~④", SN + " §1.1", "재현(수치 일치) F14"),
 ("89 운용점: 승인율 곡선·프런티어·1SE", "선/프런티어", "Sharpe최대 30.3%, 수익금최대 55.8%", "산출물/운용점_탐색/*", "재현(방식 상이): 월 빈티지 Sharpe + 구방식 횡단 Sharpe 둘 다. F01 F02 F03"),
 ("89 선택 모형 연도별 알파(표S1)·알파 CI(표S5)", "forest", "NPV 새변수 +0.23/+0.49 등", SN + " §5", "재현(수치 차): F05 F06"),
 ("89 허들 민감도/손익분기(표F3)", "선/표", "손익분기 1.35%p; h=2에서 ~2%", SN + " §6.3", "재현 F11"),
 ("89 30 vs 53(표F2)", "forest", "차 −0.32 [−0.36,−0.28]", SN + " §6.5", "재현 F13"),
 ("89 변수 재판정(표V1)", "forest", "C04·C05·C07 채택(새 기준선)", SN + " §7", "부분 재현: 변수별 사전규칙 Holm 판정은 미재현. 대신 raw vs raw+채택/FLAG/전체/nv만 알파·AUC 비교(F06 F15)"),
 ("89 D–G vs A–C / 60m vs 36m / 2016 등급 (표E1~E3)", "선/막대", "표E1~E3b", SN + " §8", "재현(수치 일치) F14 + pair60"),
 ("89 NT10 공시축소(2014-11) 전후", "표", "연관성", SN + " §9", "재현 불가: 시험 월이 2014-01~2016-12이며 2014 학습표본 부족 — 새 분할(시험 2015–16)에 경계(2014-11)가 포함되지 않음"),
 ("92_ir_v4 F01 2016 다리", "막대", "0.65→1.17", "scripts/92_ir_v4_figs.py", "재현 F14"),
 ("92 F02 귀속 워터폴(NT06)", "워터폴", "이자 19.42−조기 3.16−…=초과 5.64", V4 + " NT06", "재현 불가: 이자·손실 항목별 귀속 정의가 별도 코드(91_nt_e); 총합 초과(E_npv)는 재현"),
 ("92 F03 손실 삼각표(NT01)", "히트맵", "평생 순손실 7.19/5.99/6.70/7.46/8.73", V4 + " NT01", "부분 재현: 평생 순손실 수치 + 빈티지 누적곡선(F07); 분기×나이 삼각표 미재현"),
 ("92 F04 손실 시점 곡선(NT04)", "선", "나이 12/18/24/36: 20.2/50.4/75.0/98.0%", V4 + " NT04", "재현(원금 상각 기준) F07 + 비교표"),
 ("92 F05 원인별 위험률(NT02)", "막대", "부도 A 4.9% … G 44.1%", V4 + " NT02", "부분 재현: 등급별 부도율(F08); 조기상환·경쟁위험 분해는 월 상태가 없어 재현 불가"),
 ("92 F06 스트레스 손익분기(NT08)", "격자/막대", "기준 5.53, 손익분기 1.73", V4 + " NT08", "재현 F12"),
 ("92 F07 알파 CI / F08 허들 frontier / F09 변수 재판정 / F10 D–G vs A–C", "—", "위와 동일", "scripts/92_ir_v4_figs.py", "F05 / F11 / F06,F15 / F14"),
 ("신규(요청): 보정·ROC·리프트, SHAP, AUC 비교", "보정/ROC/리프트/SHAP", "이전 보고값 없음", "—", "신규 F09 F10 F15"),
]
ALD = pd.DataFrame(AL, columns=["이전_분석", "그림_유형", "이전_수치", "출처", "새테이블_재현_상태"])
ML = ("알파", "운용점", "선택 모형", "허들", "30 vs 53", "변수 재판정", "신규", "F07 알파")
ALD["학습규칙"] = ["strict36 헤드라인 + primary36 누수상한 병기" if any(k in r for k in ML) else "학습 없음(기술통계)" for r in ALD.이전_분석]
ALD["재현_상태_갱신"] = ["strict36 재실행(F01~F06, F09~F11, F13, F15, F18, F19 갱신); 학습 없는 항목은 영향 없음" if k.startswith("strict") else "" for k in ALD.학습규칙.str[:6]]
ALD.loc[len(ALD)] = ["신규(리뷰 반영): 누수 시연 primary36 vs strict36", "막대", "—", "—", "신규 F18", "primary36=상한, strict36=헤드라인", ""]
ALD.loc[len(ALD)] = ["신규(리뷰 반영): 순위 단위(월 vs 월×세부등급 셀) Sharpe 순서 점검 + 선택 구성", "선/막대", "구 NPV 7.07/7.74 > 부도 4.52/4.81", "새지표_재분석.md §5.3", "신규 F19, 표_선택구성_월vs셀.csv", "strict36", ""]
ALD.to_csv(O / "분석목록.csv", index=False, encoding="utf-8-sig")
# 결과 표 복사 (단위 통일: 수익률 열은 %p, 승인율은 %; 이름에 단위 표기)
def conv(df, pp=(), pct=()):
    df = df.copy()
    for c in pp:
        if c in df: df[c] = df[c] * 100; df = df.rename(columns={c: c + "_pp"})
    for c in pct:
        if c in df: df[c] = df[c] * 100; df = df.rename(columns={c: c + "_pct"})
    return df
RC = ["R30", "R53", "R100", "diff30_53", "diff30_53_lo", "diff30_53_hi", "old_excess_at_SRmax"]; AC = ["argmax_sharpe", "argmax_sharpe_lo", "argmax_sharpe_hi", "argmax_excess", "argmax_excess_lo", "argmax_excess_hi", "old_argmax_SR", "old_argmax_excess"]
conv(sm, RC, AC).to_csv(O / "표_전략요약.csv", index=False, encoding="utf-8-sig")
conv(cu, ["excess_pooled", "excess_lo", "excess_hi", "e_mean", "e_sd", "E_npv_total", "enpv_lo", "enpv_hi", "old_excess", "old_sd"], ["a"]).to_csv(O / "표_승인율곡선.csv", index=False, encoding="utf-8-sig")
conv(hu, [], ["best_a"]).to_csv(O / "표_허들.csv", index=False, encoding="utf-8-sig")
conv(pd.read_csv(W / "market.csv"), ["E_npv", "R_reinv", "R_noreinv", "safe_ann", "R_noreinv_locked_tsy", "E_npv_tsy", "lifetime_net_loss", "ret_total_pymnt_ann", "default_rate", "default_rate_amt"]).to_csv(O / "표_빈티지x등급.csv", index=False, encoding="utf-8-sig")
conv(pd.read_csv(W / "market_ci.csv"), [c for c in pd.read_csv(W / "market_ci.csv").columns if c.endswith(("_lo", "_hi"))]).to_csv(O / "표_기준선_CI.csv", index=False, encoding="utf-8-sig")
conv(pd.read_csv(W / "bridge.csv"), [c for c in pd.read_csv(W / "bridge.csv").columns if c != "year"]).to_csv(O / "표_다리표.csv", index=False, encoding="utf-8-sig")
for src, dst in [("alpha.csv", "표_알파.csv"), ("stress_grid.csv", "표_스트레스.csv"), ("stress_breakeven.csv", "표_스트레스_손익분기.csv"), ("auc.csv", "표_AUC_보정.csv"), ("model_metrics.csv", "표_모형_AUC전체.csv"), ("importance.csv", "표_중요도_SHAP.csv"), ("calibration.csv", "표_보정.csv"), ("lift.csv", "표_리프트.csv"), ("selection_composition.csv", "표_선택구성_월vs셀.csv")]:
    pd.read_csv(W / src).to_csv(O / dst, index=False, encoding="utf-8-sig")
print(len(cmp_), "rows")
