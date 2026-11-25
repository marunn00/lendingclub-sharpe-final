"""논문형 3장 3.1 예측 오차의 축소, 3.2 분류와 결정 경계. 저장된 소표본 실험 결과 표에서 수치를 읽어 본문과 표를 만든다."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def f(x, n=4):
    return f"{x:.{n}f}"


def sg(x, n=4):
    return f"{x:+.{n}f}".replace("-", "−")


def pc(x, n=1):
    return f"{x * 100:.{n}f}%"


def rd(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def build(P):
    T = P.tables
    LAD = rd(T / "랩_회귀사다리.csv")
    RD = rd(T / "랩_회귀반복.csv")
    CLS = rd(T / "랩_분류.csv")
    CLR = rd(T / "랩_분류반복.csv")
    MB = rd(T / "랩_마진구간.csv")
    RDG = rd(T / "랩_잔차진단_시드0.csv")
    PCV = rd(T / "랩_p값_선택변수.csv")
    LJ = json.loads((T / "랩_요약.json").read_text(encoding="utf-8"))
    ALT = rd(P.root / "산출물" / "대안모델" / "_work" / "results.csv").set_index("id")

    # ---- 회귀: 아이디어 사다리
    STEP = {int(r.step): r for r in LAD.itertuples()}
    PCOL = RD.groupby("step").p.median()
    IDEA = {0: "평균만(기준선)", 1: "가격·약정 변수만(OLS)", 2: "+신용이력", 3: "+재무", 4: "+시장·시기 = 전체 변수 OLS",
            5: "Ridge(L2 규제)", 6: "Lasso(L1 규제)", 7: "2단계: 부도확률 × 조건부 이득", 8: "비선형: HistGB 1단계",
            9: "v3-lite: HistGB 2단계 + 비율 순위"}
    BASE = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4}
    PH = RD.pivot(index="seed", columns="step", values="rmse_hold")
    PS = RD.pivot(index="seed", columns="step", values="SR26")

    def verdict(a, b):
        dr = PH[a] - PH[b]
        ds = PS[a] - PS[b]
        lo = float((dr < 0).mean())
        m = float(dr.median())
        tag = "설명력↑" if (m <= -0.0005 and lo >= 0.75) else ("악화" if (m >= 0.0005 and lo <= 0.25) else "차이 없음")
        return tag, m, lo, float(ds.median()), float((ds > 0).mean())

    VD = {s: verdict(s, BASE[s]) for s in BASE}
    GAP = {s: STEP[s].과적합간격 for s in STEP}
    imp = [s for s in VD if VD[s][0] == "설명력↑"]
    non = [s for s in VD if VD[s][0] != "설명력↑"]
    NLZ = float(PCV.lasso_nz.median())
    NP = int(PCV.p_total.median())
    N05 = int(PCV.p05.median())
    NBF = int(PCV.p_bonf.median())
    AR = float(PCV.ridge_alpha.median())
    AL = float(PCV.lasso_alpha.median())
    N_RMIN = int((PCV.ridge_alpha <= 0.0100001).sum())

    # ---- 분류
    names = list(CLS.model)
    M_NONE, M_L2, M_L1, M_SV, M_SW, M_LZ = names

    def cpair(a, b, col):
        x = CLR[CLR.model == a].set_index("seed")[col]
        y = CLR[CLR.model == b].set_index("seed")[col]
        dd = x - y
        return float(dd.median()), float((dd > 0).mean())

    d_l2 = cpair(M_L2, M_NONE, "auc_hold")
    d_l1 = cpair(M_L1, M_L2, "auc_hold")
    d_tgt = cpair(M_LZ, M_L2, "SR26")
    d_sv = cpair(M_SV, M_LZ, "SR26")
    d_sw = cpair(M_SW, M_LZ, "SR26")
    d_swv = cpair(M_SW, M_SV, "SR26")
    d_sv_sign = cpair(M_SV, M_LZ, "auc_sign_hold")

    # ---- 임계값 거리
    R = MB[MB.측.str.startswith("거절")].set_index("|d|")
    A = MB[MB.측.str.startswith("승인")].set_index("|d|")
    NEAR, FARB = ["0~0.25", "0.25~0.5"], ["1~2", "2 이상"]

    def wacc(df):
        return float((df.대출 * df.맞음비율).sum() / df.대출.sum())

    NR = wacc(R.loc[NEAR]); FR = wacc(R.loc[FARB]); NA = wacc(A.loc[NEAR]); FA = wacc(A.loc[FARB])
    BAND = float(MB[MB["|d|"].isin(NEAR)].비중.sum())

    # ---- 표
    ladder_rows = []
    for s in range(10):
        r = STEP[s]
        p_ = "—" if np.isnan(PCOL[s]) else f"{PCOL[s]:.0f}"
        if s == 0:
            mz, verd = "—", "기준선"
        else:
            tag, m, lo, _, _ = VD[s]
            mz = f(r.mz_hold, 2)
            verd = f"<b>{tag}</b><br><small>기준 {BASE[s]}단계 대비 {sg(m, 4)}, {lo * 100:.0f}%의 반복에서 낮음</small>"
        ladder_rows.append(([str(s), IDEA[s], p_, f(r.rmse_train), f(r.rmse_valid), f(r.rmse_hold), f(GAP[s]),
                             f(r.r2_hold, 3), mz, f"{f(r.SR26, 3)} [{f(r.SR26_q25, 3)}, {f(r.SR26_q75, 3)}]", verd],
                            "hl" if (s > 0 and VD[s][0] == "설명력↑") else ""))

    tag5, m5, lo5, _, _ = VD[5]
    tag6, m6, lo6, _, _ = VD[6]
    reg_rows = [
        ["Ridge", f"쓰지 않음({tag5})", f"α* 중앙값 {AR:.2f}; 20회 중 {N_RMIN}회에서 격자 최솟값 0.01 선택",
         f"{sg(m5, 5)} ({lo5 * 100:.0f}%의 반복에서 낮음)", "변수 수가 표본 크기에 가까워지거나 비슷한 변수가 많은 경우"],
        ["Lasso", f"쓰지 않음({tag6})", f"α* 중앙값 {AL:.1e}; 비영 계수 중앙값 {NLZ:.1f}개 / {NP}개",
         f"{sg(m6, 5)} ({lo6 * 100:.0f}%의 반복에서 낮음)", "변수 선택 결과를 보고하는 것이 목적인 경우"],
        [f"OLS(전체 {NP}개 변수)", "회귀 기준선으로 사용", f"p &lt; 0.05 중앙값 {N05}개; Bonferroni 기준 통과 중앙값 {NBF}개",
         "—", "계수의 부호와 크기를 해석해야 하는 경우"],
        ["트리 계열(전체 표본)", "본 모형에 사용", "소표본에서는 이득이 없으나 전체 표본에서 AUC가 높음(3.1 (4))", "—",
         "표본이 작아지면 일반화 간격을 먼저 확인"],
    ]

    res_rows = []
    for r in RDG.itertuples():
        res_rows.append([r.집단, r.모형, sg(r.학습_평균잔차), f(r.학습_잔차SD), sg(r.점검_평균잔차), f(r.점검_잔차SD), f"{r.대출_점검:,}"])
    rr0 = RDG.set_index(["집단", "모형"])
    sdr = float(rr0.loc[("부도", "1단계 Ridge"), "학습_잔차SD"] / rr0.loc[("완납", "1단계 Ridge"), "학습_잔차SD"])
    mean_full = float(rr0.loc[("완납", "1단계 Ridge"), "학습_평균잔차"])
    mean_def = float(rr0.loc[("부도", "1단계 Ridge"), "학습_평균잔차"])
    sd_full = float(rr0.loc[("완납", "1단계 Ridge"), "학습_잔차SD"])
    sd_def = float(rr0.loc[("부도", "1단계 Ridge"), "학습_잔차SD"])
    m2_full = float(rr0.loc[("완납", "2단계(부도확률×조건부)"), "학습_평균잔차"])
    m2_def = float(rr0.loc[("부도", "2단계(부도확률×조건부)"), "학습_평균잔차"])

    cls_rows = []
    cls_names = ["로지스틱(정규화 없음)", "로지스틱 L2(검증 선택)", "로지스틱 L1(검증 선택)", "선형 SVM 소프트마진(무가중)",
                 "선형 SVM 소프트마진(|z| 가중)", "로지스틱 L2, z&gt;0 목표(비교 기준)"]
    for i, r in enumerate(CLS.itertuples()):
        cval = "—" if r.C >= 1e5 else f"{r.C:.3g}"
        cls_rows.append(([cls_names[i], cval, f(r.auc_valid), f(r.auc_hold), f(r.auc_sign_hold), f(r.SR26, 3), f"{r.nonzero:.0f}"],
                         "hl" if i in (3, 4) else ""))

    mr_rows = []
    for side, df in (("거절측(τ 미만)", R), ("승인측(τ 이상)", A)):
        for b in df.index:
            r = df.loc[b]
            mr_rows.append([side, b, f"{r.대출:,}", pc(r.비중), pc(r.맞음비율), sg(r.평균z, 3), f(r.평균절대z, 3), pc(r.부도율)])

    def alt(k):
        return float(ALT.loc[k, "AUC"])

    # ---- 본문
    out = []
    out.append(P.h3("3.1 예측 오차의 축소"))
    out.append(P.p("수익률 예측의 목적은 대출 <i>i</i>의 초과수익 <i>z<sub>i</sub></i> = <i>r<sub>i</sub></i> − <i>r</i><sub>f,<i>i</i></sub>를 "
                   "가능한 한 정확히 맞히는 데 있다. 여기서 <i>r<sub>i</sub></i>는 대출의 수익, <i>r</i><sub>f,<i>i</i></sub>는 같은 발급월·만기의 국채 수익이다. "
                   "예측값 <i>ẑ<sub>i</sub></i>와의 차이 <i>e<sub>i</sub></i> = <i>z<sub>i</sub></i> − <i>ẑ<sub>i</sub></i>를 잔차라 하고, "
                   "학습 구간·검증 구간·점검 구간의 평균제곱근오차(RMSE)를 각각 학습 RMSE, 검증 RMSE, 점검 RMSE로 부른다."))
    out.append(P.p("학습 RMSE는 모형이 복잡해질수록 기계적으로 줄어들기 때문에, 그 감소만으로 설명력이 높아졌다고 보지 않았다. "
                   "같은 단계에서 검증 RMSE와 점검 RMSE가 함께 낮아지고 일반화 간격(검증 RMSE − 학습 RMSE)이 커지지 않을 때에만 설명력이 높아졌다고 판단하였다. "
                   "불편성은 Mincer–Zarnowitz 회귀 <i>z</i> = <i>α</i> + <i>β ẑ</i> + <i>u</i>의 기울기 <i>β</i>가 1에 가까운지로 보았고, "
                   "승인 규칙의 가치는 점수 상위 26%를 승인했을 때 점검 구간의 Sharpe 비율(SR26)로 함께 기록하였다."))

    out.append(P.h4("(1) 아이디어 사다리"))
    out.append(P.p(f"[[t:pr_ladder]]는 평균만 쓰는 기준선(0단계)에서 출발하여 가격·약정, 신용이력, 재무, 시장·시기 변수를 차례로 더한 과정(1~4단계), "
                   "정규화 회귀(5~6단계), 2단계 구조(7단계), 비선형 부스팅(8단계), 비율 순위를 결합한 부스팅(9단계)을 하나씩 적용한 10단계의 결과이다. "
                   "1~4단계는 직전 단계와, 5~9단계는 4단계의 전체 OLS와 같은 표본 쌍에서 비교하였다. "
                   "쌍체 판정 기준은 다음과 같다. 점검 RMSE의 반복별 차이 중앙값이 −0.0005 이하이고 20회 중 15회 이상에서 기준 단계보다 낮으면 「설명력↑」, "
                   "중앙값이 +0.0005 이상이고 20회 중 5회 이하에서만 낮으면 「악화」, 그 밖의 경우는 「차이 없음」으로 판정하였다."))
    out.append(P.tab("pr_ladder", "아이디어 사다리 10단계의 학습·검증·점검 RMSE와 쌍체 판정(20회 반복 중앙값)",
                     ["단계", "아이디어", "p", "학습 RMSE", "검증 RMSE", "점검 RMSE", "간격", "점검 R²", "MZ β", "SR26 [IQR]", "판정"],
                     ladder_rows, widths=["5%", "20%", "4%", "7%", "7%", "7%", "6%", "6%", "5%", "14%", "19%"],
                     foot="p = 변수 수(Lasso는 비영 계수 수의 중앙값). 간격 = 검증 RMSE − 학습 RMSE. SR26의 [ ]는 20회의 사분위 범위."))
    out.append(P.p(f"판정 결과 설명력이 높아진 단계는 가격·약정 변수 추가(1단계), 신용이력 추가(2단계), 시장·시기 변수를 더한 전체 OLS(4단계)이다. "
                   f"점검 RMSE는 평균만 쓸 때 {f(STEP[0].rmse_hold)}에서 전체 OLS의 {f(STEP[4].rmse_hold)}로 낮아졌고, 점검 R²는 {f(STEP[4].r2_hold, 3)}이었다. "
                   f"반면 재무 변수를 더한 3단계는 점검 RMSE를 줄이지 못하였다(차이 {sg(VD[3][1], 5)}). 같은 단계에서 일반화 간격은 "
                   f"{f(GAP[2])}에서 {f(GAP[3])}로 커져, 재무 정보가 신용이력에 이미 담긴 정보와 상당 부분 겹치는 것으로 해석된다(추정).", ni=False))
    out.append(P.fig("pr_ladderfig", "lab2_잔차사다리.png",
                     "단계별 학습 RMSE와 점검 RMSE(20회 중앙값). 두 막대의 간격이 일반화 간격에 해당하며, 단계 번호는 [[t:pr_ladder]]와 같다."))
    out.append(P.p(f"비선형 1단계(HistGB)는 학습 RMSE가 {f(STEP[8].rmse_train)}로 가장 낮지만 간격이 {f(GAP[8])}에 이르러 전체 OLS 간격의 약 "
                   f"{GAP[8] / GAP[4]:.0f}배이고, 점검 RMSE 차이는 기준을 넘지 못하였다. 학습 오차를 줄이는 것과 예측 설명력을 높이는 것이 다르다는 점을 보여준다. "
                   f"결정 가치인 SR26은 가격·약정만 쓸 때 {f(STEP[1].SR26, 3)}, 신용이력을 더하면 {f(STEP[2].SR26, 3)}, 전체 OLS에서 {f(STEP[4].SR26, 3)}로 단계마다 올랐다. "
                   "평균 오차의 감소 폭보다 승인 순위의 개선 폭이 상대적으로 커서, 평균 오차가 작게 줄어들어도 승인 품질은 크게 달라질 수 있다(추정)."))

    out.append(P.h4("(2) 정규화 회귀의 사용 여부"))
    out.append(P.p(f"OLS는 {NP}개 변수(결측 표지 포함)를 모두 쓰며, 학습 구간 {LJ['학습']:,}건에서 변수 수와 표본 수의 비 <i>p</i>/<i>n</i>은 약 {NP / LJ['학습'] * 100:.2f}%이다. "
                   "Ridge는 계수의 제곱합에, Lasso는 계수 절댓값의 합에 벌점 α를 주며, α는 검증 RMSE를 최소로 하는 값을 골랐다. 두 모형의 목적 함수는 다음과 같다."))
    out.append(P.eq("min<sub>β</sub> Σ<sub>i</sub> (<i>z<sub>i</sub></i> − <i>x<sub>i</sub></i>′<i>β</i>)² + α‖<i>β</i>‖<sub>2</sub>² (Ridge),&nbsp;&nbsp;"
                    "min<sub>β</sub> Σ<sub>i</sub> (<i>z<sub>i</sub></i> − <i>x<sub>i</sub></i>′<i>β</i>)² + α‖<i>β</i>‖<sub>1</sub> (Lasso)"))
    out.append(P.p(f"α의 중앙값은 Ridge {AR:.2f}, Lasso {AL:.1e}였다. Ridge의 α는 반복에 따라 크게 달라서, 20회 중 {N_RMIN}회에서는 격자의 최솟값 0.01이 골라졌고 "
                   "나머지 반복에서는 큰 값이 골라졌다. 최솟값이 고른 반복에서는 수축이 필요 없다는 뜻이다. "
                   f"Lasso도 비영 계수를 중앙값 {NLZ:.1f}개 남겨 {NP}개 가운데 약 {NP - NLZ:.1f}개만 0으로 만들었다. "
                   "정규화는 변수 수가 표본에 비해 많아 과적합이 문제 될 때 이득을 주는데, 이 소표본에서는 그러한 조건이 성립하지 않았다. "
                   "그 결과 정규화 회귀는 점검 오차를 줄이지 못하였다."))
    out.append(P.fig("pr_alpha", "lab1_알파곡선.png",
                     "규제 강도 α에 따른 학습 RMSE와 검증 RMSE(Ridge, Lasso)와 Lasso의 비영 계수 수(시드 0). 점선은 검증 RMSE가 최소인 α이다."))
    out.append(P.p(f"OLS 계수의 유의성을 보면, {NP}개 변수 가운데 p-value가 0.05 미만인 변수는 중앙값 {N05}개(약 {N05 / NP * 100:.0f}%)였고, "
                   f"Bonferroni 보정(0.05/p) 후에는 중앙값 {NBF}개만 남았다. 다수의 변수는 개별로는 유의하지 않았으며, 변수 간 상관으로 계수 추정의 불확실성이 큰 것으로 보인다(추정)."))
    out.append(P.tab("pr_reg", "정규화 회귀의 사용 여부와 근거(소표본 20회)",
                     ["대상", "판정", "하이퍼파라미터와 p-value", "점검 RMSE 차이(중앙값)", "다시 쓰는 조건"],
                     reg_rows, widths=["14%", "18%", "28%", "20%", "20%"]))
    out.append(P.p("이에 따라 Ridge와 Lasso는 본 모형에 쓰지 않았다. OLS는 계수의 부호와 크기를 해석할 수 있어 회귀 기준선으로 유지하였다. "
                   "트리 계열은 소표본에서는 이득이 없었으나 전체 표본에서 AUC가 높았으므로, 아래 (4)의 결과를 근거로 본 모형에 포함하였다."))

    out.append(P.h4("(3) 잔차 진단: 완납과 부도의 혼합"))
    out.append(P.p(f"잔차를 완납 집단과 부도 집단으로 나누어 보면([[t:pr_resid]], 시드 0 기준) 학습 구간의 잔차 표준편차가 부도 집단에서 {f(sd_def)}, "
                   f"완납 집단에서 {f(sd_full)}로, 부도 집단이 약 {sdr:.1f}배이다. 분산이 집단마다 다른 이분산 구조이다. "
                   f"평균 잔차도 완납에서 {sg(mean_full)}, 부도에서 {sg(mean_def)}로 반대 방향으로 벌어진다. "
                   "하나의 회귀식이 성격이 다른 두 집단의 혼합을 평균 쪽으로 맞추고 있다는 뜻이다."))
    out.append(P.tab("pr_resid", "완납·부도 집단별 잔차의 평균과 표준편차(시드 0)",
                     ["집단", "모형", "학습 평균 잔차", "학습 잔차 SD", "점검 평균 잔차", "점검 잔차 SD", "점검 대출 수"],
                     res_rows, widths=["10%", "26%", "13%", "13%", "13%", "13%", "12%"]))
    out.append(P.p(f"2단계 구조(부도확률과 조건부 이득의 곱)는 집단별 평균 잔차를 거의 바꾸지 못하였다(완납 {sg(m2_full)}, 부도 {sg(m2_def)}). "
                   f"조건부 이득 모형도 집단 안의 변동을 거의 설명하지 못하여 점검 R²가 {f(STEP[7].r2_hold, 3)}으로 전체 OLS의 {f(STEP[4].r2_hold, 3)}보다 약간 높은 수준이다. "
                   "2단계 구조의 가치는 평균 예측이 아니라 위험 Var(r) 계산에 있다. 승인 규칙의 분산 항 p(1 − p)(a − d)²가 이 구조에서 나오기 때문이다."))

    out.append(P.h4("(4) 소표본 결과의 전체 표본 연결"))
    out.append(P.p(f"소표본에서 얻은 결론이 전체 표본에서도 유지되는지 보기 위해, 30만 행 자료에 5겹 교차검증을 적용한 부도 분류의 AUC를 비교하였다. "
                   f"XGBoost의 AUC는 {f(alt('B0_sub'))}, EBM은 {f(alt('M1a_sub'))}, 신경망(MLP)은 {f(alt('M4a_sub'))}, 선형 WOE 점수표는 {f(alt('M1b_sub'))}였다. "
                   f"선형 점수표는 XGBoost보다 AUC가 {alt('B0_sub') - alt('M1b_sub'):.4f} 낮았다."))
    out.append(P.p("이 차이는 표본 크기에 따라 달라지는 것으로 설명된다. 유연한 모형은 표본이 작을 때 학습 자료에 과하게 맞춰져 간격이 커지고, 표본이 커질수록 이 불리함이 줄어든다. "
                   "소표본에서 비선형 모형의 이득이 없었던 것과 전체 표본에서 트리 모형의 AUC가 높았던 것은 같은 원리에서 나온다. "
                   "다만 전체 표본의 결과는 소표본 실험과 다른 분할에서 얻은 것이므로 두 결과를 수치로 직접 비교하지는 않는다."))

    out.append(P.h3("3.2 분류와 결정 경계"))
    out.append(P.p(f"승인 결정은 승인과 거절의 이분 결정이므로, 점수가 경계 근처에서 얼마나 믿을 만한지를 분류 관점에서 함께 보았다. "
                   "점수 <i>s<sub>i</sub></i>는 값이 클수록 부도 위험이 높으며, 임계값 τ는 상위 26% 경계로 정하였다. "
                   "ROC 곡선은 τ를 바꾸면서 부도 대출 가운데 거절된 비율(TPR)과 완납 대출 가운데 거절된 비율(FPR)이 이루는 궤적이고, "
                   "그 아래 면적인 AUC는 임의로 뽑은 부도 대출이 완납 대출보다 높은 점수를 받을 확률이다."))
    out.append(P.p("로지스틱 회귀는 L2 규제와 L1 규제를 각각 적용하였다. 규제 강도 C는 L2에서는 검증 logloss, L1에서는 검증 부도 AUC로 골랐다. "
                   f"정규화가 없는 모형과 비교하면 L2는 점검 AUC를 쌍체 중앙값 {sg(d_l2[0], 4)}만큼 높였고({d_l2[1] * 100:.0f}%의 반복에서 우세), "
                   f"L1은 L2 대비 {sg(d_l1[0], 4)}({d_l1[1] * 100:.0f}%의 반복에서 우세)이었다. 반면 SR26은 정규화가 없을 때 {f(CLS.iloc[0].SR26, 3)}이었고 "
                   f"L2에서 {f(CLS.iloc[1].SR26, 3)}으로 낮아졌다. L1은 비영 계수를 중앙값 {CLS.iloc[2].nonzero:.1f}개로 줄였지만 성능 차이는 작았다. "
                   "규제 종류에 따른 차이가 작다는 점은 3.1의 결과와 같다([[t:pr_cls]])."))
    out.append(P.tab("pr_cls", "분류 모형 비교(점검 구간, 20회 중앙값)",
                     ["모형", "C(검증 선택)", "검증 AUC(부도)", "점검 AUC(부도)", "점검 AUC(z&gt;0)", "SR26", "비영 계수"],
                     cls_rows, widths=["30%", "10%", "12%", "12%", "13%", "11%", "12%"],
                     foot="AUC(부도)는 부도 점수의 순위 정확도, AUC(z&gt;0)은 승인 점수로 초과수익이 0보다 큰 대출을 가려내는 정확도이다. "
                          "모형마다 C를 고른 기준이 달라 같은 기준의 비교는 아니며, 모두 검증 구간만으로 고르고 점검 구간은 평가에만 썼다."))
    out.append(P.p("선형 SVM 소프트마진은 마진을 넓히고 위반을 허용하는 정도를 C로 조절하며, 위반의 크기는 hinge 손실의 여유 변수 ξ<sub>i</sub>로 잰다. "
                   "|z| 가중 방식은 위반의 벌점에 |z<sub>i</sub>|를 곱하여, 손익의 절댓값이 큰 대출의 경계 위반을 더 무겁게 벌한다. "
                   "C는 검증 구간의 z&gt;0 AUC로 골랐다. 두 방식의 목적 함수는 다음과 같다."))
    out.append(P.eq("min<sub>w,b,ξ</sub> ½‖<i>w</i>‖² + C Σ<sub>i</sub> ξ<sub>i</sub>&nbsp;&nbsp;s.t.&nbsp;<i>t<sub>i</sub></i>(<i>w</i>′<i>x<sub>i</sub></i> + <i>b</i>) ≥ 1 − ξ<sub>i</sub>, ξ<sub>i</sub> ≥ 0"
                    "&nbsp;&nbsp;(무가중);&nbsp;&nbsp;벌점 C Σ<sub>i</sub> |<i>z<sub>i</sub></i>| ξ<sub>i</sub> (|<i>z</i>| 가중)"))
    out.append(P.p(f"모형 간 차이에는 무엇을 맞히도록 학습했는지(목표)와 어떤 손실을 쓰는지(소프트마진)가 함께 섞인다. 두 효과를 나누기 위해 "
                   f"같은 목표(z&gt;0)를 쓰는 로지스틱 회귀를 비교 기준으로 두었다. 목표를 부도에서 z&gt;0으로 바꾼 효과는 SR26 쌍체 차이 {sg(d_tgt[0], 4)}"
                   f"({d_tgt[1] * 100:.0f}%의 반복에서 우세)로 매우 컸다. 반면 같은 목표 아래에서 무가중 소프트마진의 SR26은 비교 기준과의 차이가 "
                   f"{sg(d_sv[0], 4)}({d_sv[1] * 100:.0f}%의 반복에서 우세)로 사실상 없었고, z&gt;0 AUC는 {sg(d_sv_sign[0], 4)}로 오히려 낮았다. "
                   f"|z| 가중 방식도 비교 기준과의 차이가 {sg(d_sw[0], 4)}({d_sw[1] * 100:.0f}%)에 그쳤다. 무가중 대비 |z| 가중의 차이 {sg(d_swv[0], 4)}은 "
                   f"반복의 {d_swv[1] * 100:.0f}%에서만 우세하여 안정적인 이득으로 보기 어렵다."))

    out.append(P.h4("(1) 임계값 거리와 결정의 정확도"))
    out.append(P.p("경계와의 거리는 승인 점수 <i>a<sub>i</sub></i>와 임계값 τ의 차이를 점수 표준편차로 나눈 임계값 거리 <i>d<sub>i</sub></i>로 정의하였다. "
                   "결정이 맞았다는 것은 승인 대출이 z&gt;0이거나, 거절 대출이 z ≤ 0인 경우를 뜻한다. 이 기준을 식으로 쓰면 다음과 같다."))
    out.append(P.eq("<i>d<sub>i</sub></i> = (<i>a<sub>i</sub></i> − τ) ÷ sd(<i>a</i>),&nbsp;&nbsp;결정이 맞음 ⇔ (승인 &amp; <i>z<sub>i</sub></i> &gt; 0) 또는 (거절 &amp; <i>z<sub>i</sub></i> ≤ 0)"))
    out.append(P.p(f"[[t:pr_margin]]은 점검 구간 8만 건을 풀링하여 거리 구간별 결정 정확도와 크기를 보여준다. "
                   f"거절측에서는 거리가 0.25 이하인 구간의 정확도가 {pc(R.loc['0~0.25'].맞음비율)}이었고, 거리 2 이상에서는 {pc(R.loc['2 이상'].맞음비율)}로 올랐다. "
                   f"평균 z도 {sg(R.loc['0~0.25'].평균z, 3)}에서 {sg(R.loc['2 이상'].평균z, 3)}로 낮아져, 경계에서 멀리 거절된 대출일수록 실제로 국채보다 못한 대출이 많았다. "
                   f"경계에서 0.5 미만 떨어진 거절 대출의 {pc(1 - NR, 0)}는 z&gt;0인 대출, 즉 놓친 이득이 있는 대출이지만 평균 z가 작아 놓친 이득의 크기는 작다. "
                   "[[f:pr_roc]]는 로지스틱 3종의 ROC 곡선과 거리 구간별 정확도를 함께 보여준다."))
    out.append(P.tab("pr_margin", "임계값 거리 |d| 구간별 결정 정확도와 크기(점검 구간 풀링 80,000건)",
                     ["측", "|d|", "대출 수", "비중", "결정이 맞은 비율", "평균 z", "평균 |z|", "부도율"], mr_rows,
                     widths=["16%", "11%", "10%", "9%", "17%", "13%", "12%", "12%"],
                     foot="점수는 2단계 HistGB의 비율 점수(E/위험), τ는 상위 26% 경계, d = (점수 − τ) ÷ 점수 표준편차이다."))
    out.append(P.fig("pr_roc", "lab3_ROC_마진.png",
                     "점검 구간의 ROC 곡선(로지스틱 3종, 왼쪽: 시드 0)과 임계값 거리별 결정이 맞은 비율(오른쪽: 20회 풀링)."))
    out.append(P.p(f"승인측에서도 같은 방향의 변화가 보이지만 폭이 작다. 맞은 비율은 거리 0.25 이하의 {pc(A.loc['0~0.25'].맞음비율)}에서 "
                   f"거리 1~2의 {pc(A.loc['1~2'].맞음비율)}로 완만하게 올랐고, 평균 z는 {sg(A.loc['0~0.25'].평균z, 3)}에서 {sg(A.loc['1~2'].평균z, 3)}로 작은 양수였다. "
                   "거리 2 이상 구간은 64건에 불과하여 해석에 주의가 필요하다."))
    out.append(P.p(f"두 측의 수준(맞은 비율의 절댓값)은 직접 비교할 수 없다. 맞음의 정의가 승인측에서는 z&gt;0, 거절측에서는 z ≤ 0이므로, "
                   f"기준 비율 자체가 다르기 때문이다(승인측 경계 근처 {pc(NA)}, 거절측 경계 근처 {pc(NR)}). 거리의 효과는 같은 쪽 안에서만 읽는다. "
                   f"거절측은 가까운 구간(|d| &lt; 0.5)의 {pc(NR)}에서 먼 구간(|d| ≥ 1)의 {pc(FR)}로 {(FR - NR) * 100:+.0f}%p 높아졌고, "
                   f"승인측은 {pc(NA)}에서 {pc(FA)}로 {(FA - NA) * 100:+.0f}%p 높아졌다. 거리 0.5 미만 구간은 전체 대출의 {pc(BAND)}이며, 평균 |z|가 작아 틀리더라도 손실의 크기가 작다."))
    out.append(P.p("결정 경계 근처의 결정은 신뢰도가 낮고, 그 정도는 거절측에서 뚜렷하다. 이 결과는 경계 구간을 별도로 다루어야 한다는 근거가 된다."))
    return "".join(out)
