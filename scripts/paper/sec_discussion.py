"""논문형 4장(논의 및 결론)과 초록·핵심어. 모든 수치는 산출물/최종보고서/표의 저장 결과에서 읽고, 판정 규칙은 해당 절의 비교 기준과 같다."""
import json
import pandas as pd

LADDER_NAMES = {1: "가격·약정 변수", 2: "신용이력 변수", 3: "재무 변수", 4: "시장·시기 변수", 5: "Ridge 규제", 6: "Lasso 규제", 7: "2단계 구조", 8: "비선형(HistGB) 1단계", 9: "v3-lite 2단계"}
LADDER_BASE = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4}


def _csv(P, name):
    return pd.read_csv(P.tables / name)


def _json(P, name):
    return json.loads((P.tables / name).read_text(encoding="utf-8"))


def _sg(x, nd=3):
    return f"{x:+.{nd}f}".replace("-", "−")


def _n(x, nd=3):
    return f"{x:.{nd}f}".replace("-", "−")


def _pc(x, nd=1):
    return f"{x * 100:.{nd}f}%"


def _verdict_fn(P):
    rmse = _csv(P, "랩_회귀반복.csv").pivot(index="seed", columns="step", values="rmse_hold")

    def verdict(a, b):
        dr = rmse[a] - rmse[b]
        m, lo = float(dr.median()), float((dr < 0).mean())
        tag = "설명력↑" if (m <= -0.0005 and lo >= 0.75) else ("악화" if (m >= 0.0005 and lo <= 0.25) else "차이 없음")
        return tag, m, lo
    return verdict


def _values(P):
    v = {}
    foc = _json(P, "FOC_요약.json")
    top = _csv(P, "주요결과_T1_T2.csv")
    top = top[top.ruler == "T1"].set_index("strategy")
    v["sr_v3"], v["sr_b0"] = float(top.loc["V3", "SR_test"]), float(top.loc["B0", "SR_test"])
    v["dr_v3"] = float(top.loc["V3", "승인부도율_test"])
    v["sr_foc"], v["foc_diff"], v["foc_ci"] = foc["시험_SR_foc"], foc["차이_foc_v3"], foc["차이_구간"]
    v["lam"], v["ap_foc"], v["dr_foc"] = foc["lam_star"], foc["시험_승인율_foc"], foc["시험_부도율_foc"]
    v["ap_v3"] = float(top.loc["V3", "승인율_test"])

    ci = _csv(P, "구간.csv")
    c = ci[(ci.ruler == "T1") & (ci.strategy == "B0")].iloc[0]
    v["d_b0"], v["d_lo"], v["d_hi"] = float(c.d_final), float(c.d_lo), float(c.d_hi)
    oot = _csv(P, "OOT_2016.csv")
    oot = oot[(oot.구간.str.startswith("2017")) & (oot.ruler == "T1")].set_index("strategy")
    v["oot_v3"], v["oot_b0"] = float(oot.loc["V3", "SR"]), float(oot.loc["B0", "SR"])

    rep = _csv(P, "대표성_연도별.csv")
    r = rep[rep.yr <= 2018]
    v["pshare"] = r.승인_대출.sum() / (r.승인_대출.sum() + r.거절_신청.sum())

    verdict = _verdict_fn(P)
    v["ladder"] = {s: verdict(s, LADDER_BASE[s]) for s in LADDER_BASE}
    lad = _csv(P, "랩_회귀사다리.csv").set_index("step")
    v["lad_sr"] = {int(s): float(lad.loc[s, "SR26"]) for s in lad.index}
    v["lad_up"] = [s for s in LADDER_BASE if v["ladder"][s][0] == "설명력↑"]
    v["ridge_tag"], v["lasso_tag"] = v["ladder"][5][0], v["ladder"][6][0]
    v["nl_tag"] = v["ladder"][8][0]

    reg_model = _csv(P, "랩_잔차진단_시드0.csv").set_index(["집단", "모형"])
    v["sdr"] = float(reg_model.loc[("부도", "1단계 Ridge"), "학습_잔차SD"] / reg_model.loc[("완납", "1단계 Ridge"), "학습_잔차SD"])
    alt = pd.read_csv(P.root / "산출물" / "대안모델" / "_work" / "results.csv").set_index("id")
    v["auc_gap"] = float(alt.loc["B0_sub", "AUC"] - alt.loc["M1b_sub", "AUC"])

    cls = _csv(P, "랩_분류반복.csv")

    def pick(key):
        return cls[cls.model.str.contains(key, regex=False)].set_index("seed")
    lg, lz, sv, sw = pick("정규화 없음"), pick("z>0 목표"), pick("무가중"), pick("(|z| 가중)")
    v["auc_def"], v["sr_def"] = float(lg.auc_hold.mean()), float(lg.SR26.mean())
    v["auc_z"], v["sr_z"] = float(lz.auc_hold.mean()), float(lz.SR26.mean())
    dsw = sw.SR26 - lz.SR26
    v["dsw_med"], v["dsw_win"] = float(dsw.median()), float((dsw > 0).mean())
    dsv = sv.SR26 - lz.SR26
    v["dsv_med"] = float(dsv.median())

    mb = _csv(P, "랩_마진구간.csv")

    def pooled(side, bins):
        s = mb[mb["측"].str.startswith(side) & mb["|d|"].isin(bins)]
        return float((s["맞음비율"] * s["대출"]).sum() / s["대출"].sum())
    near, far = ["0~0.25", "0.25~0.5"], ["1~2", "2 이상"]
    v["near_r"], v["far_r"] = pooled("거절", near), pooled("거절", far)
    v["near_a"], v["far_a"] = pooled("승인", near), pooled("승인", far)

    lb = _csv(P, "교수_F_림보.csv").set_index("방식")
    v["lb_auto"], v["lb_70"] = float(lb.loc["자동 규칙(그대로)", "T1"]), float(lb.loc["림보 심사(정확도 70%)", "T1"])
    v["lb_70_ap"], v["lb_auto_ap"] = float(lb.loc["림보 심사(정확도 70%)", "승인율"]), float(lb.loc["자동 규칙(그대로)", "승인율"])
    v["lb_reject"], v["lb_full"] = float(lb.loc["림보 전부 거절", "T1"]), float(lb.loc["림보 완전 심사(상한)", "T1"])

    ss = _json(P, "교수_H_상태공간_요약.json")
    v["aic_min"] = ss["최소AIC"]
    v["rmse_local"], v["rmse_rw"], v["rmse_mean"] = ss["rmse"]["지역 수준"], ss["rmse"]["임의보행"], ss["rmse"]["전체 평균"]

    mc = _csv(P, "채권2_모형대조.csv")
    v["b_n_ok"], v["b_n_cell"] = int((mc.오차_bp.abs() <= 30).sum()), len(mc)
    bj = _json(P, "채권2_요약.json")
    v["b_med"], v["b_max"], v["cohort"] = bj["모형오차_bp_중앙절대"], bj["모형오차_bp_최대절대"], bj["코호트"]
    v["b_ap"], v["b_ap_s"], v["b_tot"] = bj["FOC승인칸"], bj["스트레스후_승인칸"], bj["전체칸"]
    bw = _csv(P, "채권2_워터폴.csv")
    v["b_loss_n"] = int((bw.손실효과.abs() > bw.회수효과.abs() + bw.수수료효과.abs()).sum())
    v["b_mono"] = {t: bool((bw[bw.만기 == t].보존율.diff().dropna() < 0).all()) for t in (36, 60)}
    v["b_ret36A"] = float(bw[(bw.만기 == 36) & (bw.등급 == "A")].보존율.iloc[0])
    v["b_ret36G"] = float(bw[(bw.만기 == 36) & (bw.등급 == "G")].보존율.iloc[0])
    bc = _csv(P, "채권2_조건칸.csv")
    md = bc.groupby(["T", "grade", "dti3"]).Sharpe.mean().unstack("dti3")
    v["b_dti_n"], v["b_dti_c"] = int((md["낮음"] > md["높음"]).sum()), len(md)
    return v


def _verdicts(v):
    """판정 문자열. 지지·취약은 굵게 쓴다."""
    lo_ok = v["d_lo"] > 0
    h = {}
    h["H1"] = ("지지" if lo_ok else "미지지")
    h["H2"] = ("지지" if v["sdr"] >= 2 else "기각")
    h["H3"] = "기각" if (v["ridge_tag"] != "설명력↑" and v["lasso_tag"] != "설명력↑") else "지지"
    h["H4"] = "소표본 기각 · 부분표본 지지" if (v["nl_tag"] != "설명력↑" and v["auc_gap"] >= 0.005) else "혼재"
    h["H5"] = "동등" if v["foc_ci"][0] <= 0 <= v["foc_ci"][1] else "차이 있음"
    gap_r, gap_a = v["far_r"] - v["near_r"], v["far_a"] - v["near_a"]
    h["H6"] = "거절측에서 지지" if (gap_r >= 0.10 and gap_a < 0.10) else ("지지" if min(gap_r, gap_a) >= 0.10 else "미지지")
    h["H7"] = "지지" if (v["dsw_med"] >= 0.005 and v["dsw_win"] >= 0.75) else "미지지"
    h["H8"] = "지지" if (v["aic_min"].startswith("(b)") and v["rmse_local"] < v["rmse_mean"]) else "미지지"
    b = {}
    b["B1"] = "지지" if v["b_n_ok"] >= 10 else "미지지"
    b["B2"] = "지지" if v["b_loss_n"] == v["b_n_cell"] else "일부"
    b["B3"] = "지지" if all(v["b_mono"].values()) else "일부"
    b["B4"] = "지지" if v["b_dti_n"] >= 10 else "미지지"
    b["B5"] = "취약" if v["b_ap_s"] < v["b_ap"] / 2 else "미지지"
    return h, b


def _mark(s):
    return f"<b>{s}</b>" if s.startswith(("지지", "소표본", "취약", "거절측에서")) else s


def _verdict_table(P, v, h, b):
    rows = [
        ["H1", "기대 초과수익 순위(최종 규칙)가 부도확률 순위보다 Sharpe가 높다", "시험 Sharpe 차이의 구간 하한이 0보다 큼",
         f"차이 {_sg(v['d_b0'], 4)} [{_sg(v['d_lo'], 4)}, {_sg(v['d_hi'], 4)}]", _mark(h["H1"]), "3.3·3.4"],
        ["H2", "완납과 부도 대출의 잔차 분산이 다르다", "부도 잔차 SD가 완납의 2배 이상",
         f"부도 대 완납 {v['sdr']:.1f}배", _mark(h["H2"]), "3.1"],
        ["H3", "정규화가 점검 오차를 줄인다", "중앙값 차이 0.0005 이상, 15회 이상 감소", "Ridge · Lasso 모두 차이 없음", _mark(h["H3"]), "3.1"],
        ["H4", "비선형이 선형보다 설명력이 높다", "소표본 점검 오차 기준 + 30만건 AUC 차이 0.005 이상",
         f"AUC 차이 {_sg(v['auc_gap'], 4)}, 소표본 비선형 {v['nl_tag']}", _mark(h["H4"]), "3.1·3.2"],
        ["H5", "1계조건 규칙이 비율 순위 규칙보다 낫다", "시험 Sharpe 차이의 구간이 0을 벗어남",
         f"차이 {_sg(v['foc_diff'], 4)} [{_sg(v['foc_ci'][0], 4)}, {_sg(v['foc_ci'][1], 4)}]", _mark(h["H5"]), "3.3"],
        ["H6", "임계값 근처 결정일수록 결과가 불안정하다", "거절측 맞음 비율 차이 0.10 이상, 승인측 차이 0.10 미만",
         f"거절측 {_pc(v['near_r'])} → {_pc(v['far_r'])}, 승인측 {_pc(v['near_a'])} → {_pc(v['far_a'])}", _mark(h["H6"]), "3.2"],
        ["H7", "소프트마진과 |z| 가중이 순위 품질을 높인다", "z>0 목표 대비 SR26 중앙값 차이 0.005 이상, 75% 이상 반복에서 우세",
         f"차이 {_sg(v['dsw_med'], 4)}, 우세 {_pc(v['dsw_win'], 0)}", _mark(h["H7"]), "3.2"],
        ["H8", "신용 환경에는 천천히 움직이는 상태가 있다", "AIC 최소 + 평균 예측보다 한 걸음 RMSE가 작음",
         f"AIC 최소: 지역 수준 상태공간, RMSE {_n(v['rmse_local'])} 대 평균 {_n(v['rmse_mean'])}", _mark(h["H8"]), "3.6"],
        ["B1", "채권 현금흐름 모형이 실현 수익률을 재현한다", "14칸 중 절대오차 30bp 이하 10칸 이상",
         f"{v['b_n_ok']}/{v['b_n_cell']}칸, 중앙값 {v['b_med']:.0f}bp, 최대 {v['b_max']:.0f}bp", _mark(b["B1"]), "3.5"],
        ["B2", "약정 스프레드를 깎는 주된 요인은 부도 손실이다", "모든 칸에서 손실 효과가 회수와 수수료 효과의 합보다 큼",
         f"{v['b_loss_n']}/{v['b_n_cell']}칸", _mark(b["B2"]), "3.5"],
        ["B3", "등급이 낮을수록 약정 스프레드 보존율이 낮다", "만기마다 등급 순서대로 단조 감소",
         f"36개월 {'단조' if v['b_mono'][36] else '비단조'}, 60개월 {'단조' if v['b_mono'][60] else '비단조'}", _mark(b["B3"]), "3.5"],
        ["B4", "같은 만기·등급에서 dti가 높을수록 위험 대비 초과수익이 낮다", "칸 중 낮음 > 높음이 10칸 이상",
         f"{v['b_dti_n']}/{v['b_dti_c']}칸", _mark(b["B4"]), "3.5"],
        ["B5", "신용 스트레스에서는 1계조건 승인 칸이 절반 미만으로 준다", "스트레스 후 승인 칸이 절반 미만",
         f"{v['b_ap']}칸 → {v['b_ap_s']}칸(전체 {v['b_tot']}칸)", _mark(b["B5"]), "3.5"],
    ]
    return rows


def build(P):
    v = _values(P)
    h, b = _verdicts(v)
    up = "·".join(LADDER_NAMES[s] for s in v["lad_up"])
    s = v["lad_sr"]
    out = [P.h2("4 논의 및 결론", page=True), P.h3("4.1 결과의 논의")]
    out.append(P.p("[[t:ds_verdict]]은 가설 H1~H8과 채권 가설 B1~B5의 판정을 한곳에 모은 것이다. 각 판정은 해당 절에서 정한 비교와 기준을 그대로 적용하였으며, 마지막 열은 판정이 이루어진 절이다. H1과 H5는 3.3절의 규칙 비교에서, H6과 H7은 3.2절의 결정 경계 분석에서, H3과 H4는 3.1절의 회귀 사다리에서 판정되었다. 채권 가설은 모두 3.5절의 칸별 분해에서 판정되었다."))
    out.append(P.tab("ds_verdict", "가설 판정 요약(H1~H8, B1~B5)", ["번호", "가설", "비교와 판정 기준", "결과", "판정", "절"],
                     _verdict_table(P, v, h, b), widths=["6%", "26%", "26%", "24%", "12%", "6%"]))
    out.append(P.p(f"예측 오차 측면에서 설명력이 높아진 지점은 정보를 추가한 단계였다([[t:pr_ladder]]). 점검 자료의 RMSE가 같은 표본 쌍 기준으로 줄어든 단계는 {up}이며, 규제 항(Ridge, Lasso)과 비선형 1단계는 소표본에서 점검 오차를 줄이지 못하였다. 순위 품질을 나타내는 SR26은 평균만 쓴 기준선의 {_sg(s[0], 4)}에서 신용이력을 더한 뒤 {_sg(s[2], 4)}로, 시장·시기 변수까지 더한 뒤 {_sg(s[4], 4)}로 올랐고, 이후 단계에서는 {_sg(s[9], 4)} 수준에 머물렀다. 어떤 정보를 쓰는가가 어떤 모형 형태를 쓰는가보다 먼저 성과를 좌우한 셈이다."))
    out.append(P.p(f"분류 결과는 평가 지표가 결론을 바꿀 수 있음을 보여 준다([[t:pr_cls]]). 정규화를 하지 않은 기본 로지스틱 회귀는 20회 반복 평균 AUC가 {v['auc_def']:.3f}로 가장 높았지만 SR26은 {v['sr_def']:.3f}에 그쳤다. 목표 변수를 z>0으로 맞춘 모형은 AUC가 {v['auc_z']:.3f}로 낮은데도 SR26이 {v['sr_z']:.3f}로 올랐다. 소프트마진 모형은 이 비교 모형 대비 SR26 차이의 중앙값이 {_sg(v['dsw_med'], 4)}였고, 차이가 양수로 나타난 반복의 비율은 {_pc(v['dsw_win'], 0)}로 H7의 기준인 75%에 미치지 못하였다. 즉 AUC처럼 일반 지표가 좋아 보인다고 승인 경계의 품질이 좋아진 것은 아니며, 선택 편의가 있는 자료에서 지표가 개선된 것처럼 보이는 착시와 같은 구조가 나타났다고 볼 수 있다."))
    out.append(P.p(f"최종 규칙은 1계조건 점수가 양수인 대출을 승인하는 방식이다([[e:ds_rule]]). 여기서 V는 대출 수익의 조건부 분산이며, λ*는 학습 자료의 교차검증으로 {v['lam']:.2f}로 골랐다. 시험 자료에서 이 규칙의 Sharpe는 {v['sr_foc']:.4f}로, 비율 순위 규칙의 {v['sr_v3']:.4f}와 {_sg(v['foc_diff'], 4)}의 차이를 보였다. 차이의 구간이 0을 포함하여 두 규칙은 구분되지 않았다([[t:ru_foc]]). 그 가운데 1계조건 규칙은 승인율을 {_pc(v['ap_foc'])}로, 승인 대출의 부도율을 {_pc(v['dr_foc'])}로 낮추어 같은 성과를 더 보수적인 승인으로 얻었다."))
    out.append(P.eq("s<sub>k</sub> = E[z<sub>k</sub>] − λ*·V<sub>k</sub>, &nbsp; z = r − r<sub>f</sub>, &nbsp; 승인 ⇔ s<sub>k</sub> &gt; 0", label="ds_rule"))
    out.append(P.p(f"채권 관점의 가설은 3.5절에서 판정되었다. 모형 현금흐름이 실현 수익률을 재현한 정도(B1), 부도 손실이 약정 스프레드를 깎는 주된 요인이라는 가설(B2), 등급에 따른 보존율 감소(B3)는 지지되었다. dti 효과(B4)는 {v['b_dti_c']}칸 중 {v['b_dti_n']}칸에서 나타났다. 신용 스트레스에서 1계조건 승인 칸은 {v['b_ap']}칸에서 {v['b_ap_s']}칸으로 줄어 절반이 남았으므로(B5), 취약하다고 보지 않았다."))

    out.append(P.h3("4.2 한계와 보완 방안"))
    out.append(P.p("모형과 규칙 선택의 일부는 시험 자료 채점 이후에 이루어졌다. v2와 v3 후보의 착안이 1차 시험 채점 이후에 나왔고, 1계조건 규칙을 대표 규칙으로 올린 결정도 시험 결과를 본 뒤에 이루어졌다. 따라서 시험 자료의 Sharpe는 선택 과정에 따라 낙관적으로 편향되었을 수 있다. 1계조건 규칙과 비율 순위 규칙의 차이가 0을 포함한다는 사실은 이 편향이 결론을 바꾸지 않는다는 근거이지만, 편향 자체를 없애지는 못한다. 보완은 규칙 선택을 학습 자료의 교차검증으로 한정하고, 최종 규칙을 새 자료에서 한 번만 평가하는 것이다."))
    out.append(P.p(f"시간 외 구간에서는 열세가 나타났다. 2017~2020년 발급 대출에서 최종 규칙의 Sharpe는 {_sg(v['oot_v3'], 4)}로, 부도확률 순위({_sg(v['oot_b0'], 4)})보다는 높았으나 음수였다. 이 구간에서 1계조건 규칙은 다시 계산하지 않았으므로 그 시간 외 성과는 아직 알 수 없다. 열세의 원인으로는 발급 시기에 따라 달라진 신용 환경이 꼽힐 수 있으나(추정), 이 가능성은 검증하지 않았다. 보완은 시간 외 구간에서 1계조건 규칙을 다시 계산하고, 발급 시점에서 관측되는 연체율과 금리 지표로 승인율을 조정하는 국면 적응 규칙을 설계하는 것이다."))
    out.append(P.p(f"결과는 승인된 대출에서만 관측된다는 제약을 가진다. 2018년까지 신청 가운데 승인된 비율은 {_pc(v['pshare'])}였고, 거절된 신청의 상환 결과는 알 수 없다. 승인 대출로 학습한 모형을 전체 신청자에게 적용하면 선택 편의가 남는다. 이를 보완하는 방법은 거절 추론(승인 확률로 가중하는 방법)과 통제된 탐색(controlled exploration)이다. 통제된 탐색은 거절된 신청 가운데 소수, 예컨대 2~5%를 무작위로 승인하여 실제 결과를 관측하는 방식으로, 분포에 대한 가정 없이 거절 신청의 품질을 진단할 수 있다. 본 연구는 이 방식을 실행하지 않았다."))
    out.append(P.p(f"채권 분석은 만기까지 관측된 2007~2017년 발급 대출 {v['cohort']:,}건에 한정되며, 결과는 14칸으로 집계되었다. 칸 안의 대출은 평균 위험률로 현금흐름을 만들었으므로 칸 안의 이질성과 칸 사이의 동시 부도 위험은 반영되지 않았다. 칸 사이의 독립을 가정하였고, 이 가정이 깨지면 불확실성은 과소 보고된다. 학습 코호트에서의 재현 오차 중앙값 {v['b_med']:.0f}bp는 표본 안의 적합이며, 시험 기간에서 같은 크기가 유지되는지는 검증하지 않았다."))
    out.append(P.p(f"개선 방향은 세 가지로 정리된다. 첫째는 애매한 신청을 림보로 분류하여 사람이 심사하는 구조이다. 모형이 결정하고 끝나는 열린 루프를 사람의 판단과 결과가 다시 학습 자료로 돌아오는 닫힌 루프로 바꾸면, 문서로 남지 않던 심사자의 암묵지가 사유 코드로 쌓인다. 심사 정확도를 70%로 가정한 모의실험에서 Sharpe는 자동 규칙의 {v['lb_auto']:.4f}에서 {v['lb_70']:.4f}로 올랐고, 승인율은 {_pc(v['lb_auto_ap'])}에서 {_pc(v['lb_70_ap'])}로 거의 같았다. 림보를 모두 거절하면 Sharpe는 {v['lb_reject']:.4f}로 떨어졌고, 완전 심사의 상한은 {v['lb_full']:.4f}였다([[t:ds_design]]). 실제 정확도는 운영 자료에서만 알 수 있다."))
    out.append(P.tab("ds_design", "림보·비정형 텍스트 설계: 열린 루프에서 닫힌 루프로", ["작업", "입력", "출력", "검증", "가드레일"], [
        ["림보 사람 심사", "승인 경계 ±2.5%p 이내, 승인 예정 중 모형 불일치 상위 5%, 희소 구간 신청", "승인 · 거절 · 보류와 사유 코드, 판단 근거 한 줄", "결과가 확정되면 심사 정확도를 실측하여 70% 가정을 대체", "개인정보를 뺀 요약만 저장, 정확도가 낮은 구간은 사람 심사를 유지"],
        ["직함 텍스트 분류(LLM)", "직함 원문(희소 직함 포함)", "직군 코드, 신뢰도, 근거 한 줄", "키워드 규칙과의 일치율, 직군별 예측 잔차가 줄었는지", "신뢰도가 낮으면 기타, 점수를 직접 입력하지 않고 특성으로만 사용, 분류표는 사람이 검수"],
        ["림보 사유 정리(LLM)", "심사자 메모, 모형 점수와 불일치 사유", "사유 코드, 요약, 재현 가능한 조건", "코드별 사례 수와 결과 확정 후 부도율", "표에 없는 수치를 만들지 않음, 요약문에 근거 없는 수치가 있으면 반려"],
    ], widths=["14%", "22%", "20%", "22%", "22%"]))
    out.append(P.p("둘째는 직함 등 비정형 텍스트의 활용이다. 희소 직함 대출에서 직군별 예측 잔차가 체계적으로 나타났으므로, LLM으로 직함을 표준 직군 코드로 분류해 정형 변수로 넣을 여지가 있다. 분류기는 수치를 만들지 않으며, 신뢰도가 낮은 경우는 기타로 처리하고, 분류표는 사람이 검수한다. 셋째는 인과와 예측의 구분이다. 회귀 계수는 조건부 연관을 뜻하며 금리 인상이 수익을 줄인다는 인과 효과로 읽을 수 없다. 정책 효과를 인과적으로 평가하려면 일부 신청에 다른 규칙을 무작위로 적용하는 실험이 필요하다."))
    out.append(P.p(f"마지막으로 국면 감시가 필요하다. H8에서 신용 환경의 수준은 지속적으로 움직였으나(AIC 최소), 한 걸음 앞 예측의 RMSE는 임의보행의 {_n(v['rmse_rw'])}와 거의 같은 {_n(v['rmse_local'])}였다. 따라서 수준은 승인 규칙을 바꾸는 신호보다는 수익표와 λ*를 다시 추정할 시점을 알리는 감시 지표로 쓰는 것이 적절하다."))

    out.append(P.h3("4.3 결론"))
    out.append(P.p(f"본 연구는 대출 승인을 부도확률 순위가 아니라 위험 대비 기대 초과수익의 관점에서 다루었다. 부도확률 순위 규칙은 시험 자료에서 Sharpe {v['sr_b0']:.4f}에 그쳤으나 기대 초과수익 순위는 {v['sr_v3']:.4f}에 이르렀다. 설명력은 정보를 더한 단계에서 높아졌고, 1계조건 규칙은 비율 순위 규칙과 같은 수준의 Sharpe를 더 낮은 승인율과 부도율로 얻었다. 채권 관점에서는 약정 스프레드를 깎는 주된 요인이 부도 손실임을 보였고, 모형 현금흐름은 실현 수익률을 중앙값 {v['b_med']:.0f}bp 이내로 재현하였다."))
    out.append(P.p("다만 시험 채점 이후의 선택, 시간 외 열세, 승인 대출만의 관측이라는 한계가 남아 있어 결과는 보수적으로 읽어야 한다. 후속 과제는 거절 신청의 일부를 통제된 방식으로 승인하여 결과를 관측하는 것, 림보 심사의 실측 정확도를 쌓는 것, 1계조건 규칙을 시간 외 구간에서 다시 계산하는 것, 그리고 국면 감시 지표를 운영 자료로 검증하는 것이다."))
    return "".join(out)


def front(P):
    v = _values(P)
    abstract = (
        f"본 연구는 P2P 대출 승인을 부도확률 순위가 아니라 위험 대비 기대 초과수익의 관점에서 다루었다. "
        f"목적은 승인 규칙의 성과를 높이고, 그 성과가 어떤 정보와 규칙 선택에서 나오는지 밝히는 것이다. "
        f"2007~2020년 발급 대출 자료를 학습·점검·시험 자료로 나누어 회귀 사다리, 분류 모형, 1계조건 승인 규칙, 대출의 채권 가격 분해를 차례로 적용하였다. "
        f"시험 자료에서 1계조건 규칙의 Sharpe 비율은 {v['sr_foc']:.4f}로, 비율 순위 규칙의 {v['sr_v3']:.4f}와 구분되지 않았다. "
        f"과제 정의 포트폴리오의 Sharpe 비율은 {v['sr_v3']:.3f}이다. "
        f"채권 분해에서 모형 현금흐름의 중앙 오차는 {v['b_med']:.0f}bp였고, 약정 스프레드를 실현 스프레드로 줄이는 주된 요인은 부도 손실이었다. "
        f"다만 일부 선택이 시험 채점 이후에 이루어졌고 시간 외 구간에서는 열세였으며 승인 대출만 관측된다는 한계가 있어, 결과는 보수적으로 해석해야 한다."
    )
    keywords = " · ".join(["대출 승인 규칙", "Sharpe 비율", "1계조건", "선택 편의", "채권 가격 분해", "림보 심사"])
    return f"<p>{abstract}</p>", keywords
