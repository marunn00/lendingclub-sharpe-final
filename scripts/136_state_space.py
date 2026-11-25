"""136 상태공간(state-space) 모형: 교수님 강조 「레짐 스위칭·스테이터스 모델, 가우스 추정으로 상태 도출」의 빠진 절반(128은 마르코프 전환만 구현).
자료(128과 같음, 새로 만들지 않음): 표/교수_E_레짐_월별.csv = 학습 파일 36개월 대출의 발급월 평균 초과수익 z(%p), 월 300건 이상, ≤ 2017-03 (mi 월 코드, n 대출 수, p_bad = 128의 악화 국면 평활 확률).
모형(사전 고정, 가우스 오차 최대우도, statsmodels.UnobservedComponents):
 관측식 z_t = μ_t + ε_t, ε_t ~ N(0, σ²_ε)      상태식(local level) μ_{t+1} = μ_t + η_t, η_t ~ N(0, σ²_η)
 비교: (a) 상수 평균 (b) 지역 수준(local level, 상태 = 서서히 움직이는 신용 수준) (c) 2국면 마르코프 전환(평균·분산 전환, 128과 같은 설정).
 판정(실행 전 고정): AIC 최소 모형을 설명력 우위로 보고, 한 걸음 앞 예측(expanding window, 마지막 30개월, 매 시점 재추정)의 RMSE를 임의보행·전체 평균 예측과 비교.
 또 평활 상태(smoothed μ_t)와 128의 악화 국면 확률(p_bad)의 상관을 본다(두 방법이 같은 국면을 가리키는지).
출력: 표/교수_H_상태공간.csv, 교수_H_상태공간_월별.csv, 교수_H_상태공간_요약.json"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.statespace.structural import UnobservedComponents
warnings.filterwarnings("ignore")
TB = Path(__file__).resolve().parents[1] / "산출물" / "최종보고서" / "표"
G = pd.read_csv(TB / "교수_E_레짐_월별.csv").sort_values("mi").reset_index(drop=True); z = G.z.to_numpy(float); T = len(z); print("월", T, "z 평균 %.3f sd %.3f" % (z.mean(), z.std(ddof=1)))
# 같은 조건 비교: 지역 수준 모형은 첫 관측을 확산(diffuse) 초기값에 쓰므로 로그우도가 89개 관측분 → 상수·마르코프도 z[1:](89개)에 적합하고 AIC = −2·로그우도 + 2k를 모수 수 k로 직접 계산
z1 = z[1:]; const = sm.OLS(z1, np.ones(len(z1))).fit(); ll = UnobservedComponents(z, "local level").fit(disp=False, maxiter=500); mk = MarkovRegression(z1, k_regimes=2, switching_variance=True).fit(disp=False, maxiter=500)
aic = lambda llf, k: float(-2 * llf + 2 * k)
lb = acorr_ljungbox(ll.standardized_forecasts_error[0][1:], lags=[6], model_df=2, return_df=True)
rows = [dict(모형="(a) 상수 평균", 모수=2, 로그우도=const.llf, AIC=aic(const.llf, 2), 설명="국면 없음(기준), 평균·분산 2모수"),
        dict(모형="(b) 지역 수준 상태공간", 모수=2, 로그우도=ll.llf, AIC=aic(ll.llf, 2), 설명=f"σ²_ε={ll.params[0]:.4f}, σ²_η={ll.params[1]:.4f}, 표준화 잔차 Ljung–Box(6, 자유도 보정 2) p={float(lb.lb_pvalue.iloc[0]):.2f}"),
        dict(모형="(c) 2국면 마르코프 전환", 모수=6, 로그우도=mk.llf, AIC=aic(mk.llf, 6), 설명=f"유지 확률 정상 {mk.params[mk.model.param_names.index('p[0->0]')]:.3f}·악화 {1 - mk.params[mk.model.param_names.index('p[1->0]')]:.3f}")]
# 한 걸음 앞 예측(expanding window, 마지막 30개월)
H = 30; err = {"지역 수준": [], "임의보행": [], "전체 평균": []}
for t in range(T - H, T):
    m = UnobservedComponents(z[:t], "local level").fit(disp=False, maxiter=300); err["지역 수준"].append(z[t] - float(m.forecast(1)[0])); err["임의보행"].append(z[t] - z[t - 1]); err["전체 평균"].append(z[t] - z[:t].mean())
rm = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in err.items()}
for r in rows: r["AIC 최소"] = r["AIC"] == min(x["AIC"] for x in rows)
out = pd.DataFrame(rows); out["한걸음예측_RMSE(최근30개월)"] = [rm["전체 평균"], rm["지역 수준"], np.nan]; out.to_csv(TB / "교수_H_상태공간.csv", index=False, encoding="utf-8-sig"); print(out.round(3).to_string())
sm_lvl = ll.smoothed_state[0]; p_bad = G.p_bad.to_numpy(float); pbad_low = float(np.corrcoef(sm_lvl, p_bad)[0, 1])
pd.DataFrame(dict(mi=G.mi, ym=G.ym, n=G.n, z=z, 평활_수준=sm_lvl, 필터_수준=ll.filtered_state[0], p_bad=p_bad)).to_csv(TB / "교수_H_상태공간_월별.csv", index=False, encoding="utf-8-sig")
S = dict(월수=T, sigma2_eps=float(ll.params[0]), sigma2_eta=float(ll.params[1]), signal_noise=float(ll.params[1] / ll.params[0]), aic={r["모형"]: float(r["AIC"]) for r in rows}, 최소AIC=[r["모형"] for r in rows if r["AIC 최소"]][0],
         rmse=rm, 상관_수준_대_악화확률=pbad_low, 평활수준_최저=float(sm_lvl.min()), 평활수준_최고=float(sm_lvl.max()), ljungbox_p=float(lb.lb_pvalue.iloc[0]))
json.dump(S, open(TB / "교수_H_상태공간_요약.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(S, ensure_ascii=False, indent=1))
