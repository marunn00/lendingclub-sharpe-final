"""137 대출을 채권처럼 가격 매기기(최종 보고서 v4 Part Ⅱ의 근거): 36·60개월 대출을 「약정 쿠폰 + 조기상환 + 부도·회수」를 가진 상각 채권으로 보고,
(1) 기대 현금흐름 모형으로 만기수익률(YTM)·공정 쿠폰을 계산하고 실현 수익률과 맞춰 보고(검증),
(2) 약정 스프레드 → 실현 스프레드로 줄어드는 과정을 단계별로 분해하고(워터폴),
(3) 시기별 현금흐름과 누적 손실이 어디서 생기는지(취약 구간)·충격(부도율·회수율·회수 지연·조기상환)이 수익률을 얼마나 깎는지 계량하고,
(4) 조건(만기·등급·dti) 칸별 위험 대비 초과수익과 1계조건(FOC) 점수·충격 후 초과수익으로 조건 가중 방향을 진단한다. 스트레스 ΔYTM(IRR 기준)은 z(계약만기 연율화 r1)와 같은 척도로 환산계수 κ_T = 평균 r1 ÷ 평균 실현 IRR(만기별)을 곱해 z에 더한다.
자료·기준(실행 전 고정): 학습 파일(lending_club_2020_train.csv)의 결과 확정 대출 중 만기 기준 코호트(36개월 ≤2017-03, 60개월 ≤2015-03 발급, 121_rulers.frame의 ref)만 사용. 시험 파일은 쓰지 않고 채점도 하지 않는 진단·설계 분석.
현금흐름은 96_common.recon(설계상 가정: 서비스 수수료 1%, 상각 = 마지막 납입 + 6개월), 무위험 rf = 발급월 만기 맞춤 국채.
 - 기대 현금흐름: 칸(만기×등급)별 나이 t의 SMM_t·MDR_t를 강의안 p.88 원금잔액 가중으로 추정(131과 같은 정의, 코호트가 만기까지 관측되므로 절단 없음), 손실률 = 1 − Σ(회수−회수수수료) ÷ Σ(대출액−상환원금).
 - 가격 P(y) = Σ CF_t (1 + y/12)^(−t), 만기수익률 YTM: P = 1, 공정 쿠폰 c_rn: 할인율을 rf로 둘 때 P = 1이 되는 쿠폰(= 위험 중립 손익분기 쿠폰), 위험 프리미엄 = 약정금리 − c_rn.
 - 충격(각각 따로): 부도율 MDR ×1.25, 손실률 +10%p, 회수 지연 +6개월, 조기상환 SMM ×1.5와 ×0.5, 그리고 위 세 가지 신용 충격을 동시에 준 스트레스(MDR ×1.25, 손실률 +10%p, 지연 +6개월).
 - 조건 칸 진단: 칸별 평균 초과수익 z = r1 − rf, 표준편차 sd(r1), Sharpe = 평균 z ÷ sd(r1), FOC 점수 = 평균 z − λ*·Var(r1)(λ* = FOC_요약.json의 학습 선택값; 칸 안 위험만 반영하는 단순 점검).
출력: 산출물/최종보고서/표/채권2_*.csv, 채권2_요약.json, fig/채권2_*.png"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib, json
import numpy as np, pandas as pd, polars as pl, scipy.sparse as sp, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import brentq
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False})
R = importlib.import_module("121_rulers"); CC = R.K.C; log = R.K.log
OUT = CC.ROOT / "산출물" / "최종보고서"; TB = OUT / "표"; FG = OUT / "fig"; TRAIN = CC.ROOT / "lending_club_2020_train.csv"
GR = list("ABCDEFG"); L = CC.L_MAIN; FEE = CC.FEE

# ---- 자료: 학습 결과 확정 대출의 만기 기준 코호트
d = R.frame(TRAIN)
ex = pl.read_csv(TRAIN, columns=["id", "int_rate", "collection_recovery_fee", "last_pymnt_d", "grade", "dti"], infer_schema_length=0, null_values=["", "NA"]).to_pandas(); ex["id"] = ex.id.astype(str)
d = d.merge(ex, on="id", validate="1:1"); d = d[d.ref].reset_index(drop=True)
d["rate"] = pd.to_numeric(d.int_rate.str.strip().str.replace("%", ""), errors="coerce") / 100; d["cfee"] = pd.to_numeric(d.collection_recovery_fee, errors="coerce").fillna(0.0)
d["dti"] = pd.to_numeric(d.dti, errors="coerce"); assert d.rate.notna().all()
lp = CC.mcode_str(d.last_pymnt_d); d["K"] = np.where(lp >= 0, np.maximum(lp - d.mi.to_numpy(), 0), 0)
d["gi"] = d.grade.map({g: i for i, g in enumerate(GR)}).astype(int); d["ti"] = (d["T"] == 60).astype(int); d["cell"] = d.ti * 7 + d.gi
log("137 코호트", len(d), "36개월", int((d["T"] == 36).sum()), "60개월", int((d["T"] == 60).sum()))
f, Tm, K, co, rate, rf, cell = (d[c].to_numpy() for c in ("funded_amnt", "T", "K", "dflt", "rate", "rf", "cell")); f = f.astype(float)
TRP, TRI, TLF, REC, CFE = (d[c].to_numpy(float) for c in ("total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "cfee"))

# ---- 나이별 실현 현금흐름(칸별 합계, 수수료 전 납입 / 회수금 / 회수수수료 / 상각 / 원금 회수)
WM = int(K.max()) + L + 1; NCELL = 14
ACC = {a: np.zeros((NCELL, WM)) for a in ("pay", "rec", "cf", "wo", "prin")}
o = np.argsort(K, kind="stable")
for s0 in range(0, len(o), 25000):
    s = o[s0:s0 + 25000]; W = int(K[s].max()) + L + 1
    net, wo, prin = CC.recon(f[s], rate[s] * 100, Tm[s], K[s], co[s], TRP[s], TRI[s], TLF[s], REC[s], CFE[s], W)
    j = np.arange(W)[None, :]; atwo = co[s][:, None] & (j == (K[s] + L)[:, None])
    rec_net = np.where(atwo, (REC[s] - CFE[s])[:, None], 0.0); recg = np.where(atwo, REC[s][:, None], 0.0); cf = np.where(atwo, CFE[s][:, None], 0.0)
    pay = (net - rec_net) / (1 - FEE)
    for c in range(NCELL):
        m = cell[s] == c
        if m.any():
            for a, M_ in (("pay", pay), ("rec", recg), ("cf", cf), ("wo", wo), ("prin", prin)): ACC[a][c, :W] += M_[m].sum(0)
irr = lambda x, F: 12 * float(CC.irr_monthly(np.r_[x[0] - F, x[1:]])[0])

# ---- 모형 입력: 칸별 나이별 SMM·MDR(원금잔액 가중), 손실률
hT = Tm; r_m = rate / 12; ev = np.where(co, 1, 2); mm = K; tt = np.where(ev == 1, mm + 1, np.maximum(mm, 1)).astype(int)
AGE = np.arange(1, 61)[None, :]; RP, EP, RD, ED_ = (np.zeros((NCELL, 60)) for _ in range(4))
for s0 in range(0, len(d), 100000):
    s = slice(s0, s0 + 100000); r, nT, ttc, evc = r_m[s][:, None], hT[s][:, None], tt[s][:, None], ev[s][:, None]
    mb = lambda k: f[s][:, None] * np.where(k <= nT, (1 - (1 + r) ** (k - nT)) / (1 - (1 + r) ** (-nT)), 0.0)
    MBt, MBt1 = mb(AGE), mb(AGE - 1)
    rp = (AGE <= ttc) & (AGE < nT) & ~((evc == 1) & (AGE == ttc)); ep = rp & (evc == 2) & (AGE == ttc); rd = (AGE <= ttc) & (AGE <= nT); edm = rd & (evc == 1) & (AGE == ttc)
    Gm = sp.csr_matrix((np.ones(len(f[s])), (cell[s], np.arange(len(f[s])))), shape=(NCELL, len(f[s])))
    RP += Gm @ (MBt * rp); EP += Gm @ (MBt * ep); RD += Gm @ (MBt1 * rd); ED_ += Gm @ (MBt1 * edm)
SMM, MDR = EP / np.maximum(RP, 1e-12), ED_ / np.maximum(RD, 1e-12)

def flows(c, smm, mdr, sv, n, lag):
    B = 1.0; W = n + lag + 1; I, PO = np.zeros(W), np.zeros(W); rr = c / 12
    for t in range(1, n + 1):
        Dd = B * mdr[t - 1]; B1 = B - Dd; PO[t + lag] += (1 - sv) * Dd; I[t] = rr * B1
        sch = B1 * rr / ((1 + rr) ** (n - t + 1) - 1); Bs = B1 - sch; pre = Bs * smm[t - 1]; PO[t] += sch + pre; B = Bs - pre
    return I + PO
price = lambda cf, y: float((cf * (1 + y / 12) ** (-np.arange(len(cf)))).sum())
ytm_safe = lambda cf: 12 * brentq(lambda m: float((cf * (1 + m) ** (-np.arange(len(cf)))).sum()) - 1.0, -0.2, 1.0)    # 받는 돈이 원금보다 적으면 음의 수익률

# ---- 칸별 표: 워터폴·모형 대조·공정 쿠폰·취약 구간·충격
WF, MD, VU, SH = [], [], [], []
SHOCKS = {"부도율 ×1.25": dict(mdr=1.25), "손실률 +10%p": dict(sv=0.10), "회수 지연 +6개월": dict(lag=6), "조기상환 ×1.5": dict(smm=1.5), "조기상환 ×0.5": dict(smm=0.5),
          "신용 스트레스(부도 ×1.25·손실률 +10%p·지연 +6개월)": dict(mdr=1.25, sv=0.10, lag=6)}
BINS = [(1, 6), (7, 12), (13, 24), (25, 36), (37, 60)]
for c in range(NCELL):
    t = 60 if c >= 7 else 36; g = GR[c % 7]; m = cell == c
    if m.sum() < 500: continue
    F = f[m].sum(); c0 = float((f[m] * rate[m]).sum() / F); rfc = float((f[m] * rf[m]).sum() / F)
    pay, recg, cfa, wo, prin = (ACC[a][c] for a in ("pay", "rec", "cf", "wo", "prin")); rec_net = recg - cfa
    y1 = irr(pay, F); y2 = irr(pay + recg, F); y3 = irr(pay * (1 - FEE) + recg - cfa, F)
    sv = float(1 - (REC[m & co] - CFE[m & co]).sum() / (f[m & co] - TRP[m & co]).sum()); n = t; smm, mdr = SMM[c], MDR[c]
    cf0 = flows(c0, smm, mdr, sv, n, L); cf_nd = flows(c0, smm, 0 * mdr, sv, n, L); cf_np = flows(c0, 0 * smm, mdr, sv, n, L)
    y_m, y_nd, y_np = ytm_safe(cf0), ytm_safe(cf_nd), ytm_safe(cf_np)
    c_rn = brentq(lambda cc: price(flows(cc, smm, mdr, sv, n, L), rfc) - 1.0, 0.005, 0.9)
    WF.append(dict(만기=t, 등급=g, 건수=int(m.sum()), 약정금리=c0, rf=rfc, 약정스프레드=c0 - rfc, 손실효과=y1 - c0, 회수효과=y2 - y1, 수수료효과=y3 - y2, 실현수익률=y3, 실현스프레드=y3 - rfc,
                   보존율=(y3 - rfc) / (c0 - rfc), 수수료전수익률=y2))
    MD.append(dict(만기=t, 등급=g, 약정금리=c0, 모형_YTM=y_m, 실현_수수료전=y2, 오차_bp=(y_m - y2) * 1e4, 조기상환제거시_bp=(y_np - y_m) * 1e4, 부도제거시_bp=(y_nd - y_m) * 1e4,
                   rf=rfc, 공정쿠폰=c_rn, 위험프리미엄=c0 - c_rn, 손실률=sv, 평균CDR=float(1 - np.prod(1 - mdr[:n]) ** (12 / n)) if n else np.nan))
    cum = np.cumsum(pay * (1 - FEE) + rec_net) / F; pb = int(np.argmax(cum >= 1.0)) if (cum >= 1.0).any() else np.nan
    loss = (wo - rec_net); intr = pay - prin; age = np.arange(len(pay))
    for lo, hi in BINS:
        if lo > t + L: continue
        mk = (age >= lo) & (age <= hi); VU.append(dict(만기=t, 등급=g, 나이구간=f"{lo}~{hi}", 이자비중=float(intr[mk].sum() / intr.sum()), 순손실비중=float(loss[mk].sum() / max(loss.sum(), 1e-12)), 원금회수비중=float(prin[mk].sum() / prin.sum())))
    VU.append(dict(만기=t, 등급=g, 나이구간="전체", 이자비중=1.0, 순손실비중=1.0, 원금회수비중=1.0, 회수월=pb, 누적순손실률=float(loss.sum() / F), 이자_대_손실=float(intr.sum() / max(loss.sum(), 1e-12))))
    base = y_m
    for nm, kw in SHOCKS.items():
        cfx = flows(c0, np.minimum(smm * kw.get("smm", 1.0), 1.0), np.minimum(mdr * kw.get("mdr", 1.0), 1.0), min(sv + kw.get("sv", 0.0), 1.0), n, L + kw.get("lag", 0))
        add = np.nan
        if nm.startswith("신용 스트레스"):                                        # 충격 후에도 충격 전 YTM을 지키려면 쿠폰을 얼마나 올려야 하나
            fx = lambda cc: ytm_safe(flows(cc, np.minimum(smm * kw.get("smm", 1.0), 1.0), np.minimum(mdr * kw.get("mdr", 1.0), 1.0), min(sv + kw.get("sv", 0.0), 1.0), n, L + kw.get("lag", 0))) - base
            add = (brentq(fx, c0, 0.95) - c0) * 1e4 if fx(0.95) > 0 else np.nan
        SH.append(dict(만기=t, 등급=g, 충격=nm, 방어쿠폰가산_bp=add, 수익률변화_bp=(ytm_safe(cfx) - base) * 1e4, 가격변화_pct=(price(cfx, base) / price(cf0, base) - 1) * 100,
                       충격후_모형스프레드_bp=(ytm_safe(cfx) - rfc) * 1e4, 충격전_모형스프레드_bp=(base - rfc) * 1e4))
WFD, MDD, VUD, SHD = (pd.DataFrame(x) for x in (WF, MD, VU, SH))
for nm, x in (("워터폴", WFD), ("모형대조", MDD), ("취약구간", VUD), ("충격", SHD)): x.to_csv(TB / f"채권2_{nm}.csv", index=False, encoding="utf-8-sig"); print(nm); print(x.round(4).to_string())

# ---- 조건 칸(만기×등급×dti 3분위): 위험 대비 초과수익, FOC 점수
lam = float(json.load(open(TB / "FOC_요약.json"))["lam_star"]); d["z"] = d.r1 - d.rf
q1, q2 = d.dti.quantile([1 / 3, 2 / 3]); d["dti3"] = np.where(d.dti <= q1, "낮음", np.where(d.dti <= q2, "중간", "높음"))
cg = d.groupby(["T", "grade", "dti3"]).agg(건수=("z", "size"), 평균z=("z", "mean"), sd_r=("r1", "std"), 부도율=("dflt", "mean")).reset_index()
cg["Sharpe"] = cg.평균z / cg.sd_r; cg["FOC점수"] = cg.평균z - lam * cg.sd_r ** 2; cg["FOC승인"] = cg.FOC점수 > 0
st = SHD[SHD.충격.str.startswith("신용 스트레스")][["만기", "등급", "수익률변화_bp"]].rename(columns={"만기": "T", "등급": "grade", "수익률변화_bp": "스트레스_bp"})
KAP = {t_: float(d[d["T"] == t_].r1.mean() / np.average(WFD[WFD.만기 == t_].실현수익률, weights=WFD[WFD.만기 == t_].건수)) for t_ in (36, 60)}   # z는 계약만기 연율화 수익(r1)이라 IRR 기준 충격을 같은 척도로 환산
cg = cg.merge(st, on=["T", "grade"], how="left"); cg["환산"] = cg["T"].map(KAP); cg["스트레스후_평균z"] = cg.평균z + cg.스트레스_bp / 1e4 * cg.환산; cg["스트레스후_FOC승인"] = (cg.스트레스후_평균z - lam * cg.sd_r ** 2) > 0
cg.to_csv(TB / "채권2_조건칸.csv", index=False, encoding="utf-8-sig"); print(cg.round(4).to_string())
SUM = dict(코호트=len(d), lam_star=lam, dti_분위경계=[float(q1), float(q2)], FOC승인칸=int(cg.FOC승인.sum()), 전체칸=len(cg), 스트레스후_승인칸=int(cg.스트레스후_FOC승인.sum()), 환산계수=KAP,
           모형오차_bp_최대절대=float(MDD.오차_bp.abs().max()), 모형오차_bp_중앙절대=float(MDD.오차_bp.abs().median()))
json.dump(SUM, open(TB / "채권2_요약.json", "w"), ensure_ascii=False, indent=1); print(SUM)

# ---- 그림 F6: 시기별 현금흐름(대출액 1당 이자·원금·회수·순손실), 그림 F7: 약정 스프레드 → 실현 스프레드 워터폴
fig, axs = plt.subplots(2, 3, figsize=(13, 6.4), sharey=False)
for r_, t in enumerate((36, 60)):
    for k_, g in enumerate(("A", "D", "G")):
        c = (t == 60) * 7 + GR.index(g); m = cell == c; ax = axs[r_, k_]; F = f[m].sum()
        pay, recg, cfa, wo, prin = (ACC[a][c] for a in ("pay", "rec", "cf", "wo", "prin")); intr = pay - prin; age = np.arange(WM); rec_net = recg - cfa; loss = wo - rec_net
        ax.bar(age, intr / F * 100, color="#4C78A8", label="이자·연체료"); ax.bar(age, prin / F * 100, bottom=intr / F * 100, color="#9ecae9", label="원금 회수")
        ax.bar(age, rec_net / F * 100, bottom=(intr + prin) / F * 100, color="#59A14F", label="회수금(상각+6개월)"); ax.bar(age, -loss / F * 100, color="#E15759", label="순손실(상각−회수)")
        ax2 = ax.twinx(); ax2.plot(age, np.cumsum(loss) / F * 100, color="#B6992D", lw=1.8, label="누적 순손실(오른쪽)"); ax2.set_ylim(0, None); ax2.grid(False); ax2.spines["right"].set_visible(True)
        ax.set_title(f"{t}개월 · 등급 {g}"); ax.set_xlabel("나이(개월)"); ax.set_ylabel("대출액 대비 월 현금흐름(%)"); ax2.set_ylabel("누적 순손실(%)")
        if r_ == 0 and k_ == 0: h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=5, fontsize=9, frameon=False); fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(FG / "채권2_시기별현금흐름.png", dpi=150); plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
for ax, t in zip(axs, (36, 60)):
    w = WFD[WFD.만기 == t]; x = np.arange(len(w)); s0 = w.약정스프레드.to_numpy() * 100
    ax.bar(x, s0, color="#4C78A8", label="약정 스프레드"); b = s0.copy()
    for col, cl, lb in (("손실효과", "#E15759", "부도 손실"), ("회수효과", "#59A14F", "회수금"), ("수수료효과", "#F28E2B", "수수료")):
        v = w[col].to_numpy() * 100; ax.bar(x, v, bottom=np.where(v < 0, b, b), color=cl, width=0.45, label=lb); b = b + v
    ax.plot(x, w.실현스프레드.to_numpy() * 100, "k_", ms=22, mew=2, label="실현 스프레드"); ax.set_xticks(x); ax.set_xticklabels(w.등급); ax.set_title(f"{t}개월"); ax.set_xlabel("등급"); ax.axhline(0, color="gray", lw=0.8)
axs[0].set_ylabel("연율 %p (rf 대비)"); axs[0].legend(fontsize=8.5, loc="upper left"); fig.tight_layout(); fig.savefig(FG / "채권2_워터폴.png", dpi=150); plt.close(fig)
log("137 완료")
