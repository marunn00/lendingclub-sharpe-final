"""114 최종 보고서 HTML(도식·그림 포함, 자급). 수치는 산출물/테스트평가/표·결과.json·로그에서 읽는다. 출력: 산출물/테스트평가/보고서.html → PDF는 visualize export."""
import base64, json, re, sys, importlib, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parents[1]; O = ROOT / "산출물" / "테스트평가"; T = O / "표"
F = importlib.import_module("98_features"); rc = json.load(open(ROOT / "산출물/앙상블_v1/model/recipe.json"))
res = json.load(open(O / "결과.json")); K = pd.read_csv(T / "운영지점_요약.csv"); D = pd.read_csv(T / "쌍체차이.csv")
W = pd.read_csv(T / "다르게_고른_대출.csv").set_index("그룹"); G = pd.read_csv(T / "승인_등급구성.csv").set_index("등급"); R = pd.read_csv(T / "ROE_곡선.csv")
gc = json.load(open(O / "model_gain_const.json")); log = (O / "fit_apply.log").read_text()
n_gate_tr = log.split("[데이터 관문(시험 적용)]")[0].count("  통과  "); n_gate_te = log.split("[데이터 관문(시험 적용)]")[1].count("  통과  ")
r_rep, r_def = (int(re.search(rf"{k} rounds (\d+)", log).group(1)) for k in ("repaid", "default"))
ref = res["ROE_ref"]["all"]; n_all, n_res = res["n_test_rows"], res["n_resolved"]
k = lambda s, a=0.25: K[(K.strategy == s) & (abs(K.rate - a) < 1e-9)].iloc[0]
dd = lambda c, a=0.25: D[(D["compare"] == c) & (abs(D.rate - a) < 1e-9)].iloc[0]
yg = lambda s, g: R[(R.strategy == s) & (R.group == g) & (abs(R.rate - .25) < 1e-9)].ROE.iloc[0]
m6, m6v1, b0, gr = k("M6v2"), k("M6"), k("B0"), k("GRADE"); sh = G / G.sum() * 100      # 최종 = v2(부도확률 3모형 평균), v1 = 사전 계획 모델
BP = ROOT / "산출물" / "우수사례평가"; S3 = pd.read_csv(BP / "표/우3_요약.csv"); S3w = pd.read_csv(BP / "표/우3_seed승률.csv").set_index("rule"); U1 = pd.read_csv(BP / "표/우1_혼동행렬_금액위험.csv")
AP = pd.read_csv(BP / "검수_빠진요소_쌍체.csv"); AR = pd.read_csv(ROOT / "산출물/대안모델/_work/results.csv").drop_duplicates("id", keep="last").set_index("id")
s3 = lambda m, c, r="R1": S3[(S3.rule == r) & (S3.model == m)][c].iloc[0]; u1 = lambda pt, m, c: U1[U1.지점.str.startswith(pt) & (U1.모델 == m)][c].iloc[0]
ap = lambda v, a, c: AP[(AP.variant == v) & (abs(AP.rate - a) < 1e-9)][c].iloc[0]
LP = pd.read_csv(BP / "이수용/표/정책표.csv").set_index("정책"); LC = pd.read_csv(BP / "이수용/표/신뢰구간.csv"); LB = pd.read_csv(BP / "이수용/표/만기_연도별.csv").set_index("구분")
lc = lambda rf, p, c: LC[(LC.rf == rf) & (LC.정책 == p)][c].iloc[0]
def img3(n, w="100%"): return f'<img src="data:image/png;base64,{base64.b64encode((BP / "이수용" / "fig" / n).read_bytes()).decode()}" style="width:{w}">'
_q = pd.read_csv(BP / "표/우3_seed별.csv").query("rule == 'R1'").pivot_table(index="seed", columns="model", values="test_SR"); n_g_b = int((_q.GRADE > _q.B0).sum())
UM = {"M6v2": "우리 방식 v2(3모형 부도확률)", "M6": "우리 방식(위험 대비 수익)", "B0": "부도확률 XGBoost", "GRADE": "세부등급 규칙", "LC": "LendingClub(전부 승인)"}
def img2(n, w="100%"): return f'<img src="data:image/png;base64,{base64.b64encode((BP / "fig" / n).read_bytes()).decode()}" style="width:{w}">'
SRC = pd.read_csv(T / "샤프_곡선.csv"); sr_all = lambda t: SRC[(SRC.strategy == "M6") & (SRC.term == t) & (abs(SRC.rate - 1.0) < 1e-9)].SR.iloc[0]   # 승인 100% = 전부 승인(전략 무관)
man = lambda v: f"{v * 100:+,.0f}만 원"
def img(n, w="100%"): return f'<img src="data:image/png;base64,{base64.b64encode((O / "fig" / n).read_bytes()).decode()}" style="width:{w}">'
P = rc["params"]

# ---------- SVG 도식 ----------
FONT = 'font-family="Apple SD Gothic Neo, AppleGothic, sans-serif"'
def box(x, y, w, h, lines, fill="#eef1f7", stroke="#8a97b0", dash=False, bold_first=True, fs=15):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"{" stroke-dasharray=\"7 5\"" if dash else ""}/>'
    y0 = y + h / 2 - (len(lines) - 1) * (fs + 5) / 2 + fs / 3
    for i, t in enumerate(lines):
        s += f'<text x="{x + w / 2}" y="{y0 + i * (fs + 5)}" text-anchor="middle" font-size="{fs if i else fs + 1}" {"font-weight=\"700\"" if (i == 0 and bold_first) else ""} fill="#1b2433">{t}</text>'
    return s
def arrow(x1, y1, x2, y2, c="#4a5568", dash=False):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="2.5"{" stroke-dasharray=\"6 4\"" if dash else ""} marker-end="url(#ah)"/>'
def svg(w, h, body): return f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" {FONT}><defs><marker id="ah" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#4a5568"/></marker></defs>{body}</svg>'
def lane(y, h, label, fill): return f'<rect x="5" y="{y}" width="990" height="{h}" rx="12" fill="{fill}" opacity=".55"/><text x="18" y="{y + 22}" font-size="15" font-weight="700" fill="#33415c">{label}</text>'

BW, GAP, X0 = 172, 27, 18; X = [X0 + i * (BW + GAP) for i in range(5)]
d1 = lane(8, 200, "학습 단계 · 학습 파일만 사용", "#e3ecfa") + lane(222, 200, "시험 단계 · 시험 파일의 입력 열만 사용", "#fbeaea") + lane(436, 150, "채점 단계 · 이때 처음으로 시험 결과를 봄", "#eaf5e6")
top = [["① 학습 파일", "1,755,294건", "141개 열"], ["② 데이터 정리", f"사후 정보 {len(F.POST)}열 삭제", "결측은 그대로", "파생 변수·전달 국채"], ["③ 데이터 검사", f"{n_gate_tr}개 항목 통과", "(통과 못 하면 중단)"],
       ["④ 모델 학습", "결과 확정 대출", "1,116,156건", "예측 모델 5개"], ["⑤ 완성된 모델", "부도확률 3종 평균", "+ 이득 예측 2개"]]
bot = [["Ⓐ 시험 파일", f"{n_all:,}건", "결과 열은 잠금"], ["Ⓑ 같은 정리 규칙", "학습 때 만든 규칙", "그대로 적용"], ["Ⓒ 데이터 검사", f"{n_gate_te}개 항목 통과"], ["Ⓓ 점수 계산", "기대 이득 ÷ 위험"], ["Ⓔ 승인 결정", "매달 점수 순", "금액 25% 승인", "나머지는 국채"]]
for i in range(5):
    d1 += box(X[i], 45, BW, 145, top[i]) + box(X[i], 259, BW, 145, bot[i], fill="#fff4f4", stroke="#c98b8b")
    if i < 4: d1 += arrow(X[i] + BW, 117, X[i + 1] - 3, 117) + arrow(X[i] + BW, 331, X[i + 1] - 3, 331)
d1 += arrow(X[4] + BW / 2, 190, X[3] + BW / 2 + 40, 256)
d1 += box(X[1], 470, BW + 60, 100, ["시험 결과 열", "갚았나 · 떼였나 · 받은 돈"], fill="#fff", stroke="#d62728", dash=True)
d1 += box(X[3], 470, BW * 2 + GAP, 100, ["Ⓕ 채점", f"결과 확정 {n_res:,}건", "샤프 비율 · 연 수익률"], fill="#e6f4e1", stroke="#5c9a4d")
d1 += arrow(X[1] + BW + 60, 520, X[3] - 3, 520, c="#d62728", dash=True) + arrow(X[4] + BW / 2, 404, X[4] + BW / 2, 467)
d1 += f'<text x="{X[2] + 40}" y="510" font-size="13" fill="#d62728">채점에만 사용</text>'
SVG1 = svg(1000, 595, d1)

d2 = ""
L3 = [["부도확률 p", "갚지 못할 가능성"], ["다 갚을 때 이득 a", "국채보다 더 버는 몫"], ["떼일 때 이득 d", "국채 대비 손실(음수)"]]
for i, l in enumerate(L3): d2 += box(10, 15 + i * 100, 230, 80, l, fill="#e3ecfa")
d2 += box(320, 30, 280, 105, ["기대 이득", "(1 − p) × a + p × d", "평균적으로 남는 몫"], fill="#fff4dc", stroke="#c9a23c")
d2 += box(320, 180, 280, 105, ["위험", "결과가 출렁이는 폭", "(갚음/떼임 차이 + 잔여 오차)"], fill="#fff4dc", stroke="#c9a23c")
d2 += box(680, 100, 150, 105, ["점수", "기대 이득", "÷ 위험"], fill="#fde2e2", stroke="#d62728")
d2 += box(860, 100, 130, 105, ["승인", "매달", "상위 25%"], fill="#e6f4e1", stroke="#5c9a4d")
for i in range(3): d2 += arrow(240, 55 + i * 100, 317, 82) + arrow(240, 55 + i * 100, 317, 232)
d2 += arrow(600, 82, 677, 140) + arrow(600, 232, 677, 170) + arrow(830, 152, 857, 152)
SVG2 = svg(1000, 310, d2)

d3 = '<rect x="20" y="40" width="96" height="70" fill="#f2d7a6" stroke="#b38a3a" stroke-width="2"/><text x="68" y="70" text-anchor="middle" font-size="14" font-weight="700">튜닝 10%</text><text x="68" y="92" text-anchor="middle" font-size="12">111,640건</text>'
for i in range(5):
    x = 130 + i * 118; d3 += f'<rect x="{x}" y="40" width="112" height="70" fill="{"#f4a582" if i == 2 else "#cfe0f5"}" stroke="#6b85a8" stroke-width="2"/><text x="{x + 56}" y="70" text-anchor="middle" font-size="14" font-weight="700">조각 {i + 1}</text><text x="{x + 56}" y="92" text-anchor="middle" font-size="12">{"점검용" if i == 2 else "학습용"}</text>'
d3 += '<text x="425" y="28" text-anchor="middle" font-size="15" font-weight="700" fill="#33415c">학습 파일의 결과 확정 대출 1,116,156건 (조각 5개를 돌아가며 점검)</text>'
d3 += box(770, 30, 215, 90, ["시험 파일", f"{n_all:,}건", "이 단계에선 열지 않음"], fill="#fff", stroke="#d62728", dash=True)
steps = [["1. 설정 찾기", "튜닝 10%에서만"], ["2. 방법 비교", "5조각 교차검증"], ["3. 계획 고정", "모델·승인율 25%", "사전계획.md"], ["4. 시험 적용", "한 번만 실행"], ["5. 채점", "결과 그대로 보고"]]
for i, s in enumerate(steps):
    x = 20 + i * 196; d3 += box(x, 160, 170, 95, s, fill="#e6f4e1" if i >= 3 else "#eef1f7", stroke="#5c9a4d" if i >= 3 else "#8a97b0")
    if i < 4: d3 += arrow(x + 170, 207, x + 193, 207)
SVG3 = svg(1000, 270, d3)

CSS = """@page { size: A4; margin: 15mm 14mm 16mm }
body { font-family: "Apple SD Gothic Neo","AppleGothic",sans-serif; color:#1b2433; line-height:1.62; font-size:11pt }
h1 { font-size:25pt; margin:0 0 2pt } h2 { font-size:15.5pt; border-left:6px solid #d62728; padding-left:9px; margin:20pt 0 7pt; break-after:avoid }
h3 { font-size:12.3pt; margin:12pt 0 4pt; color:#2b3a55; break-after:avoid } .sub { color:#566; font-size:11pt; margin-bottom:8pt }
.box { background:#f6efe6; border-radius:8px; padding:10pt 13pt; margin:9pt 0; break-inside:avoid } .key { background:#fde9e9; border-left:5px solid #d62728; border-radius:6px; padding:7pt 11pt; margin:8pt 0; break-inside:avoid }
.note { background:#eef4fb; border-left:5px solid #4a78b5; border-radius:6px; padding:7pt 11pt; margin:8pt 0; break-inside:avoid; font-size:10.3pt }
.big { font-size:14.5pt; font-weight:700 } .red { color:#d62728 } table { border-collapse:collapse; width:100%; margin:7pt 0; font-size:10pt; break-inside:avoid }
th,td { border:1px solid #ccd; padding:4.5pt 6pt; text-align:center; vertical-align:middle } th { background:#eef1f7 } td.l { text-align:left } tr.win td { background:#fde9e9; font-weight:700 }
figure { margin:7pt 0; break-inside:avoid } figcaption { font-size:9.6pt; color:#566; text-align:center; margin-top:3pt } .small { font-size:9.3pt; color:#566 }
ul, ol { margin:3pt 0 3pt 17pt; padding:0 } li { margin:1.5pt 0 } .pb { break-before:page } code { font-size:9pt; background:#f1f2f5; padding:0 3px; border-radius:3px }
.kpi { display:flex; gap:8pt; margin:8pt 0 } .kpi div { flex:1; border:1px solid #e0c9c9; border-radius:8px; padding:7pt; text-align:center; background:#fff } .kpi b { display:block; font-size:17pt; color:#d62728 } .kpi span { font-size:9.5pt; color:#566 }"""

gate_rows = [("결과를 담은 열이 없는가", f"대출이 끝난 뒤에야 생기는 {len(F.POST)}개 열(받은 돈, 마지막 납입일, 회수액 등)이 표에도 모델 입력에도 없음"),
             ("한 열만으로 답이 너무 잘 맞지 않는가", "어떤 열 하나로 부도를 맞히는 점수(AUC)가 0.95 이상이면 '답이 새어 들어왔다'는 신호 → 최대 0.68(금리)"),
             ("결측을 몰래 채우지 않았는가", "빈 값은 빈 값 그대로 둠(예: 공공기록 경과월은 85%가 빈 값으로 유지)"),
             ("미래 정보가 섞이지 않았는가", "시장 금리는 대출 나간 달의 전달 값이고 같은 달·같은 만기에는 한 값, 신용 이력 길이는 0 이상"),
             ("범주를 숫자로 바꿀 때 답을 보지 않았는가", "가짜 범주를 넣어 시험: 자기 답을 빼고 계산하면 상관 0.002, 자기 답을 넣으면 0.458 → 빼고 계산하고 있음 확인"),
             ("행이 빠지거나 섞이지 않았는가", "행 수·대출 번호·순서가 원본과 같음")]
H = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>누구에게 돈을 빌려줄까 — LendingClub 대출 승인 전략 보고서</title><style>{CSS}</style></head><body>
<h1>누구에게 돈을 빌려줄까</h1><div class="sub">LendingClub 대출 승인 전략 · 최종 보고서 (시험 데이터 평가 포함)</div>

<div class="box"><div class="big">결론: "떼일 위험을 감안해도 이자가 충분히 남는 대출"만 골라 승인하면, 처음 보는 대출에서 1억 원당 1년에 <span class="red">{m6.ROE_test * 100:,.0f}만 원</span>을 법니다.</div>
같은 돈을 국채에만 넣으면 {ref['treasury'] * 100:,.0f}만 원, 업계 표준(부도확률 XGBoost)으로 고르면 {b0.ROE_test * 100:,.0f}만 원, 등급 순서로 고르면 {gr.ROE_test * 100:,.0f}만 원이고, 신청을 전부 승인하면 {abs(ref['all_in']) * 100:,.0f}만 원을 잃습니다.</div>
<div class="kpi"><div><b>{m6.ROE_test:.2f}%</b><span>연 수익률 (국채 {ref['treasury']:.2f}%)</span></div><div><b>{m6.SR36_test:.2f}</b><span>샤프 비율 36개월 (등급 순서 {gr.SR36_test:.2f})</span></div>
<div><b>+{dd('M6v2 − B0').dROE:.2f}%p</b><span>표준 방식 대비 추가 수익 (95% 구간 +{dd('M6v2 − B0').dROE_lo:.2f}~+{dd('M6v2 − B0').dROE_hi:.2f})</span></div><div><b>{n_res:,}건</b><span>처음 보는 시험 대출로 채점</span></div></div>

<h3>핵심 결과 6가지</h3>
<ol><li><b>수익:</b> 승인 25%에서 연 수익률 {m6.ROE_test:.2f}%로, 비교한 모든 방법과 국채({ref['treasury']:.2f}%)보다 높습니다.</li>
<li><b>위험 대비 수익:</b> 샤프 비율이 등급 순서 방식의 약 2배입니다(36개월 {m6.SR36_test:.2f} 대 {gr.SR36_test:.2f}, 60개월 {m6.SR60_test:.2f} 대 {gr.SR60_test:.2f}).</li>
<li><b>일관성:</b> 승인율을 4%~95% 어디로 잡아도, 발행 시기를 셋으로 나눠도 표준 방식과 등급 순서보다 수익이 높습니다.</li>
<li><b>재현성:</b> 학습 데이터 안 점검 {m6.ROE_cv:.2f}%와 시험 {m6.ROE_test:.2f}%가 거의 같습니다. 외워서 나온 결과가 아닙니다.</li>
<li><b>공정성:</b> 모델과 승인율은 학습 데이터만으로 정했고, 시험 결과는 채점에만 썼습니다. 학습·적용 전 자동 검사 {n_gate_tr + n_gate_te}개 항목을 모두 통과했습니다.</li>
<li><b>다른 잣대로도 1등:</b> 우수사례 3(대출별 IRR 샤프, 100번 반복), 우수사례 1(놓친 손실 + 놓친 수익), 이수용 보고서(만기 지난 대출 전부, rf 0~3%)의 평가 방식으로 다시 재도 우리 방식이 가장 좋습니다(10장).</li></ol>

<h2>1. 무엇을 푸는 문제인가</h2>
<ul><li><b>상황:</b> 투자자가 대출 신청서를 보고 <b>빌려줄지(승인), 말지(거절)</b> 정합니다. 거절한 돈은 가장 안전한 <b>미국 국채</b>에 넣습니다.</li>
<li><b>목표:</b> 과제 안내서가 정한 목표는 <b>샤프 비율 최대화</b>입니다. 샤프 비율 = (포트폴리오 수익률 − 국채 수익률) ÷ 수익률의 출렁임. "위험 1만큼 질 때 국채보다 얼마나 더 버나"입니다.</li>
<li><b>함께 보는 지표:</b> 연 수익률(ROE)로 "실제로 얼마를 버나"를 봅니다. 부도를 얼마나 잘 맞히나(AUC)는 참고로만 봅니다.</li></ul>
<div class="note"><b>샤프 계산 방법:</b> 같은 달에 나간 대출을 한 묶음으로 보고, 묶음마다 "승인한 대출 + 국채에 넣은 돈"의 수익률에서 국채만 들었을 때의 수익률을 뺍니다. 이 초과수익의 평균을 표준편차로 나눕니다. 36개월·60개월 대출은 기간이 달라 따로 계산합니다.<br>
<b>연 수익률 계산 방법:</b> 승인한 대출마다 (받은 돈 − 빌려준 돈) ÷ 실제 상환 기간(년)을 더하고, 거절한 돈은 같은 기간 국채 이자를 더해, 전체 투자금으로 나눕니다.</div>

<h2>2. 데이터 한눈에</h2>
<table><tr><th></th><th>학습 파일</th><th>시험 파일</th></tr>
<tr><td class="l">대출 건수</td><td>1,755,294건</td><td>{n_all:,}건</td></tr>
<tr><td class="l">열(정보 항목)</td><td>141개</td><td>141개 (같은 형식)</td></tr>
<tr><td class="l">발행 기간</td><td>2007~2020년</td><td>2007~2020년</td></tr>
<tr><td class="l">결과가 확정된 대출(다 갚음·떼임)</td><td>1,116,156건 (부도율 19.5%)</td><td>{n_res:,}건 (채점용)</td></tr>
<tr><td class="l">두 파일에 함께 있는 대출</td><td colspan="2">0건 — 시험 대출은 모델이 한 번도 보지 못한 대출</td></tr></table>
<table><tr><th>등급 (학습 파일)</th><th>A</th><th>B</th><th>C</th><th>D</th><th>E</th><th>F</th><th>G</th></tr>
<tr><td class="l">대출 수</td><td>207,851</td><td>327,862</td><td>317,075</td><td>164,785</td><td>69,743</td><td>22,411</td><td>6,429</td></tr>
<tr><td class="l">부도율</td><td>6.5%</td><td>13.5%</td><td>22.4%</td><td>30.0%</td><td>37.5%</td><td>44.2%</td><td>50.1%</td></tr></table>
<div class="key"><b>데이터가 알려 준 것:</b> 다섯 명 중 한 명꼴로 떼입니다. 그래서 신청을 전부 승인하면 연 {ref['all_in']:.1f}%로 큰 손해입니다. 등급이 낮을수록 잘 떼이지만 이자도 높습니다. 따라서 "안전한 등급만"이 아니라 <b>"위험에 비해 이자가 충분한 대출"을 골라야</b> 합니다. 이 관찰이 우리 방식의 출발점입니다.</div>

<h2 class="pb">3. 전체 처리 흐름</h2>
학습 파일로 규칙과 모델을 만들고(위), 그 규칙을 시험 파일에 그대로 적용해 승인을 정한 뒤(가운데), 마지막에 한 번 채점합니다(아래). 빨간 점선은 <b>시험 결과 열이 채점 단계에만 들어간다</b>는 뜻입니다.
<figure>{SVG1}<figcaption>도식 1. 전체 처리 흐름 — 학습 → 시험 적용 → 채점</figcaption></figure>
<table><tr><th>단계</th><th>하는 일</th><th>결과물</th></tr>
<tr><td>① ~ ③</td><td class="l">학습 파일을 정리하고 자동 검사</td><td class="l">모델 입력 표 (71개 변수)</td></tr>
<tr><td>④ ~ ⑤</td><td class="l">부도확률 모델 3종(XGBoost·LightGBM·CatBoost)과 이득 예측 모델 2개 학습</td><td class="l">저장된 모델 파일</td></tr>
<tr><td>Ⓐ ~ Ⓒ</td><td class="l">시험 파일에 같은 정리 규칙 적용 후 검사 (결과 열은 읽지 않음)</td><td class="l">시험 입력 표</td></tr>
<tr><td>Ⓓ ~ Ⓔ</td><td class="l">대출마다 점수 계산, 매달 상위 25% 승인</td><td class="l">117만 건 전부의 점수·승인 여부</td></tr>
<tr><td>Ⓕ</td><td class="l">결과가 확정된 시험 대출로 채점</td><td class="l">샤프·연 수익률·비교표</td></tr></table>

<h2>4. 단계별 처리 방법</h2>
<h3>4-1. 데이터 정리: "대출을 내줄 때 알 수 있는 정보만"</h3>
<table><tr><th>처리</th><th>무엇을</th><th>왜</th></tr>
<tr><td class="l">사후 정보 삭제</td><td class="l">대출이 끝난 뒤에야 생기는 {len(F.POST)}개 열: 받은 원금·이자, 마지막 납입일·금액, 회수액, 남은 원금, 채무조정 기록, 최근 신용점수 등</td><td class="l">답을 미리 보고 푸는 셈이라, 넣으면 점수가 0.9999까지 부풀지만 실제로는 쓸 수 없음</td></tr>
<tr><td class="l">식별·중복 열 제외</td><td class="l">대출 번호, 웹 주소, 등급(세부등급과 중복), 자유 서술 제목 등</td><td class="l">예측에 의미가 없거나 같은 정보의 반복</td></tr>
<tr><td class="l">결측 그대로</td><td class="l">빈 값을 평균 등으로 채우지 않음</td><td class="l">"기록이 없다"는 것 자체가 정보(예: 연체 기록이 없으면 빈 값). 모델(XGBoost)이 빈 값을 따로 다룸</td></tr>
<tr><td class="l">시장 금리 붙이기</td><td class="l">미국 국채(3년·5년·10년)·기준금리의 <b>대출 나간 달의 전달</b> 평균값</td><td class="l">대출 결정 시점에 이미 알려진 값만 쓰기 위해</td></tr></table>
<b>새로 만든 변수(쉬운 설명):</b>
<table><tr><th>변수</th><th>뜻</th></tr>
<tr><td class="l">금리 − 기준금리</td><td class="l">시장보다 이자를 얼마나 더 받나 (위험에 대한 보상)</td></tr>
<tr><td class="l">신용 이력 길이</td><td class="l">첫 신용카드·대출을 연 뒤 몇 년이 지났나</td></tr>
<tr><td class="l">열린 계좌 비율, 최근 2년 새 계좌 비율</td><td class="l">최근 갑자기 빚을 늘렸나</td></tr>
<tr><td class="l">국채 금리(10년, 만기 맞춤)</td><td class="l">대출이 나간 때의 시장 분위기</td></tr>
<tr><td class="l">동네·직업·주·용도의 과거 부도율</td><td class="l">같은 우편번호 앞 3자리·같은 직업 사람들이 과거에 얼마나 떼였나. <b>자기 자신의 결과는 빼고</b> 계산(5조각으로 나눠 다른 조각에서 계산)</td></tr>
<tr><td class="l">동네·직업의 흔한 정도</td><td class="l">그 동네·직업이 데이터에 얼마나 자주 나오나</td></tr></table>
<div class="small">최종 입력: 숫자 변수 57개 + 범주 7개 + 흔한 정도 2개 + 과거 부도율 4개 + 발행연도 1개 = 71개.</div>

<h3>4-2. 데이터 검사: 학습·적용 전에 자동으로 멈추는 관문</h3>
<table><tr><th>검사</th><th>확인 내용과 결과</th></tr>{''.join(f"<tr><td class='l'>{a}</td><td class='l'>{b}</td></tr>" for a, b in gate_rows)}</table>
<div class="small">학습 전 {n_gate_tr}개, 시험 적용 전 {n_gate_te}개 항목이 모두 "통과"였습니다(<code>fit_apply.log</code>). 하나라도 실패하면 학습·적용이 시작되지 않게 만들었습니다.</div>

<h3>4-3. 모델: 세 가지를 예측합니다</h3>
<table><tr><th>모델</th><th>무엇을 배우나</th><th>학습 데이터</th></tr>
<tr><td class="l">부도확률 모델 3종</td><td class="l">이 대출이 떼일 확률 (0~1). XGBoost·LightGBM·CatBoost 세 모델의 값을 평균</td><td class="l">결과 확정 대출 1,116,156건</td></tr>
<tr><td class="l">"다 갚을 때" 이득 모델</td><td class="l">다 갚으면 국채보다 1년에 몇 % 더 버나</td><td class="l">다 갚은 대출만</td></tr>
<tr><td class="l">"떼일 때" 이득 모델</td><td class="l">떼이면 국채 대비 몇 % 잃나 (회수액 포함)</td><td class="l">떼인 대출만</td></tr></table>
<div class="note"><b>XGBoost란:</b> "금리가 15%보다 높은가? → 소득 대비 빚이 30%보다 큰가? → …" 같은 스무고개 나무를 수백 그루 만들고, 앞 나무가 틀린 부분을 뒤 나무가 고쳐 가며 답을 모으는 방법입니다. 표 형태 데이터에서 가장 널리 쓰이는 방법입니다.<br>
<b>세 모델을 평균하는 이유:</b> 같은 스무고개 방식이라도 나무를 키우는 규칙이 조금씩 다른 세 모델(XGBoost, LightGBM, CatBoost)의 의견을 평균하면 한 모델의 실수가 줄어듭니다. 범주(동네·직업 등)는 CatBoost가 자체 방식으로 처리합니다.<br>
<b>설정:</b> XGBoost 부도확률 모델은 나무 깊이 {P['max_depth']}, 학습 속도 {P['learning_rate']:.3f}, 나무 {rc['rounds']}그루이고, 시작값을 바꾼 {rc['seeds']}개를 평균했습니다. 설정은 학습 데이터 중 따로 떼어 둔 10%에서만 25번 시도해 골랐습니다. 이득 모델 두 개는 같은 설정에 대출 금액만큼 무게를 주었고, 따로 떼어 둔 10%에서 더 나아지지 않으면 멈추게 했습니다(각 {r_rep}·{r_def}그루).</div>

<h3>4-4. 점수: 기대 이득 ÷ 위험</h3>
<figure>{SVG2}<figcaption>도식 2. 세 예측을 합쳐 점수를 만드는 방법</figcaption></figure>
<ul><li><b>기대 이득</b> = (갚을 확률 × 갚을 때 이득) + (떼일 확률 × 떼일 때 이득). 예: 떼일 확률 10%, 갚으면 +4%, 떼이면 −40%라면 0.9×4 + 0.1×(−40) = <b>−0.4%</b> → 국채보다 못하니 뒤로 밉니다.</li>
<li><b>위험</b> = 결과가 얼마나 출렁이는지. "갚음과 떼임의 차이가 클수록, 떼일 확률이 50%에 가까울수록" 커집니다. 같은 갚음·떼임 안에서 남는 흔들림(갚음 {gc['sa']:.3f}, 떼임 {gc['sd']:.3f})도 더합니다.</li>
<li><b>점수</b> = 기대 이득 ÷ 위험. 샤프 비율과 같은 모양이라, 과제 목표와 같은 방향으로 대출 순서를 매깁니다.</li></ul>

<h3>4-5. 승인 규칙</h3>
매달 그달에 나온 대출을 점수 순서로 세우고, 위에서부터 <b>금액 기준 25%</b>까지 승인합니다. 나머지 돈은 국채에 넣습니다. 달마다 같은 비율로 승인하므로 특정 시기에 몰리지 않습니다.

<h2 class="pb">5. 방법은 어떻게 골랐나</h2>
시험 데이터는 잠가 두고, 학습 데이터 안에서만 여러 방법을 견주었습니다. 결과가 확정된 학습 대출을 튜닝용 10%와 점검용 90%로 나누고, 90%는 5조각으로 나눠 "4조각으로 학습 → 1조각으로 점검"을 다섯 번 돌렸습니다(교차검증).
<figure>{SVG3}<figcaption>도식 3. 검증 설계 — 시험 데이터는 계획을 고정한 뒤 한 번만 사용</figcaption></figure>
<table><tr><th>방법</th><th>한 줄 설명</th><th>부도 맞히기(AUC)</th><th>연 수익률 승인 30%</th><th>연 수익률 승인 50%</th><th>샤프 승인 53%</th></tr>
<tr><td class="l">부도확률 XGBoost</td><td class="l">업계 표준, 기준선</td><td><b>0.739</b></td><td>1.56%</td><td>0.70%</td><td>0.95</td></tr>
<tr class="win"><td class="l">위험 대비 수익 순위 (최종, 부도확률 3모델 평균)</td><td class="l">기대 이득 ÷ 위험</td><td>{AR.loc['AUD_blend', 'AUC']:.3f}</td><td>{AR.loc['AUD_blend', 'ROE30']:.2f}%</td><td>{AR.loc['AUD_blend', 'ROE50']:.2f}%</td><td>{AR.loc['AUD_blend', 'SR53_36']:.2f}</td></tr>
<tr><td class="l">위험 대비 수익 순위 (사전 계획, XGBoost 부도확률)</td><td class="l">기대 이득 ÷ 위험</td><td>0.707</td><td>1.96%</td><td>1.49%</td><td>0.99</td></tr>
<tr><td class="l">기대수익 순위</td><td class="l">기대 이득만으로 순서</td><td>0.706</td><td>1.96%</td><td>1.48%</td><td>0.97</td></tr>
<tr><td class="l">부도·조기상환 위험 곡선</td><td class="l">매달의 부도·조기상환 위험 학습</td><td>0.715</td><td>1.84%</td><td>1.31%</td><td>0.91</td></tr>
<tr><td class="l">EBM (설명형 모델)</td><td class="l">변수별 효과를 더한 투명한 모델</td><td>0.732</td><td>1.44%</td><td>0.55%</td><td>0.90</td></tr></table>
<div class="small">이 밖에 신용평가표, 신경망, 순위 직접 학습 등도 시험했습니다. 순위 직접 학습은 부도가 아니라 금리 순서를 배워 결과가 거꾸로 나와 제외했습니다.</div>
<div class="key"><b>선택 이유:</b> 부도를 가장 잘 맞히는 것은 표준 방식이지만, 과제가 묻는 것은 <b>돈과 위험</b>입니다. 위험 대비 수익 순위는 같은 학습 데이터에서 연 수익률이 가장 높았고, 샤프도 가장 높았습니다(다만 샤프 차이는 작아 오차 범위 안). 승인율 25%는 이 점검에서 연 수익률이 가장 높았던 지점입니다. 둘 다 시험 전에 정해 <code>사전계획.md</code>에 기록했습니다. 이후 검수(9장)에서 부도확률을 세 모델 평균으로 바꾼 것이 학습 데이터에서 더 좋아 최종 모델로 삼았습니다.</div>

<h2 class="pb">6. 시험 결과</h2>
시험 파일 {n_all:,}건 전부에 점수와 승인 여부를 매겼고, 그중 결과가 확정된 {n_res:,}건으로 채점했습니다(승인 25%).
<table><tr><th>방법</th><th>연 수익률</th><th>1억 원당 1년</th><th>샤프 36개월</th><th>샤프 60개월</th></tr>
<tr class="win"><td class="l">우리 방식 (최종)</td><td>{m6.ROE_test:.2f}%</td><td>{man(m6.ROE_test)}</td><td>{m6.SR36_test:.2f}</td><td>{m6.SR60_test:.2f}</td></tr>
<tr><td class="l">우리 방식 (사전 계획 v1)</td><td>{m6v1.ROE_test:.2f}%</td><td>{man(m6v1.ROE_test)}</td><td>{m6v1.SR36_test:.2f}</td><td>{m6v1.SR60_test:.2f}</td></tr>
<tr><td class="l">부도확률 XGBoost</td><td>{b0.ROE_test:.2f}%</td><td>{man(b0.ROE_test)}</td><td>{b0.SR36_test:.2f}</td><td>{b0.SR60_test:.2f}</td></tr>
<tr><td class="l">등급 순서대로 승인</td><td>{gr.ROE_test:.2f}%</td><td>{man(gr.ROE_test)}</td><td>{gr.SR36_test:.2f}</td><td>{gr.SR60_test:.2f}</td></tr>
<tr><td class="l">전부 국채</td><td>{ref['treasury']:.2f}%</td><td>{man(ref['treasury'])}</td><td>–</td><td>–</td></tr>
<tr><td class="l">전부 승인</td><td>{ref['all_in']:.2f}%</td><td>{man(ref['all_in'])}</td><td>{sr_all(36):.2f}</td><td>{sr_all(60):.2f}</td></tr></table>
<table><tr><th>차이 (승인 25%)</th><th>연 수익률 차이 [95% 구간]</th><th>샤프 36개월 차이 [95% 구간]</th><th>샤프 60개월 차이 [95% 구간]</th></tr>
<tr><td class="l">우리 방식 − XGBoost</td><td><b>+{dd('M6v2 − B0').dROE:.2f}%p</b> [+{dd('M6v2 − B0').dROE_lo:.2f}, +{dd('M6v2 − B0').dROE_hi:.2f}]</td><td>{dd('M6v2 − B0').dSR36:+.2f} [{dd('M6v2 − B0').dSR36_lo:+.2f}, {dd('M6v2 − B0').dSR36_hi:+.2f}]</td><td>{dd('M6v2 − B0').dSR60:+.2f} [{dd('M6v2 − B0').dSR60_lo:+.2f}, {dd('M6v2 − B0').dSR60_hi:+.2f}]</td></tr>
<tr><td class="l">우리 방식 − 등급 순서</td><td><b>+{dd('M6v2 − GRADE').dROE:.2f}%p</b> [+{dd('M6v2 − GRADE').dROE_lo:.2f}, +{dd('M6v2 − GRADE').dROE_hi:.2f}]</td><td><b>{dd('M6v2 − GRADE').dSR36:+.2f}</b> [{dd('M6v2 − GRADE').dSR36_lo:+.2f}, {dd('M6v2 − GRADE').dSR36_hi:+.2f}]</td><td><b>{dd('M6v2 − GRADE').dSR60:+.2f}</b> [{dd('M6v2 − GRADE').dSR60_lo:+.2f}, {dd('M6v2 − GRADE').dSR60_hi:+.2f}]</td></tr></table>
<div class="small">95% 구간: 연 수익률은 대출을 다시 뽑기 200회, 샤프는 발행월 묶음을 다시 뽑기로 계산했습니다.</div>
<h3>승인 25%와 절반(53%) 비교</h3>
그동안 분석의 기준 지점은 승인 53%(약 절반)였습니다. 시험 계획을 고정할 때 학습 데이터 점검에서 연 수익률이 가장 높았던 25%를 대표 지점으로 정했습니다. 두 지점의 결과를 함께 공개합니다.
<table><tr><th rowspan="2">방법</th><th colspan="3">승인 25%</th><th colspan="3">승인 53%</th></tr><tr><th>연 수익률</th><th>샤프 36개월</th><th>샤프 60개월</th><th>연 수익률</th><th>샤프 36개월</th><th>샤프 60개월</th></tr>
{''.join(f"<tr{' class=win' if s == 'M6v2' else ''}><td class='l'>{lab}</td><td>{k(s).ROE_test:.2f}%</td><td>{k(s).SR36_test:.2f}</td><td>{k(s).SR60_test:.2f}</td><td>{k(s, .53).ROE_test:.2f}%</td><td>{k(s, .53).SR36_test:.2f}</td><td>{k(s, .53).SR60_test:.2f}</td></tr>" for s, lab in [('M6v2', '우리 방식'), ('B0', '부도확률 XGBoost'), ('GRADE', '등급 순서')])}</table>
<div class="small">승인을 절반까지 넓히면 수익률과 샤프가 모두 낮아집니다. 그래도 같은 승인율에서 우리 방식이 XGBoost와 등급 순서보다 높습니다(53%에서 연 수익률 +{dd('M6v2 − B0', .53).dROE:.2f}%p, 샤프 36개월 {dd('M6v2 − B0', .53).dSR36:+.2f} [{dd('M6v2 − B0', .53).dSR36_lo:+.2f}, {dd('M6v2 − B0', .53).dSR36_hi:+.2f}]).</div>
<figure>{img("fig3_1억원_예시.png")}<figcaption>그림 1. 1억 원을 맡기면 1년에 얼마를 벌까</figcaption></figure>
<figure>{img("fig1_승인율별_수익률.png")}<figcaption>그림 2. 승인율을 어디로 잡아도 우리 방식(빨강)이 표준 방식(파랑)과 등급 순서(회색)보다 높습니다</figcaption></figure>
<figure>{img("fig2_승인율별_샤프.png")}<figcaption>그림 3. 위험 대비 수익(샤프 비율)도 등급 순서 방식의 약 2배입니다</figcaption></figure>

<h2 class="pb">7. 왜 더 버나</h2>
"안 떼일 사람"만 고르면 금리가 낮은 대출이 많이 섞입니다. 이런 대출은 몇 건만 떼여도 남는 이자로 손실을 메우지 못해 국채보다 못합니다. 우리 방식은 <b>떼일 위험을 감안해도 이자가 충분히 남는 대출</b>을 고릅니다.
<table><tr><th>승인 25%에서</th><th>건수</th><th>평균 금리</th><th>부도율</th><th>국채보다 더 번 연 수익</th></tr>
<tr class="win"><td class="l">우리 방식만 승인한 대출</td><td>{int(W.loc['M6만 승인'].건수):,}</td><td>{W.loc['M6만 승인'].평균금리:.1f}%</td><td>{W.loc['M6만 승인'].부도율 * 100:.1f}%</td><td>{W.loc['M6만 승인'].초과수익_금액가중:+.2f}%p</td></tr>
<tr><td class="l">XGBoost만 승인한 대출</td><td>{int(W.loc['B0만 승인'].건수):,}</td><td>{W.loc['B0만 승인'].평균금리:.1f}%</td><td>{W.loc['B0만 승인'].부도율 * 100:.1f}%</td><td>{W.loc['B0만 승인'].초과수익_금액가중:+.2f}%p</td></tr></table>
<figure>{img("fig5_무엇을_다르게_고르나.png")}<figcaption>그림 4. 승인한 대출의 등급 구성(왼쪽)과 두 방식이 서로 다르게 고른 대출의 수익(오른쪽)</figcaption></figure>
<div class="key">우리 방식이 승인한 대출 중 A등급은 {sh.loc['A', 'M6']:.0f}%로, 표준 방식({sh.loc['A', 'B0']:.0f}%)과 등급 순서({sh.loc['A', '등급규칙']:.0f}%)보다 적습니다. 대신 이자를 충분히 받는 B~C등급을 더 담습니다. 부도율은 조금 높아도 남는 돈이 더 많습니다.</div>

<h2>8. 믿을 수 있는 근거</h2>
<table><tr><th>점검 항목</th><th>근거</th></tr>
<tr><td class="l">처음 보는 데이터인가</td><td class="l">시험 대출 {n_all:,}건 중 학습 파일과 겹치는 대출 0건</td></tr>
<tr><td class="l">시험 결과를 보고 고르지 않았나</td><td class="l">승인율·지표는 시험 전에 <code>사전계획.md</code>에 고정. 모델 보강(9장)은 학습 데이터 교차검증만으로 판정했고, 보강 전·후 시험 결과를 모두 공개</td></tr>
<tr><td class="l">미래 정보를 쓰지 않았나</td><td class="l">사후 정보 {len(F.POST)}열 삭제, 시장 금리는 전달 값, 자동 검사 {n_gate_tr + n_gate_te}개 항목 통과</td></tr>
<tr><td class="l">시험 결과가 학습에 새지 않았나</td><td class="l">학습·적용 스크립트(<code>110</code>)는 시험 파일의 입력 열만 읽고, 결과 열은 채점 스크립트(<code>111</code>)만 읽음</td></tr>
<tr><td class="l">연습과 시험 성적이 비슷한가</td><td class="l">연 수익률 점검 {m6.ROE_cv:.2f}% → 시험 {m6.ROE_test:.2f}%, 샤프 60개월 {m6.SR60_cv:.2f} → {m6.SR60_test:.2f}</td></tr>
<tr><td class="l">특정 시기만 좋은 것 아닌가</td><td class="l">아래 표: 세 시기 모두 우리 방식이 가장 높음</td></tr></table>
<table><tr><th>발행 시기 (승인 25%)</th><th>우리 방식</th><th>XGBoost</th><th>등급 순서</th></tr>
{''.join(f"<tr><td class='l'>{t}</td><td><b>{yg('M6v2', g):.2f}%</b></td><td>{yg('B0', g):.2f}%</td><td>{yg('GRADE', g):.2f}%</td></tr>" for t, g in [("2007~2011년", "y0711"), ("2012~2016년", "y1216"), ("2017~2020년", "y1720")])}</table>
<figure>{img("fig4_연습_대_시험.png", "76%")}<figcaption>그림 5. 연습(학습 데이터 안 점검)과 실제 시험의 성적</figcaption></figure>

<h2 class="pb">9. 검수: 좋았던 방식이 모두 들어갔나</h2>
그동안 학습 데이터 점검에서 "좋았다"고 판정한 방식을 최종 모델과 하나씩 대조했습니다.
<table><tr><th>방식</th><th>이전 판정</th><th>최종 모델</th><th>근거</th></tr>
<tr><td class="l">결측 수정한 신용 이력 길이 등 정리 규칙</td><td>채택</td><td>포함</td><td class="l">입력 71개 변수가 탐색 때와 같음(이름·개수 대조)</td></tr>
<tr><td class="l">Optuna 튜닝 설정 + 시작값 3개 평균</td><td>채택</td><td>포함</td><td class="l">저장된 설정 파일과 같음</td></tr>
<tr><td class="l">기준금리 대비 금리, 국채 참고 금리 등 파생 변수 6개</td><td>채택</td><td>포함</td><td class="l">위와 같음</td></tr>
<tr><td class="l">위험 대비 수익 점수 (기대 이득 ÷ 위험)</td><td>채택</td><td>포함</td><td class="l">다시 돌린 교차검증이 이전과 소수 셋째 자리까지 같음 (연 수익률 승인 30% {AR.loc['AUD_M6', 'ROE30']:.3f}%, 샤프 {AR.loc['AUD_M6', 'SR53_36']:.3f})</td></tr>
<tr class="win"><td class="l">부도확률 세 모델 평균 (XGBoost·LightGBM·CatBoost)</td><td>AUC 최고 (0.741)</td><td><b>빠져 있었음 → 추가</b></td><td class="l">위험 대비 수익 점수에 넣으면 연 수익률 승인 30% <b>+{ap('AUD_blend', .3, 'dROE'):.3f}%p</b> [+{ap('AUD_blend', .3, 'lo'):.3f}, +{ap('AUD_blend', .3, 'hi'):.3f}], 샤프 승인 50% +{ap('AUD_blend', .5, 'dSR36'):.2f} [+{ap('AUD_blend', .5, 'dSR_lo'):.2f}, +{ap('AUD_blend', .5, 'dSR_hi'):.2f}]</td></tr>
<tr><td class="l">지역·시장 충격 변수 8개</td><td>AUC +0.0005, 수익 효과 없음</td><td>제외 유지</td><td class="l">위험 대비 수익 점수에 넣어도 연 수익률 승인 30% {ap('AUD_shock', .3, 'dROE'):+.3f}%p [{ap('AUD_shock', .3, 'lo'):+.3f}, {ap('AUD_shock', .3, 'hi'):+.3f}]로 차이 없음</td></tr></table>
<div class="note"><b>판정 방식:</b> 추가 여부는 <b>학습 데이터 교차검증만</b>으로 정했습니다(같은 5조각, 같은 지표, 짝지은 비교의 95% 구간). 다만 사전 계획 모델(v1)의 시험 채점이 먼저 끝난 뒤에 이 검수를 했으므로, 두 모델의 시험 결과를 모두 공개합니다. 시험에서도 최종 모델이 v1보다 연 수익률 <b>+{dd('M6v2 − M6').dROE:.3f}%p</b> [+{dd('M6v2 − M6').dROE_lo:.3f}, +{dd('M6v2 − M6').dROE_hi:.3f}], 샤프 60개월 +{dd('M6v2 − M6').dSR60:.2f} [+{dd('M6v2 − M6').dSR60_lo:.2f}, +{dd('M6v2 − M6').dSR60_hi:.2f}] 높았습니다.</div>

<h2>10. 우수사례의 평가 방식으로 다시 재 보기</h2>
우리 지표만으로 평가하면 "우리에게 유리한 잣대를 골랐다"는 의심을 받을 수 있습니다. 그래서 과거 우수사례 두 팀의 평가 방식을 <b>그대로</b> 가져와 같은 시험 데이터에 적용했습니다.
<h3>10-1. 우수사례 3 방식: 대출별 IRR 샤프, 100번 반복</h3>
<table><tr><th>우수사례 3의 절차</th><th>그대로 적용한 내용</th></tr>
<tr><td class="l">대출마다 IRR 계산 (0월 −대출금, 매달 +할부금, 부도 시 마지막 납입 다음 달 +회수액 −회수 수수료)</td><td class="l">같은 현금흐름으로 시험·학습 대출 전부 계산. 조기 상환은 마지막 달에 남은 금액을 받는 것으로 처리</td></tr>
<tr><td class="l">초과수익 = IRR − 국채 (36개월 → 3년, 60개월 → 5년, 발행일 기준)</td><td class="l">같음 (발행월 평균)</td></tr>
<tr><td class="l">샤프 = 승인한 대출들의 초과수익 평균 ÷ 표준편차</td><td class="l">같음</td></tr>
<tr><td class="l">6:2:2로 나눠 100번 반복, 검증셋에서 샤프가 최대인 기준선을 골라 시험셋에 적용</td><td class="l">100번 반복. 검증셋 = 학습 파일의 교차검증 예측, 시험셋 = 시험 파일 (각 148,922건 = 20%). 시험 파일은 기준선 선택에 쓰지 않음</td></tr>
<tr><td class="l">IRR이 계산 안 될 때의 규칙 (발표에서 문제로 지적)</td><td class="l">발표가 제안한 수정안대로 −100%로 처리. 원래 규칙(0)도 함께 계산</td></tr></table>
<table><tr><th>모델</th><th>검증에서 고른 기준선의 시험 샤프 (중앙값, 5~95%)</th><th>승인 25%의 시험 샤프</th><th>전부 승인 (기존 LC 정책)</th><th>XGBoost보다 높았던 횟수</th></tr>
<tr class="win"><td class="l">우리 방식</td><td>{s3('M6v2', '시험SR_중앙'):.2f} ({s3('M6v2', '시험SR_5'):.2f}~{s3('M6v2', '시험SR_95'):.2f})</td><td>{s3('M6v2', '승인25_시험SR_중앙'):.2f}</td><td rowspan="3">{s3('M6v2', '전부승인_시험SR'):.2f}</td><td>{S3w.loc['R1', 'M6v2>B0_seed수']}/100</td></tr>
<tr><td class="l">부도확률 XGBoost</td><td>{s3('B0', '시험SR_중앙'):.2f} ({s3('B0', '시험SR_5'):.2f}~{s3('B0', '시험SR_95'):.2f})</td><td>{s3('B0', '승인25_시험SR_중앙'):.2f}</td><td>–</td></tr>
<tr><td class="l">등급 순서</td><td>{s3('GRADE', '시험SR_중앙'):.2f} ({s3('GRADE', '시험SR_5'):.2f}~{s3('GRADE', '시험SR_95'):.2f})</td><td>{s3('GRADE', '승인25_시험SR_중앙'):.2f}</td><td>{n_g_b}/100</td></tr></table>
<figure>{img2("fig6_우3_샤프분포.png")}<figcaption>그림 6. 우수사례 3 방식으로 100번 반복한 시험 샤프의 분포</figcaption></figure>
<figure>{img2("fig7_우3_승인비율별.png", "88%")}<figcaption>그림 7. 우수사례 3 방식: 승인 비율별 시험 샤프</figcaption></figure>
<div class="key"><b>읽는 법:</b> 우수사례 3의 발표처럼 전부 승인하면 샤프가 0 근처({s3('M6v2', '전부승인_시험SR'):.2f})입니다. 우리 방식은 승인 25%에서 {s3('M6v2', '승인25_시험SR_중앙'):.2f}로 XGBoost({s3('B0', '승인25_시험SR_중앙'):.2f})와 등급 순서({s3('GRADE', '승인25_시험SR_중앙'):.2f})보다 높고, 100번 모두 분포가 겹치지 않습니다. 검증으로 고른 기준선은 세 모델 모두 "승인 1~2%"처럼 아주 좁게 잡혀(우수사례 3이 지적한 '낮은 기준선' 현상과 같음), 그 지점에서는 XGBoost와의 차이가 작습니다({S3w.loc['R1', 'M6v2>B0_seed수']}/100번 우세). 원래 규칙(계산 불가 = 0)으로도 결론은 같습니다(우리 방식 {s3('M6v2', '시험SR_중앙', 'R0'):.2f}, XGBoost {s3('B0', '시험SR_중앙', 'R0'):.2f}).</div>

<h3>10-2. 우수사례 1 방식: 혼동행렬과 금액 위험</h3>
우수사례 1은 "부도를 맞혔나"(혼동행렬)와 "돈으로 얼마를 놓쳤나"(Total Risk = 놓친 부도 손실 + 잘못 거절해 놓친 수익)를 함께 봤습니다. 비교를 공정하게 하려고 모든 모델이 같은 비율(시험 부도율 19.5%)을 거절하게 맞췄습니다.
<table><tr><th>모델</th><th>정확도</th><th>정밀도</th><th>재현율</th><th>F1</th><th>AUC</th><th>놓친 부도 손실</th><th>놓친 수익</th><th>Total Risk</th><th>공통 판단 제외</th></tr>
{''.join(f"<tr{' class=win' if m == 'M6v2' else ''}><td class='l'>{lab}</td><td>{u1('나', UM[m], 'Accuracy') * 100:.1f}%</td><td>{u1('나', UM[m], 'Precision') * 100:.1f}%</td><td>{u1('나', UM[m], 'Recall') * 100:.1f}%</td><td>{u1('나', UM[m], 'F1') * 100:.1f}%</td><td>{u1('나', UM[m], 'AUC'):.3f}</td><td>${u1('나', UM[m], 'loss_amnt') / 1e6:,.0f}M</td><td>${u1('나', UM[m], 'loss_amnt_implicit') / 1e6:,.0f}M</td><td><b>${u1('나', UM[m], 'total_risk') / 1e6:,.0f}M</b></td><td>${u1('나', UM[m], 'total_risk_공통제외') / 1e6:,.0f}M</td></tr>" for m, lab in [('M6v2', '우리 방식'), ('B0', '부도확률 XGBoost'), ('GRADE', '등급 순서')])}
<tr><td class="l">LendingClub (전부 승인)</td><td>{u1('나', UM['LC'], 'Accuracy') * 100:.1f}%</td><td>–</td><td>0%</td><td>–</td><td>–</td><td>${u1('나', UM['LC'], 'loss_amnt') / 1e6:,.0f}M</td><td>$0M</td><td>${u1('나', UM['LC'], 'total_risk') / 1e6:,.0f}M</td><td>–</td></tr></table>
<figure>{img2("fig8_우1_금액위험.png")}<figcaption>그림 8. 우수사례 1 방식의 Total Risk (낮을수록 좋음)</figcaption></figure>
<div class="key"><b>읽는 법:</b> "부도를 맞히는" 지표(정확도·정밀도·재현율·AUC)는 부도확률 XGBoost가 조금 높습니다. 그런데 <b>돈으로 잰 Total Risk는 우리 방식이 가장 작습니다</b>(${u1('나', UM['M6v2'], 'total_risk') / 1e6:,.0f}M 대 ${u1('나', UM['B0'], 'total_risk') / 1e6:,.0f}M). 세 모델이 똑같이 판단한 대출을 빼고 비교해도 같습니다. 우리 방식은 부도를 몇 건 더 놓치더라도 이자를 많이 내는 대출을 덜 거절해, 놓친 수익이 훨씬 작기 때문입니다. 승인 비율을 어디로 잡아도 이 순서는 같습니다(그림 8 오른쪽). 우수사례 1의 결론("부도 예측력만으로는 위험 크기를 줄이지 못한다")을 우리 방식이 해결한 셈입니다.</div>
<div class="small">주: 우수사례 1 슬라이드에서 loss_amnt_implicit의 조건식은 loss_amnt와 같게 적혀 있으나, 설명문("부도날 것이라 예측했지만 실제로는 부도가 나지 않아 얻지 못한 기대 수익")에 따라 "실제 상환인데 거절"로 계산했습니다. 우리 사전 계획 지점(승인 25%)에서도 Total Risk 순서는 같습니다(우리 방식 ${u1('가', UM['M6v2'], 'total_risk') / 1e6:,.0f}M, XGBoost ${u1('가', UM['B0'], 'total_risk') / 1e6:,.0f}M, 등급 순서 ${u1('가', UM['GRADE'], 'total_risk') / 1e6:,.0f}M).</div>

<h3>10-3. 이수용 보고서 방식: 만기 지난 대출 전부, rf를 바꿔 가며</h3>
<table><tr><th>이수용 보고서의 절차</th><th>그대로 적용한 내용</th></tr>
<tr><td class="l">평가 대상 = 만기가 지난 대출(36개월은 2017년 3월 전, 60개월은 2015년 3월 전 발급). 상태로 거르지 않음(생존편향 방지)</td><td class="l">같은 기준으로 시험 파일에서 {int(LP.loc['전부 승인', '승인건수']):,}건 (이수용 467,101건)</td></tr>
<tr><td class="l">대출별 값 = 실현 총수익률 − rf × 잔액·년 (잔액·년 = 받은 이자 ÷ 금리)</td><td class="l">같음</td></tr>
<tr><td class="l">Sharpe = 승인 대출 값의 평균 ÷ 표준편차, rf = 0·1·2·3%와 발급월 국채</td><td class="l">같음</td></tr>
<tr><td class="l">95% 구간 = 발급월 블록 부트스트랩 2,000회</td><td class="l">같음</td></tr></table>
<table><tr><th>정책</th><th>승인율</th><th>부도율</th><th>연수익률</th><th>rf 0%</th><th>rf 1%</th><th>rf 2%</th><th>rf 3%</th><th>국채</th></tr>
{''.join(f"<tr{' class=win' if p == '우리 방식 20%' else ''}><td class='l'>{p}</td><td>{LP.loc[p, '승인율'] * 100:.1f}%</td><td>{LP.loc[p, '부도율'] * 100:.1f}%</td><td>{LP.loc[p, '연수익률'] * 100:.2f}%</td>" + ''.join(f"<td>{LP.loc[p, c]:.3f}</td>" for c in ('rf 0%', 'rf 1%', 'rf 2%', 'rf 3%', '국채')) + "</tr>" for p in ['전부 승인', 'A~B등급만', 'A등급만', '부도확률 낮은 순 20%', '우리 방식 20%', '우리 방식 25%', '우리 방식 50%'])}
<tr><td class="l">(참고) 이수용 RAROC 모델 — 보고서 값</td><td>17.0%</td><td>7.6%</td><td>6.42%</td><td>0.485</td><td>0.415</td><td>0.344</td><td>0.269</td><td>0.414</td></tr></table>
<figure>{img3("fig9_이수용방식.png")}<figcaption>그림 9. 이수용 보고서 방식의 Sharpe — rf별(왼쪽), 승인율별(오른쪽)</figcaption></figure>
<div class="key"><b>읽는 법:</b> 이수용 보고서에서는 "A등급만" 규칙(국채 기준 0.439)이 RAROC 모델(0.414)을 이겼습니다. 같은 잣대로 재면 우리 방식은 비슷한 승인율(20%)에서 <b>{LP.loc['우리 방식 20%', '국채']:.3f}</b>으로 A등급만({LP.loc['A등급만', '국채']:.3f})보다 <b>+{lc('국채', '우리 방식 20%', '−A등급만'):.3f}</b> [+{lc('국채', '우리 방식 20%', '−A등급만_lo'):.3f}, +{lc('국채', '우리 방식 20%', '−A등급만_hi'):.3f}] 높고, rf를 0~3% 어디로 두어도 구간이 0을 넘습니다. 같은 20%를 부도확률 낮은 순으로 고른 것보다도 +{lc('국채', '우리 방식 20%', '−부도확률 낮은 순 20%'):.3f} [+{lc('국채', '우리 방식 20%', '−부도확률 낮은 순 20%_lo'):.3f}, +{lc('국채', '우리 방식 20%', '−부도확률 낮은 순 20%_hi'):.3f}] 높습니다. 부도율은 A등급만과 비슷한데({LP.loc['우리 방식 20%', '부도율'] * 100:.1f}% 대 {LP.loc['A등급만', '부도율'] * 100:.1f}%) 연수익률은 {LP.loc['우리 방식 20%', '연수익률'] * 100:.2f}% 대 {LP.loc['A등급만', '연수익률'] * 100:.2f}%로 훨씬 높기 때문입니다. 승인 50%로 넓혀도 A등급만(승인 19%)보다 Sharpe가 높습니다({LP.loc['우리 방식 50%', '국채']:.3f}). 발급연도 2010~2017년 8개 해 모두에서 우리 방식(승인 25%)이 A등급만과 전부 승인보다 높았습니다.</div>
<div class="small">이수용 RAROC 값은 그 보고서에 인쇄된 수치이며 평가 대상 건수가 조금 다릅니다(467,101건 대 {int(LP.loc['전부 승인', '승인건수']):,}건). 같은 잣대의 "전부 승인" 값이 0.283 대 {LP.loc['전부 승인', '국채']:.3f}로 비슷해, 계산 방식이 같게 재현됐음을 확인했습니다.</div>

<h2>11. 한계와 가정</h2>
<ul><li><b>채점 범위:</b> 결과가 확정된 대출만 채점했습니다. 아직 갚는 중인 약 41만 건은 결과가 없어 예측만 저장했습니다.</li>
<li><b>현금흐름 가정:</b> 상환 기록 합계로 매달 현금흐름을 복원했습니다. 부도 처리 시점은 마지막 납입 후 6개월, 수수료는 1%로 두었습니다.</li>
<li><b>샤프의 차이:</b> 표준 방식(XGBoost)과의 샤프 차이는 승인 25%에서 60개월은 확실히 앞서지만({dd('M6v2 − B0').dSR60:+.2f} [{dd('M6v2 − B0').dSR60_lo:+.2f}, {dd('M6v2 − B0').dSR60_hi:+.2f}]), 36개월은 오차 범위 안입니다({dd('M6v2 − B0').dSR36:+.2f} [{dd('M6v2 − B0').dSR36_lo:+.2f}, {dd('M6v2 − B0').dSR36_hi:+.2f}]). 승인 53%에서는 반대로 36개월이 확실히 앞서고({dd('M6v2 − B0', .53).dSR36:+.2f} [{dd('M6v2 − B0', .53).dSR36_lo:+.2f}, {dd('M6v2 − B0', .53).dSR36_hi:+.2f}]), 60개월은 오차 범위 안입니다. 어느 지점에서나 확실히 앞서는 것은 수익입니다.</li>
<li><b>부도 맞히기 점수:</b> AUC는 표준 방식 {res['AUC']['B0']:.3f}, 우리 방식 {res['AUC']['M6']:.3f}입니다. 우리 방식은 부도 맞히기가 아니라 수익과 위험을 기준으로 순서를 매깁니다.</li>
<li><b>많이 승인할 때:</b> 승인 53%까지 늘리면 2017~2020년 대출은 모든 방법에서 손해입니다. 승인은 좁게 유지하는 것이 좋습니다.</li></ul>

<h2>12. 다시 만들어 보려면</h2>
<table><tr><th>순서</th><th>명령</th><th>하는 일 (걸린 시간)</th></tr>
<tr><td>1</td><td class="l"><code>python3 scripts/98_e_returns.py lending_club_2020_test.csv 산출물/테스트평가/loans_test.parquet</code></td><td class="l">시험 대출의 실제 현금흐름 계산, 채점용 (10초)</td></tr>
<tr><td>2</td><td class="l"><code>python3 scripts/110_test_fit_apply.py</code></td><td class="l">학습 → 시험 파일에 점수·승인 (약 3분)</td></tr>
<tr><td>3</td><td class="l"><code>python3 scripts/111_test_eval.py</code></td><td class="l">채점·비교·신뢰구간 (약 2분)</td></tr>
<tr><td>4</td><td class="l"><code>python3 scripts/115_audit_cv.py</code></td><td class="l">검수: 빠진 요소를 넣은 교차검증 (약 7분)</td></tr>
<tr><td>5</td><td class="l"><code>python3 scripts/117_test_blend.py</code></td><td class="l">최종 모델(부도확률 3모델 평균) 학습 → 시험 점수 (약 9분), 이후 3번 다시 실행</td></tr>
<tr><td>6</td><td class="l"><code>python3 scripts/116_best_practice_eval.py</code></td><td class="l">우수사례 1·3 평가 (약 4분)</td></tr>
<tr><td>7</td><td class="l"><code>python3 scripts/119_lsy_eval.py</code></td><td class="l">이수용 보고서 방식 평가 (약 1분)</td></tr>
<tr><td>8</td><td class="l"><code>scripts/112_test_figs.py</code>, <code>118_bp_figs.py</code>, <code>120_lsy_fig.py</code>, <code>114_report.py</code></td><td class="l">그림과 이 보고서</td></tr></table>
<div class="small">시험 대출 전체의 점수와 승인 여부(<code>approve_25</code>)는 <code>산출물/테스트평가/test_scores.parquet</code>에 있습니다.</div>

<h2>부록. 과제 요구사항과 보고서 대응</h2>
<table><tr><th>과제 안내서 요구</th><th>이 보고서에서</th></tr>
<tr><td class="l">데이터 전처리: 분석에 쓸 표본과 변수 정리</td><td class="l">4-1 정리 규칙, 4-2 자동 검사</td></tr>
<tr><td class="l">EDA: 변수 분포와 관계 파악</td><td class="l">2장 등급별 부도율과 "위험에 비해 이자가 충분한 대출"이라는 관찰</td></tr>
<tr><td class="l">모델 학습·선정: train으로 학습, 최종 모델 선택</td><td class="l">4-3 ~ 4-5 모델·점수·승인 규칙, 5장 비교와 선택 이유</td></tr>
<tr><td class="l">평가: test 데이터로 평가</td><td class="l">6장 결과, 7장 이유, 8장 근거</td></tr>
<tr><td class="l">목표: 샤프 비율 최대화</td><td class="l">점수 자체가 샤프와 같은 모양(기대 이득 ÷ 위험), 6장 샤프 비교</td></tr>
<tr><td class="l">선택의 근거와 결과를 일관되게 설명</td><td class="l">학습 데이터로 계획 고정 → 같은 지표로 시험 채점, 9장 검수, 11장 한계</td></tr>
<tr><td class="l">(추가) 다른 평가 잣대에서도 성립하는가</td><td class="l">10장 우수사례 1·3, 이수용 보고서 평가 방식 재현</td></tr></table>
<h3>용어 풀이</h3>
<table><tr><th>용어</th><th>쉬운 뜻</th></tr>
<tr><td class="l">샤프 비율</td><td class="l">위험 1만큼 질 때 국채보다 얼마나 더 버나</td></tr>
<tr><td class="l">연 수익률(ROE)</td><td class="l">1년에 투자금의 몇 %를 벌었나</td></tr>
<tr><td class="l">AUC</td><td class="l">떼일 대출을 안 떼일 대출보다 위험하게 줄 세울 확률 (0.5 = 동전 던지기, 1 = 완벽)</td></tr>
<tr><td class="l">교차검증</td><td class="l">데이터를 조각으로 나눠 "일부로 공부, 나머지로 시험"을 돌아가며 해 보는 연습 시험</td></tr>
<tr><td class="l">95% 구간</td><td class="l">데이터를 다시 뽑아도 결과가 대부분 이 범위에 들어온다는 뜻. 0을 포함하지 않으면 우연이 아님</td></tr>
<tr><td class="l">사후 정보</td><td class="l">대출이 끝난 뒤에야 알 수 있는 정보 (받은 돈, 마지막 납입일 등)</td></tr></table>
</body></html>"""
(O / "보고서.html").write_text(H, encoding="utf-8"); print("saved", len(H))
