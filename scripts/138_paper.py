"""138 논문형 최종 보고서 조립기. 절 모듈(scripts/paper/sec_*.py)의 build(P)를 순서대로 이어 붙이고 표지·초록·차례를 붙여
산출물/최종보고서/최종보고서_논문형.html 과 보고서표_논문형/ 의 표 CSV를 만든다.
사용: python3 scripts/138_paper.py            (전체 조립)
      python3 scripts/138_paper.py sec_x ...   (지정한 모듈만 미리보기, 번호 토큰 해석 없이 $TMPDIR/preview.html 에 저장)"""
import sys, os, re, importlib
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "paper"))
import paper_fmt as PF
ROOT = HERE.parent; OUT = ROOT / "산출물" / "최종보고서"; FIG = OUT / "fig"
ORDER = [("sec_intro_methods", None), ("sec_results_prediction", "3 연구 결과"), ("sec_results_rule", None), ("sec_results_market", None), ("sec_discussion", None)]
only = sys.argv[1:]
P = PF.Paper(FIG); P.root, P.out = ROOT, OUT; P.tables = OUT / "표"
body = []
for mod, h in ORDER:
    if only and mod not in only: continue
    if not (HERE / 'paper' / f'{mod}.py').exists(): print('missing', mod); continue
    m = importlib.import_module(mod)
    if h: body.append(P.h2(h))
    body.append(m.build(P))
if only:
    html = "<meta charset='utf-8'><style>" + PF.CSS + "</style>" + "".join(body); Path(os.environ["TMPDIR"], "preview.html").write_text(html, encoding="utf-8")
    print("preview", len("".join(body)), "chars; tables", P.tn, "figures", P.fn, "equations", P.en); sys.exit()
abstract, keywords = importlib.import_module("sec_discussion").front(P)
doc = "".join(body); doc = P.resolve(doc)
heads = re.findall(r"<h([23])[^>]*>(.*?)</h\1>", doc)
toc = "<div class='toc pg'><h2>차 례</h2>" + "".join(f"<p class='l{'1' if k == '2' else '2'}'>{t}</p>" for k, t in heads) + "</div>"
TITLE = "Sharpe 1계조건을 이용한 대출 승인 규칙: 예측 오차의 축소, 채권 가격 분해, 시장 상태"
cover = (f"<div class='title'>{TITLE}</div><div class='subtitle'>Loan Approval Rules from the First-Order Condition of the Sharpe Ratio:<br>Forecast Error Reduction, Bond-Style Pricing, and Market States</div>"
         f"<div class='au'>○조 (조원 이름 기입)</div><div class='abs'><div class='abs-h'>요 약</div>{abstract}</div><div class='kw'><b>핵심어</b>: {keywords}</div>")
html = f"<!doctype html><html><head><meta charset='utf-8'><title>{TITLE}</title><style>{PF.CSS}</style></head><body>{cover}{toc}<div class='pg'></div>{doc}</body></html>"
(OUT / "최종보고서_논문형.html").write_text(html, encoding="utf-8"); PF.dump_tables(P, OUT / "보고서표_논문형")
print("saved paper", len(html), "tables", P.tn, "figures", P.fn, "equations", P.en)
