"""논문 형식(Applied Stochastic Models in Business and Industry의 절 구성 + 국내 학술 논문 서술체) 공통 서식.
절 모듈(scripts/paper/sec_*.py)은 build(P)에서 P의 메서드로 HTML 조각을 만들고, 138_paper.py가 순서대로 이어 붙인다.
표·그림·식 번호는 등장 순서로 자동 부여하고, 본문에서는 [[t:라벨]] · [[f:라벨]] · [[e:라벨]] 토큰으로 가리킨다(최종 조립 때 「표 3」 「그림 2」 「식 (4)」로 바뀜).
"""
import base64, html, re, csv

CSS = """
@page { size: A4; margin: 22mm 20mm 22mm 20mm; @bottom-center { content: counter(page); font-size: 9pt; color: #333; } }
* { box-sizing: border-box; }
body { font-family: "AppleMyungjo", "Batang", serif; font-size: 10.4pt; line-height: 1.78; color: #111; margin: 0; text-align: justify; word-break: keep-all; }
h1, h2, h3, h4, .cap, .fcap, table, .fm, .kw, .au, .abs-h { font-family: "Apple SD Gothic Neo", "AppleGothic", sans-serif; }
h2 { font-size: 13.5pt; margin: 20px 0 6px 0; color: #111; } h3 { font-size: 11.5pt; margin: 14px 0 4px 0; color: #111; } h4 { font-size: 10.6pt; margin: 10px 0 2px 0; font-weight: 700; }
p { margin: 0 0 5px 0; text-indent: 1em; } p.ni { text-indent: 0; }
.pg { break-before: page; }
.title { font-family: "Apple SD Gothic Neo", sans-serif; font-size: 19pt; font-weight: 800; text-align: center; line-height: 1.4; margin: 26mm 0 4mm 0; }
.subtitle { font-family: "Apple SD Gothic Neo", sans-serif; font-size: 11pt; text-align: center; color: #333; margin-bottom: 10mm; }
.au { text-align: center; font-size: 11pt; margin: 4mm 0; }
.abs { margin: 8mm 6mm 4mm 6mm; padding: 8px 0; border-top: 1.5px solid #111; border-bottom: 1.5px solid #111; } .abs p { text-indent: 0; font-size: 10pt; line-height: 1.7; }
.abs-h { font-weight: 800; font-size: 10.5pt; margin-bottom: 3px; } .kw { margin: 4px 6mm; font-size: 10pt; }
.toc { font-family: "Apple SD Gothic Neo", sans-serif; font-size: 10pt; line-height: 1.6; margin: 8mm 6mm; } .toc p { text-indent: 0; margin: 1px 0; } .toc .l2 { padding-left: 1.6em; color: #333; } .toc .l1 { font-weight: 700; margin-top: 5px; }
.tbl, .fig { margin: 9px 0 11px 0; break-inside: avoid; }
.cap { font-size: 9.6pt; font-weight: 700; margin-bottom: 3px; text-align: left; } .fcap { font-size: 9.4pt; margin-top: 3px; text-align: left; }
table { width: 100%; border-collapse: collapse; font-size: 8.8pt; line-height: 1.4; }
th { border-top: 1.6px solid #111; border-bottom: 1px solid #111; padding: 3px 4px; font-weight: 700; background: none; }
td { border-bottom: 0.5px solid #bbb; padding: 3px 5px; text-align: center; } td:first-child { text-align: left; } tbody tr:last-child td { border-bottom: 1.6px solid #111; }
tr.hl td { font-weight: 700; background: #f2f2f2; } .tfoot { font-size: 8.4pt; color: #333; margin-top: 2px; text-indent: 0; }
.fig img { display: block; margin: 2px auto 0 auto; width: auto; height: auto; max-height: 76mm; }
.fm { margin: 6px 0; padding: 6px 4mm; display: flex; justify-content: space-between; align-items: center; font-size: 10pt; background: none; } .fm .b { flex: 1; text-align: center; line-height: 1.7; } .fm .n { flex: 0 0 auto; }
ol.st { margin: 3px 0 6px 0; padding-left: 2.1em; } ol.st li { margin: 1px 0; text-indent: 0; }
sub, sup { line-height: 0; }
"""

class Paper:
    def __init__(self, figdir):
        self.tn = self.fn = self.en = 0; self.figdir = figdir; self.labels = {}; self.rec = []
    def _reg(self, kind, label, n):
        if label:
            assert (kind, label) not in self.labels, (kind, label); self.labels[(kind, label)] = n
    def h2(self, t, page=False): return f"<h2{' class=pg' if page else ''}>{t}</h2>"
    def h3(self, t): return f"<h3>{t}</h3>"
    def h4(self, t): return f"<h4>{t}</h4>"
    def p(self, t, ni=False): return f"<p{' class=ni' if ni else ''}>{t}</p>"
    def steps(self, items): return "<ol class='st'>" + "".join(f"<li>{t}</li>" for t in items) + "</ol>"
    def eq(self, body, label=None):
        self.en += 1; self._reg("e", label, self.en); return f"<div class='fm'><span class='b'>{body}</span><span class='n'>({self.en})</span></div>"
    def tab(self, label, caption, head, rows, widths=None, foot=None):
        """rows의 원소는 셀 리스트 또는 (셀 리스트, 'hl'). foot는 표 아래 짧은 설명(약어·단위 정의에만 사용)."""
        self.tn += 1; self._reg("t", label, self.tn); self.rec.append((self.tn, caption, head, [r[0] if isinstance(r, tuple) else r for r in rows]))
        h = "".join(f"<th{f' style=width:{w}' if widths else ''}>{c}</th>" for c, w in zip(head, widths or [None] * len(head))); body = ""
        for r in rows:
            cells, rc = (r if isinstance(r, tuple) else (r, "")); body += f"<tr class='{rc}'>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
        return (f"<div class='tbl'><div class='cap'>표 {self.tn}. {caption}</div><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"
                + (f"<p class='tfoot'>{foot}</p>" if foot else "") + "</div>")
    def fig(self, label, file, caption, h=None):
        self.fn += 1; self._reg("f", label, self.fn); b = base64.b64encode((self.figdir / file).read_bytes()).decode()
        return (f"<div class='fig'><img src='data:image/png;base64,{b}'{f' style=max-height:{h}' if h else ''}><div class='fcap'><b>그림 {self.fn}.</b> {caption}</div></div>")
    def resolve(self, doc):
        """본문의 [[t:라벨]] 등을 번호로 바꾼다. 없는 라벨은 오류."""
        def sub(m):
            k, lab = m.group(1), m.group(2); n = self.labels[(k, lab)]
            return {"t": f"표 {n}", "f": f"그림 {n}", "e": f"식 ({n})"}[k]
        return re.sub(r"\[\[([tfe]):([^\]]+)\]\]", sub, doc)

def plain(s): return html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"<br\s*/?>", " / ", str(s)))).strip()

def dump_tables(P, outdir, prefix="표"):
    outdir.mkdir(parents=True, exist_ok=True); idx = []
    for n, cap, head, rows in P.rec:
        fn = f"{prefix}{n:02d}_" + re.sub(r"[^0-9A-Za-z가-힣]+", "_", plain(cap)).strip("_")[:40] + ".csv"
        with open(outdir / fn, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow([plain(c) for c in head]); w.writerows([[plain(c) for c in r] for r in rows])
        idx.append(dict(표=n, 제목=plain(cap), 열=" | ".join(plain(c) for c in head), 행수=len(rows), 파일=fn))
    with open(outdir / "표_색인.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(idx[0])); w.writeheader(); w.writerows(idx)
