"""보고서 공통 형식(통계청 보도자료형): CSS·표·그림·개조식 문단. 130_final_report.py(최종 보고서)와 132_foc_report.py(1계조건 보고서 재작성)가 함께 쓴다.
형식 규칙의 글 설명은 산출물/보고서형식_프롬프트.md."""
import base64

CSS = """
@page { size: A4; margin: 15mm 14mm 15mm 14mm; @bottom-center { content: "- " counter(page) " -"; font-size: 8.5pt; color: #555; } }
* { box-sizing: border-box; }
body { font-family: "AppleMyungjo", "Batang", serif; font-size: 10.3pt; line-height: 1.62; color: #111; margin: 0; }
h1, h2, h3, .g, table, .tt, .ft, .note, .src, .kpi, .cover, .toc, .box, .sc { font-family: "Apple SD Gothic Neo", "AppleGothic", sans-serif; }
.page { break-before: page; }
h2 { font-size: 15pt; color: #fff; background: #1F4E79; padding: 5px 12px; margin: 0 0 10px 0; letter-spacing: 0.5px; }
h3 { font-size: 12pt; color: #1F4E79; border-bottom: 1.5px solid #1F4E79; padding-bottom: 2px; margin: 14px 0 6px 0; }
h4 { font-family: "Apple SD Gothic Neo", sans-serif; font-size: 10.8pt; margin: 10px 0 4px 0; color: #222; }
p { margin: 3px 0; }
.L1, .L2, .L3 { display: flex; margin: 3px 0; } .L1 > .tx, .L2 > .tx, .L3 > .tx { flex: 1 1 0; width: auto; min-width: 0; } .L1 > span, .L2 > span, .L3 > span { flex: 0 0 auto; width: 1.4em; }
.L1 { font-weight: 600; font-family: "Apple SD Gothic Neo", sans-serif; margin-top: 7px; } .L2 { padding-left: 1.1em; } .L3 { padding-left: 2.4em; font-size: 9.8pt; color: #333; }
.nt { font-size: 9pt; color: #444; padding-left: 1.1em; }
.tbl, .fig { margin: 8px 0 10px 0; break-inside: avoid; }
.tt, .ft { display: flex; justify-content: space-between; font-size: 9.6pt; font-weight: 700; margin-bottom: 3px; color: #1F4E79; }
.unit { font-weight: 400; color: #444; font-size: 8.8pt; }
table { width: 100%; border-collapse: collapse; font-size: 8.7pt; line-height: 1.35; }
th { background: #DDEBF7; border-top: 2px solid #1F4E79; border-bottom: 1px solid #1F4E79; padding: 4px 4px; font-weight: 700; }
td { border-bottom: 1px solid #d0d7e2; padding: 3px 5px; text-align: center; }
td:first-child { text-align: left; } tbody tr:last-child td { border-bottom: 1.5px solid #1F4E79; }
tr.hl td { background: #EEF4FB; font-weight: 700; } tr.sub td { color: #555; } tr.ref td { color: #C00000; }
.note, .src { font-size: 8.2pt; color: #444; margin-top: 2px; } .src { color: #666; }
.fig img { display: block; margin: 2px auto 0 auto; width: auto; height: auto; max-height: 74mm; }
h3, h4, .ft, .tt { break-after: avoid; }
.summ .L2 { font-size: 9.7pt; line-height: 1.5; margin: 2px 0; } .summ h4 { margin: 6px 0 2px 0; }
.box { border: 1.5px solid #1F4E79; padding: 8px 12px; margin: 8px 0; background: #F7FAFD; font-size: 9.6pt; }
.formula { font-family: "Apple SD Gothic Neo", sans-serif; background: #F3F6FA; border-left: 4px solid #5B9BD5; padding: 6px 12px; margin: 6px 0; font-size: 9.8pt; }
.cover { padding-top: 6mm; }
.cover .hdr { width: 100%; border-collapse: collapse; font-size: 10pt; }
.cover .hdr td { border: 1px solid #1F4E79; padding: 6px 10px; text-align: left; }
.cover .hdr .lab { background: #1F4E79; color: #fff; font-weight: 700; width: 18%; text-align: center; }
.cover .badge { display: inline-block; border: 2px solid #1F4E79; color: #1F4E79; font-weight: 800; font-size: 16pt; padding: 3px 10px; letter-spacing: 3px; white-space: nowrap; }
.cover .title { font-size: 21pt; font-weight: 800; line-height: 1.35; color: #111; margin: 16mm 0 3mm 0; text-align: center; }
.cover .sub { font-size: 11.5pt; text-align: center; color: #1F4E79; margin-bottom: 10mm; }
.cover .keys { border-top: 3px solid #1F4E79; border-bottom: 3px solid #1F4E79; padding: 8px 6px; font-family: "AppleMyungjo", serif; }
.cover .foot { margin-top: 12mm; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 6px 0 10px 0; }
.kpi { border: 1.5px solid #1F4E79; border-radius: 4px; padding: 8px 6px; text-align: center; }
.kpi .v { font-size: 21pt; font-weight: 800; color: #1F4E79; line-height: 1.15; } .kpi .l { font-size: 9pt; font-weight: 700; } .kpi .c { font-size: 8.3pt; color: #555; }
.toc { font-size: 9.8pt; line-height: 1.45; column-count: 2; column-gap: 9mm; } .toc .lv1 { break-after: avoid; } .toc p { margin: 1px 0; } .toc .lv1 { font-weight: 700; margin-top: 5px; } .toc .lv2 { padding-left: 1.6em; color: #333; }
.diag { display: flex; align-items: stretch; gap: 4px; font-family: "Apple SD Gothic Neo", sans-serif; font-size: 8.6pt; margin: 6px 0 8px 0; }
.diag .n { flex: 1; border: 1.5px solid #1F4E79; border-radius: 4px; padding: 6px; background: #F7FAFD; line-height: 1.35; } .diag .n b { color: #1F4E79; display: block; margin-bottom: 2px; font-size: 9.2pt; }
.diag .a { flex: 0 0 auto; align-self: center; color: #1F4E79; font-size: 15pt; font-weight: 800; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
small { font-size: 85%; color: #555; }
"""

class Numbering:
    """표·그림 번호를 등장 순서로 매기고, 그림은 figdir 의 PNG를 문서 안에 넣는다."""
    def __init__(self, figdir, prefix=""): self.tn, self.fn, self.figdir, self.rec, self.px = 0, 0, figdir, [], prefix
    def table(self, title, head, rows, unit=None, note=None, src=None, widths=None, cls=""):
        self.rec.append((self.tn + 1, title, unit, note, src, head, [r[0] if isinstance(r, tuple) else r for r in rows]))
        self.tn += 1; h = "".join(f"<th{f' style=width:{w}' if widths else ''}>{c}</th>" for c, w in zip(head, widths or [None] * len(head)))
        body = ""
        for r in rows:
            cells, rc = (r if isinstance(r, tuple) else (r, ""))
            body += f"<tr class='{rc}'>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
        return (f"<div class='tbl {cls}'><div class='tt'><span>&lt;표 {self.px}{self.tn}&gt; {title}</span><span class='unit'>{'(단위: ' + unit + ')' if unit else ''}</span></div>"
                f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>" + (f"<div class='note'>주) {note}</div>" if note else "") + (f"<div class='src'>자료: {src}</div>" if src else "") + "</div>")
    def figure(self, file, title, note=None, src=None, w="100%", h=None):
        self.fn += 1; b = base64.b64encode((self.figdir / file).read_bytes()).decode()
        return (f"<div class='fig'><div class='ft'>[그림 {self.px}{self.fn}] {title}</div><img src='data:image/png;base64,{b}' style='max-width:{w}{";max-height:" + h if h else ""}'>" + (f"<div class='note'>주) {note}</div>" if note else "")
                + (f"<div class='src'>자료: {src}</div>" if src else "") + "</div>")

def plain(s):
    """셀의 HTML을 보이는 글자로: <br> → ' / ', 나머지 태그 제거."""
    import html, re
    return html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"<br\s*/?>", " / ", str(s)))).strip()

def dump_tables(num, outdir, idx_name="표_색인.csv"):
    """보고서에 실린 표를 보이는 그대로 CSV로 저장: 표NN_제목.csv + 표_색인.csv(번호·제목·단위·열·행수·주·자료)."""
    import csv, re
    outdir.mkdir(parents=True, exist_ok=True); idx = []
    for n, title, unit, note, src, head, rows in num.rec:
        fn = f"표{num.px}{n:02d}_" + re.sub(r"[^0-9A-Za-z가-힣]+", "_", plain(title)).strip("_")[:40] + ".csv"
        with open(outdir / fn, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow([plain(c) for c in head]); w.writerows([[plain(c) for c in r] for r in rows])
        idx.append(dict(표=f"{num.px}{n}", 제목=plain(title), 단위=unit or "", 열=" | ".join(plain(c) for c in head), 행수=len(rows), 파일=fn, 주=plain(note or ""), 자료=plain(src or "")))
    with open(outdir / idx_name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(idx[0])); w.writeheader(); w.writerows(idx)
    return idx

def L1(t): return f"<p class='L1'><span>□</span><span class='tx'>{t}</span></p>"
def L2(t): return f"<p class='L2'><span>○</span><span class='tx'>{t}</span></p>"
def L3(t): return f"<p class='L3'><span>-</span><span class='tx'>{t}</span></p>"
def note(t): return f"<p class='nt'>※ {t}</p>"
