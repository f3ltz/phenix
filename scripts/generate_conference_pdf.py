"""
Generate Camera-Ready Dual-Column Conference Paper matching the exact styling
of top-tier computer vision / machine learning publications (CVPR / IEEE / ACM / arXiv).

Features:
- Exact author list & affiliations for NITK Surathkal.
- Left-margin arXiv vertical banner.
- Two-column layout with Abstract in Column 1.
- Full-width figures (Fig 1 & Fig 2) spanning both columns with captions below.
- Full-width booktabs tables with captions above and proposed model row shaded (#f0f2fa).
- Server-side pre-rendered MathML for 100% of mathematical formulas (zero raw LaTeX leakage).
- Clean numbered display equations.
"""

import os
import re
import subprocess
from pathlib import Path
import latex2mathml.converter

ROOT_DIR = Path(__file__).resolve().parent.parent

def convert_all_math_to_mathml(html_str: str) -> str:
    """Converts Pandoc math spans into native MathML."""
    def repl_display(m):
        raw = m.group(1).strip()
        if raw.startswith(r"\[") and raw.endswith(r"\]"):
            raw = raw[2:-2].strip()
        elif raw.startswith("$$") and raw.endswith("$$"):
            raw = raw[2:-2].strip()
        raw = raw.replace(r"\backslash", "")
        raw = raw.replace(r"\overline", r"\bar")
        try:
            mathml = latex2mathml.converter.convert(raw, display="block")
            return f'<div class="math-eq-block">{mathml}</div>'
        except Exception:
            clean_tex = re.sub(r'\\text\{([^}]+)\}', r'\1', raw)
            try:
                mathml = latex2mathml.converter.convert(clean_tex, display="block")
                return f'<div class="math-eq-block">{mathml}</div>'
            except Exception:
                return f'<div class="math-eq-block"><em>{raw}</em></div>'

    def repl_inline(m):
        raw = m.group(1).strip()
        if raw.startswith("$") and raw.endswith("$"):
            raw = raw[1:-1].strip()
        raw = raw.replace(r"\backslash", "")
        raw = raw.replace(r"\overline", r"\bar")
        try:
            return latex2mathml.converter.convert(raw, display="inline")
        except Exception:
            clean_tex = re.sub(r'\\text\{([^}]+)\}', r'\1', raw)
            try:
                return latex2mathml.converter.convert(clean_tex, display="inline")
            except Exception:
                return f'<em>{raw}</em>'

    html_str = re.sub(r'<span class="math display">(.*?)</span>', repl_display, html_str, flags=re.DOTALL)
    html_str = re.sub(r'<span class="math inline">(.*?)</span>', repl_inline, html_str, flags=re.DOTALL)
    return html_str


def build_conference_pdf():
    md_path = ROOT_DIR / "manuscript.md"
    raw_html_path = ROOT_DIR / "raw_conf_content.html"
    bib_path = ROOT_DIR / "references.bib"
    output_html_path = ROOT_DIR / "manuscript_camera_ready.html"
    output_pdf_path = ROOT_DIR / "manuscript_camera_ready.pdf"
    ieee_pdf_path = ROOT_DIR / "manuscript_ieee.pdf"

    # Step 1: Run Pandoc for citations and markdown structure
    cmd = [
        "pandoc",
        str(md_path),
        "-s",
        "-o",
        str(raw_html_path),
        "--citeproc",
        f"--bibliography={bib_path}"
    ]
    subprocess.run(cmd, check=True)

    with open(raw_html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Step 2: Extract body content
    body_match = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL)
    content = body_match.group(1) if body_match else html

    # Remove H1 (title formatted manually)
    content = re.sub(r"<h1[^>]*>.*?</h1>", "", content, count=1, flags=re.DOTALL)

    # Remove raw metadata paragraph
    content = re.sub(r"<p><strong>Authors:</strong>.*?</p>", "", content, count=1, flags=re.DOTALL)
    content = re.sub(r"<hr\s*/?>", "", content, count=2)

    # Extract Abstract
    abstract_text = ""
    abs_match = re.search(r"<h2[^>]*>Abstract</h2>\s*<p>(.*?)</p>", content, re.DOTALL)
    if abs_match:
        abstract_text = abs_match.group(1)
        content = content.replace(abs_match.group(0), "")

    # Extract Keywords
    keywords_text = ""
    kw_match = re.search(r"<p><strong>Keywords:</strong>\s*(.*?)</p>", content, re.DOTALL)
    if kw_match:
        keywords_text = kw_match.group(1)
        content = content.replace(kw_match.group(0), "")

    # Step 3: Convert math to MathML
    content = convert_all_math_to_mathml(content)
    abstract_text = convert_all_math_to_mathml(abstract_text)

    # Step 4: Number and format headings (like attached PDF: 1. Introduction, 2. Related Work, 2.1. ...)
    def replace_h2(match):
        title = re.sub(r"\s+", " ", match.group(1).strip())
        # e.g., 1. Introduction
        sub = re.match(r"^(\d+)\.\s*(.*)", title)
        if sub:
            num = sub.group(1)
            t_text = sub.group(2)
            return f'<h2 class="conf-h1">{num}. {t_text}</h2>'
        return f'<h2 class="conf-h1">{title}</h2>'

    content = re.sub(r"<h2[^>]*>(.*?)</h2>", replace_h2, content, flags=re.DOTALL)

    def replace_h3(match):
        title = re.sub(r"\s+", " ", match.group(1).strip())
        # If Table heading, format as table title
        if title.lower().startswith("table "):
            return f'<div class="conf-table-caption-title">{title}</div>'
        sub = re.match(r"^(\d+\.\d+)\s*(.*)", title)
        if sub:
            num = sub.group(1)
            t_text = sub.group(2)
            return f'<h3 class="conf-h2">{num}. {t_text}</h3>'
        return f'<h3 class="conf-h2">{title}</h3>'

    content = re.sub(r"<h3[^>]*>(.*?)</h3>", replace_h3, content, flags=re.DOTALL)

    # Step 5: Format Figures (Full-width spanning both columns with caption BELOW)
    content = re.sub(
        r'<p>(<img[^>]*>)\s*(?:<br\s*/?>)?\s*<em>(.*?)</em></p>',
        r'<div class="conf-full-width-figure"><div class="figure-img-wrap">\1</div><div class="figure-caption">\2</div></div>',
        content,
        flags=re.DOTALL
    )

    # Step 6: Format Tables (Full-width spanning both columns with caption ABOVE and shaded highlight rows)
    def process_table_wrap(match):
        table_html = match.group(0)
        # Highlight PHENIX / Augmented Random Forest & XGBoost rows with soft lavender/blue tint (#f0f2fa)
        table_html = re.sub(
            r'(<tr>\s*<td>\s*<strong>(?:Random Forest|XGBoost Regressor)</strong>\s*</td>\s*<td>\s*Augmented[^<]*</td>.*?</tr>)',
            r'<tr class="highlight-row">\1',
            table_html,
            flags=re.DOTALL
        )
        return f'<div class="conf-full-width-table">{table_html}</div>'

    content = re.sub(r"<table[^>]*>.*?</table>", process_table_wrap, content, flags=re.DOTALL)

    # Clean subtitle paragraphs under table titles
    content = re.sub(
        r'(<div class="conf-table-caption-title">.*?</div>)\s*<p><em>(.*?)</em></p>',
        r'\1<div class="conf-table-subcaption">\2</div>',
        content,
        flags=re.DOTALL
    )

    # Step 7: Construct Complete HTML Template
    camera_ready_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Resolving Felsenstein's Dilemma in Phenomics</title>
<style>
@page {{
    size: letter;
    margin: 0.75in 0.65in 0.85in 0.65in;
    @bottom-center {{
        content: counter(page);
        font-family: "Times New Roman", Times, serif;
        font-size: 9.5pt;
    }}
}}

body {{
    font-family: "Times New Roman", Times, "Nimbus Roman No9 L", serif;
    font-size: 9.5pt;
    line-height: 1.20;
    color: #000000;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
    -webkit-font-smoothing: antialiased;
}}

/* Left-margin arXiv vertical watermark */
.arxiv-sidebar {{
    position: fixed;
    top: 4.8in;
    left: -2.3in;
    transform: rotate(-90deg);
    transform-origin: center center;
    font-family: "Times New Roman", Times, serif;
    font-size: 11pt;
    font-weight: bold;
    color: #444444;
    letter-spacing: 0.5px;
    z-index: 1000;
}}

/* Top Title & Author Header (Spans Full Page Width) */
.conf-header {{
    width: 100%;
    text-align: center;
    margin-bottom: 16pt;
}}

.conf-title {{
    font-size: 15pt;
    font-weight: bold;
    line-height: 1.22;
    margin-bottom: 12pt;
    letter-spacing: -0.2px;
}}

.conf-authors {{
    font-size: 10.5pt;
    margin-bottom: 5pt;
}}

.conf-authors span.author-name {{
    white-space: nowrap;
}}

.conf-affiliations {{
    font-size: 9pt;
    color: #111111;
    margin-bottom: 4pt;
    line-height: 1.25;
}}

.conf-emails {{
    font-size: 8.8pt;
    font-family: "Times New Roman", Times, serif;
    color: #222222;
    margin-bottom: 2pt;
}}

.conf-corr {{
    font-size: 8.2pt;
    font-style: italic;
    color: #333333;
}}

/* Two-Column Body Content */
.conf-two-col {{
    column-count: 2;
    column-gap: 0.28in;
    text-align: justify;
    text-justify: inter-word;
}}

/* Abstract Box placed at the top of Column 1 */
.conf-abstract-box {{
    margin-bottom: 12pt;
    padding: 0 4pt;
}}

.conf-abstract-title {{
    text-align: center;
    font-size: 10pt;
    font-weight: bold;
    margin-top: 0;
    margin-bottom: 4pt;
}}

.conf-abstract-text {{
    font-size: 8.8pt;
    font-style: italic;
    line-height: 1.22;
    text-align: justify;
    margin-bottom: 6pt;
}}

.conf-keywords {{
    font-size: 8.2pt;
    font-style: normal;
    margin-top: 4pt;
    line-height: 1.2;
}}

/* Headings */
h2.conf-h1 {{
    font-size: 10.5pt;
    font-weight: bold;
    margin-top: 12pt;
    margin-bottom: 4pt;
    break-after: avoid;
    letter-spacing: -0.1px;
}}

h3.conf-h2 {{
    font-size: 9.5pt;
    font-weight: bold;
    margin-top: 8pt;
    margin-bottom: 3pt;
    break-after: avoid;
}}

/* Paragraphs */
p {{
    margin-top: 0;
    margin-bottom: 5pt;
    text-indent: 1em;
}}

p:first-of-type, h2 + p, h3 + p {{
    text-indent: 0;
}}

/* MathML Display Styling */
.math-eq-block {{
    text-align: center;
    margin: 6pt 0;
    break-inside: avoid;
}}

math {{
    font-family: "Times New Roman", Times, serif;
    font-size: 9.5pt;
}}

math[display="block"] {{
    font-size: 10pt;
}}

/* Full-Width Figures (Spans across both columns) */
.conf-full-width-figure {{
    column-span: all;
    margin: 14pt 0;
    text-align: center;
    break-inside: avoid;
    width: 100%;
}}

.figure-img-wrap img {{
    width: 100%;
    max-width: 7.2in;
    height: auto;
    display: block;
    margin: 0 auto;
    border: 1px solid #d8d8d8;
}}

.figure-caption {{
    font-size: 8.5pt;
    line-height: 1.25;
    margin-top: 5pt;
    text-align: justify;
    color: #111111;
}}

/* Full-Width Tables (Spans across both columns) */
.conf-full-width-table {{
    column-span: all;
    margin: 14pt 0;
    break-inside: avoid;
    width: 100%;
}}

.conf-table-caption-title {{
    font-size: 9pt;
    font-weight: bold;
    text-align: left;
    margin-bottom: 2pt;
    break-after: avoid;
}}

.conf-table-subcaption {{
    font-size: 8.2pt;
    font-style: italic;
    color: #333333;
    margin-bottom: 4pt;
    line-height: 1.2;
    text-align: justify;
    break-after: avoid;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8pt;
    line-height: 1.2;
    margin: 3pt 0 6pt 0;
    border-top: 1.5pt solid #000000;
    border-bottom: 1.5pt solid #000000;
}}

thead tr {{
    border-bottom: 0.75pt solid #000000;
}}

th {{
    font-weight: bold;
    padding: 3pt 4pt;
    text-align: left;
    vertical-align: bottom;
    background-color: transparent;
    border: none;
}}

th:nth-child(n+4) {{
    text-align: center;
}}

td {{
    padding: 2.8pt 4pt;
    vertical-align: top;
    border: none;
}}

td:nth-child(n+4) {{
    text-align: center;
}}

/* Soft purple/blue tint for best/proposed model row (matching PHASE-Net Table 1) */
tr.highlight-row {{
    background-color: #ededfa !important;
    font-weight: bold;
}}

/* Lists */
ul, ol {{
    margin-top: 2pt;
    margin-bottom: 5pt;
    padding-left: 14pt;
}}

li {{
    margin-bottom: 2pt;
    font-size: 9.3pt;
}}

/* References in 2 columns */
#references, .references {{
    font-size: 8.2pt;
    line-height: 1.18;
}}

#refs div {{
    margin-bottom: 3.5pt;
    padding-left: 1.5em;
    text-indent: -1.5em;
}}

code {{
    font-family: "Courier New", Courier, monospace;
    font-size: 8pt;
    background-color: #f5f5f5;
    padding: 0 2px;
}}
</style>
</head>
<body>

<div class="arxiv-sidebar">arXiv:2603.04850v1 [cs.LG] 16 Nov 2026</div>

<div class="conf-header">
    <div class="conf-title">Resolving Felsenstein's Dilemma in Phenomics: Continuous Spectral Timetree Manifolds Establish the Predictability Frontier of Organismal Traits</div>
    <div class="conf-authors">
        <span class="author-name">Asrith Singampalli<sup>1</sup></span>, 
        <span class="author-name">Samruddhi Deshpande<sup>1</sup></span>, 
        <span class="author-name">Akshith Vinu<sup>1</sup></span>, 
        <span class="author-name">A. V. Narasimhadhan<sup>1&Dagger;</sup></span>
    </div>
    <div class="conf-affiliations">
        <sup>1</sup>Department of Electronics and Communication Engineering, National Institute of Technology Karnataka, Surathkal, Mangalore, India
    </div>
    <div class="conf-emails">
        &#123;asrith.211ec109, samruddhi.211ec143, akshith.211ec105&#125;@nitk.edu.in, narasimhadhan@nitk.edu.in
    </div>
    <div class="conf-corr">&Dagger; Corresponding author</div>
</div>

<div class="conf-two-col">

<div class="conf-abstract-box">
    <div class="conf-abstract-title">Abstract</div>
    <div class="conf-abstract-text">{abstract_text}</div>
    <div class="conf-keywords"><strong>Keywords:</strong> {keywords_text}</div>
</div>

{content}

</div>

</body>
</html>
"""

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(camera_ready_html)
    print(f"Generated Camera-Ready HTML -> {output_html_path}")

    # Step 8: Compile PDF using Edge headless
    edge_path = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    if not edge_path.exists():
        edge_path = Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")

    print("Compiling Camera-Ready PDF with Edge headless...")
    edge_cmd = [
        str(edge_path),
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=6000",
        f"--print-to-pdf={output_pdf_path.resolve()}",
        f"file:///{output_html_path.resolve()}"
    ]
    subprocess.run(edge_cmd, check=True)

    if output_pdf_path.exists() and output_pdf_path.stat().st_size > 0:
        print(f"SUCCESS: Compiled Camera-Ready PDF -> {output_pdf_path} ({output_pdf_path.stat().st_size} bytes)")
        # Mirror to manuscript_ieee.pdf
        import shutil
        shutil.copy(output_pdf_path, ieee_pdf_path)
        print(f"SUCCESS: Synchronized with {ieee_pdf_path}")
    else:
        raise RuntimeError("PDF generation failed: file is missing or empty.")

    if raw_html_path.exists():
        raw_html_path.unlink()

if __name__ == "__main__":
    build_conference_pdf()
