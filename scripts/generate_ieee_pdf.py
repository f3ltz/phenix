"""
Script to transform manuscript.md into a high-fidelity IEEE Conference PDF.
Uses Pandoc for citeproc and KaTeX, post-processes HTML for exact IEEE styling,
and compiles to PDF via headless Microsoft Edge.
"""

import os
import re
import subprocess
from pathlib import Path

def generate_ieee_html():
    raw_html_path = Path("raw_content.html")
    # Generate raw HTML using pandoc
    cmd = [
        "pandoc",
        "manuscript.md",
        "-s",
        "-o",
        str(raw_html_path),
        "--citeproc",
        "--bibliography=references.bib",
        "--math-method=katex"
    ]
    subprocess.run(cmd, check=True)

    with open(raw_html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Extract body content
    body_match = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL)
    if body_match:
        content = body_match.group(1)
    else:
        content = html

    # Remove the first H1 if present (we will format the IEEE header manually)
    content = re.sub(r"<h1[^>]*>.*?</h1>", "", content, count=1, flags=re.DOTALL)

    # Remove the metadata paragraph (Authors:, Target Venues:, etc.)
    content = re.sub(r"<p><strong>Authors:</strong>.*?</p>", "", content, count=1, flags=re.DOTALL)
    content = re.sub(r"<hr\s*/?>", "", content, count=2)

    # Extract Abstract and Keywords
    abstract_text = ""
    abs_match = re.search(r"<h2[^>]*>Abstract</h2>\s*<p>(.*?)</p>", content, re.DOTALL)
    if abs_match:
        abstract_text = abs_match.group(1)
        content = content.replace(abs_match.group(0), "")

    keywords_text = ""
    kw_match = re.search(r"<p><strong>Keywords:</strong>\s*(.*?)</p>", content, re.DOTALL)
    if kw_match:
        keywords_text = kw_match.group(1)
        content = content.replace(kw_match.group(0), "")

    # Clean up Figure 1 ascii box art
    content = re.sub(r"<pre><code>.*?\[High-Dimensional.*?</pre>", "", content, flags=re.DOTALL)
    content = re.sub(r"<pre><code>.*?STAGE 1: MULTIMODAL INGESTION.*?</pre>", "", content, flags=re.DOTALL)

    # Format Section Headings to IEEE Roman Numerals
    def replace_h2(match):
        title = match.group(1).strip()
        # Map titles to Roman Numerals
        num_map = {
            "1. Introduction": "I. INTRODUCTION",
            "2. Related Work": "II. RELATED WORK",
            "3. The PHENIX Framework &amp; Methodology": "III. THE PHENIX METHODOLOGY",
            "3. The PHENIX Framework & Methodology": "III. THE PHENIX METHODOLOGY",
            "4. Empirical Benchmark Results": "IV. EMPIRICAL BENCHMARK RESULTS",
            "5. Discussion": "V. DISCUSSION",
            "6. Conclusion": "VI. CONCLUSION",
            "References": "REFERENCES"
        }
        clean_title = num_map.get(title, title.upper())
        return f'<h2 class="ieee-heading-1">{clean_title}</h2>'

    content = re.sub(r"<h2[^>]*>(.*?)</h2>", replace_h2, content)

    # Format Subsections to IEEE Letters
    def replace_h3(match):
        title = match.group(1).strip()
        # Subsections like 2.1 The Non-Independence... -> A. The Non-Independence...
        sub_match = re.match(r"^\d+\.(\d+)\s+(.*)", title)
        if sub_match:
            sub_num = int(sub_match.group(1))
            letter = chr(ord('A') + sub_num - 1)
            sub_title = sub_match.group(2)
            return f'<h3 class="ieee-heading-2">{letter}. {sub_title}</h3>'
        return f'<h3 class="ieee-heading-2">{title}</h3>'

    content = re.sub(r"<h3[^>]*>(.*?)</h3>", replace_h3, content)

    # Make Table 1 and Table 2 span both columns
    content = re.sub(r"(<table[^>]*>)", r'<div class="ieee-full-width-table">\1', content)
    content = re.sub(r"(</table>)", r"\1</div>", content)

    # IEEE Document Template with strict CSS
    ieee_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Resolving Felsenstein's Dilemma in Phenomics</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<style>
@page {{
    size: letter;
    margin: 0.75in 0.625in 1.0in 0.625in;
}}

body {{
    font-family: "Times New Roman", Times, Georgia, serif;
    font-size: 10pt;
    line-height: 1.18;
    color: #000;
    margin: 0;
    padding: 0;
    text-align: justify;
    background-color: #fff;
    -webkit-print-color-adjust: exact;
}}

/* Header Banner - Full Width */
.ieee-title-banner {{
    text-align: center;
    margin-bottom: 1.5em;
    width: 100%;
}}

.ieee-title {{
    font-size: 20pt;
    font-weight: bold;
    line-height: 1.15;
    margin-bottom: 0.6em;
}}

.ieee-authors {{
    font-size: 10.5pt;
    margin-bottom: 0.3em;
}}

.ieee-affil {{
    font-size: 9.5pt;
    font-style: italic;
    color: #333;
    margin-bottom: 0.3em;
}}

.ieee-email {{
    font-size: 9pt;
    font-family: "Courier New", monospace;
    margin-bottom: 1.5em;
}}

/* Two Column Layout */
.ieee-content {{
    column-count: 2;
    column-gap: 0.25in;
    text-align: justify;
    width: 100%;
}}

/* Abstract and Keywords block */
.ieee-abstract-box {{
    margin-bottom: 1.2em;
    font-size: 9pt;
    line-height: 1.15;
}}

.ieee-abstract-title {{
    font-weight: bold;
    font-style: italic;
}}

.ieee-keywords-box {{
    margin-bottom: 1.4em;
    font-size: 9pt;
    line-height: 1.15;
}}

.ieee-keywords-title {{
    font-weight: bold;
    font-style: italic;
}}

/* Headings */
h2.ieee-heading-1 {{
    font-size: 10pt;
    font-weight: normal;
    font-variant: small-caps;
    text-align: center;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
    letter-spacing: 0.05em;
    break-after: avoid;
    page-break-after: avoid;
}}

h3.ieee-heading-2 {{
    font-size: 10pt;
    font-weight: normal;
    font-style: italic;
    text-align: left;
    margin-top: 0.9em;
    margin-bottom: 0.25em;
    break-after: avoid;
    page-break-after: avoid;
}}

/* Paragraphs */
p {{
    text-indent: 0.25in;
    margin-top: 0;
    margin-bottom: 0;
}}

p.no-indent {{
    text-indent: 0;
}}

/* Lists */
ul, ol {{
    margin-top: 0.3em;
    margin-bottom: 0.4em;
    padding-left: 1.2em;
    font-size: 9.5pt;
}}

li {{
    margin-bottom: 0.2em;
}}

/* Tables */
.ieee-full-width-table {{
    column-span: all;
    margin: 1.4em 0;
    width: 100%;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8pt;
    line-height: 1.15;
    margin: 0.6em auto;
}}

th, td {{
    padding: 3px 5px;
    border-top: 0.5pt solid #000;
    border-bottom: 0.5pt solid #000;
    text-align: left;
}}

thead tr:first-child th {{
    border-top: 1.2pt solid #000;
}}

thead tr:last-child th {{
    border-bottom: 0.8pt solid #000;
}}

tbody tr:last-child td {{
    border-bottom: 1.2pt solid #000;
}}

caption {{
    font-variant: small-caps;
    font-size: 8.5pt;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
    caption-side: top;
}}

/* Equations */
.katex-display {{
    margin: 0.4em 0 !important;
    text-align: center;
    font-size: 9.5pt;
}}

/* References */
#references, .references {{
    font-size: 8pt;
    line-height: 1.15;
}}

.references p {{
    text-indent: -0.22in;
    padding-left: 0.22in;
    margin-bottom: 0.35em;
}}

/* Avoid orphan lines */
p, li {{
    orphans: 2;
    widows: 2;
}}
</style>
</head>
<body>

<div class="ieee-title-banner">
    <div class="ieee-title">Resolving Felsenstein’s Dilemma in Phenomics: Continuous Spectral Timetree Manifolds Establish the Predictability Frontier of Organismal Traits</div>
    <div class="ieee-authors">The PHENIX Consortium</div>
    <div class="ieee-affil">Computational Biology & Evolutionary Machine Learning Group</div>
    <div class="ieee-email">contact@phenix-project.org &nbsp;|&nbsp; https://github.com/phenix-project/phenix</div>
</div>

<div class="ieee-content">

    <div class="ieee-abstract-box">
        <span class="ieee-abstract-title">Abstract—</span>{abstract_text}
    </div>

    <div class="ieee-keywords-box">
        <span class="ieee-keywords-title">Index Terms—</span>{keywords_text}
    </div>

    {content}

</div>

</body>
</html>
"""

    out_html = Path("manuscript_ieee.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(ieee_html)
    print(f"Generated IEEE HTML -> {out_html}")
    return out_html

def compile_pdf(html_path):
    pdf_out = Path("manuscript_ieee.pdf").resolve()
    html_abs = Path(html_path).resolve()

    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    cmd = [
        edge_path,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--virtual-time-budget=6000",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_out}",
        f"{html_abs.as_uri()}"
    ]

    print("Compiling PDF with Edge headless...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if pdf_out.exists() and pdf_out.stat().st_size > 50000:
        print(f"SUCCESS: Compiled IEEE PDF -> {pdf_out} ({pdf_out.stat().st_size} bytes)")
    else:
        print(f"FAILED to generate PDF: {res.stderr}")

if __name__ == "__main__":
    html_path = generate_ieee_html()
    compile_pdf(html_path)
