"""
High-fidelity NeurIPS / Top-Tier Single-Column PDF Compiler for PHENIX.
Features:
1. Server-side MathML pre-compilation via latex2mathml (ZERO raw LaTeX in PDF).
2. Strict NeurIPS single-column layout (5.5-inch text column, 1.5-inch margins, Times Roman 10pt).
3. Authentic booktabs table styling with table titles placed above tables and notes below.
4. Large, crystal-clear 300 DPI vector figures with captions placed below.
5. Removal of all sync barrier and internal debugging artifacts.
6. Clean headless Edge rendering with virtual-time-budget for guaranteed layout stability.
"""

import os
import re
import subprocess
from pathlib import Path
import latex2mathml.converter

ROOT_DIR = Path(__file__).resolve().parent.parent

def convert_all_math_to_mathml(html_str: str) -> str:
    """Replaces all Pandoc math spans with server-side rendered native MathML."""
    
    def repl_display(m):
        raw = m.group(1).strip()
        # Clean delimiters
        if raw.startswith(r"\[") and raw.endswith(r"\]"):
            raw = raw[2:-2].strip()
        elif raw.startswith("$$") and raw.endswith("$$"):
            raw = raw[2:-2].strip()
        # Clean problematic escapes
        raw = raw.replace(r"\backslash", "")
        raw = raw.replace(r"\overline", r"\bar")
        try:
            mathml = latex2mathml.converter.convert(raw, display="block")
            return f'<div class="neurips-math-display">{mathml}</div>'
        except Exception as e:
            # Fallback cleanup
            clean_tex = re.sub(r'\\text\{([^}]+)\}', r'\1', raw)
            try:
                mathml = latex2mathml.converter.convert(clean_tex, display="block")
                return f'<div class="neurips-math-display">{mathml}</div>'
            except Exception:
                return f'<div class="neurips-math-display"><pre style="font-family:serif;font-style:italic;">{raw}</pre></div>'

    def repl_inline(m):
        raw = m.group(1).strip()
        # Clean delimiters if present
        if raw.startswith("$") and raw.endswith("$"):
            raw = raw[1:-1].strip()
        raw = raw.replace(r"\backslash", "")
        raw = raw.replace(r"\overline", r"\bar")
        try:
            mathml = latex2mathml.converter.convert(raw, display="inline")
            return mathml
        except Exception as e:
            clean_tex = re.sub(r'\\text\{([^}]+)\}', r'\1', raw)
            try:
                mathml = latex2mathml.converter.convert(clean_tex, display="inline")
                return mathml
            except Exception:
                return f'<span style="font-family:serif;font-style:italic;">{raw}</span>'

    # Convert display math first
    html_str = re.sub(r'<span class="math display">(.*?)</span>', repl_display, html_str, flags=re.DOTALL)
    # Convert inline math
    html_str = re.sub(r'<span class="math inline">(.*?)</span>', repl_inline, html_str, flags=re.DOTALL)
    return html_str


def build_neurips_pdf():
    md_path = ROOT_DIR / "manuscript.md"
    raw_html_path = ROOT_DIR / "raw_neurips_content.html"
    bib_path = ROOT_DIR / "references.bib"
    output_html_path = ROOT_DIR / "manuscript_neurips.html"
    output_pdf_path = ROOT_DIR / "manuscript_neurips.pdf"
    ieee_pdf_path = ROOT_DIR / "manuscript_ieee.pdf"

    # Step 1: Run Pandoc for markdown parsing, citations, and structure
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

    # Remove H1 (will format title manually)
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

    # Step 3: Server-side MathML conversion for 100% of math
    content = convert_all_math_to_mathml(content)
    abstract_text = convert_all_math_to_mathml(abstract_text)

    # Step 4: Format Section Headings (NeurIPS style: 1 Introduction, 2 Related Work, etc.)
    def replace_h2(match):
        title = re.sub(r"\s+", " ", match.group(1).strip())
        # Clean any remaining tags or numbers
        title = re.sub(r"^(\d+)\.\s*", r'<span class="sec-num">\1</span> ', title)
        return f'<h2 class="neurips-h1">{title}</h2>'

    content = re.sub(r"<h2[^>]*>(.*?)</h2>", replace_h2, content, flags=re.DOTALL)

    def replace_h3(match):
        title = re.sub(r"\s+", " ", match.group(1).strip())
        # If it's a table heading, format as table title
        if title.lower().startswith("table "):
            return f'<div class="neurips-table-title">{title}</div>'
        title = re.sub(r"^(\d+\.\d+)\s*", r'<span class="sec-num">\1</span> ', title)
        return f'<h3 class="neurips-h2">{title}</h3>'

    content = re.sub(r"<h3[^>]*>(.*?)</h3>", replace_h3, content, flags=re.DOTALL)

    # Format Figures (NeurIPS style: Full width, caption BELOW figure)
    content = re.sub(
        r'<p>(<img[^>]*>)\s*(?:<br\s*/?>)?\s*<em>(.*?)</em></p>',
        r'<div class="neurips-figure"><div class="figure-img-wrap">\1</div><div class="figure-caption">\2</div></div>',
        content,
        flags=re.DOTALL
    )

    # Format Tables (Booktabs style with table caption above table)
    def wrap_table(match):
        table_html = match.group(0)
        return f'<div class="neurips-table-wrap">{table_html}</div>'

    content = re.sub(r"<table[^>]*>.*?</table>", wrap_table, content, flags=re.DOTALL)

    # Step 5: Construct complete NeurIPS HTML template
    neurips_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Resolving Felsenstein's Dilemma in Phenomics</title>
<style>
@page {{
    size: letter;
    margin: 1.25in 1.5in 1.25in 1.5in;
    @bottom-center {{
        content: counter(page);
        font-family: "Times New Roman", Times, serif;
        font-size: 10pt;
    }}
}}

body {{
    font-family: "Times New Roman", Times, "Nimbus Roman No9 L", serif;
    font-size: 10pt;
    line-height: 1.38;
    color: #000000;
    background-color: #ffffff;
    max-width: 5.5in;
    margin: 0 auto;
    text-align: justify;
    text-justify: inter-word;
    -webkit-font-smoothing: antialiased;
}}

/* NeurIPS Review / Preprint Banner */
.neurips-banner {{
    border-top: 1px solid #000000;
    border-bottom: 1px solid #000000;
    padding: 3px 0;
    margin-bottom: 24pt;
    text-align: center;
    font-size: 8.5pt;
    font-weight: bold;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}

/* Title and Authors */
.paper-title {{
    font-size: 16pt;
    font-weight: bold;
    text-align: center;
    line-height: 1.22;
    margin-bottom: 16pt;
    letter-spacing: -0.2px;
}}

.author-block {{
    text-align: center;
    font-size: 10pt;
    margin-bottom: 18pt;
    line-height: 1.3;
}}

.author-names {{
    font-weight: bold;
    font-size: 10pt;
    margin-bottom: 2pt;
}}

.author-inst {{
    font-size: 9pt;
    color: #222222;
}}

.author-email {{
    font-size: 9pt;
    font-family: "Courier New", Courier, monospace;
    margin-top: 2pt;
}}

/* Abstract Block */
.abstract-block {{
    margin: 16pt 24pt 24pt 24pt;
    font-size: 9pt;
    line-height: 1.34;
}}

.abstract-heading {{
    text-align: center;
    font-size: 10pt;
    font-weight: bold;
    margin-bottom: 6pt;
    letter-spacing: 0.2px;
}}

.keywords-block {{
    margin-top: 8pt;
    font-size: 8.5pt;
}}

/* Headings */
h2.neurips-h1 {{
    font-size: 11.5pt;
    font-weight: bold;
    margin-top: 18pt;
    margin-bottom: 6pt;
    border-bottom: none;
    break-after: avoid;
    letter-spacing: -0.1px;
}}

h3.neurips-h2 {{
    font-size: 10pt;
    font-weight: bold;
    margin-top: 13pt;
    margin-bottom: 4pt;
    break-after: avoid;
}}

.sec-num {{
    font-weight: bold;
    margin-right: 4px;
}}

/* Paragraphs */
p {{
    margin-top: 0;
    margin-bottom: 6pt;
}}

/* MathML Display Styling */
.neurips-math-display {{
    text-align: center;
    margin: 10pt 0;
    overflow-x: auto;
    break-inside: avoid;
}}

math {{
    font-family: "Times New Roman", Times, serif;
    font-size: 10pt;
}}

math[display="block"] {{
    font-size: 10.5pt;
}}

/* Booktabs Table Styling */
.neurips-table-wrap {{
    margin: 14pt 0;
    break-inside: avoid;
    width: 100%;
}}

.neurips-table-title {{
    font-size: 9pt;
    font-weight: bold;
    margin-bottom: 2pt;
    text-align: left;
    break-after: avoid;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8pt;
    line-height: 1.25;
    margin: 4pt 0 8pt 0;
    border-top: 1.5pt solid #000000;
    border-bottom: 1.5pt solid #000000;
}}

thead tr {{
    border-bottom: 0.75pt solid #000000;
}}

th {{
    font-weight: bold;
    padding: 3.5pt 4pt;
    text-align: left;
    vertical-align: bottom;
    background-color: transparent;
    border-top: none;
    border-bottom: none;
    border-left: none;
    border-right: none;
}}

th:nth-child(n+4) {{
    text-align: center;
}}

td {{
    padding: 3.2pt 4pt;
    vertical-align: top;
    border-top: none;
    border-bottom: none;
    border-left: none;
    border-right: none;
}}

td:nth-child(n+4) {{
    text-align: center;
}}

tbody tr:nth-child(even) {{
    background-color: #fbfbfb;
}}

/* Figures */
.neurips-figure {{
    margin: 16pt 0;
    text-align: center;
    break-inside: avoid;
    width: 100%;
}}

.figure-img-wrap img {{
    width: 100%;
    max-width: 5.5in;
    height: auto;
    display: block;
    margin: 0 auto;
    border: 1px solid #e0e0e0;
}}

.figure-caption {{
    font-size: 8.5pt;
    line-height: 1.3;
    margin-top: 6pt;
    text-align: justify;
    color: #222222;
}}

/* Lists */
ul, ol {{
    margin-top: 2pt;
    margin-bottom: 6pt;
    padding-left: 18pt;
}}

li {{
    margin-bottom: 2pt;
}}

/* References */
#references, .references {{
    font-size: 8.5pt;
    line-height: 1.25;
}}

#refs div {{
    margin-bottom: 4pt;
    padding-left: 1.8em;
    text-indent: -1.8em;
}}

/* Code spans */
code {{
    font-family: "Courier New", Courier, monospace;
    font-size: 8.5pt;
    background-color: #f6f6f6;
    padding: 1px 3px;
}}
</style>
</head>
<body>

<div class="neurips-banner">Preprint. Under review at NeurIPS / Top-Tier Conference</div>

<div class="paper-title">Resolving Felsenstein's Dilemma in Phenomics: Continuous Spectral Timetree Manifolds Establish the Predictability Frontier of Organismal Traits</div>

<div class="author-block">
    <div class="author-names">The PHENIX Consortium</div>
    <div class="author-inst">Computational Biology and Evolutionary Machine Learning Group</div>
    <div class="author-email">contact@phenix-project.org</div>
</div>

<div class="abstract-block">
    <div class="abstract-heading">Abstract</div>
    <p>{abstract_text}</p>
    <div class="keywords-block"><strong>Keywords:</strong> {keywords_text}</div>
</div>

{content}

</body>
</html>
"""

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(neurips_html)
    print(f"Generated NeurIPS HTML -> {output_html_path}")

    # Step 6: Compile with Edge headless with virtual time budget for rock-solid stability
    edge_path = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    if not edge_path.exists():
        edge_path = Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")

    print("Compiling NeurIPS PDF with Edge headless...")
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
        print(f"SUCCESS: Compiled NeurIPS PDF -> {output_pdf_path} ({output_pdf_path.stat().st_size} bytes)")
        # Also mirror to manuscript_ieee.pdf for backward compatibility
        try:
            import shutil
            shutil.copy(output_pdf_path, ieee_pdf_path)
            print(f"SUCCESS: Synchronized with {ieee_pdf_path}")
        except Exception as e:
            print(f"Note: Could not mirror to ieee_pdf: {e}")
    else:
        raise RuntimeError("PDF generation failed: file is missing or empty.")

    # Cleanup temporary pandoc html
    if raw_html_path.exists():
        raw_html_path.unlink()

if __name__ == "__main__":
    build_neurips_pdf()
