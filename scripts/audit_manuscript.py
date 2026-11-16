from pathlib import Path
import re

def audit():
    md = Path("manuscript.md").read_text(encoding="utf-8")
    tex = Path("manuscript.tex").read_text(encoding="utf-8")
    html_path = Path("manuscript_ieee.html")
    
    print("=== AUDIT MANUSCRIPT.MD ===")
    broken_text = re.findall(r"(\\\s*\$?ext\{[^\}]+\})", md)
    print("Broken \\text patterns:", broken_text)
    rev_markers = re.findall(r"\(?REV-[0-9]+\)?", md)
    print("REV markers:", rev_markers)
    sec1_5 = md.split("## 6. Code & Data Availability")[0]
    inline_paths = re.findall(r"`[a-zA-Z0-9_\-\./\\]+\.(?:py|json|csv|h5|pt|pkl|tsv)`", sec1_5)
    print("Inline file paths in Sec 1-5:", inline_paths)
    print("Blomberg equation present:", "Blomberg’s $K$" in md or "Blomberg's $K$" in md)
    print("Table 3 footnote present:", "statistical suppression" in md or "suppression" in md)
    print("Figure 1 in md:", "fig_sys_overview_bw.png" in md)
    print("Figure 2 in md:", "fig_flowchart_bw.png" in md)
    print("Table 1b (K-Pg) in md:", "Table 1b" in md)
    print("Table 1c (Buffer) in md:", "Table 1c" in md)
    print("Table 4 (Multi-trait) in md:", "Table 4" in md)
    
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8")
        print("\n=== AUDIT MANUSCRIPT_IEEE.HTML ===")
        print("Figures in html:", len(re.findall(r"<img[^>]+>", html)))
        print("Tables in html:", len(re.findall(r"<table[^>]*>", html)))
        print("Headings in html:", re.findall(r'<h2 class="ieee-heading-1">([^<]+)</h2>', html))
        
    print("\n=== AUDIT MANUSCRIPT.TEX ===")
    print("REV markers in tex:", re.findall(r"\(?REV-[0-9]+\)?", tex))
    print("Figures in tex:", "fig_sys_overview_bw" in tex, "fig_flowchart_bw" in tex)
    print("Table 1 in tex has pooled:", "pooled" in tex.lower())

if __name__ == "__main__":
    audit()
