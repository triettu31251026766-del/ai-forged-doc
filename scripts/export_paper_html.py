import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: "Segoe UI", Arial, sans-serif; max-width: 920px; margin: 32px auto;
       padding: 0 28px; color: #1f2328; line-height: 1.6; font-size: 16px; }
h1 { font-size: 28px; border-bottom: 2px solid #d0d7de; padding-bottom: 10px; }
h2 { font-size: 22px; margin-top: 34px; border-bottom: 1px solid #d0d7de; padding-bottom: 6px; page-break-after: avoid; }
h3 { font-size: 18px; margin-top: 26px; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 14px; page-break-inside: avoid; }
th, td { border: 1px solid #c9d1d9; padding: 6px 9px; text-align: left; vertical-align: top; }
th { background: #f2f4f7; }
tr:nth-child(even) td { background: #fafbfc; }
code { background: #f0f2f5; padding: 1px 5px; border-radius: 4px; font-size: 13px; }
pre { background: #f0f2f5; padding: 12px; border-radius: 6px; overflow-x: auto; }
img { max-width: 100% !important; height: auto; margin: 14px 0; page-break-inside: avoid; display: block; margin-left: auto; margin-right: auto; }
blockquote { border-left: 4px solid #d0d7de; margin: 12px 0; padding: 4px 16px; color: #57606a; }
hr { border: 0; border-top: 1px solid #d0d7de; margin: 30px 0; }
p, li { orphans: 2; widows: 2; }
@media print {
  body { max-width: none; margin: 0; padding: 0; font-size: 10.5pt; }
  h1 { font-size: 20pt; } h2 { font-size: 15pt; } h3 { font-size: 12.5pt; }
  table { font-size: 8.5pt; }
  img { max-width: 100% !important; max-height: 235mm; }
  a { color: inherit; text-decoration: none; }
}
"""

def convert(md_path: Path) -> Path:
    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "attr_list", "sane_lists"])
    title = next((l.lstrip("# ").strip() for l in text.splitlines() if l.startswith("# ")), md_path.stem)
    html = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{CSS}</style></head>
<body>{body}</body></html>"""
    out = md_path.with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    return out


def main():
    files = sys.argv[1:] or ["paper/main.md", "paper/TOM_TAT_TIENG_VIET.md"]
    for f in files:
        p = ROOT / f
        if p.exists():
            print("Wrote:", convert(p))


if __name__ == "__main__":
    main()
