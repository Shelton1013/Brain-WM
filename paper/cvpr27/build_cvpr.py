"""Build the CVPR 2027 Overleaf project from the arXiv master (paper/arxiv/specbram_arxiv.tex).

The arXiv file stays the single source; this script splits it into the CVPR author-kit layout
(main.tex + preamble.tex + sec/*.tex + fig/teaser.tex), anonymizes it for review, copies the
generated tables, figures and bibliography, and zips the project for Overleaf.

    python paper/cvpr27/build_cvpr.py <path to cvpr author-kit dir>
"""
import re
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "arxiv"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("kit", help="CVPR author kit (or an existing derived project)")
parser.add_argument("--output-dir", type=Path, default=HERE / "specbram_cvpr27")
parser.add_argument("--archive", type=Path, default=HERE / "specbram_cvpr27_overleaf.zip")
args = parser.parse_args()
KIT, OUT = Path(args.kit), args.output_dir
if OUT.exists() and any(OUT.iterdir()):
    parser.error("Output directory is not empty; use --output-dir with a fresh directory")

s = (SRC / "specbram_arxiv.tex").read_text(encoding="utf-8")


def between(a, b, text=None):
    text = s if text is None else text
    i = text.index(a) + len(a)
    return text[i:text.index(b, i)]


title = between(r"\title{", "}\n%")
abstract = between(r"\begin{abstract}", r"\end{abstract}").strip()
teaser = between(r"\begin{figure*}[!t]\centering", r"\end{figure*}")
body = between(r"\end{abstract}", r"\appendix")
appendix = between(r"\appendix", r"\clearpage" + "\n" + r"\setlength{\bibsep}")

# anonymity: drop the statement about the earlier submission
body = body.replace(
    "This work was completed and submitted for peer review in July 2026; FAME\n"
    r"\cite{yu2026fame} and MANAS-2 \cite{kulkarni2026manas2}, which also use band log-power"
    "\ntargets, appeared afterwards and are concurrent.",
    r"FAME \cite{yu2026fame} and MANAS-2 \cite{kulkarni2026manas2}, which also use band"
    "\nlog-power targets, are concurrent work.")
assert "submitted for peer review" not in body

(OUT / "sec").mkdir(parents=True)
(OUT / "fig").mkdir()

# ---- sections: one file per \section of the main text
parts = re.split(r"(?=^% =+\n\\section\{|^\\section\{)", body, flags=re.M)
names, n = [], 0
for part in parts:
    m = re.search(r"\\section\{([^}]*)\}", part)
    if not m:
        continue
    n += 1
    slug = re.sub(r"[^a-z]+", "_", m.group(1).lower()).strip("_")[:24]
    name = f"{n}_{slug}"
    (OUT / "sec" / f"{name}.tex").write_text(part.strip() + "\n", encoding="utf-8")
    names.append(name)
(OUT / "sec" / "0_abstract.tex").write_text(
    "\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n", encoding="utf-8")
(OUT / "sec" / "X_suppl.tex").write_text(
    "\\clearpage\n\\setcounter{page}{1}\n\\maketitlesupplementary\n\\appendix\n"
    + appendix.strip() + "\n", encoding="utf-8")

# ---- teaser as a two-column strip on the first page
inc = re.search(r"\\includegraphics\[[^]]*\]\{([^}]*)\}", teaser).group(1)
cap = between(r"\caption{", "\n\\label{fig:contrast}", teaser).rstrip()
assert cap.endswith("}")
(OUT / "fig" / "teaser.tex").write_text(
    "\\begin{strip}\n\\centering\n\\includegraphics[width=\\linewidth]{" + inc + "}\n"
    "\\captionsetup{hypcap=false}\n\\captionof{figure}{" + cap[:-1] + "}\n"
    "\\captionsetup{hypcap=true}\n\\label{fig:contrast}\n\\end{strip}\n", encoding="utf-8")

# ---- preamble: kit preamble + our packages and macros
macros = between(r"\pdfinfo{/TemplateVersion (2027.1)}", r"\title{")
macros = macros.replace(r"\setcounter{secnumdepth}{2}   % arXiv version: numbered sections so Sec./App. refs resolve", "")
pre = (KIT / "preamble.tex").read_text(encoding="utf-8").split("%% ---- SpecBraM", 1)[0]
pre += "\n%% ---- SpecBraM\n\\usepackage{multirow}\n\\usepackage{amsthm}\n\\usepackage{bm}\n" + macros.strip() + "\n"
(OUT / "preamble.tex").write_text(pre, encoding="utf-8")

main = r"""% SpecBraM -- CVPR 2027 submission (built from paper/arxiv/specbram_arxiv.tex by build_cvpr.py;
% CVPR 2026 author kit, conference year set to 2027).
\documentclass[10pt,twocolumn,letterpaper]{article}

%%%%%%%%% PAPER TYPE
% \usepackage{cvpr}              % camera-ready
\usepackage[review]{cvpr}      % review version
% \usepackage[pagenumbers]{cvpr} % arXiv version

\input{preamble}

\definecolor{cvprblue}{rgb}{0.21,0.49,0.74}
\usepackage[pagebackref,breaklinks,colorlinks,allcolors=cvprblue]{hyperref}

\def\paperID{*****} % *** Enter the Paper ID here
\def\confName{CVPR}
\def\confYear{2027}

\title{""" + title + r"""}

\author{Anonymous CVPR submission}

\begin{document}
\maketitle
\input{fig/teaser}
\input{sec/0_abstract}
""" + "".join(f"\\input{{sec/{nm}}}\n" for nm in names) + r"""
{
    \small
    \bibliographystyle{ieeenat_fullname}
    \bibliography{main}
}

% Supplementary material, compiled after the references (split it off for submission).
\input{sec/X_suppl}

\end{document}
"""
(OUT / "main.tex").write_text(main, encoding="utf-8")

# ---- assets
for f in ("cvpr.sty", "ieeenat_fullname.bst"):
    shutil.copy(KIT / f, OUT / f)
shutil.copy(SRC / "arxiv.bib", OUT / "main.bib")
alltex = "\n".join(p.read_text(encoding="utf-8") for p in OUT.rglob("*.tex"))
(OUT / "tables").mkdir()
need = set(re.findall(r"\\input\{tables/([^}]*)\}", alltex))
for _ in range(2):                                     # tables can \input other tables
    for t in list(need):
        src = SRC / "tables" / f"{t}.tex"
        need |= set(re.findall(r"\\input\{tables/([^}]*)\}", src.read_text(encoding="utf-8")))
for t in sorted(need):
    shutil.copy(SRC / "tables" / f"{t}.tex", OUT / "tables" / f"{t}.tex")
(OUT / "figures").mkdir()
alltex = "\n".join(p.read_text(encoding="utf-8") for p in OUT.rglob("*.tex"))   # incl. tables
for fig in sorted(set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{(figures/[^}]*)\}", alltex))):
    shutil.copy(SRC / fig, OUT / fig)

zp = args.archive
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and f.suffix not in (".aux", ".log", ".bbl", ".blg", ".out", ".pdf") \
                or (f.is_file() and f.parent.name == "figures"):
            z.write(f, f.relative_to(OUT))
print("sections:", names)
print("tables:", sorted(need))
print("wrote", zp)
