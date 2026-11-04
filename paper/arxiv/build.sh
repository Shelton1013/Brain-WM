#!/bin/sh
# usage: sh build.sh <jobname>
j=${1:-specbram_draft}
pdflatex -interaction=nonstopmode -jobname=$j specbram_arxiv.tex >/dev/null
bibtex $j >/dev/null
pdflatex -interaction=nonstopmode -jobname=$j specbram_arxiv.tex >/dev/null
pdflatex -interaction=nonstopmode -jobname=$j specbram_arxiv.tex >/dev/null
grep -E "Output written|^!|undefined|Float too large" $j.log | head -5
