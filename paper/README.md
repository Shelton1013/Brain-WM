# Manuscript sources

The current master is `arxiv/specbram_arxiv.tex`, together with `arxiv/arxiv.bib`,
`arxiv/appendix_detail.tex`, and the `arxiv/tables/` and `arxiv/figures/` dependencies.
The `cvpr27/specbram_cvpr27/` tree is a derived conference-layout project.

The source is a working research draft. The arXiv-layout author block must be checked
before a public arXiv submission; the CVPR-layout project is anonymous and has a paper-ID
placeholder. A template labeled with a conference year is not evidence of acceptance.

`arxiv/arxiv_abstract.txt` is the shortened plain-text abstract for the submission form
(under 1,920 characters). The PDF abstract is the full version.

Build the arXiv-layout paper from its directory with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error specbram_arxiv.tex
bibtex specbram_arxiv
pdflatex -interaction=nonstopmode -halt-on-error specbram_arxiv.tex
pdflatex -interaction=nonstopmode -halt-on-error specbram_arxiv.tex
```

Run `python paper/arxiv/make_arxiv_tables.py` from the repository root to regenerate
tables from aggregate result JSONs. It writes the generated tables and figures in place.
Do not replace main-model rows with historical Hann/log1p/SIGReg result families.

`cvpr27/build_cvpr.py <author-kit-dir> --output-dir /path/to/fresh/project
--archive /path/to/output.zip` derives the CVPR project and Overleaf ZIP. It refuses to
overwrite a nonempty project directory, preserving existing manual changes.

The older files directly under `paper/` and their figures are retained as historical
material. They do not describe the current classical-band, no-SIGReg configuration.
