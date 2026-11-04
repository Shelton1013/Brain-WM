"""Rebuild generated paper tables in isolation and compare them to the source tree.

No existing paper or result files are changed. Requires numpy/matplotlib and the
checked-in aggregate JSONs, but not EEG recordings or model checkpoints.
"""
import contextlib
import io
import json
from pathlib import Path
import runpy
import shutil
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="specbram_table_audit_") as name:
        stage = Path(name)
        paper = stage / "paper/arxiv"
        paper.mkdir(parents=True)
        for source in (root / "results").rglob("*.json"):
            dest = stage / source.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
        for name in ("make_arxiv_tables.py", "appendix_detail.tex"):
            shutil.copyfile(root / "paper/arxiv" / name, paper / name)
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(paper / "make_arxiv_tables.py"))
        rebuilt = sorted((paper / "tables").glob("*.tex"))
        mismatch = []
        for path in rebuilt:
            existing = root / "paper/arxiv/tables" / path.name
            if not existing.exists() or existing.read_text(encoding="utf-8") != path.read_text(encoding="utf-8"):
                mismatch.append(path.name)
        generated = {p.name for p in rebuilt}
        other = sorted(p.name for p in (root / "paper/arxiv/tables").glob("*.tex") if p.name not in generated)
        print(json.dumps({"rebuilt_tables": len(rebuilt), "mismatches": mismatch,
                          "other_tables_not_produced_by_this_generator": other}, indent=2))
        if mismatch:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
