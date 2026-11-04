"""Results workbook built from the SAME result JSONs and loaders as the paper tables.

    python paper/arxiv/make_results_xlsx.py [out.xlsx]

Runs make_arxiv_tables.py (which also rewrites tables/*.tex) and reuses its loaders, so
every number here is the number the paper prints. Per-seed values are written as inputs
(blue); means, standard deviations and differences are Excel formulas (black). Values
quoted from other papers are grey italic with a cell comment.
"""
import io
import runpy
import sys
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as CL

HERE = Path(__file__).resolve().parent
with redirect_stdout(io.StringIO()):
    G = runpy.run_path(str(HERE / "make_arxiv_tables.py"))
DS, SEEDS, FR, MAIN = G["DS"], G["SEEDS"], G["FR"], G["MAIN"]
lp, le, FAM, hc, hcle = G["lp"], G["le"], G["FAM"], G["hc"], G["hcle"]
tk, TFAM, ftv, SV, SVFT = G["tk"], G["TFAM"], G["ftv"], G["SV"], G["SVFT"]
seeds_of = G["seeds_of"]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE.parents[1] / "results" / f"SpecBraM_results_{date.today().isoformat()}.xlsx"

FONT = "Arial"
F_TITLE = Font(name=FONT, size=13, bold=True)
F_NOTE = Font(name=FONT, size=9, italic=True, color="555555")
F_HEAD = Font(name=FONT, size=10, bold=True, color="FFFFFF")
F_BODY = Font(name=FONT, size=10)
F_BOLD = Font(name=FONT, size=10, bold=True)
F_IN = Font(name=FONT, size=10, color="0000FF")
F_QUOTE = Font(name=FONT, size=10, italic=True, color="777777")
FILL_HEAD = PatternFill("solid", fgColor="2A78D6")
FILL_MAIN = PatternFill("solid", fgColor="E8F0FB")
FILL_STAT = PatternFill("solid", fgColor="F3F3F1")
BORDER = Border(bottom=Side(style="thin", color="D0D0CC"))

NAMES = {
    "m": "SpecBraM (classical bands, periodogram, log P; no SIGReg)",
    "d2n": "Main target, multitaper (3 DPSS)",
    "sbp": "Band power, 4.2M parameters",
    "srr": "Raw-waveform recon, 4.2M parameters",
    "d1": "SpecBraM (classical bands, periodogram, log P)",
    "g": "Main target without SIGReg",
    "d2": "Main target, multitaper (3 DPSS)",
    "nd": "Main target without the delta band (4-45 Hz)",
    "linraw": "Raw-waveform recon, linear tok",
    "fbd1": "Main target, filterbank tok",
    "fbraw": "Raw-waveform recon, filterbank tok",
    "bandrecon": "Band-waveform recon, filterbank tok",
}

wb = Workbook()
wb.remove(wb.active)


class Sheet:
    """Writes stacked blocks; remembers where each family's mean row is."""

    def __init__(self, name, title, note):
        self.ws = wb.create_sheet(name)
        self.ws["A1"], self.ws["A1"].font = title, F_TITLE
        self.ws["A2"], self.ws["A2"].font = note, F_NOTE
        self.r = 4
        self.mean_row = {}
        self.ws.column_dimensions["A"].width = 58
        self.ws.freeze_panes = "B4"

    def header(self, cols, label="Model / seed"):
        ws = self.ws
        for j, c in enumerate([label] + list(cols), 1):
            cell = ws.cell(self.r, j, c)
            cell.font, cell.fill = F_HEAD, FILL_HEAD
            cell.alignment = Alignment(horizontal="center" if j > 1 else "left", wrap_text=True)
            if j > 1:
                ws.column_dimensions[CL(j)].width = max(ws.column_dimensions[CL(j)].width or 0, 13)
        self.ncol = len(cols)
        self.r += 1

    def section(self, text):
        self.ws.cell(self.r, 1, text).font = F_BOLD
        self.r += 1

    def row(self, label, vals, font=F_IN, fmt="0.0000", fill=None, comment=None):
        ws = self.ws
        ws.cell(self.r, 1, label).font = F_BOLD if fill else F_BODY
        for j, v in enumerate(vals, 2):
            c = ws.cell(self.r, j, v if v is not None else "--")
            c.font = font if v is not None else F_BODY
            c.number_format = fmt
            c.alignment = Alignment(horizontal="center")
            if fill:
                c.fill = fill
        if fill:
            ws.cell(self.r, 1).fill = fill
        if comment:
            ws.cell(self.r, 1).comment = Comment(comment, "results")
        for j in range(1, self.ncol + 2):
            ws.cell(self.r, j).border = BORDER
        self.r += 1
        return self.r - 1

    def family(self, key, per_seed, fmt="0.0000", main=False):
        """per_seed: {seed: [values]} -> seed rows + Mean/Std formula rows."""
        first = self.r
        for s in per_seed:
            self.row(f"{NAMES.get(key, key)} · seed {s}", per_seed[s], fmt=fmt)
        last = self.r - 1
        rng = lambda j: f"{CL(j)}{first}:{CL(j)}{last}"
        mean = self.row(f"{NAMES.get(key, key)} · MEAN", [f'=IFERROR(AVERAGE({rng(j)}),"--")' for j in range(2, self.ncol + 2)],
                        font=F_BOLD, fmt=fmt, fill=FILL_MAIN if main else FILL_STAT)
        self.row(f"{NAMES.get(key, key)} · STD (sample)", [f'=IFERROR(STDEV({rng(j)}),"--")' for j in range(2, self.ncol + 2)],
                 font=F_BODY, fmt=fmt)
        self.mean_row[key] = mean
        return mean

    def delta(self, label, a_row, b_row, cols=None, scale=100):
        cols = cols or range(2, self.ncol + 2)
        vals = [None] * self.ncol
        for j in cols:
            vals[j - 2] = f'=IFERROR(({CL(j)}{a_row}-{CL(j)}{b_row})*{scale},"--")'
        return self.row(label, vals, font=F_BODY, fmt="+0.0;-0.0;0.0")

    def gap(self):
        self.r += 1


# ---------------------------------------------------------------- README
ws = wb.create_sheet("README")
ws.column_dimensions["A"].width = 120
lines = [
    ("SpecBraM results workbook", F_TITLE),
    (f"Generated {date.today().isoformat()} by paper/arxiv/make_results_xlsx.py from results/arxiv_data "
     "(the same JSONs and loaders as paper/arxiv/tables/*.tex).", F_NOTE),
    ("", F_BODY),
    ("Conventions", F_BOLD),
    ("Blue = measured value read from a result JSON. Black = Excel formula (mean, sample std, differences).", F_BODY),
    ("Grey italic = quoted from another paper (see the row comment). '--' = not run / not reported / still running.", F_BODY),
    ("Means/stds are over the available PRETRAINING seeds (42/43/44); each seed is one pretraining run.", F_BODY),
    ("Differences (Δ) are in balanced-accuracy points (×100).", F_BODY),
    ("", F_BODY),
    ("Main model: linear patch embedding; target = log power in the classical bands 0.5-4/4-8/8-13/13-30/30-45 Hz "
     "from the per-patch periodogram, log(P+1e-3); SIGReg 0.05; 16 epochs on 2,388 h of TUEG.", F_BODY),
    ("Only the theory-aligned target is listed (as in the paper); runs of the earlier implementation are kept in the "
     "result files only.", F_BODY),
    ("", F_BODY),
    ("Sheets", F_BOLD),
    ("Sleep_LP — strict linear probe on ISRUC/HMC vs. foundation models and handcrafted features", F_BODY),
    ("Target_Study — band power vs waveform reconstruction, estimator, SIGReg, delta band, filterbank tokenizer", F_BODY),
    ("Paired_BP_vs_Raw — per-seed paired differences", F_BODY),
    ("Label_Eff — balanced accuracy vs label fraction", F_BODY),
    ("FT30 — 30-s sleep fine-tuning", F_BODY),
    ("Tasks_LP — MA, Mumtaz, FACED, PhysioNet-MI, TUAB, TUEV strict linear probe", F_BODY),
    ("SEED_VIG — vigilance regression, LP at 0.5/4 Hz input and FT", F_BODY),
    ("Along_Pretraining — linear probe per pretraining epoch (main target vs raw, seed 42)", F_BODY),
]
for i, (t, fnt) in enumerate(lines, 1):
    ws.cell(i, 1, t).font = fnt

M3 = ("balanced_accuracy", "cohen_kappa", "weighted_f1")
COLS4 = ["ISRUC BA", "HMC BA", "ISRUC BA @1%", "HMC BA @1%"]


def four(key, s):
    k = FAM[key][s]
    return [lp("isruc", k), lp("hmc", k), le("isruc", k, "0.01"), le("hmc", k, "0.01")]


# ---------------------------------------------------------------- Sleep_LP
sh = Sheet("Sleep_LP", "Strict linear probe — sleep staging (ISRUC / HMC)",
           "Frozen encoder, mean-pooled tokens, standardized, logistic regression with C in {0.01,0.1,1} "
           "chosen on validation; 30-s epoch = mean of three 10-s window features.")
sh.header(["ISRUC BA", "ISRUC κ", "ISRUC F1", "HMC BA", "HMC κ", "HMC F1"])
sh.section("Our models (pretraining seeds)")
for key in (MAIN, "linraw"):
    sh.family(key, {s: [lp(ds, FAM[key][s], m) for ds in DS for m in M3] for s in seeds_of(key)},
              main=key == MAIN)
sh.gap()
sh.section("External models (official weights, single checkpoint; probe is deterministic)")
for key, nm in (("random_init", "Random init (our arch.)"), ("biot", "BIOT (six-dataset)"),
                ("biot_sleep", "BIOT (SHHS+PREST)"), ("cbramod", "CBraMod"), ("labram", "LaBraM-Base")):
    sh.row(nm, [lp(ds, key, m) for ds in DS for m in M3])
for size, nm in (("base", "REVE-Base"), ("large", "REVE-Large")):
    ks = {ds: G["reve_sleep_key"](ds, size) for ds in DS}
    if all(ks.values()):
        sh.row(f"{nm} (readout chosen on validation: {ks['isruc'].rsplit('_', 1)[1]}/{ks['hmc'].rsplit('_', 1)[1]})",
               [lp(ds, ks[ds], m) for ds in DS for m in M3],
               comment="Measured with our strict probe (official HF weights, REVE input convention).")
sh.gap()
sh.section("Handcrafted spectral features (same samples and probe; mean over the three 10-s windows)")
sh.row("Handcrafted: 5-band log/relative power", [hc(ds, "basic", m) for ds in DS for m in M3])
hr = sh.row("Handcrafted: rich spectral set", [hc(ds, "rich", m) for ds in DS for m in M3])
sh.gap()
sh.section("Differences (points)")
sh.delta("SpecBraM − raw-waveform recon", sh.mean_row[MAIN], sh.mean_row["linraw"])
sh.delta("SpecBraM − handcrafted rich", sh.mean_row[MAIN], hr)

# ---------------------------------------------------------------- Target_Study
sh = Sheet("Target_Study", "Pretext-target study (matched backbone, data, 16 epochs, launcher)",
           "Δ rows: mean − mean of the band-power row with the same tokenizer.")
sh.header(COLS4, "Target / seed")
sh.section("Linear patch embedding")
for key in (MAIN, "d2n", "nd", "linraw", "sbp", "srr"):
    if seeds_of(key):
        sh.family(key, {s: four(key, s) for s in seeds_of(key)}, main=key == MAIN)
sh.section("Filterbank tokenizer")
for key in ("fbd1", "fbraw", "bandrecon"):
    if seeds_of(key):
        sh.family(key, {s: four(key, s) for s in seeds_of(key)})
sh.gap()
sh.section("Δ vs band power (points)")
for key in ("d2n", "nd", "linraw"):
    if key in sh.mean_row:
        sh.delta(f"{NAMES[key]} − main", sh.mean_row[key], sh.mean_row[MAIN])
if "fbd1" in sh.mean_row:
    for key in ("fbraw", "bandrecon"):
        sh.delta(f"{NAMES[key]} − main target (filterbank)", sh.mean_row[key], sh.mean_row["fbd1"])
sh.delta("Band-waveform − raw recon (filterbank)", sh.mean_row["bandrecon"], sh.mean_row["fbraw"])

# ---------------------------------------------------------------- Paired_BP_vs_Raw
sh = Sheet("Paired_BP_vs_Raw", "Band power vs raw reconstruction, paired per pretraining seed",
           "Δ rows are per-seed differences (points); wins = number of seeds with Δ > 0.")
sh.header(COLS4, "Comparison / seed")
for bp, raw, nm in ((MAIN, "linraw", "Linear patch embedding"), ("fbd1", "fbraw", "Filterbank tokenizer")):
    ss = [s for s in seeds_of(bp) if s in seeds_of(raw)]
    if not ss:
        continue
    sh.section(nm)
    rows = []
    for s in ss:
        a = sh.row(f"{NAMES[bp]} · seed {s}", four(bp, s))
        b = sh.row(f"{NAMES[raw]} · seed {s}", four(raw, s))
        rows.append(sh.delta(f"Δ seed {s}", a, b))
    refs = lambda j: ",".join(f"{CL(j)}{r}" for r in rows)
    sh.row("Δ mean", [f"=AVERAGE({refs(j)})" for j in range(2, 6)], font=F_BOLD, fmt="+0.0;-0.0;0.0", fill=FILL_STAT)
    sh.row(f"wins (of {len(rows)})", ["=" + "+".join(f"({CL(j)}{r}>0)" for r in rows) for j in range(2, 6)],
           font=F_BOLD, fmt="0", fill=FILL_STAT)
    sh.gap()

# ---------------------------------------------------------------- Label_Eff
sh = Sheet("Label_Eff", "Label efficiency — strict linear probe, balanced accuracy",
           "<100%: mean of five fixed random subsets of the training epochs, identical for all models.")
for ds in DS:
    sh.section(ds.upper())
    sh.header([f"{float(x) * 100:g}%" for x in FR], "Model / seed")
    for key in (MAIN, "linraw"):
        sh.family(key, {s: [le(ds, FAM[key][s], fr) for fr in FR] for s in seeds_of(key)}, fmt="0.000",
                  main=key == MAIN)
    for key, nm in (("random_init", "Random init (our arch.)"), ("biot", "BIOT"), ("cbramod", "CBraMod"),
                    ("labram", "LaBraM-Base")):
        sh.row(nm, [le(ds, key, fr) for fr in FR], fmt="0.000")
    for kind, nm in (("basic", "Handcrafted, 5-band"), ("rich", "Handcrafted, rich")):
        sh.row(nm, [hcle(ds, kind, fr) for fr in FR], fmt="0.000")
    sh.gap()

# ---------------------------------------------------------------- FT30
sh = Sheet("FT30", "Sleep staging, full fine-tuning (30-s epochs in sequences of 20)",
           "Our rows: one fine-tuning run per pretraining seed. Baselines quoted from CSBrain (REVE: its paper).")
sh.header(["ISRUC BA", "ISRUC κ", "ISRUC wF1", "HMC BA", "HMC κ", "HMC wF1"])
for tgt, key in (("fftstdlog", MAIN), ("rawrecon", "linraw")):
    per = {m + ds: ftv(tgt, ds, m) for ds in DS for m in M3}
    sh.family(key, {s: [per[m + ds][i] for ds in DS for m in M3] for i, s in enumerate(SEEDS)}, main=key == MAIN)
sh.gap()
sh.section("Published results (mean of their fine-tuning seeds)")
for grp, n, h, v in G["PAPER_FT"]:
    vals = [float(x.split("/")[0]) if x != "--" else None for x in v]
    sh.row(n.split("~")[0], vals, font=F_QUOTE, comment="Quoted from CSBrain (sleep staging); REVE from its paper.")
sh.gap()
sh.section("Differences (points)")
sh.delta("SpecBraM − raw-waveform recon", sh.mean_row[MAIN], sh.mean_row["linraw"])

# ---------------------------------------------------------------- Tasks_LP
BIN, MUL = G["BIN"], G["MUL"]
TASKS = [("ma", BIN, "MA"), ("mumtaz", BIN, "Mumtaz"), ("faced", MUL, "FACED"),
         ("physio", MUL, "PhysioNet-MI"), ("tuab", BIN, "TUAB"), ("tuev", MUL, "TUEV")]
SHORT = {"balanced_accuracy": "BA", "auc_pr": "AUC-PR", "auroc": "AUROC", "cohen_kappa": "κ", "weighted_f1": "F1"}
sh = Sheet("Tasks_LP", "Strict linear probe beyond sleep (CBraMod's samples and splits)",
           "TUAB/TUEV belong to TUH, of which TUEG (pretraining corpus of every model) is the superset; "
           "patients were not excluded.")
sh.header([f"{nm} {SHORT[m]}" for t, ms_, nm in TASKS for m in ms_])
for key in (MAIN, "linraw"):
    sh.family(key, {s: [tk(t, TFAM[key][s], m) for t, ms_, _ in TASKS for m in ms_] for s in SEEDS},
              fmt="0.000", main=key == MAIN)
for key, nm in (("cbramod", "CBraMod (official)"), ("labram", "LaBraM-Base (official)"),
                ("hc_basic", "Handcrafted: 5-band"), ("hc_rich", "Handcrafted: rich"),
                ("nd_s42", "Main target without delta band · seed 42"),
                ("random_init", "Random init (our arch.)")):
    vals = [tk(t, key, m) for t, ms_, _ in TASKS for m in ms_]
    if any(v is not None for v in vals):
        sh.row(nm, vals, fmt="0.000")
QUOTED = {
    "CBraMod fine-tuned (paper)": [0.726, 0.627, 0.791, 0.956, 0.992, 0.992, 0.551, 0.504, 0.562,
                                   0.642, 0.522, 0.643, 0.829, 0.926, 0.923, 0.667, 0.677, 0.834],
    "CBraMod frozen backbone + own head (paper)": [None] * 6 + [0.315, 0.258, 0.308, 0.385, 0.298, 0.395]
                                                   + [None] * 6,
}
for nm, v in QUOTED.items():
    sh.row(nm, v, font=F_QUOTE, fmt="0.000", comment="Quoted from the CBraMod paper.")
sh.gap()
sh.section("Differences (points)")
sh.delta("SpecBraM − raw-waveform recon", sh.mean_row[MAIN], sh.mean_row["linraw"])

# ---------------------------------------------------------------- SEED_VIG
sh = Sheet("SEED_VIG", "SEED-VIG vigilance regression (PERCLOS)",
           "Split 1-15/16-19/20-23 by subject (as stated in the CBraMod paper). LP = frozen encoder + ridge; "
           "FT = full fine-tuning. '--' = not run / still running.")
sh.header(["LP 0.5Hz Pearson", "LP 0.5Hz R²", "LP 4Hz Pearson", "LP 4Hz R²", "FT Pearson", "FT R²"])
for key in (MAIN, "linraw"):
    per = {}
    for s in SEEDS:
        k = f"{key}_s{s}"
        v = [G["svm"](k, b, m) for b in ("0.5", "4") for m in ("pearson", "r2")]
        v += [SVFT[k][m] if k in SVFT else None for m in ("pearson", "r2")]
        per[s] = v
    sh.family(key, per, fmt="0.000", main=key == MAIN)
for key, nm in (("nd_s42", "Main target without delta band · seed 42"), ("labram", "LaBraM-Base (official)"),
                ("cbramod", "CBraMod (official)"), ("random_init", "Random init")):
    v = [G["svm"](key, b, m) for b in ("0.5", "4") for m in ("pearson", "r2")]
    if any(x is not None for x in v):
        sh.row(nm, v + [None, None], fmt="0.000")
sh.row("CBraMod fine-tuned (paper)", [None] * 4 + [0.646, 0.337], font=F_QUOTE, fmt="0.000",
       comment="Quoted from the CBraMod paper.")

# ---------------------------------------------------------------- Along_Pretraining
if G["HAS_CURVE"]:
    sh = Sheet("Along_Pretraining", "Linear probe along pretraining (linear tok, seed 42)", "")
    sh.header(["ISRUC BP", "ISRUC Raw", "HMC BP", "HMC Raw"], "Epoch")
    for e in G["CURVE_EP"]:
        sh.row(f"epoch {e}", [lp(ds, G["curve_key"](k, e)) for ds in DS for k in ("bp", "raw")])

OUT.parent.mkdir(parents=True, exist_ok=True)
wb.calculation.fullCalcOnLoad = True          # openpyxl stores no cached values
wb.save(OUT)
print(OUT)
