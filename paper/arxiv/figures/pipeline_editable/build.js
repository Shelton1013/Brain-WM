// Editable one-slide version of the SpecBraM pipeline figure.
// Native shapes: boxes, arrows, token grid, every bar, all labels and formulas.
// Images: only the plotted signals and the periodogram (assets/*.png).
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const A = (f) => path.join(__dirname, "assets", f);
const V = JSON.parse(fs.readFileSync(A("values.json"), "utf8"));
const OUT = process.argv[2] || path.join(__dirname, "specbram_pipeline.pptx");

const W = 15, H = 9.2;
const pres = new pptxgen();
pres.defineLayout({ name: "FIG", width: W, height: H });
pres.layout = "FIG";
pres.theme = { headFontFace: "Arial", bodyFontFace: "Arial" };
const s = pres.addSlide();
s.background = { color: "FFFFFF" };

const INK = "222222", MUTED = "666666";
const ENC = "8FC1A9", ENC_EDGE = "5F9E82", GREEN = "4F9A5F", PURPLE = "8A5FA8", RED = "B5475B";
const TEAL = "16848F", MASKC = "F4B183", TOKEN_M = "9E9E9E", TOKEN_M_EDGE = "6F6F6F";
const BCOL = ["5B5BD6", "8E5BD0", "1F9E89", "E0A030", "D1495B"];
const BAND = ["δ", "θ", "α", "β", "γ"];

// figure-fraction helpers (same coordinates as make_pipeline_ref.py)
const fx = (x) => x * W;
const fy = (y, h = 0) => (1 - y - h) * H;

function text(t, x, y, w, h, o = {}) {
  s.addText(t, Object.assign({ x, y, w, h, fontFace: "Arial", fontSize: 10, color: INK, margin: 0,
    valign: "middle", isTextBox: true }, o));
}
function box(t, x, y, w, h, o = {}) {
  s.addText(t, Object.assign({ x, y, w, h, shape: pres.ShapeType.roundRect, rectRadius: 0.12,
    fill: { color: ENC }, line: { color: ENC_EDGE, width: 1 }, fontFace: "Arial", fontSize: 13,
    bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 2 }, o));
}
function badge(x, y) {      // "trainable" marker, centred at x
  s.addText("trainable", { x: x - 0.42, y, w: 0.84, h: 0.22, shape: pres.ShapeType.roundRect,
    rectRadius: 0.05, fill: { color: "E4572E" }, line: { color: "E4572E" }, fontFace: "Arial",
    fontSize: 8.5, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
}
function arrow(x0, y0, x1, y1, color = GREEN) {
  const flipH = x1 < x0, flipV = y1 < y0;
  s.addShape(pres.ShapeType.line, { x: Math.min(x0, x1), y: Math.min(y0, y1), w: Math.abs(x1 - x0) || 0.0001,
    h: Math.abs(y1 - y0) || 0.0001, flipH, flipV, line: { color, width: 1.75, endArrowType: "triangle" } });
}
// bar chart from native rectangles; mode: "pred", "tgt" (dashed outline) or "both"
function bars(x, y, w, h, mode, title) {
  const all = V.y_hat.concat(V.y_tgt);
  const lo = Math.min(...all) - 0.6, hi = Math.max(...all) + 0.15;
  const labH = 0.24, areaH = h - labH - (title ? 0.24 : 0), top = y + (title ? 0.24 : 0);
  if (title) s.addText(title, { x: x - 0.25, y, w: w + 0.5, h: 0.22, fontFace: "Arial", fontSize: 9, color: MUTED,
    align: "center", valign: "middle", margin: 0, isTextBox: true });
  const slot = w / 5;
  for (let i = 0; i < 5; i++) {
    const cx = x + slot * (i + 0.5);
    const hgt = (v) => (v - lo) / (hi - lo) * areaH;
    if (mode !== "pred") {
      const th = hgt(V.y_tgt[i]);
      s.addShape(pres.ShapeType.rect, { x: cx - slot * 0.32, y: top + areaH - th, w: slot * 0.64, h: th,
        fill: { type: "none" }, line: { color: BCOL[i], width: 1.25, dashType: "dash" } });
    }
    if (mode !== "tgt") {
      const ph = hgt(V.y_hat[i]);
      const bw = mode === "both" ? slot * 0.42 : slot * 0.62;
      s.addShape(pres.ShapeType.rect, { x: cx - bw / 2, y: top + areaH - ph, w: bw, h: ph,
        fill: { color: TEAL }, line: { color: TEAL, width: 0.5 } });
    }
    text(BAND[i], cx - slot / 2, top + areaH + 0.02, slot, labH - 0.02,
      { fontSize: 11, bold: true, color: BCOL[i], align: "center" });
  }
  s.addShape(pres.ShapeType.line, { x, y: top + areaH, w, h: 0, line: { color: "444444", width: 0.75 } });
  s.addShape(pres.ShapeType.line, { x, y: top, w: 0, h: areaH, line: { color: "444444", width: 0.75 } });
}
const sub = (t) => ({ text: t, options: { subscript: true } });
const sup = (t) => ({ text: t, options: { superscript: true } });
const r = (t, o = {}) => ({ text: t, options: o });

// ===================== (a) Pretraining =====================
text("(a) Pretraining", 0.12, 0.08, 4, 0.35, { fontSize: 15, bold: true });

// raw EEG
const eeg = { x: fx(0.035), y: fy(0.73, 0.17), w: 2.25, h: 1.5 };
text("Multichannel EEG, 0.78-s patches", eeg.x - 0.2, eeg.y - 0.3, eeg.w + 0.4, 0.25,
  { fontSize: 10, color: MUTED, align: "center" });
s.addImage({ path: A("eeg_raw.png"), x: eeg.x, y: eeg.y, w: eeg.w, h: eeg.h });
["Fp1", "Cz", "O1"].forEach((c, k) => {
  const yc = eeg.y + eeg.h * 0.82 * ((0.6 + k) / 3.2);
  text(c, eeg.x - 0.5, yc - 0.1, 0.42, 0.2, { fontSize: 9, color: MUTED, align: "right" });
});
text("time (s)", eeg.x, eeg.y + eeg.h - 0.04, eeg.w, 0.18, { fontSize: 8, color: MUTED, align: "center" });

// patchify + linear embedding
box("Patchify +\nlinear\nembedding", fx(0.215), fy(0.775, 0.075), 1.43, 0.69, { fontSize: 11.5 });
badge(fx(0.262), fy(0.775, 0.075) - 0.3);
arrow(eeg.x + eeg.w + 0.05, 1.72, fx(0.215) - 0.03, 1.72);

// token grid (masking acts on embeddings)
const grid = { x: fx(0.345) + 0.3, y: fy(0.735, 0.15), w: 2.25, h: 1.38 };
text("token grid (channel × time patch)", grid.x - 0.3, grid.y - 0.3, grid.w + 0.6, 0.25,
  { fontSize: 10, color: MUTED, align: "center" });
const cw = grid.w / 12, ch = grid.h / 3;
for (let k = 0; k < 3; k++) {
  text(["Fp1", "Cz", "O1"][k], grid.x - 0.45, grid.y + ch * k + ch / 2 - 0.1, 0.38, 0.2,
    { fontSize: 9, color: MUTED, align: "right" });
  for (let j = 0; j < 12; j++) {
    const m = V.masked[k][j] === 1;
    s.addText(m ? "M" : "", { x: grid.x + cw * j + cw * 0.08, y: grid.y + ch * k + ch * 0.1, w: cw * 0.84,
      h: ch * 0.8, shape: pres.ShapeType.roundRect, rectRadius: 0.04,
      fill: { color: m ? TOKEN_M : ENC }, line: { color: m ? TOKEN_M_EDGE : ENC_EDGE, width: 0.75 },
      fontFace: "Arial", fontSize: 8, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
  }
}
text("masked tokens (p = 0.3) → learnable [M] token", grid.x - 0.6, grid.y + grid.h + 0.08, grid.w + 1.2, 0.24,
  { fontSize: 10, align: "center" });
text("the raw signal itself is not altered", grid.x - 0.6, grid.y + grid.h + 0.32, grid.w + 1.2, 0.2,
  { fontSize: 9, color: MUTED, align: "center" });
arrow(fx(0.215) + 1.43 + 0.04, 1.72, grid.x - 0.47, 1.72);

// encoder + head
const enc = { x: fx(0.535), y: fy(0.765, 0.095), w: 1.8, h: 0.874 };
s.addText([r("Transformer\nencoder", { fontSize: 13, bold: true, breakLine: true }),
           r("criss-cross: channel / time attention", { fontSize: 8, bold: false })],
  { x: enc.x, y: enc.y, w: enc.w, h: enc.h, shape: pres.ShapeType.roundRect, rectRadius: 0.12,
    fill: { color: ENC }, line: { color: ENC_EDGE, width: 1 }, fontFace: "Arial", color: "FFFFFF",
    align: "center", valign: "middle", margin: 2 });
badge(enc.x + enc.w / 2, enc.y - 0.3);
arrow(grid.x + grid.w + 0.05, 1.72, enc.x - 0.03, 1.72);
const head = { x: fx(0.68), y: fy(0.775, 0.075), w: 1.35, h: 0.69 };
box("Band-power\nhead", head.x, head.y, head.w, head.h, { fontSize: 12.5 });
badge(head.x + head.w / 2, head.y - 0.3);
arrow(enc.x + enc.w + 0.03, 1.72, head.x - 0.03, 1.72);
const pb = { x: fx(0.80), y: fy(0.735, 0.15) - 0.15, w: 1.3, h: 1.6 };
bars(pb.x, pb.y, pb.w, pb.h, "pred", [r("prediction ŷ"), sub("c,t,b")]);
arrow(head.x + head.w + 0.03, 1.72, pb.x - 0.05, 1.72);

// target branch
arrow(1.65, eeg.y + eeg.h + 0.22, 1.65, 3.62, PURPLE);
text("same, uncorrupted\nsignal", 1.74, 2.85, 1.6, 0.4, { fontSize: 9, color: PURPLE });
const pt = { x: fx(0.035), y: 3.95, w: 2.25, h: 0.95 };
s.addText([r("one masked patch x"), sub("c,t"), r(" (200 samples)")],
  { x: pt.x - 0.2, y: pt.y - 0.3, w: pt.w + 0.4, h: 0.25, fontFace: "Arial", fontSize: 10, color: MUTED,
    align: "center", margin: 0, isTextBox: true });
s.addImage({ path: A("patch.png"), x: pt.x, y: pt.y, w: pt.w, h: pt.h });
text("time (s)", pt.x, pt.y + pt.h, pt.w, 0.18, { fontSize: 8, color: MUTED, align: "center" });

const pg = { x: fx(0.235), y: 3.7, w: 3.0, h: 1.5 };
text("per-patch periodogram (bins every 1.28 Hz)", pg.x - 0.2, pg.y - 0.28, pg.w + 0.4, 0.25,
  { fontSize: 10, color: MUTED, align: "center" });
s.addImage({ path: A("periodogram.png"), x: pg.x, y: pg.y, w: pg.w, h: pg.h });
arrow(pt.x + pt.w + 0.05, 4.42, pg.x - 0.03, 4.42, PURPLE);
text("FFT", pt.x + pt.w + 0.1, 4.12, 0.6, 0.22, { fontSize: 10, color: PURPLE, align: "center" });

const tx = fx(0.455);
text("Parameter-free target", tx, 3.72, 3.4, 0.3, { fontSize: 12, bold: true, color: PURPLE });
const formula = (runs, y) => s.addText(runs, { x: tx, y, w: 3.3, h: 0.3, fontFace: "Arial", fontSize: 11.5,
  color: INK, margin: 0, valign: "middle", isTextBox: true });
formula([r("1. Periodogram   |X"), sub("k"), r("|"), sup("2")], 4.12);
formula([r("2. Band power   P"), sub("c,t,b"), r(" = (2/P"), sup("2"), r(") Σ"), sub("f_k ∈ B_b"),
         r(" |X"), sub("k"), r("|"), sup("2")], 4.47);
formula([r("3. Log   y"), sub("c,t,b"), r(" = log(P"), sub("c,t,b"), r(" + ε)")], 4.82);
text("δ 0.5–4, θ 4–8, α 8–13, β 13–30, γ 30–45 Hz", tx, 5.15, 3.6, 0.22, { fontSize: 9.5, color: MUTED });
arrow(pg.x + pg.w + 0.05, 4.42, tx - 0.05, 4.42, PURPLE);

const tb = { x: fx(0.70), y: 3.62, w: 1.3, h: 1.6 };
bars(tb.x, tb.y, tb.w, tb.h, "tgt", [r("target y"), sub("c,t,b")]);
arrow(tx + 3.3, 4.42, tb.x - 0.08, 4.42, PURPLE);
const lb = { x: fx(0.83), y: 3.62, w: 1.5, h: 1.6 };
bars(lb.x, lb.y, lb.w, lb.h, "both", "masked patches only");
arrow(tb.x + tb.w + 0.05, 4.42, lb.x - 0.05, 4.42, PURPLE);
arrow(pb.x + pb.w / 2 + 0.35, pb.y + pb.h + 0.05, lb.x + lb.w / 2 + 0.2, lb.y - 0.03, GREEN);

s.addText([r("ℒ = 1/(5|ℳ|)  Σ"), sub("(c,t) ∈ ℳ"), r("  Σ"), sub("b"), r("  (ŷ"), sub("c,t,b"),
           r(" − y"), sub("c,t,b"), r(")"), sup("2")],
  { x: fx(0.70), y: 5.38, w: 4.2, h: 0.32, fontFace: "Arial", fontSize: 13, color: INK, margin: 0,
    valign: "middle", isTextBox: true });
text("no reconstruction term, no regularizer", fx(0.70), 5.7, 4.2, 0.22, { fontSize: 9.5, color: MUTED });

// legend
badge(1.05, 5.65);
s.addShape(pres.ShapeType.rect, { x: 1.75, y: 5.68, w: 0.18, h: 0.16, fill: { color: TEAL }, line: { color: TEAL } });
text("predicted band power", 2.0, 5.65, 1.9, 0.22, { fontSize: 10 });
s.addShape(pres.ShapeType.rect, { x: 3.85, y: 5.68, w: 0.18, h: 0.16, fill: { type: "none" },
  line: { color: INK, width: 1, dashType: "dash" } });
text("target band power", 4.1, 5.65, 1.6, 0.22, { fontSize: 10 });
s.addShape(pres.ShapeType.rect, { x: 5.6, y: 5.68, w: 0.24, h: 0.16, fill: { color: MASKC, transparency: 45 },
  line: { color: MASKC } });
text("hidden patch (panel b)", 5.9, 5.65, 1.9, 0.22, { fontSize: 10 });
s.addShape(pres.ShapeType.line, { x: 0.15, y: 6.03, w: W - 0.3, h: 0, line: { color: "2A6F97", width: 1.25 } });

// ===================== (b) ours vs alternative =====================
text("(b) What the target asks the encoder to learn", 0.12, 6.1, 7, 0.3, { fontSize: 15, bold: true });
const panels = [{ x0: fx(0.02), col: "79C9C4", tag: "ours", title: "Predict band power" },
                { x0: fx(0.51), col: RED, tag: "alternative", title: "Reconstruct the raw waveform" }];
panels.forEach((p) => {
  s.addShape(pres.ShapeType.roundRect, { x: p.x0, y: 6.42, w: 7.05, h: 2.62, rectRadius: 0.15,
    fill: { type: "none" }, line: { color: p.col, width: 1.5 } });
  s.addText(p.tag, { x: p.x0 + 0.18, y: 6.52, w: 0.95, h: 0.28, shape: pres.ShapeType.roundRect, rectRadius: 0.08,
    fill: { color: p.col }, line: { color: p.col }, fontFace: "Arial", fontSize: 11, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0 });
  text(p.title, p.x0 + 1.4, 6.52, 4.5, 0.3, { fontSize: 13, align: "center" });
  s.addShape(pres.ShapeType.line, { x: p.x0 + 3.52, y: 6.95, w: 0, h: 1.95, line: { color: p.col, width: 1 } });
});
// ours
s.addImage({ path: A("eeg_masked_2ch.png"), x: fx(0.035), y: 7.05, w: 1.95, h: 1.15 });
bars(fx(0.19), 7.0, 0.75, 1.25, "both", null);
arrow(fx(0.035) + 1.98, 7.62, fx(0.19) - 0.04, 7.62, "4FB3AD");
text("infer the hidden patch's power from its context", fx(0.02) + 0.1, 8.5, 3.35, 0.28,
  { fontSize: 10, italic: true, align: "center" });
text("many waveforms", fx(0.27) - 0.1, 6.92, 1.55, 0.22, { fontSize: 10, align: "center" });
s.addImage({ path: A("waves_same_power.png"), x: fx(0.27), y: 7.15, w: 1.35, h: 1.2 });
text("the same power", fx(0.40) - 0.25, 6.92, 1.4, 0.22, { fontSize: 10, align: "center" });
bars(fx(0.40), 7.15, 0.9, 1.15, "tgt", null);
arrow(fx(0.27) + 1.38, 7.62, fx(0.40) - 0.05, 7.62, "4FB3AD");
text("phase is unpredictable and irrelevant:\nlearn the spectral state", fx(0.02) + 3.55, 8.42, 3.45, 0.45,
  { fontSize: 10, bold: true, align: "center" });
// alternative
s.addImage({ path: A("eeg_masked_2ch.png"), x: fx(0.525), y: 7.05, w: 1.95, h: 1.15 });
text("every sample of\nthe masked patch", fx(0.665) - 0.25, 6.88, 1.5, 0.4, { fontSize: 9.5, align: "center" });
s.addImage({ path: A("recon_patch.png"), x: fx(0.665), y: 7.3, w: 1.0, h: 1.0 });
arrow(fx(0.525) + 1.98, 7.62, fx(0.665) - 0.04, 7.62, RED);
text("infer the hidden patch's waveform from its context", fx(0.51) + 0.1, 8.5, 3.4, 0.28,
  { fontSize: 10, italic: true, align: "center" });
text("one power", fx(0.76) - 0.3, 6.92, 1.3, 0.22, { fontSize: 10, align: "center" });
bars(fx(0.76), 7.15, 0.75, 1.15, "pred", null);
text("many possible waveforms", fx(0.85) - 0.3, 6.92, 1.95, 0.22, { fontSize: 10, align: "center" });
s.addImage({ path: A("waves_many.png"), x: fx(0.85), y: 7.15, w: 1.35, h: 1.2 });
arrow(fx(0.76) + 0.8, 7.62, fx(0.85) - 0.05, 7.62, RED);
text("capacity spent on phase and noise", fx(0.51) + 3.55, 8.5, 3.45, 0.28,
  { fontSize: 10, bold: true, align: "center" });

pres.writeFile({ fileName: OUT }).then((f) => console.log("wrote", f));
