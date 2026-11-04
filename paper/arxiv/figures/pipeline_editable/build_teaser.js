// Cover (teaser) figure: one masked input, one encoder, three targets. All shapes editable.
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const A = (f) => path.join(__dirname, "assets", f);
const V = JSON.parse(fs.readFileSync(A("values.json"), "utf8"));
const OUT = process.argv[2] || path.join(__dirname, "specbram_teaser.pptx");

const W = 13.33, H = 5.9;
const pres = new pptxgen();
pres.defineLayout({ name: "TEASER", width: W, height: H });
pres.layout = "TEASER";
const s = pres.addSlide();
s.background = { color: "FFFFFF" };
const F = "Times New Roman";
const INK = "1F1F1F", MUTED = "5E5E5E";
const GREEN = "5BB89A", GREEN_BG = "EEF8F4";       // LaBraM
const ORANGE = "E08A5F", ORANGE_BG = "FDF3EE";     // CBraMod / CSBrain / REVE
const BLUE = "3F7CC0", BLUE_BG = "E8F1FB";         // SpecBraM
const ENC = "8A9BA8", TEAL = "16848F";
const BCOL = ["5B5BD6", "8E5BD0", "1F9E89", "E0A030", "D1495B"];
const BAND = ["δ", "θ", "α", "β", "γ"];

const T = (t, x, y, w, h, o = {}) => s.addText(t, Object.assign({ x, y, w, h, fontFace: F, fontSize: 13,
  color: INK, margin: 0, valign: "middle", isTextBox: true }, o));
function arrow(x0, y0, x1, y1, color = "6B6B6B", width = 1.75) {
  s.addShape(pres.ShapeType.line, { x: Math.min(x0, x1), y: Math.min(y0, y1), w: Math.abs(x1 - x0) || 0.0001,
    h: Math.abs(y1 - y0) || 0.0001, flipH: x1 < x0, flipV: y1 < y0,
    line: { color, width, endArrowType: "triangle" } });
}
function bars(x, y, w, h, fs = 12) {                         // predicted (filled) vs target (dashed)
  const all = V.y_hat.concat(V.y_tgt), lo = Math.min(...all) - 0.6, hi = Math.max(...all) + 0.15;
  const labH = 0.24, areaH = h - labH, slot = w / 5, hgt = (v) => (v - lo) / (hi - lo) * areaH;
  for (let i = 0; i < 5; i++) {
    const cx = x + slot * (i + 0.5), th = hgt(V.y_tgt[i]), ph = hgt(V.y_hat[i]);
    s.addShape(pres.ShapeType.rect, { x: cx - slot * 0.33, y: y + areaH - th, w: slot * 0.66, h: th,
      fill: { type: "none" }, line: { color: BCOL[i], width: 1.25, dashType: "dash" } });
    s.addShape(pres.ShapeType.rect, { x: cx - slot * 0.21, y: y + areaH - ph, w: slot * 0.42, h: ph,
      fill: { color: TEAL }, line: { color: TEAL, width: 0.5 } });
    T(BAND[i], cx - slot / 2, y + areaH + 0.02, slot, labH - 0.02,
      { fontFace: "Arial", fontSize: fs, bold: true, color: BCOL[i], align: "center" });
  }
  s.addShape(pres.ShapeType.line, { x, y: y + areaH, w, h: 0, line: { color: "444444", width: 0.75 } });
}

// title
T("What should an EEG pretext task predict?", 0, 0.12, W, 0.45, { fontSize: 22, bold: true, align: "center" });

// shared masked input
const inp = { x: 0.3, y: 1.1, w: 3.25, h: 3.3 };
s.addShape(pres.ShapeType.roundRect, { x: inp.x, y: inp.y, w: inp.w, h: inp.h, rectRadius: 0.1,
  fill: { color: "F6F6F6" }, line: { color: "BDBDBD", width: 1, dashType: "dash" } });
T("Masked EEG input", inp.x, inp.y + 0.1, inp.w, 0.32, { fontSize: 15, bold: true, align: "center" });
// raw EEG (patch boundaries dotted) -> patchify + embed -> token grid with [M] tokens,
// each token aligned under its patch (same style as the pipeline figure)
const img = { x: inp.x + 0.12, y: inp.y + 0.45, w: 3.0, h: 1.1 };
s.addImage({ path: A("eeg_raw_wide.png"), x: img.x, y: img.y, w: img.w, h: img.h });
const gx0 = img.x + 0.035 * img.w, gx1 = img.x + 0.965 * img.w;   // plotted time axis
const gy = inp.y + 1.83, gcw = (gx1 - gx0) / 12, gch = 0.2;
arrow(inp.x + inp.w / 2, img.y + img.h + 0.02, inp.x + inp.w / 2, gy - 0.04, "8A9BA8", 1.25);
T("patchify + embed", inp.x + inp.w / 2 + 0.08, img.y + img.h - 0.01, 1.4, 0.2,
  { fontSize: 10, italic: true, color: MUTED });
for (let k = 0; k < 3; k++) {
  for (let j = 0; j < 12; j++) {
    const m = V.masked[k][j] === 1;
    s.addText(m ? "M" : "", { x: gx0 + gcw * j + gcw * 0.08, y: gy + k * (gch + 0.05), w: gcw * 0.84, h: gch,
      shape: pres.ShapeType.roundRect, rectRadius: 0.03, fill: { color: m ? "9E9E9E" : "8FC1A9" },
      line: { color: m ? "6F6F6F" : "5F9E82", width: 0.5 }, fontFace: "Arial", fontSize: 7, bold: true,
      color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
  }
}
T("M = masked token (learnable [M]);", inp.x, inp.y + inp.h - 0.56, inp.w, 0.24,
  { fontSize: 11, color: MUTED, align: "center" });
T("same masking for every model", inp.x, inp.y + inp.h - 0.33, inp.w, 0.26,
  { fontSize: 12, italic: true, color: MUTED, align: "center" });

// shared encoder
const enc = { x: 3.95, y: 2.25, w: 1.5, h: 0.95 };
s.addText("Transformer\nencoder", { x: enc.x, y: enc.y, w: enc.w, h: enc.h, shape: pres.ShapeType.roundRect,
  rectRadius: 0.12, fill: { color: ENC }, line: { color: "6F7F8C", width: 1 }, fontFace: F, fontSize: 15,
  bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 2 });
T("same architecture", enc.x - 0.2, enc.y + enc.h + 0.06, enc.w + 0.4, 0.28,
  { fontSize: 12, italic: true, color: MUTED, align: "center" });
arrow(inp.x + inp.w + 0.05, enc.y + enc.h / 2, enc.x - 0.03, enc.y + enc.h / 2);

// three target cards
const cx0 = 6.05, cw = W - cx0 - 0.3, chh = 1.32;
const cards = [
  { y: 0.78, col: GREEN, bg: GREEN_BG, who: "LaBraM", what: "Discrete code",
    desc: "index in a separately trained VQ codebook", verdict: "target defined by a tokenizer, not by the signal's physics", ours: false },
  { y: 2.27, col: ORANGE, bg: ORANGE_BG, who: "CBraMod · CSBrain · REVE", what: "Raw waveform",
    desc: "every sample of the masked patch", verdict: "spends capacity on phase and noise that context cannot predict", ours: false },
  { y: 3.76, col: BLUE, bg: BLUE_BG, who: "SpecBraM (ours)", what: "Band power",
    desc: "log power of the masked patch in five bands", verdict: "learns the spectral state that EEG labels are defined by", ours: true },
];
cards.forEach((c, i) => {
  s.addShape(pres.ShapeType.roundRect, { x: cx0, y: c.y, w: cw, h: chh, rectRadius: 0.1,
    fill: { color: c.bg }, line: { color: c.col, width: c.ours ? 2.5 : 1.25, dashType: c.ours ? "solid" : "dash" } });
  // who
  T(c.who, cx0 + 0.15, c.y + 0.08, 2.6, 0.3, { fontSize: 13, bold: true, color: c.col });
  // what + description
  T(c.what, cx0 + 0.15, c.y + 0.42, 2.0, 0.34, { fontSize: 17, bold: true });
  T(c.desc, cx0 + 0.15, c.y + 0.76, 2.35, 0.5, { fontSize: 11.5, color: MUTED, valign: "top" });
  // verdict
  T(c.verdict, cx0 + cw - (c.ours ? 2.05 : 2.35), c.y + 0.2, c.ours ? 1.95 : 2.2, chh - 0.4,
    { fontSize: 12.5, italic: !c.ours, bold: c.ours, color: c.ours ? BLUE : MUTED, align: "left" });
  // arrow from the encoder into the card
  arrow(enc.x + enc.w + 0.03, enc.y + enc.h / 2, cx0 - 0.04, c.y + chh / 2, c.col, c.ours ? 2.5 : 1.5);
});
// card visuals (middle column)
const vx = cx0 + 2.55, vw = 2.15;
// LaBraM: codebook tiles, one selected
[["code_tile1.png", false], ["code_tile2.png", true], ["code_tile1.png", false], ["code_tile2.png", false]]
  .forEach(([img, sel], j) => {
    const tx = vx + 0.02 + j * 0.5, ty = cards[0].y + 0.4;
    s.addImage({ path: A(img), x: tx, y: ty, w: 0.44, h: 0.37 });
    if (sel) s.addShape(pres.ShapeType.rect, { x: tx - 0.03, y: ty - 0.03, w: 0.5, h: 0.43,
      fill: { type: "none" }, line: { color: "6A3D9A", width: 2 } });
  });
T("codebook index k", vx, cards[0].y + 0.88, vw, 0.25, { fontSize: 11, color: MUTED, align: "center" });
// raw waveform: the masked patch, sample by sample
s.addImage({ path: A("recon_patch.png"), x: vx + 0.35, y: cards[1].y + 0.12, w: 1.45, h: 0.95 });
T("200 samples", vx, cards[1].y + 1.03, vw, 0.25, { fontSize: 11, color: MUTED, align: "center" });
// band power: five numbers
// band power: spectrum of the masked patch -> five band powers (phase-free)
const cy = cards[2].y;
s.addImage({ path: A("spectrum_mini.png"), x: vx - 0.12, y: cy + 0.12, w: 1.1, h: 0.8 });
arrow(vx + 1.0, cy + 0.5, vx + 1.22, cy + 0.5, BLUE, 1.5);
bars(vx + 1.27, cy + 0.12, 0.95, 0.86, 10.5);
T("phase-free, from the patch's spectrum", vx - 0.2, cy + 1.02, 2.55, 0.24,
  { fontSize: 11, italic: true, color: BLUE, align: "center" });

// takeaway strip
s.addText([
  { text: "Same backbone, data and training steps; only the target changes: ", options: { bold: false } },
  { text: "band power beats waveform reconstruction on every pretraining seed", options: { bold: true, color: BLUE } },
  { text: " (+2.6 / +2.8 balanced accuracy on ISRUC / HMC, +4.5 / +5.3 with 1% of the labels).", options: { bold: false } },
], { x: 0.3, y: 5.28, w: W - 0.6, h: 0.42, fontFace: F, fontSize: 13, color: INK, align: "center",
  valign: "middle", margin: 0, isTextBox: true });

pres.writeFile({ fileName: OUT }).then((f) => console.log("wrote", f));
