"""Draw the architecture of the new approach (P1, P2, P3: five single-task models) as reports/architecture.png and .svg.

Every number in the drawing comes from cdccp/single_task.py and scripts/make_final_model_notebook.py (25 Nov 2026 build,
code version 0777e2f70c39). Run: python D:\\dev\\uni\\AML\\reports\\make_architecture.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
W, H = 100, 182
PHASE_COLOURS = {"P1": "#0072B2", "P2": "#D55E00", "P3": "#009E73"}
BODY, SMALL, HEAD = 11.5, 10.5, 13

fig = plt.figure(figsize=(15, 15 * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")


def box(cx, top, w, h, title, lines=(), *, edge="black", lw=1.6, style="-", size=BODY, title_size=None, fill="white"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, top), w, h, boxstyle="round,pad=0,rounding_size=0.8", linewidth=lw,
                                edgecolor=edge, facecolor=fill, linestyle=style))
    text = [(title, "bold", title_size or size)] + [(line, "normal", size) for line in lines]
    step = h / (len(text) + 1)
    for i, (t, weight, s) in enumerate(text, start=1):
        ax.text(cx, top + i * step, t, ha="center", va="center", fontsize=s, fontweight=weight)


def arrow(x0, y0, x1, y1, colour="black", lw=1.6, style="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, color=colour, linewidth=lw,
                                 linestyle=style, shrinkA=0, shrinkB=0))


ax.text(50, 3.5, "CDC-CP new approach: P1, P2, P3 from five single-task models", ha="center", va="center",
        fontsize=19, fontweight="bold")

# input
box(50, 7, 46, 11, "Leaf photo", ["resized to 256, centre crop 224 x 224 RGB",
                                   "dev and test: the photo and its mirror image; the two answers are averaged"])

# five models
COLS = [10.25, 29.75, 49.25, 68.75, 88.25]
CW = 18.5
CONV_TINY = ["ConvNeXt-Tiny", "ImageNet weights", "4 convolution stages", "+ average pool + LayerNorm",
             "stochastic depth 0.1"]
CLIP_IMG = ["CLIP ViT-B/16 image encoder", "OpenAI CLIP weights", "12 transformer layers,", "16 x 16 patches",
            "+ projection"]
MODELS = [
    dict(name="crop_b1", task="CROP model (55 classes)", recipe=["standard augmentation", "cross-entropy, wd 0.001"],
         backbone=CONV_TINY, feat=["768 numbers"], head=["Dropout 0.3", "Linear 768 -> 55"], out="log P(crop)"),
    dict(name="crop_clip", task="CROP model (55 classes)", recipe=["standard augmentation", "cross-entropy, wd 0.1"],
         backbone=CLIP_IMG, feat=["512 numbers,", "scaled to length 1"],
         head=["cosine with 55 crop", "text vectors", "x a learned scale"], out="log P(crop)"),
    dict(name="disease_b1", task="DISEASE model (175 classes)", recipe=["standard augmentation", "cross-entropy, wd 0.001"],
         backbone=CONV_TINY, feat=["768 numbers"], head=["Dropout 0.2", "Linear 768 -> 175"], out="log P(disease)"),
    dict(name="disease_clip", task="DISEASE model (175 classes)", recipe=["standard augmentation", "cross-entropy, wd 0.1"],
         backbone=CLIP_IMG, feat=["512 numbers,", "scaled to length 1"],
         head=["cosine with 175 disease", "text vectors", "x a learned scale"], out="log P(disease)"),
    dict(name="disease_small_reg", task="DISEASE model (175 classes)",
         recipe=["strong augmentation", "label smoothing 0.1, wd 0.05"],
         backbone=["ConvNeXt-Small", "ImageNet weights", "4 convolution stages", "+ average pool + LayerNorm",
                   "stochastic depth 0.4"], feat=["768 numbers"], head=["Dropout 0.2", "Linear 768 -> 175"],
         out="log P(disease)"),
]
ROWS = dict(header=(22, 15), backbone=(41, 19), feat=(64, 7), head=(75, 11), out=(90.5, 6))
for i, (cx, m) in enumerate(zip(COLS, MODELS)):
    def cell(row, title, lines, **kw):
        top, h = ROWS[row]
        box(cx, top, CW, h, title, lines, **kw)

    arrow(50, 18, cx, ROWS["header"][0])
    cell("header", m["name"], [m["task"]] + m["recipe"], size=SMALL, title_size=HEAD, lw=2.2)
    arrow(cx, 37, cx, ROWS["backbone"][0])
    cell("backbone", m["backbone"][0], m["backbone"][1:], lw=2.4)
    arrow(cx, 60, cx, ROWS["feat"][0])
    cell("feat", "Feature", m["feat"], size=SMALL)
    arrow(cx, 71, cx, ROWS["head"][0])
    cell("head", "Head", m["head"], size=SMALL)
    arrow(cx, 86, cx, ROWS["out"][0])
    cell("out", m["out"], [], fill="#F2F2F2")
ax.text(50, 20.3, "Each model looks at the photo on its own and answers one question only: which crop, or which disease.",
        ha="center", va="center", fontsize=SMALL, style="italic", bbox=dict(facecolor="white", edgecolor="none", pad=1))

# phases: which model goes into which phase, as coloured tags under each model
PHASES = {"P1": (17, ["crop: crop_b1", "disease: disease_b1"]),
          "P2": (50, ["crop: crop_b1", "disease: disease_clip"]),
          "P3": (83, ["crop: average of crop_b1 + crop_clip", "disease: average of disease_clip", "+ disease_small_reg",
                      "(average of the log P, equal weights)"])}
USES = {"P1": ("crop_b1", "disease_b1"), "P2": ("crop_b1", "disease_clip"),
        "P3": ("crop_b1", "crop_clip", "disease_clip", "disease_small_reg")}
for cx, m in zip(COLS, MODELS):
    used = [phase for phase in PHASES if m["name"] in USES[phase]]
    ax.text(cx - 8.6, 100, "used in:", ha="left", va="center", fontsize=SMALL)
    for k, phase in enumerate(used):
        ax.text(cx - 1.2 + k * 3.6, 100, phase, ha="center", va="center", fontsize=SMALL, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PHASE_COLOURS[phase], edgecolor="none"))
ax.text(50, 105.5, "CLIP text vectors: the frozen CLIP text encoder reads 4 sentences per class "
        "(e.g. \"a photo of a plant leaf with early blight.\"); their average is the class vector.",
        ha="center", va="center", fontsize=SMALL, style="italic")
PTOP, PH = 112, 17
for phase, (cx, lines) in PHASES.items():
    colour = PHASE_COLOURS[phase]
    box(cx, PTOP, 30, PH, phase, lines, edge=colour, lw=3, title_size=16)
    arrow(cx, PTOP + PH, 50 + (cx - 50) * 0.5, 134, colour=colour, lw=2.2)

# pair score
box(50, 134, 92, 13, "Pair score, for each phase (one photo, every crop c and disease d)",
    ["S(c, d) = log P(crop c) + r x log P(disease d) - gamma x [pair (c, d) was seen in training]",
     "the pair with the highest score is the answer"], size=13, title_size=14, lw=2.6)
arrow(30, 147, 27, 151)
arrow(70, 147, 73, 151)
box(27, 151, 50, 17, "1. Raw results (reported first)",
    ["gamma = 0, r = 1",
     "all 9,625 pairs (55 x 175): top crop + top disease",
     "= the Baseline 1 rule",
     "333-pair list: best of the 323 seen + 10 unseen pairs"], size=SMALL, title_size=HEAD, lw=2)
box(77, 151, 42, 17, "2. Calibrated results",
    ["gamma (and r) chosen on the dev split, per phase",
     "rule 1 keep seen (main): best dev unseen,",
     "dev seen at most 3 points lower",
     "rule 2 best dev HM: r from 0.5 to 2"], size=SMALL, title_size=HEAD, lw=2, style="--")
arrow(27, 168, 45, 171)
arrow(77, 168, 55, 171)
box(50, 171, 70, 9, "Official test, once per phase",
    ["seen, unseen and HM for crop, disease and pair; Top-1 and Top-5; compared with Baseline 1"],
    size=SMALL, title_size=HEAD, lw=2.6)

fig.savefig(OUT / "architecture.png", dpi=150)
fig.savefig(OUT / "architecture.svg")
print("written", OUT / "architecture.png", OUT / "architecture.svg", file=sys.stdout)
