"""Build REPORT_TABLES.md from the Colab result files in final_model_results/. Run: python make_report_tables.py"""
import ast
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "final_model_results"
FINAL_MODEL_PY = HERE.parent / "cdc-cp" / "cdccp" / "final_model.py"
OUT = HERE / "REPORT_TABLES.md"
MODELS = ("crop_b1", "crop_clip", "disease_b1", "disease_clip", "disease_small_reg")
PHASES = ("P1", "P2", "P3")


def rows(name):
    with open(RES / "results" / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def hist(stage, model):
    with open(RES / stage / model / "history.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f2(x):
    return f"{float(x):.2f}"


def hm(a, b):
    a, b = float(a), float(b)
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def table(header, body):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(lines)


def settings_table():
    body = []
    for m in MODELS:
        c1 = json.loads((RES / "stage1" / m / "config.json").read_text(encoding="utf-8"))
        rv, b1, clip = c1["recipe_values"], c1["b1"], c1["backbone"] == "clip_vit_b16"
        lr1 = (rv.get("backbone_lr", b1["backbone_lr"]), rv.get("head_lr", b1["head_lr"]))
        lr2 = (rv.get("stage2_backbone_lr", b1["stage2_backbone_lr"]), rv.get("stage2_head_lr", b1["stage2_head_lr"]))
        sd = rv["stochastic_depth"].get(c1["backbone"])
        sd_text = "none" if clip else ("0.1 (torchvision default)" if sd is None else sd)
        head = "frozen CLIP text prompts, cosine x learned scale" if clip else f"dropout {b1['dropout'][c1['task']]} + linear 768->N"
        body.append([m, c1["task"], c1["backbone"], c1["aug"], head, f"{lr1[0]:g} / {lr1[1]:g}", f"{lr2[0]:g} / {lr2[1]:g}",
                     rv["weight_decay"], rv["label_smoothing"], sd_text, b1["batch_size"], b1["seed"]])
    return table(["Model", "Task", "Backbone", "Augmentation", "Head", "Stage 1 lr (backbone / head)",
                  "Stage 2 lr (backbone / head)", "Weight decay", "Label smoothing", "Stochastic depth", "Batch", "Seed"], body)


def training_table():
    board = {r["arm"]: r for r in rows("dev_stage1_board.csv")}
    body = []
    for m in MODELS:
        b, h1, h2 = board[m], hist("stage1", m), hist("stage2", m)
        last2 = h2[-1]
        body.append([m, len(h1), b["best_epoch"], f2(100 * float(b["seen_acc"])), f2(100 * float(b["unseen_acc"])),
                     f2(100 * float(b["individual_h"])), last2["best_epoch"], f2(100 * float(last2["best_train_acc"]))])
    return table(["Model", "Stage 1 epochs run", "Best dev epoch", "Dev seen Top-1 (%)", "Dev unseen Top-1 (%)", "Dev HM (%)",
                  "Stage 2 epoch kept", "Stage 2 train Top-1 (%)"], body)


def baseline1():
    """Baseline 1 official test values, read from EARLIER_RUNS in cdccp/final_model.py (the values the notebook compares with)."""
    tree = ast.parse(FINAL_MODEL_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "EARLIER_RUNS" for t in node.targets):
            return ast.literal_eval(node.value)["Baseline 1"]
    raise SystemExit(f"EARLIER_RUNS not found in {FINAL_MODEL_PY}")


def main_table(pair_list):
    body = []
    for task, (s1, u1, s5, u5) in baseline1().items():
        body.append(["Baseline 1", task, 1, f2(s1), f2(u1), f2(hm(s1, u1))])
        body.append(["Baseline 1", task, 5, f2(s5), f2(u5), f2(hm(s5, u5))])
    for r in rows("official_test_results.csv"):
        if r["pair list"] == pair_list:
            body.append([r["phase"], r["Task"], r["Top-K"], f2(r["Seen (%)"]), f2(r["Unseen (%)"]), f2(r["HM (%)"])])
    return table(["Phase", "Task", "Top-K", "Seen (%)", "Unseen (%)", "HM (%)"], body)


def compare_table():
    body = [[r["model"], f2(r["crop HM"]), f2(r["disease seen"]), f2(r["disease unseen"]), f2(r["disease HM"]), f2(r["pair seen"]),
             f2(r["pair unseen"]), f2(r["pair HM"]), f2(r["pair Top-5 HM"])] for r in rows("compare_phases.csv")]
    return table(["Model", "Crop HM", "Disease seen", "Disease unseen", "Disease HM", "Pair seen", "Pair unseen", "Pair HM",
                  "Pair Top-5 HM"], body)


def calibrated_table():
    data = rows("official_test_raw_and_calibrated.csv")
    body = []
    for phase in PHASES:
        for model in ("Raw, heads only", "Raw, pair list", "Calibrated, pair list", "Calibrated (best dev HM), pair list",
                      "Reference: unseen pairs only"):
            pick = {r["Test split"]: r for r in data if r["phase"] == phase and r["Model"] == model}
            s, u = pick["Seen pairs"], pick["Unseen pairs"]
            label = {"Raw, heads only": "Raw, all 9,625 pairs", "Raw, pair list": "Raw, 333-pair list",
                     "Calibrated, pair list": "Calibrated rule 1 (keep seen), 333 list",
                     "Calibrated (best dev HM), pair list": "Calibrated rule 2 (best dev HM), 333 list",
                     "Reference: unseen pairs only": "Reference only: never predicts a seen pair, 333 list"}[model]
            gamma = "very large" if model.startswith("Reference") else f2(s["gamma"])
            body.append([phase, label, gamma, s["r"], f2(s["Crop-only Top-1"]), f2(u["Crop-only Top-1"]),
                         f2(s["Disease-only Top-1"]), f2(u["Disease-only Top-1"]), f2(s["Pair Top-1"]), f2(u["Pair Top-1"]),
                         f2(hm(s["Pair Top-1"], u["Pair Top-1"]))])
    return table(["Phase", "Setting", "Gamma", "r", "Crop seen (%)", "Crop unseen (%)", "Disease seen (%)",
                  "Disease unseen (%)", "Pair seen (%)", "Pair unseen (%)", "Pair HM (%)"], body)


def dev_gamma_table():
    body = [[r["phase"], f2(r["gamma_keep"]), f2(r["A_S_raw"]), f2(r["A_S_keep"]), f2(r["A_U_raw"]), f2(r["A_U_keep"]),
             r["r_hm"], f2(r["gamma_hm"]), f2(r["A_S_hm"]), f2(r["A_U_hm"]), f2(r["H_hm"])] for r in rows("dev_calibration.csv")]
    return table(["Phase", "Rule 1 gamma", "Dev seen raw", "Dev seen rule 1", "Dev unseen raw", "Dev unseen rule 1",
                  "Rule 2 r", "Rule 2 gamma", "Dev seen rule 2", "Dev unseen rule 2", "Dev HM rule 2"], body)


def unseen_pairs_table():
    data = rows("unseen_pairs_P3.csv")
    total = sum(int(r["images"]) for r in data)
    body = [[r["composition_name"], r["images"], f2(100 * int(r["images"]) / total), f2(r["right_crop"]), f2(r["right_disease"]),
             f2(r["right_pair"])] for r in data]
    biggest = max(data, key=lambda r: int(r["images"]))
    right = {r["composition_name"]: int(r["images"]) * float(r["right_pair"]) / 100 for r in data}
    top_right = max(right, key=right.get)
    top3 = sum(sorted((int(r["images"]) for r in data), reverse=True)[:3])
    note = (f"{biggest['composition_name']} is the largest unseen pair ({100 * int(biggest['images']) / total:.1f}% of unseen "
            f"images) and P3 gets {float(biggest['right_pair']):.2f}% of it right, so it pulls the unseen score down. "
            f"Most correct unseen pairs ({right[top_right]:.0f} of {sum(right.values()):.0f}) are {top_right}. "
            f"The unseen score rests on only 10 pairs; the 3 largest hold {100 * top3 / total:.1f}% of unseen images.")
    return table(["Unseen test pair", "Images", "Share of unseen test (%)", "Right crop (%)", "Right disease (%)",
                  "Right pair (%)"], body), total, note


def paper_table():
    """Paper-style sweep numbers (paper_metrics.py) on test and dev, or None when the test file is missing."""
    body = []
    for split, name in (("test", "paper_metrics_test.csv"), ("dev", "paper_metrics_dev.csv")):
        path = HERE / name
        if not path.exists():
            return None
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                world = "closed (333 list)" if r["world"] == "closed" and split == "test" else (
                    "closed (323 dev list)" if r["world"] == "closed" else "open (9,625 pairs)")
                body.append([split, r["phase"], world, f2(r["best_seen"]), f2(r["best_unseen"]), f2(r["best_hm"]),
                             f"{f2(r['seen_at_best_hm'])} / {f2(r['unseen_at_best_hm'])}", f2(r["auc"])])
    return table(["Split", "Phase", "Candidate pairs", "Best seen (%)", "Best unseen (%)", "Best HM (%)",
                  "Seen / unseen at best HM (%)", "AUC (%)"], body)


def main():
    pairs, total, pairs_note = unseen_pairs_table()
    paper = paper_table()
    parts = [
        "# Report tables: new approach P1-P3 (generated by make_report_tables.py; do not edit by hand)",
        "Source: Colab run New_approach_done.ipynb, NVIDIA L4, seed 42, one run. Files: final_model_results/.",
        "All numbers are on the official test set unless the table says dev. Raw results come first; calibrated second.",
        "## Terms",
        "- P1: crop_b1 + disease_b1. P2: crop_b1 + disease_clip. P3: crop = mean of crop_b1 and crop_clip; disease = mean of "
        "disease_clip and disease_small_reg (means of log-probabilities).",
        "- Pair score = log P(crop) + r x log P(disease) - gamma x [pair seen in training]. Raw results use gamma = 0 and r = 1.",
        "- gamma: a penalty taken off every seen pair so that unseen pairs can win. r: the weight of the disease score.",
        "- Augmentation \"b1\": random crop 224 (area 85-100%), horizontal and vertical flip, rotation +-10 degrees, brightness "
        "and contrast +-0.15 (the Baseline 1 recipe). \"strong\": crop area 30-100%, flips, TrivialAugmentWide, random erasing "
        "(p 0.25).",
        "## Table 1. Training settings of the five models",
        "All models: AdamW, cross-entropy loss, cosine learning-rate decay (stage 1 T_max 30, stage 2 T_max 10), no warm-up, "
        "mixed precision, 224 x 224 input. Stage 1: dev split (6,905 train images), max 25 epochs, early stop after 5 epochs "
        "with no better dev HM (seen and unseen Top-1 of the model's own task), keep the best dev epoch. Stage 2: 10 epochs on "
        "all 8,000 training images from the best stage-1 weights; keep the epoch with the best train accuracy. Test: average "
        "of the image and its mirror image (flip). CLIP models use CLIP's own image normalisation.",
        settings_table(),
        "## Table 2. Training results (dev split, stage 1) and stage 2",
        training_table(),
        "## Table 3. Main result: official test, raw, all 9,625 pairs (same pair rule as Baseline 1)",
        main_table("all 9,625 pairs (Baseline 1 rule)"),
        "Baseline 1 values come from the team Baseline 1 notebook; its HM here is recomputed from its rounded seen and unseen "
        "values (the Baseline 1 notebook prints pair HM 11.90 for Top-1 and 30.80 for Top-5). P1 retrains the Baseline 1 "
        "recipe (the same ConvNeXt-Tiny models and settings). It differs in three ways: flip averaging at test (Baseline 1 "
        "has none); stage 1 stops early (max 25 epochs, patience 5), where Baseline 1 ran 30 epochs with no early stop; and "
        "the GPU (NVIDIA L4 here, Tesla T4 for Baseline 1).",
        "## Table 4. Comparison with Baseline 1 (raw, all 9,625 pairs, Top-1 in %; last column Top-5)",
        compare_table(),
        "## Table 5. Raw and calibrated results (Top-1 in %)",
        calibrated_table(),
        "The 333-pair list is the 323 training pairs plus the 10 unseen test pairs (closed-world setting). Calibrated results "
        "use this list, so they assume the 10 possible new pairs are known. Rule 1 was named the main calibrated result "
        "before the test was run.",
        "Read the calibrated unseen numbers with care. In 8 of the 10 unseen pairs, the crop has only one unseen pair in the "
        "list (cucumber has two). So crop accuracy caps calibrated unseen pair accuracy. With rule 2 and in the reference row, "
        "unseen pair accuracy is within 2 points of unseen crop accuracy, because the model only has to pick the crop's one "
        "unseen pair over its seen pairs, not one of 175 diseases. With rule 1 the gap is larger (for example P3: 74.86 pair "
        "against 84.74 crop).",
        "## Table 6. How gamma was chosen (dev split only, before the test labels were read; pair Top-1 in %)",
        dev_gamma_table(),
        "These are pair Top-1 values on the dev pair list (seen dev pairs plus the 10 hidden dev pairs), from the stage-1 "
        "models with flip averaging; they are not the single-model values of Table 2. Rule 1 (keep seen): best dev unseen "
        "accuracy while dev seen accuracy drops at most 3 points, r = 1. Rule 2 (best dev HM): best dev HM over r from 0.5 "
        "to 2.",
        f"## Table 7. P3 on each unseen test pair (raw, all 9,625 pairs; {total} images)",
        pairs,
        pairs_note,
        *([
            "## Table 8. Paper-style results: gamma swept over the whole curve (pair Top-1, r = 1)",
            paper,
            "Computed by paper_metrics.py from the saved log-probabilities (checks: check_paper_metrics.py). It follows "
            "the reporting idea of the CZSL papers (TMN, CGE, CompCos): gamma is swept from \"always a seen pair\" to "
            "\"always an unseen pair\". Here the sweep visits every distinct operating point; public CZSL evaluation "
            "code samples a smaller set of bias values, which can give slightly lower best HM and AUC (a few tenths of a "
            "point on our scores). Best seen, best unseen and best HM are each the best point of the sweep on the split "
            "shown, so on test they are optimistic and are not a model or gamma choice. Table 3 holds the raw results "
            "(gamma = 0) and Table 5 the results with gamma fixed on dev before the test. AUC is the area under the "
            "seen-against-unseen accuracy curve, in % of the unit square.",
            "Open-world best seen equals closed-world best seen, and is higher than the raw seen value of Table 3: at the "
            "\"always a seen pair\" end only the seen pairs compete (323 on test, 313 on dev), in both worlds. Closed-world "
            "best unseen equals the reference row of Table 5, and like it is capped by crop accuracy, because in 8 of "
            "the 10 unseen pairs the crop has only one unseen pair in the list (see the note under Table 5). Published benchmarks have many more unseen "
            "pairs (for example MIT-States, UT-Zappos, C-GQA), so our closed-world AUC is not comparable with theirs; "
            "the open-world rows are the harder setting. The dev rows come from the stage-1 models on the dev pair list "
            "(313 seen + 10 hidden pairs), not from the stage-2 models used on test.",
        ] if paper else []),
        "## Figures",
        "- Framework: ../new_approach/architecture.png",
        "- Training curves: final_model_results/results/stage1_curves.png",
        "- Confusion matrix of seen test pairs: final_model_results/results/confusion_seen_333_P3.png. It is P3, raw, on the "
        "333-pair list, so its pair Top-1 (72.15%) is higher than the 9,625-pair value in Table 3 (70.55%): fewer wrong "
        "pairs can be chosen.",
        "## Points the report must state",
        "- One seed (42) and one run, so there are no error bars. Gaps of 1-3 points between phases may be noise.",
        "- The main comparison with Baseline 1 is Table 3 / Table 4 (all 9,625 pairs). Calibrated results (Table 5) use the "
        "closed-world 333-pair list, and their unseen accuracy is capped by crop recognition (see the note under Table 5).",
        "- " + pairs_note,
        "- On test, calibration rule 1 lowered seen pair accuracy by more than the 3-point dev limit in all three phases "
        "(Table 5, against the \"Raw, 333-pair list\" row: P1 -3.65, P2 -3.80, P3 -5.50 points). Gamma was chosen with the stage-1 models on dev, but the test uses "
        "the stage-2 models, which were trained again on all 8,000 images, so the dev limit does not carry over exactly.",
        "- Gamma for this run was fixed on dev before the test labels were read. But the P2 and P3 designs came from earlier "
        "runs that were scored on the same official test set (P5 had the P2 design; P6 had the P3 design, pair Top-1 70.95 "
        "seen / 10.01 unseen). So the test is not fully untouched by design choices.",
        "- Report rule 1 and rule 2; do not pick one by its test score.",
        "- The brief needs two baselines. These tables compare with Baseline 1 only; Baseline 2 (multi-task CNN) results "
        "must be added.",
    ]
    OUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print("written", OUT)


if __name__ == "__main__":
    main()
