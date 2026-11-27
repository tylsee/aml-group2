# Tyra — CDC-CP experiments (V12b-focused)

Selected, **original** training/evaluation evidence imported from the Google Drive account `plumtys@gmail.com` into the AML Group 2 GitHub repository. The source artifacts are preserved byte-for-byte. This is a **curated evidence export**, not a claim that all research files or an executable final notebook are included.

**Source Drive folder:** [DPD-Small](https://drive.google.com/drive/folders/1bTOnArjpgS1bABqGOHB-cxwB6-ILVtJy).

## Where to start

| Experiment | Contents | Interpretation |
| --- | --- | --- |
| [V12b frozen full-8,000](experiments/v12b_full8000/) | Run identity, internal epoch selection, pair-head training curves, locked-bundle metadata, post-hoc official evaluation | Core V12b study |
| [V14A — cRT from V12b](experiments/v14a_crt/) | Development comparison, summary, learning histories and plot | Follow-up ablation |
| [V14B — Balanced Softmax from V12b](experiments/v14b_balanced_softmax/) | Development comparison, ablation summaries, run and model manifests | Follow-up ablation |
| [Cross-version evidence](comparisons/) | Retrospective evaluation table and checkpoint inventory | **Post-hoc exploratory** comparisons |
| [Checkpoint locations](checkpoints/README.md) | Verified Drive links; no large `.pt` files committed | Reproducibility reference |

## Selected results — distinguish evaluation protocols

| Experiment / protocol | Seen composition Top-1 | Unseen composition Top-1 | Top-1 harmonic mean |
| --- | ---: | ---: | ---: |
| Original V12b pair head — **development validation** | 50.94% | 7.50% | 13.07% |
| V14A cRT + new pair head — **development validation** | 45.61% | 5.00% | 9.01% |
| V14B Balanced Softmax + pair head — **development validation** | 56.98% | 7.50% | 13.26% |
| V12b **frozen CNN / 8,000-image refit**, official evaluation **post-hoc exploratory** | 40.55% | 5.18% | 9.19% |

**These are not four independent official-test evaluations.** Development values are from V14A/V14B saved baseline/comparison CSVs; the frozen full-8,000 post-hoc values are from [V12b evaluation summary](experiments/v12b_full8000/evaluation/v12b_full_evaluation_summary.csv). Do not use the post-hoc official results for checkpoint selection.

The full-8,000 V12b refit reports frozen crop/disease CNN weights, 323 observed training pairs, 9,625 crop-disease candidate pairs, and a 9-epoch pair-head selection from an internal training holdout. Its original [full-refit summary](experiments/v12b_full8000/reports/full_refit_summary.json) explicitly distinguishes refitting from official evaluation.

## Scope and limitations

- Selected **24** small result/configuration files, not the entire Drive experiment archive.
- No raw dataset, image dataset, model `.pt` checkpoints, feature caches, large predictions, or synthetic image corpus were copied.
- The V12b image-to-image ablation folder was inspected, but its reports directory contained no files at inspection time, so it was **not** represented as a completed numerical study here.
- The inspected Drive experiment folders did not provide a complete runnable V12b notebook/source package; run logs/configs are *evidence*, not executable source code.
- [IMPORT_MANIFEST.md](IMPORT_MANIFEST.md) records every imported file's source Drive ID and Git blob SHA.

The source Drive materials remain unmodified. Verify Google Drive sharing permissions separately before giving checkpoint links to assessors.
