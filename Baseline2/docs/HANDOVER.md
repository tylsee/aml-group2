# Baseline 2 teammate handover

Open [the main notebook](../baseline2.ipynb) for the method, training stages and official results. The [experiment summary](EXPERIMENT_SUMMARY.md) compares the saved development runs. The shareable summaries under outputs/development/, outputs/final_training/, and outputs/official/ cover Exp04, Exp06, Exp07, the final fit and both official scoring modes. The original run directories below are local artifacts. CSV and JSON accuracies are fractions unless labelled otherwise.

## Local artifact map

| Location | Contents |
|---|---|
| splits_v2/ | Original development CSVs and split manifest required for split verification. |
| outputs/exp01/ through outputs/exp05/ | Stage 1 checkpoints, configurations, histories and validation metrics. |
| outputs/stage2_development/ft01/ through ft04/ | Development fine-tuning results and checkpoints. |
| outputs/exp06_calibration/ and outputs/exp07_tta/ | Original post-processing reports and per-image predictions. |
| outputs/stage2_final/b2_final_exp04/ | Final checkpoint, history, configuration and calibration lock. Keep the checkpoint, lock and original splits together. |
| outputs/stage2_final/b2_final_exp04/official_evaluation_both/ | Full official reports, predictions, errors and confusion matrices for both modes. |
| archive/ | Earlier notebook versions, the legacy evaluator and superseded instructions. |
| archive/experiments/ | Original executed Exp06 and Exp07 notebooks. The two Exp06 files have identical cell source and distinct saved outputs. |
| archive/backups/ | Pre-change files in their original relative locations and a manifest of hashes and moves for rollback. |

Place approved local data and artifacts at the relative paths above. The training-image folder defaults to DPD_Small_dataset beside Baseline2. The public-facing reports are sufficient to review results without loading weights or running experiments.

## Repository review

Run `python scripts/audit_before_push.py` to check core files, run switches and Git status when available. Review the files before staging. Ignore rules cover local data, environments, raw outputs, checkpoints, caches and the archive. Files already tracked by Git require separate review.
