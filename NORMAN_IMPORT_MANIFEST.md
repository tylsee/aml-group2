# Norman source-branch import manifest

- **Source repository:** `tylsee/aml-group2`
- **Source branch:** `new-approach`
- **Source commit:** `96dbc8a935ed70139ee2629eeea72172a0ddb0b9`
- **Destination:** `main` after review via `chore/import-new-approach-into-norman`
- **Source tracked files:** 86
- **Policy:** Copy into the reorganized Norman folders, keep the source branch unchanged, preserve original file bytes, and do not fabricate checkpoints or rerun experiments.

## Exact source-to-destination path mapping

| Source path on `new-approach` | Destination path |
| --- | --- |
| `README.md` | `new_proposed_approach/norman_convnext_clip/SOURCE_BRANCH_README.md` |
| `NewApproach/.gitattributes` | `new_proposed_approach/norman_convnext_clip/experiments/source_repo_metadata/.gitattributes` |
| `NewApproach/.gitignore` | `new_proposed_approach/norman_convnext_clip/experiments/source_repo_metadata/.gitignore` |
| `NewApproach/cdc-cp/**` | `new_proposed_approach/norman_convnext_clip/experiments/cdc-cp/**` |
| `NewApproach/new_approach/New_approach.ipynb` | `new_proposed_approach/norman_convnext_clip/experiments/New_approach.ipynb` |
| `NewApproach/new_approach/New_approach_done.ipynb` | `new_proposed_approach/norman_convnext_clip/final/New_approach_done.ipynb` |
| `NewApproach/new_approach/Final_model_colab_steps.txt` | `new_proposed_approach/norman_convnext_clip/final/Final_model_colab_steps.txt` |
| `NewApproach/new_approach/architecture.png` | `new_proposed_approach/norman_convnext_clip/final/architecture.png` |
| `NewApproach/results/final_model_results/**` | `new_proposed_approach/norman_convnext_clip/final/final_model_results/**` |
| Other `NewApproach/results/*` files | `new_proposed_approach/norman_convnext_clip/final/reporting/*` |

## Reproducibility notes

The original instructions refer to `NewApproach/cdc-cp` when building `cdc-cp_code_final_model.zip`. The source code was copied into `experiments/cdc-cp` without rewriting paths or changing behavior. Recreate the ZIP from its new location or use the untouched `new-approach` branch layout.

All original tracked `.npz`, `.csv`, `.png`, `.json`, `.py`, `.txt` and `.ipynb` files are included. Large trained `.pt` checkpoint files were not tracked on the source branch and have not been invented.

## Verification

The imported files are checked by mapping every source path to its destination and comparing Git blob SHA values. This ensures no missing, overwritten or content-modified original file. Verification findings are recorded in the pull request description.
