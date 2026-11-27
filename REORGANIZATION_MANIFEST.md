# File reorganization manifest

**Source repository:** `tylsee/aml-group2`  
**Original main commit:** `aced08474c158134bdf4e13731a46253471591c1`  
**Original tree:** 1,784 tracked files (blobs), 388 directory trees  
**Policy:** no experiment file deleted, no original experiment blob changed, and no notebook code modified.

## Directory-prefix moves

| Old path prefix | New path prefix |
| --- | --- |
| `baseline_1_independent_cnn/experiments/AML_B1_Clean_Results/` | `baseline_1_independent_cnn/experiments/consolidated_results/` |
| `baseline_1_independent_cnn/experiments/efficientNetB0/` | `baseline_1_independent_cnn/experiments/efficientnet_b0/` |
| `baseline_2_shared_multitask/experiments/Baseline2_Stage1_Handoff/` | `baseline_2_shared_multitask/experiments/resnet50_research_handoff/` |
| `baseline_2_shared_multitask/experiments/ConvNeXt/` | `baseline_2_shared_multitask/experiments/convnext/` |
| `baseline_2_shared_multitask/experiments/EfficientNet/` | `baseline_2_shared_multitask/experiments/efficientnet_b0/` |
| `baseline_2_shared_multitask/experiments/ResNet/` | `baseline_2_shared_multitask/experiments/resnet50/` |
| `new_proposed_approach/emily-cdc-cp-experiments/` | `new_proposed_approach/emily_cdc_cp/experiments/` |

Everything else retains the same relative path, except that new README indexes and empty-submission placeholders are added. The original `README.md` heading `# aml-group2` is preserved and expanded for navigation.

## Verification

Each renamed subtree is referenced by its original *tree object SHA*, which preserves all child paths, blob SHAs and bytes beneath the moved prefix. The final branch is checked against the source tree by mapping every original blob path to its destination and comparing exact Git blob SHAs.

## Notes

- **No Norman files** were tracked in the source repository. Placeholder READMEs document the expected destination only.
- No checkpoint data is generated or inferred.
- Existing absolute filesystem paths in notebook source or config files are not rewritten.
- `main` is unchanged until the pull request is deliberately reviewed and merged.
