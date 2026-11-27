# Baseline2 Stage 1 handoff for Emily

This folder contains individual files for the ResNet-50 Exp09–18 development experiments. No ZIP is delivered. Original notebooks, training logs, tables, plots and validation metrics are copied without execution or modification. Begin with Stage1_Experimental_Findings.md, Experiment_Comparison.csv and Notebook_Output_Audit.csv.

## Folder guide

- 01_Core_Baseline_ResNet50: original Exp09 notebook, audit notebooks, outputs and configurations.
- 02_Single_Stage_Optimisation: Exp10, calibration Exp11/12, Exp13/14 and Exp15/16 notebooks and results.
- 03_Classifier_Retraining_cRT: Exp17 seed 42 and separate Exp18 seed 123/2026 results. The seed-123 executed notebook was recovered byte-for-byte from Downloads/Baseline2_Stage1_Handoff_Exp09-18.zip. The project working notebook contains seed-2026 outputs.
- 04_Source_Code_and_Configs: shared model/evaluation source, original requirements and exact splits. Per-run config.json files contain configuration, environment, class mappings, composition mapping and candidate pairs.
- 05_Checkpoints: the only copies of the five requested checkpoints, each in its original experiment-named directory. Follow CHECKPOINT_REFERENCES.md in each experiment group.

## Notebook cautions

Output presence is audited per file in Notebook_Output_Audit.csv; it does not imply every cell executed. The original Exp10 runall notebook is partial and contains a CUDA out-of-memory error. The cpu_smoke-named version actually contains completed CUDA training logs and final results. baseline2_exp18_multiseed.ipynb and the Downloads audit_fixed notebook have no saved outputs and are templates, not completed executions. Exp15/16 share an executed working notebook; their saved outputs/configurations are distinct. Seed-123 and seed-2026 executed Exp18 records are both available. Do not use Run All merely to inspect results; several originals can start training.

## Reproduction and limitations

Download individual files, then reconstruct a project root with notebooks at the root, shared source files at the root, splits_v2/ at the root, and each group's outputs/<experiment>/ at root outputs/<experiment>/. Copy 05_Checkpoints/<experiment>/best_development.pt into the matching outputs/<experiment>/ directory. Restore exp09_audit outputs similarly. Original paths and filenames inside notebooks/configurations are unchanged, so the Drive display layout itself is not directly runnable.

The ResNet notebooks expect a sibling DPD_Small_dataset image directory; inspect IMAGE_ROOT/DATASET_ROOT and PROJECT_DIR in the original notebook for a different machine. Raw images are intentionally excluded and Emily must obtain authorised access separately. The original full training CSV may be needed for a fresh data/split rebuild; exact Stage 1 split CSVs are included, so no split regeneration is needed for these runs. No Official Test CSVs are included.

The original requirements.txt is preserved but does not match the recorded run environment: Python 3.12.10, torch 2.11.0+cu128, torchvision 0.26.0+cu128, numpy 2.5.4, pandas 3.1.1, CUDA runtime 12.8 (RTX 4050 Laptop GPU). It pins torch 2.6.0/torchvision 0.21.0 and restricts numpy/pandas below recorded versions. Use per-run config.json as historical evidence; a complete matching dependency lock is unavailable, including exact Pillow, matplotlib and Jupyter versions. ImageNet1K V2 pretrained weights may be required for a fresh backbone training run.

Notebook imports use model_resnet50.load_resnet50_checkpoint; evaluate.py's standalone command imports model.load_checkpoint (the EfficientNet helper). Shared model.py is preserved for dependency completeness, but do not assume the standalone evaluate.py CLI loads ResNet checkpoints. Follow the ResNet notebook evaluation route; no source code compatibility changes were made.

Only the five requested checkpoints are included to avoid redundant large uploads. Exp10/13/14/16 checkpoints exist locally but are intentionally omitted; their metrics and histories remain included. Exp11 raw logits caches are omitted as redundant artifacts. Historical Exp08–14 PDF from Downloads is reference material and does not replace current saved metrics.

Read File_Manifest.csv for source provenance, exact sizes and hashes. Restricted sharing is retained; Emily has not been added. No originals were overwritten, no checkpoints were changed, and no training/evaluation was run.
