# Norman — Proposed ConvNeXt + CLIP plant-disease identification

Imported from branch [`new-approach`](https://github.com/tylsee/aml-group2/tree/new-approach), preserving every tracked source file byte-for-byte. The source branch remains intact.

- [experiments/](experiments/) — initial notebook, complete `cdc-cp` Python source package, mapping and split files.
- [final/](final/) — finished `New_approach_done.ipynb`, original architecture and setup instructions, all saved result directories and report tables.
- [SOURCE_BRANCH_README.md](SOURCE_BRANCH_README.md) — original root README from the source branch, retained verbatim.

## Model variants

| Phase | Crop | Disease |
| --- | --- | --- |
| P1 | ConvNeXt-Tiny | ConvNeXt-Tiny |
| P2 | ConvNeXt-Tiny | CLIP ViT-B/16 |
| P3 | ConvNeXt-Tiny + CLIP ViT-B/16 ensemble | CLIP ViT-B/16 + ConvNeXt-Small ensemble |

The original report contains raw and calibration-adjusted results. [Read the saved report tables](final/reporting/REPORT_TABLES.md). Training checkpoints excluded from Git must be provided separately if needed for reproducible inference.

**Provenance:** Imported from `new-approach` commit `96dbc8a935ed70139ee2629eeea72172a0ddb0b9`. Notebook code, results and plots are unchanged. See [import manifest](../../NORMAN_IMPORT_MANIFEST.md).
