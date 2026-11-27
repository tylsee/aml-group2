# V14B — Balanced Softmax from V12b

Source: [results_CDC_CP_V14B_BalancedSoftmax_from_V12b_v1](https://drive.google.com/drive/folders/126Y1b7xQP8cEySqJe5tApq2thAwk3WhR).

Included: run identity, model manifest, matched development comparison, per-composition metrics, disease ablation CSV, pipeline comparison, training history and curve plot.

Development validation (695 seen-like, 400 unseen-like images):
- V12b pair head: seen 50.94%, unseen 7.50%, composition Top-1 HM **13.07%**.
- V14B Balanced Softmax + pair head: seen 56.98%, unseen 7.50%, HM **13.26%**.
- V14B Balanced Softmax + prototypes: seen 52.52%, unseen 7.00%, HM **12.35%**.

These are **development comparisons**, not confirmed final official-test scores.
