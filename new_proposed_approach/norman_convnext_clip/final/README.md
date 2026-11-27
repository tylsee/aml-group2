# Norman — completed P1/P2/P3 notebook and evaluated results

These artifacts are copied **unchanged** from branch `new-approach`, not regenerated.

## Key files

- [New_approach_done.ipynb](New_approach_done.ipynb) — executed notebook.
- [Final_model_colab_steps.txt](Final_model_colab_steps.txt) — original Google Colab setup, rerun and reproducibility guidance.
- [architecture.png](architecture.png) — original architecture figure.
- [final_model_results/](final_model_results/) — original results, per-member stage 1/2 histories, configs and validation artifacts.
- [reporting/REPORT_TABLES.md](reporting/REPORT_TABLES.md) — detailed P1/P2/P3 tables; [official results](final_model_results/results/official_test_results.csv).

The source report identifies P3 as crop ensemble ConvNeXt-Tiny + CLIP ViT-B/16 and disease ensemble CLIP ViT-B/16 + ConvNeXt-Small. It reports **16.47% raw official composition Top-1 harmonic mean on the full 9,625-pair candidate set**. Report both seen and unseen performance and distinguish calibration separately.

**Weights:** No large `.pt` trained-model checkpoints were present as tracked blobs in the source branch. The `.npz` artifacts in `final_model_results` are preserved; these must not be misrepresented as standalone model weights.
