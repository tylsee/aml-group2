"""Review/copy only small saved training summary files to reports, never weights or test labels.

Previews allowlisted configs, aggregate metrics, histories and plots.
Run with --copy to copy missing files to outputs/experiment_runs without overwriting.
"""
from pathlib import Path
import sys, shutil

project = Path(__file__).resolve().parents[1]
source = project / "outputs"
dest = project / "outputs" / "experiment_runs"
allowed_names = {
    "config.json", "training_history.csv", "validation_metrics.json",
    "experiment_summary.csv", "full_data_history.csv", "training_curves.png",
    "accuracy_curves.png", "task_accuracy_curves.png", "loss_curves.png", "development_metrics.json",
}
if not source.exists():
    print("No local outputs folder found; nothing to copy.")
    raise SystemExit(0)
copy = "--copy" in sys.argv
count = 0
for file in sorted(source.rglob("*")):
    if not file.is_file() or file.name not in allowed_names:
        continue
    rel = file.relative_to(source)
    parts = rel.parts
    if not (parts[0].startswith("exp0") or parts[0] == "stage2_development"):
        continue
    target = dest / rel
    print(("COPY" if copy else "PREVIEW"), rel, "->",target.relative_to(project))
    if copy:
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            print("SKIP existing:",target)
        else:
            shutil.copy2(file,target)
    count += 1
print(f"Found {count} safe-to-review summary files. Original outputs unchanged.")
