"""Read-only pre-push checks for Baseline2. Usage: python scripts/audit_before_push.py"""
from pathlib import Path
import json
import re
import shutil
import subprocess

project = Path(__file__).resolve().parents[1]
print("Project:", project)
for file in ("baseline2.ipynb", "model.py", "evaluate.py", "README.md", ".gitignore",
             "outputs/development/exp04/validation_metrics.json",
             "outputs/development/exp06/calibration_summary.json",
             "outputs/development/exp07/tta_summary.json",
             "outputs/official/comparison.json"):
    p = project / file
    print(f"{'OK' if p.is_file() else 'MISSING'}: {file}")
nb = json.loads((project / "baseline2.ipynb").read_text(encoding="utf8"))
source = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")
for name in ("RUN_SMOKE_CHECKS", "RUN_STAGE1", "RUN_DIAGNOSTICS",
             "RUN_STAGE2_DEVELOPMENT", "RUN_STAGE2_FINAL",
             "APPROVE_FULL_DATA_TRAINING", "RUN_FINAL_PREFLIGHT", "RUN_OFFICIAL_TEST"):
    match = re.search(r"^\s*"+name+r"\s*=\s*(True|False)", source, re.M)
    print(f"{name}: {match.group(1) if match else 'NOT FOUND'}")
    if not match or match.group(1) != "False":
        raise SystemExit("STOP: verify run switches before committing")
for path in (project / "outputs").rglob("*.pt"):
    print('LOCAL CHECKPOINT (check Git ignore status):',path.relative_to(project))
git = shutil.which("git")
if git is None:
    print("Git status unavailable because Git is missing from PATH.")
else:
    status = subprocess.run(
        [git, "status", "--short", "--branch"], cwd=project,
        capture_output=True, text=True,
    )
    if status.returncode == 0:
        print("\nGit status:\n" + status.stdout)
    else:
        print("Git status unavailable. " + status.stderr.strip())
print('Audit finished. This script does not stage, delete or push anything.')
