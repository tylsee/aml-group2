import ast
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cdccp.config import REPO, load_config


@pytest.fixture
def notebook(tmp_path):
    nb = json.loads((REPO / "notebooks/P1f_features_kaggle.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for source in code:
        ast.parse(source)
    env = {"pd": pd, "torch": torch, "shutil": shutil, "np": np, "load_config": load_config}
    exec(code[0], env)
    env.update(SMOKE=False, CODE_DIR=REPO, INPUT=tmp_path / "input", RUNS_DIR=tmp_path / "runs")
    exec(code[3], env)
    return code, env


def test_p1f_recipes_isolate_learning_rate_and_freezing():
    recipes = [load_config(REPO / "configs" / f"{name}.yaml")
               for name in ("p1f_ctl", "p1f_low_lr", "p1f_frozen")]
    control, low, frozen = recipes
    assert control["model"]["backbone"] == "convnext_tiny"
    assert control["optim"]["lr_backbone"] == 1e-4 and not control["model"]["freeze_backbone"]
    assert low["optim"]["lr_backbone"] == 3e-5 and frozen["model"]["freeze_backbone"]
    for recipe in recipes:
        assert recipe["epochs"] == 10 and recipe["batch_size"] == 16 and recipe["views"] == 2
        assert recipe["alpha"] == recipe["beta"] == 0
        assert recipe["optim"]["lr_heads"] == 1e-3
        recipe.pop("name")
    low["optim"]["lr_backbone"] = control["optim"]["lr_backbone"]
    frozen["model"]["freeze_backbone"] = False
    assert control == low == frozen


def fake_run(env, config, seed=42, applied=4310, recipe=None, epochs=None):
    folder = env["RUNS_DIR"] / env["run_name"](config, seed)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"epoch": epochs or list(range(1, 11)), "applied": np.arange(1, 11) * 431}).to_csv(folder / "history.csv", index=False)
    torch.save({"applied": applied, "recipe": recipe or env["expected_recipe"](config, seed)}, folder / "last.pt")
    (folder / "val_10.npz").touch()
    (folder / "epoch_10.pt").touch()
    return folder


def test_finished_refuses_wrong_recipe_crash_and_missing_p3_checkpoint(notebook):
    _, env = notebook
    folder = fake_run(env, "p1f_ctl")
    assert env["finished"]("p1f_ctl", 42)
    fake_run(env, "p1f_ctl", applied=3879)
    assert not env["finished"]("p1f_ctl", 42)
    fake_run(env, "p1f_ctl", recipe={"name": "wrong"})
    assert not env["finished"]("p1f_ctl", 42)
    fake_run(env, "p1f_ctl", epochs=[1, 2, 3, 4, 5, 6, 7, 8, 8, 10])
    assert not env["finished"]("p1f_ctl", 42)
    fake_run(env, "p1f_ctl")
    (folder / "epoch_10.pt").unlink()
    assert not env["finished"]("p1f_ctl", 42)


def test_p3_queue_only_scores_completed_runs(notebook):
    code, env = notebook
    env.update(N_GPU=2, sys=__import__("sys"))
    exec(code[5], env)
    fake_run(env, "p1f_ctl")
    fake_run(env, "p1f_low_lr", applied=3879)
    jobs = []
    env["run_queue"] = lambda queue: jobs.extend(queue) or []
    exec(code[7], env)
    assert [j[0] for j in jobs] == ["p3_p1f_ctl_seed42"]
    args = jobs[0][1]
    assert args[1] == "scripts/p3_eval.py" and args[args.index("--epoch") + 1] == 10
    jobs.clear()
    env["SMOKE"] = True
    exec(code[7], env)
    assert jobs == []


def test_restore_skips_smoke_and_incompatible_recipes(notebook):
    code, env = notebook
    source = {**env, "RUNS_DIR": env["INPUT"] / "v1" / "p1f_runs"}
    fake_run(source, "p1f_ctl", applied=3879)
    fake_run({**source, "RUNS_DIR": env["INPUT"] / "v2" / "p1f_runs"}, "p1f_ctl", recipe={"name": "wrong"})
    fake_run({**source, "RUNS_DIR": env["INPUT"] / "v3" / "p1f_smoke"}, "p1f_ctl")
    exec(code[3], env)
    assert env["applied"](env["RUNS_DIR"] / "p1f_ctl_seed42") == 3879
    assert len(env["restored"]) == 1


def test_smoke_uses_separate_outputs_and_two_epoch_commands(notebook):
    code, env = notebook
    settings = {}
    exec(code[0], settings)
    assert settings["SMOKE"] and settings["RUNS_DIR"].name == "p1f_smoke"
    assert settings["CONFIGS"] == ["p1f_ctl", "p1f_low_lr", "p1f_frozen"]
    env.update(sys=__import__("sys"), SMOKE=True, N_GPU=2)
    exec(code[5], env)
    _, args = env["job"]("p1f_frozen", 43)
    assert args[-2:] == ["--max-epochs", "2"] and "seed=43" in args
    assert "os.chdir('/kaggle/working')" in code[1]
    assert code[1].index("os.chdir('/kaggle/working')") < code[1].index("shutil.rmtree(CODE_DIR)")


def test_smoke_rejects_zero_trainable_heads(notebook):
    code, env = notebook
    env.update(sys=__import__("sys"), SMOKE=True, N_GPU=2)
    exec(code[5], env)
    env["run_queue"] = lambda jobs: []
    log = env["RUNS_DIR"] / "p1f_frozen_seed42.log"
    good = "trainable parameters: backbone=0 heads=489190\n"
    for text in ("trainable parameters: backbone=0 heads=0\n",            # heads do not train
                 "trainable parameters: backbone=27820128 heads=489190\n",  # backbone is not frozen
                 good + "trainable parameters: backbone=27820128 heads=489190\n",  # newest line of an appended log wins
                 "no count line\n"):
        log.write_text(text)
        with pytest.raises(AssertionError, match="rozen"):
            exec(code[6], env)
    log.write_text(good)
    exec(code[6], env)


def _picker(code):
    """The code-folder picker functions from the notebook's GPU and code cell, without the GPU and copy steps."""
    import hashlib
    import zipfile
    tree = ast.parse(code[1])
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("is_p1f_code", "code_digest", "pick_code")]
    assert len(funcs) == 3
    env = {"hashlib": hashlib, "zipfile": zipfile, "shutil": shutil, "Path": Path}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "picker", "exec"), env)
    return env["pick_code"]


def _fake_code(root, p1f=True, extra=""):
    for rel, text in (("cdccp/__init__.py", ""), ("cdccp/models.py", "freeze_backbone" if p1f else "old"),
                      ("cdccp/evaluate.py", "def pair_breakdown"), ("scripts/p3_eval.py", '"unseen disease"' if p1f else "old"),
                      ("configs/p1f_ctl.yaml", "epochs: 10" + extra)):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def test_code_cell_picks_the_p1f_folder_and_skips_old_code(notebook, tmp_path):
    code, _ = notebook
    pick = _picker(code)
    inp, unzip = tmp_path / "in", tmp_path / "unz"
    _fake_code(inp / "cdccp-p1" / "a_cdc-cp_code_p2e" / "cdc-cp", p1f=False)    # old version, sorts first
    new = _fake_code(inp / "cdccp-p1" / "cdc-cp_code_p1f" / "cdc-cp")
    assert pick(inp, unzip) == new
    # a second P1f copy with the same training code (e.g. only tests differ) is fine
    _fake_code(inp / "other" / "cdc-cp")
    (inp / "other" / "cdc-cp" / "tests").mkdir()
    (inp / "other" / "cdc-cp" / "tests" / "t.py").write_text("x")
    assert pick(inp, unzip) in (new, inp / "other" / "cdc-cp")
    # two P1f copies whose training code differs in any of cdccp/, scripts/ or configs/ are refused
    other = inp / "other" / "cdc-cp"
    for rel in ("cdccp/models.py", "scripts/p3_eval.py", "configs/p1f_ctl.yaml"):
        _fake_code(other)
        (other / rel).write_text((other / rel).read_text(encoding="utf-8") + "\n# changed", encoding="utf-8")
        with pytest.raises(AssertionError, match="different P1f code folders"):
            pick(inp, unzip)


@pytest.mark.parametrize("missing", ["cdccp/models.py", "scripts/p3_eval.py", "cdccp/evaluate.py"])
def test_code_cell_skips_code_that_lacks_any_one_p1f_feature(notebook, tmp_path, missing):
    code, _ = notebook
    pick = _picker(code)
    inp, unzip = tmp_path / "in", tmp_path / "unz"
    (_fake_code(inp / "ds" / "cdc-cp") / missing).write_text("old", encoding="utf-8")
    with pytest.raises(AssertionError, match="No P1f code"):
        pick(inp, unzip)


def test_code_cell_refuses_only_old_code_and_unpacks_a_p1f_zip(notebook, tmp_path):
    import zipfile
    code, _ = notebook
    pick = _picker(code)
    inp, unzip = tmp_path / "in", tmp_path / "unz"
    _fake_code(inp / "ds" / "cdc-cp", p1f=False)
    with pytest.raises(AssertionError, match="No P1f code"):
        pick(inp, unzip)
    src = _fake_code(tmp_path / "src" / "cdc-cp")
    with zipfile.ZipFile(inp / "ds" / "cdc-cp_code_p1f (1).zip", "w") as z:   # Kaggle may rename a re-uploaded file
        for p in src.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(src.parent))
    got = pick(inp, unzip)
    assert got.is_relative_to(unzip) and (got / "cdccp" / "models.py").read_text() == "freeze_backbone"
