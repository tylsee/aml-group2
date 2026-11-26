import ast
import json

import pandas as pd
import pytest
import torch

from cdccp.config import REPO, load_config
from cdccp.data import COLUMNS


@pytest.fixture
def nb(tmp_path):
    notebook = json.loads((REPO / "notebooks/P_final_kaggle.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for source in code:
        ast.parse(source)
    env = {}
    exec(code[0], env)  # settings
    env.update(CODE_DIR=REPO)
    exec(code[2], env)  # helpers only, no side effects
    return code, env


def fake_parent(root, config, seed, epoch=10, meta_name=None):
    folder = root / "p1f_runs" / f"{config}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"config": {"name": meta_name or config}, "seed": seed, "epoch": epoch}
    torch.save({"model": {}, "meta": meta}, folder / f"epoch_{epoch:02d}.pt")
    (folder / f"p3_board_ep{epoch:02d}.csv").write_text("scorer,gamma keep\nhead,1.0\n")
    return folder


def test_one_run_trains_locks_then_tests(nb):
    code, env = nb
    assert env["MODE"] == "smoke" and env["SEEDS"] == (42,)
    whole = "\n".join(code[3:])
    find, gate, lock, copy, test = (whole.index(k) for k in (
        "find_official(INPUT)", "assert ALL_GATES_PASS", "scripts/make_lock.py", "copy_official(OFFICIAL", "--confirm-final-test"))
    assert find < gate < lock < copy < test  # missing test CSVs fail before training; they are copied only after the lock
    assert "if not SMOKE:" in code[8].split("copy_official(OFFICIAL")[0]
    assert "DPD_Small_test" not in "\n".join(code[3:8])


def test_find_parent_checks_run_and_skips_smoke(nb, tmp_path):
    _, env = nb
    folder = fake_parent(tmp_path, "p1f_ctl", 42)
    smoke = tmp_path / "p1f_smoke" / "p1f_ctl_seed42"
    smoke.mkdir(parents=True)
    (smoke / "epoch_10.pt").write_bytes(b"different")
    pt, board = env["find_parent"]("p1f_ctl", 42, tmp_path)
    assert pt == folder / "epoch_10.pt" and board == folder / "p3_board_ep10.csv"
    fake_parent(tmp_path / "other", "p2c_plain", 43, meta_name="p2c_cross")
    with pytest.raises(AssertionError, match="expected"):
        env["find_parent"]("p2c_plain", 43, tmp_path / "other")
    with pytest.raises(AssertionError, match="no p2c_plain_seed44"):
        env["find_parent"]("p2c_plain", 44, tmp_path)


def test_find_parent_refuses_two_different_copies(nb, tmp_path):
    _, env = nb
    fake_parent(tmp_path / "a", "p1f_ctl", 42)
    fake_parent(tmp_path / "b", "p1f_ctl", 42, meta_name="p1f_ctl")
    torch.save({"model": {"w": torch.ones(1)}, "meta": {}}, tmp_path / "b" / "p1f_runs" / "p1f_ctl_seed42" / "epoch_10.pt")
    with pytest.raises(AssertionError, match="different"):
        env["find_parent"]("p1f_ctl", 42, tmp_path)


def test_lock_spec_names_kinds_and_boards(nb, tmp_path):
    _, env = nb
    seeds = (42, 43, 44)
    parents = {(c, s): (tmp_path / f"{c}{s}.pt", tmp_path / f"{c}{s}.csv") for c in ("p1f_ctl", "p2c_plain") for s in seeds}
    spec = env["lock_spec"](parents, ["p1f_ctl", "p2c_plain"], seeds, tmp_path / "lock")
    by_name = {m["name"]: m for m in spec["models"]}
    assert sorted(by_name) == sorted(f"{g}_seed{s}" for g in ("P1", "P2", "P3pooled", "P3balanced") for s in seeds)
    assert by_name["P1_seed43"]["kind"] == "head"
    assert by_name["P1_seed43"]["checkpoint"] == str(tmp_path / "lock" / "p1f_ctl_seed43" / "stage2_final.pt")
    assert by_name["P3pooled_seed44"]["kind"] == "proto-pooled"
    assert by_name["P3balanced_seed42"]["checkpoint"] == str(tmp_path / "lock" / "p2c_plain_seed42" / "stage2_final.pt")
    assert by_name["P2_seed42"]["boards"] == [str(tmp_path / f"p2c_plain{s}.csv") for s in seeds]


def test_official_csvs_found_by_name_then_copied(nb, tmp_path):
    _, env = nb
    src, dst = tmp_path / "input" / "dpd", tmp_path / "data"
    src.mkdir(parents=True)
    dst.mkdir()
    with pytest.raises(AssertionError, match="Add DPD_Small_test_seen.csv"):
        env["find_official"](tmp_path / "input")
    (src / "DPD_Small_test_seen.csv").write_text("seen")
    (src / "dpd_small_test_unseen.csv").write_text("unseen")
    found = env["find_official"](tmp_path / "input")
    env["copy_official"](found, dst)
    assert (dst / "DPD_Small_test_seen.csv").read_text() == "seen" and (dst / "DPD_Small_test_unseen.csv").read_text() == "unseen"
    (tmp_path / "input" / "copy").mkdir()
    (tmp_path / "input" / "copy" / "DPD_Small_test_seen.csv").write_text("other")
    with pytest.raises(AssertionError, match="expected one DPD_Small_test_seen.csv"):
        env["find_official"](tmp_path / "input")


def test_standins_use_official_format(nb, tmp_path):
    _, env = nb
    env["write_standins"](tmp_path, load_config()["paths"]["splits_dir"])
    seen = pd.read_csv(tmp_path / "DPD_Small_test_seen.csv", header=None, names=COLUMNS)
    val = pd.read_csv(REPO / load_config()["paths"]["splits_dir"] / "val_seen.csv")
    assert len(seen) == len(val) == 695 and list(seen.filename) == list(val.filename)


def test_stage2_job_smoke_is_short(nb, tmp_path):
    _, env = nb
    name, args = env["stage2_job"]("p1f_ctl", 42, tmp_path / "e.pt", tmp_path / "lock", True)
    assert name == "p1f_ctl_seed42" and args[1] == "scripts/stage2.py" and args[3] == tmp_path / "lock" / "p1f_ctl_seed42"
    assert "stage2.epochs=1" in args and "stage2.train_split=train_fit" in args
    _, full = env["stage2_job"]("p1f_ctl", 42, tmp_path / "e.pt", tmp_path / "lock", False)
    assert not any(a.startswith("stage2.") for a in map(str, full))


def test_gate_table_and_summary(nb, tmp_path):
    _, env = nb
    for name, ok in (("a", True), ("b", False)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "stage2_gate.json").write_text(json.dumps({"ok": ok, "reason": "r", "train_loss_by_epoch": [2.0, 1.5]}))
    gates = env["gate_table"](tmp_path, ["a", "b"])
    assert list(gates.gate) == ["PASS", "FAIL"] and list(gates.loss_ep2) == [1.5, 1.5]
    rows = [{"model": f"P1_seed{s}", "catalogue": "locked_raw", "split": "combined", "metric": "H", "value": v}
            for s, v in ((42, 10.0), (43, 20.0), (44, 30.0))]
    pd.DataFrame(rows).to_csv(tmp_path / "m.csv", index=False)
    summary = env["summarise"](tmp_path / "m.csv")
    row = summary.iloc[0]
    assert row.group == "P1" and row["mean"] == 20.0 and row.sd == 10.0 and row.seeds == 3
