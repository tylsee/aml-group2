import ast
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

from cdccp.config import REPO

GENERATOR = REPO / "scripts" / "make_p6_colab_notebook.py"
P5_NOTEBOOK = Path(r"D:\dev\uni\AML\runs\P5_colab.ipynb")
DUMMY_SHA = "ab" * 32
P5_PINNED_SHA = {"lock": "783b913f02fd42bc7a0a8ef1cab9468a56be78ba51901b75c350588f2032385f",
                 "marker": "af44acc2118239386db0f59f574a8d91e434ea914237a4abc13de938f8f2bc71",
                 "dev_logprobs": "89b2cd043b6920d05ef4add01fff44958cc689ee6e0ed6b52d6cc6dd2d34ed08",
                 "test_logprobs": "b017be11d34d8e804e8db6159882cc156816dc6d217bc36b797e3492e64a2652",
                 "test_results": "a9e67f1d7b4821ceafe7e71a130e47b25098567ac87b73b6496d77917bdff1b9",
                 "crop_b1_model": "427e9d8e7cb0b1dc4cfe002e5bfce9069e4693204d451f32d1233b74eb132df1"}
P5_RESULTS = Path(r"D:\dev\uni\AML\runs\p5_colab_L4\p5_results")
COPIES = {"copy_file", "copy_tree", "copy2", "copyfile", "copytree"}
WRITERS = {"copy_file", "copy_tree", "save_results", "rename", "replace", "write_text", "write_bytes", "unlink", "rmtree", "move",
           "copy2", "copyfile", "copytree", "to_csv", "savez", "savez_compressed", "save", "mkdir", "touch", "symlink_to", "write",
           "remove", "makedirs", "open", "exec", "system", "run", "Popen"}
READERS = {"read_labelled_csv", "read_csv", "read_text", "read_bytes", "open", "load"}


def load_generator():
    spec = importlib.util.spec_from_file_location("make_p6_colab_notebook", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return load_generator()


@pytest.fixture(scope="module")
def cells(gen, tmp_path_factory):
    out = tmp_path_factory.mktemp("nb") / "P6_colab.ipynb"
    gen.build(out, DUMMY_SHA)
    notebook = json.loads(out.read_text(encoding="utf-8"))
    return [(c["cell_type"], "".join(c["source"])) for c in notebook["cells"]]


def code_cells(cells):
    return [(i, src) for i, (kind, src) in enumerate(cells) if kind == "code"]


def find(cells, prefix):
    hits = [i for i, src in code_cells(cells) if src.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


def call_name(node):
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None


def names_in(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def written_names(node):
    """Names a call can write to. A copy only writes to its destination (every argument after the first), never to its source."""
    if call_name(node) in COPIES:
        parts = [*node.args[1:], *(keyword.value for keyword in node.keywords)]
        return set().union(*(names_in(part) for part in parts)) if parts else set()
    return names_in(node)


def p5_write_violations(source):
    """Calls that could write, move or delete something, and mention a P5_ name where they write."""
    return [ast.unparse(node)[:80] for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
            and call_name(node) in WRITERS and any(n.startswith("P5_") for n in written_names(node))]


def test_every_code_cell_parses_and_the_notebook_has_the_expected_shape(cells):
    for _, src in code_cells(cells):
        ast.parse(src)
    assert len(cells) == 23 and cells[0][0] == "markdown"
    assert "@@" not in "\n".join(src for _, src in cells)


def test_settings_cell_is_the_p6_design(cells, monkeypatch, tmp_path):
    src = cells[find(cells, "# 1. Settings")][1]
    monkeypatch.delenv("P6_SMOKE", raising=False)
    env = {}
    exec(src, env)
    assert env["SMOKE"] is False and env["SEED"] == 42
    assert (env["STAGE1_EPOCHS"], env["PATIENCE"], env["STAGE2_EPOCHS"]) == (25, 5, 10)
    assert env["GPU_JOBS"] == {(0, 0): ["disease_small_reg"], (0, 1): ["crop_clip"]}
    assert env["FLIP"] is True and env["PRETRAINED"] is True and env["AMP"] is True and env["MAX_TRAIN_BATCHES"] is None
    assert env["EXPECT_TEST"] == (2000, 1448) and env["EXPECT_IMAGES"] == 11448 and env["TEST_DISJOINT_FROM_TRAIN"] is True
    assert env["DRIVE_RESULTS"] == env["DRIVE_DIR"] / "p6_results" and env["P5_RESULTS"] == env["DRIVE_DIR"] / "p5_results"
    assert env["CODE_ZIP"] == "cdc-cp_code_p6.zip"
    assert env["P5_EXPECTED_SHA"] == P5_PINNED_SHA
    monkeypatch.setenv("P6_SMOKE", str(tmp_path))
    fake_pins = {name: "cd" * 32 for name in P5_PINNED_SHA}
    (tmp_path / "p5_expected_sha.json").write_text(json.dumps(fake_pins))
    smoke = {}
    exec(src, smoke)
    assert smoke["P5_EXPECTED_SHA"] == fake_pins
    assert smoke["SMOKE"] is True and (smoke["STAGE1_EPOCHS"], smoke["STAGE2_EPOCHS"]) == (1, 1)
    assert smoke["MAX_TRAIN_BATCHES"] == 2 and smoke["PRETRAINED"] is False and smoke["EXPECT_TEST"] == (695, 400)
    for name in ("INPUT", "OUT", "CODE_DIR", "DRIVE_DIR", "DRIVE_RESULTS", "P5_RESULTS"):
        assert tmp_path in smoke[name].parents, name
    assert smoke["DRIVE_RESULTS"].name == "p6_results" and smoke["P5_RESULTS"].name == "p5_results"


def test_the_pinned_crop_b1_hash_is_the_p5_lock_hash_and_the_hash_of_the_real_p5_model():
    lock = json.loads((P5_RESULTS / "lock.json").read_text())
    assert lock["files"]["crop_b1"]["sha256"] == P5_PINNED_SHA["crop_b1_model"]
    assert lock["files"]["crop_b1"]["path"].endswith("/stage2/crop_b1/best_train_acc.pt")
    model = P5_RESULTS / "stage2" / "crop_b1" / "best_train_acc.pt"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == P5_PINNED_SHA["crop_b1_model"]


def test_the_smoke_variable_is_the_only_environment_read(cells):
    uses = [i for i, src in code_cells(cells) if "environ" in src or "getenv" in src]
    assert uses == [find(cells, "# 1. Settings")]
    assert "".join(src for _, src in cells).count("P6_SMOKE") >= 1
    settings = cells[uses[0]][1]
    assert settings.count("os.environ") == 1 and "os.environ.get('P6_SMOKE', '')" in settings


def test_test_csvs_are_read_only_in_the_official_test_cell_after_the_lock(cells):
    lock_cell = find(cells, "# 10. Calibration")
    test_cell = find(cells, "# 11. Official test")
    assert test_cell > lock_cell
    assert find(cells, "# 11b.") > test_cell
    lock_src = cells[lock_cell][1]
    assert lock_src.count("OFFICIAL") == 1
    assert "**{f'test_csv_{split}': str(path) for split, path in OFFICIAL.items()}" in lock_src
    assert "**{f'drive_test_csv_{split}': str(path) for split, path in DRIVE_TEST_CSVS.items()}" in lock_src
    for index, src in code_cells(cells):
        tree = ast.parse(src)
        parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and call_name(node) in READERS and "OFFICIAL" in names_in(node):
                assert index == test_cell, f"cell {index} reads an official test CSV"
            if isinstance(node, ast.Name) and node.id == "OFFICIAL" and isinstance(node.ctx, ast.Load) and index != test_cell:
                ancestors, up = [], node
                while up in parent:
                    up = parent[up]
                    ancestors.append(up)
                assigns_item = isinstance(parent[node], ast.Subscript) and isinstance(parent[node].ctx, ast.Store)
                prints = any(isinstance(a, ast.Call) and call_name(a) == "print" for a in ancestors)
                hashed_in_lock = index == lock_cell and any(isinstance(a, ast.Call) and call_name(a) == "write_lock" for a in ancestors)
                assert assigns_item or prints or hashed_in_lock, f"cell {index} uses OFFICIAL outside print, item assignment and the lock's file list"
    locate = cells[find(cells, "# 3. Find the 8,000")][1]
    assert "read_labelled_csv(" in locate and "OFFICIAL[split] = hits[0]" in locate
    assert not any(n.value.id == "OFFICIAL" for n in ast.walk(ast.parse(locate)) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                   and isinstance(n.ctx, ast.Load))
    assert "st.read_labelled_csv(path) for split, path in OFFICIAL.items()" in cells[test_cell][1]


def test_official_test_cell_order_marker_and_p5_check_come_first(cells):
    src = cells[find(cells, "# 11. Official test")][1]
    order = ["LOCK_INFO = st.verify_lock(OUT / 'lock.json')['info']", "assert not STARTED.exists()", "p6*/official_test_started.json",
             "p6_results*/official_test_started.json",
             "pm.read_logprobs(P5_FILES['test_logprobs'], pm.P5_ARMS.values(), {'seen': EXPECT_TEST[0], 'unseen': EXPECT_TEST[1]})",
             "P5_SAVED_TABLE = pm.check_results_layout(P5_FILES['test_results'])",
             "ALLOWED_333 = pm.pair_mask(CAT_FINAL_PATH, CROP_IDS, DISEASE_IDS)",
             "STARTED.write_text(", "copy_file(STARTED, DRIVE_RESULTS / STARTED.name",
             "st.read_labelled_csv(path) for split, path in OFFICIAL.items()", "pm.check_test_frames(",
             "row_model = st.load_trained(LOCAL_CROP_B1, DEVICE)", "pm.check_rows_match_saved(ROW_CHECK_NOW",
             "pm.assert_matches_table(P5_TABLE, P5_SAVED_TABLE)", "LOGP_NEW = {}", "st.load_trained(path, DEVICE)", "pm.p6_parts(P5_TEST, LOGP_NEW",
             "results.to_csv(RESULTS / 'official_test_results.csv'"]
    positions = [src.index(item) for item in order]
    assert positions == sorted(positions), dict(zip(order, positions))
    assert src.count("st.predict(") == 3 and "flip=FLIP)" in src and src.index("flip=FLIP)") < src.index("amp=AMP, workers=WORKERS)\n")
    assert "p5_results*" not in src and "p5_results*" not in "".join(s for _, s in cells)
    assert "official_test_started.json" in src and "STARTED = OUT / 'official_test_started.json'" in src


def test_p5_test_files_are_touched_only_in_cells_after_the_lock_and_the_lock_cell(cells):
    lock_cell, test_cell = find(cells, "# 10. Calibration"), find(cells, "# 11. Official test")
    for index, src in code_cells(cells):
        if "P5_FILES['test_" in src or "P5_TEST" in src:
            assert index >= test_cell, f"cell {index} uses P5 test files before the official test cell"
    assert "p5_{name}" in cells[lock_cell][1] and "P5_FILES.items()" in cells[lock_cell][1]


def test_p5_files_are_only_read(cells):
    for index, src in code_cells(cells):
        assert p5_write_violations(src) == [], f"cell {index}"
    assigned = {}
    for index, src in code_cells(cells):
        for node in ast.walk(ast.parse(src)):
            assert not (isinstance(node, ast.AugAssign) and any(n.startswith("P5_") for n in names_in(node.target))), index
            if isinstance(node, ast.Name) and node.id.startswith("P5_") and isinstance(node.ctx, ast.Store):
                assigned.setdefault(node.id, set()).add(index)
    assert assigned and all(len(where) == 1 for where in assigned.values()), assigned
    whole = "\n".join(src for _, src in code_cells(cells))
    assert whole.count("'p5_results'") == 1


def test_the_p5_write_checker_catches_real_violations():
    bad = ["save_results(OUT, P5_RESULTS)", "copy_file(a, P5_FILES['lock'])", "P5_RESULTS.rename(x)", "P5_FILES['lock'].write_text('x')",
           "shutil.rmtree(P5_RESULTS)", "shutil.copy2(a, P5_RESULTS / 'x')", "np.savez_compressed(P5_RESULTS / 'a.npz')",
           "table.to_csv(P5_RESULTS / 'a.csv')", "open(P5_FILES['lock'], 'w')", "P5_RESULTS.with_name('x').mkdir()"]
    for source in bad:
        assert p5_write_violations(source), source
    assert p5_write_violations("x = pm.read_logprobs(P5_FILES['dev_logprobs'], arms, rows)\nprint(P5_RESULTS)") == []
    assert p5_write_violations("shutil.copy2(P5_FILES['crop_b1_model'], LOCAL_CROP_B1)") == []
    assert p5_write_violations("shutil.copy2(P5_FILES['crop_b1_model'], P5_RESULTS / 'x')")
    assert p5_write_violations("shutil.copy2(src=P5_FILES['crop_b1_model'], dst=P5_RESULTS / 'x')")
    assert p5_write_violations("shutil.move(P5_FILES['crop_b1_model'], LOCAL_CROP_B1)")


def test_results_go_only_to_p6_results(cells):
    for index, src in code_cells(cells):
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call) and call_name(node) in ("save_results", "copy_file", "copy_tree"):
                targets = [ast.unparse(a) for a in node.args[1:2]]
                assert all(t.startswith("DRIVE_RESULTS") or t == "INPUT / 'DPD-Small' / 'DPD_Small_dataset'" for t in targets), (index, targets)
    assert "DRIVE_RESULTS = DRIVE_DIR / 'p6_results'" in cells[find(cells, "# 1. Settings")][1]


def test_stage_commands_and_jobs(cells):
    stage1 = cells[find(cells, "# 6. Stage 1")][1]
    assert "'--epochs', str(STAGE1_EPOCHS), '--patience', str(PATIENCE)" in stage1 and "run_on_gpus(GPU_JOBS" in stage1
    stage2 = cells[find(cells, "# 9. Stage 2")][1]
    assert "jobs = {(0, 0): [FINAL_ARMS[0]], (0, 1): [FINAL_ARMS[1]]}" in stage2 and "FINAL_ARMS = [pm.NEW_ARMS['crop'], pm.NEW_ARMS['disease']]" in stage2
    dev = cells[find(cells, "# 8. Flip dev outputs")][1]
    assert "STAGE1 / arm / 'best.pt'" in dev and "flip=FLIP" in dev
    assert cells.index(("code", stage2)) > cells.index(("code", dev)) > cells.index(("code", stage1))


def test_stage_2_command_uses_the_stage_2_epochs_and_the_full_8000_image_train_csv(cells):
    stage2 = cell_src(cells, "# 9. Stage 2")
    assert ("    cmd = [sys.executable, 'scripts/train_job.py', 'stage2', '--arm', arm, '--stage1', str(STAGE1), '--train-csv', str(train_csv),\n"
            "           '--images', str(IMAGE_DIR), '--out', str(STAGE2), '--epochs', str(STAGE2_EPOCHS), '--workers', str(WORKERS), "
            "'--device', DEVICE_NAME]\n") in stage2
    assert stage2.count("'--epochs'") == 1 and stage2.count("--train-csv") == 1
    assert "STAGE1_EPOCHS" not in stage2 and "PATIENCE" not in stage2 and "splits" not in stage2.lower()
    stage1 = cell_src(cells, "# 6. Stage 1")
    assert "'--epochs', str(STAGE1_EPOCHS)" in stage1 and "STAGE2_EPOCHS" not in stage1 and "train_csv" not in stage1
    locate = cell_src(cells, "# 3. Find the 8,000")
    assert "train_csv = next(p for p in INPUT.rglob('*') if p.name.lower() == 'dpd_small_train.csv')\n" in locate
    assert "startswith('9f1b729a'), 'unexpected DPD_Small_train.csv'" in locate and "assert len(train_all) == 8000\n" in locate
    assigned = [index for index, src in code_cells(cells) for node in ast.walk(ast.parse(src))
                if isinstance(node, ast.Name) and node.id == "train_csv" and isinstance(node.ctx, ast.Store)]
    assert assigned == [find(cells, "# 3. Find the 8,000")]


def test_dev_calibration_uses_catalogue_dev_and_the_test_uses_catalogue_final(cells):
    src = cell_src(cells, "# 10. Calibration")
    assert "CAT_DEV_PATH = CODE_DIR / 'data' / 'mappings' / 'catalogue_dev.json'\n" in src
    assert "CAT_FINAL_PATH = CODE_DIR / 'data' / 'mappings' / 'catalogue_final.json'\n" in src
    assert "CAT_DEV = load_catalogue(CAT_DEV_PATH)\n" in src
    calls = [line for line in src.splitlines() if "dev_calibration(" in line or "dev_unseen_numbers(" in line]
    assert len(calls) == 4, calls
    for line in calls:
        assert "CAT_DEV" in line and "CAT_FINAL" not in line, line
    assert "'catalogue': str(CAT_FINAL_PATH), 'catalogue_dev': str(CAT_DEV_PATH)," in src
    final_uses = [index for index, text in code_cells(cells) if "CAT_FINAL_PATH" in text]
    assert final_uses == [find(cells, prefix) for prefix in ("# 10.", "# 11. Official", "# 11b.", "# 14.")], final_uses
    assert "CAT_DEV" not in cell_src(cells, "# 11. Official test") and "CAT_DEV" not in cell_src(cells, "# 11b.")


def test_lock_records_p6_settings_gammas_code_and_p5_hashes(cells):
    src = cells[find(cells, "# 10. Calibration")][1]
    for item in ("'code_sha256': CODE_SHA", "'weights': list(pm.WEIGHTS)", "'gamma': GAMMA", "'gamma_hm': GAMMA_HM", "'r_hm': R_HM",
                 "'p5_sha256': P5_SHA", "'patience': PATIENCE", "'stage1_epochs': STAGE1_EPOCHS", "'stage2_epochs': STAGE2_EPOCHS",
                 "'seed': SEED", "'flip': FLIP", "pm.dev_unseen_numbers(", "p5_{name}", "OUT / 'lock.json'"):
        assert item in src, item
    assert src.index("st.write_lock(") > src.index("pm.dev_calibration(*DEV_PARTS")


def test_cell_two_fingerprint_matches_the_generator_helper(gen, cells, tmp_path):
    src = cells[find(cells, "# 2. Check the GPU")][1]
    assert f"EXPECTED_CODE_SHA = '{DUMMY_SHA}'" in src and "CODE_ZIP" in src
    root = tmp_path / "cdc-cp"
    for rel, data in (("cdccp/gpu_runner.py", b"x = 1\n"), ("cdccp/a.py", b"a\n"), ("data/z.csv", b"1,2\n"), ("scripts/train_job.py", b"j\n")):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    digest = hashlib.sha256()
    for rel in sorted(["cdccp/gpu_runner.py", "cdccp/a.py", "data/z.csv", "scripts/train_job.py"]):
        digest.update(rel.encode() + b"\0" + (root / rel).read_bytes())
    assert gen.code_sha_of_dir(root) == digest.hexdigest()
    (root / "cdccp" / "__pycache__").mkdir()
    (root / "cdccp" / "__pycache__" / "a.pyc").write_bytes(b"junk")
    assert gen.code_sha_of_dir(root) == digest.hexdigest()
    archive = tmp_path / "code.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for path in root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                z.write(path, Path("cdc-cp") / path.relative_to(root))
    assert gen.code_sha_of_zip(archive) == digest.hexdigest()


def test_build_refuses_a_bad_fingerprint(gen, tmp_path):
    for bad in ("", "xyz", "AB" * 32, "ab" * 31):
        with pytest.raises(ValueError):
            gen.build(tmp_path / "n.ipynb", bad)
    assert not (tmp_path / "n.ipynb").exists()


def cell_src(cells, prefix):
    return cells[find(cells, prefix)][1]


def lines_of(src):
    return src.splitlines()


def test_cell_1b_pins_the_five_p5_files_and_refuses_a_local_marker(cells):
    src = cell_src(cells, "# 1b.")
    keys = [k.value for node in ast.walk(ast.parse(src)) if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "P5_FILES" for t in node.targets)
            for k in node.value.keys]
    assert keys == ["lock", "marker", "dev_logprobs", "test_logprobs", "test_results", "crop_b1_model"] and sorted(keys) == sorted(P5_PINNED_SHA)
    assert "'crop_b1_model': P5_RESULTS / 'stage2' / 'crop_b1' / 'best_train_acc.pt'}" in src
    copy_model = ["LOCAL_CROP_B1 = INPUT / 'p5_crop_b1' / 'best_train_acc.pt'", "LOCAL_CROP_B1.parent.mkdir(parents=True, exist_ok=True)",
                  "shutil.copy2(P5_FILES['crop_b1_model'], LOCAL_CROP_B1)"]
    positions = [src.index(line) for line in copy_model]
    assert positions == sorted(positions) and positions[0] > src.index("print('copied to', INPUT")
    assert "'marker': P5_RESULTS / 'official_test_started.json'," in src
    local = "assert not list(OUT.parent.glob('p6*/official_test_started.json')), ("
    drive = "assert not list(DRIVE_DIR.glob('p6_results*/official_test_started.json')), ("
    assert local in src and drive in src
    assert src.index(local) < src.index("DRIVE_RESULTS.rename(old)") and src.index(local) < src.index("shutil.copy2(")


def test_cell_3_checks_test_csv_times_and_the_pinned_hashes_before_anything_else_uses_p5(cells):
    src = cell_src(cells, "# 3. Find the 8,000")
    steps = ["DRIVE_TEST_CSVS = {split: DRIVE_DIR / f'DPD_Small_test_{split}.csv' for split in ('seen', 'unseen')}",
             "pm.check_test_csvs_predate_marker(DRIVE_TEST_CSVS.values(), P5_FILES['marker'])",
             "P5_SHA = {name: pm.sha256_of(path) for name, path in P5_FILES.items()}",
             "pm.check_pinned_hashes(P5_SHA, P5_EXPECTED_SHA)",
             "assert pm.sha256_of(LOCAL_CROP_B1) == P5_SHA['crop_b1_model']",
             "P5_LOCK = json.loads(P5_FILES['lock'].read_text())", "pm.check_p5_lock(P5_LOCK)",
             "assert P5_LOCK['files']['crop_b1']['sha256'] == P5_SHA['crop_b1_model']",
             "P5_DEV = pm.read_logprobs(P5_FILES['dev_logprobs']"]
    positions = [src.index(step) for step in steps]
    assert positions == sorted(positions), dict(zip(steps, positions))
    assert "evidence, not proof" in src and "has not changed" not in src and "then they are the files P5 read" not in src
    assert find(cells, "# 3. Find the 8,000") < find(cells, "# 10. Calibration") < find(cells, "# 11. Official test")
    settings = cell_src(cells, "# 1. Settings")
    assert settings.count("P5_EXPECTED_SHA") == 2 and "P5_EXPECTED_SHA = json.loads((root / 'p5_expected_sha.json').read_text())" in settings


def test_cell_4_measures_the_amp_rerun_noise_of_the_p5_crop_model_before_the_marker(cells):
    src = cell_src(cells, "# 4. Print the settings")
    steps = ["noise_model = st.load_trained(LOCAL_CROP_B1, DEVICE)",
             "NOISE_NOW = st.predict(noise_model, noise_rows, IMAGE_DIR, CROP_IDS, DISEASE_IDS, DEVICE, amp=AMP, workers=WORKERS, flip=FLIP)",
             "NOISE_FULL = st.predict(noise_model, noise_rows, IMAGE_DIR, CROP_IDS, DISEASE_IDS, DEVICE, amp=False, workers=WORKERS, flip=FLIP)",
             "NOISE_LIMIT = pm.ROW_MAX_ABS_DIFF / 4",
             "NOISE = pm.check_rows_match_saved({'dev': NOISE_NOW}, {'dev': NOISE_FULL}, tolerance=NOISE_LIMIT)['dev']",
             "raise RuntimeError(", ") from err"]
    positions = [src.index(step) for step in steps]
    assert positions == sorted(positions), dict(zip(steps, positions))
    assert "noise_rows = val_seen.head(256)" in src and "official_test_started" not in src
    assert find(cells, "# 3. Find the 8,000") < find(cells, "# 4. Print the settings") < find(cells, "# 11. Official test")


def test_dev_outputs_wiring_of_cell_8(cells):
    src = cell_src(cells, "# 8. Flip dev outputs")
    assert "model = st.load_trained(STAGE1 / arm / 'best.pt', DEVICE)" in src
    assert ("NEW_DEV[arm, split] = st.predict(model, df, IMAGE_DIR, CROP_IDS, DISEASE_IDS, DEVICE, amp=AMP, workers=WORKERS, flip=FLIP)\n"
            in src)
    assert "for split, df in (('seen', val_seen), ('unseen', val_unseen)):" in src
    assert "np.savez_compressed(RESULTS / 'dev_flip_logprobs_p6_new.npz', **{f'{a}_{s}': v for (a, s), v in NEW_DEV.items()})" in src


def test_calibration_wiring_of_cell_10(cells):
    src = cell_src(cells, "# 10. Calibration")
    for line in ("DEV_PARTS = pm.p6_parts(P5_DEV, NEW_DEV, (val_seen, val_unseen))",
                 "DEV_CAL = pm.dev_calibration(*DEV_PARTS, CAT_DEV)",
                 "GAMMA, GAMMA_HM, R_HM = DEV_CAL['gamma_keep'], DEV_CAL['gamma_hm'], DEV_CAL['r_hm']",
                 "P5_DEV_PARTS = tuple(pg.parts_from_logprobs(*pm.member_heads(P5_DEV, pm.P5_ARMS)[s], df) for s, df in zip(pm.SPLITS, (val_seen, val_unseen)))",
                 "P5_DEV_CAL = pm.dev_calibration(*P5_DEV_PARTS, CAT_DEV)",
                 "FINAL = {arm: STAGE2 / arm / 'best_train_acc.pt' for arm in FINAL_ARMS}",
                 "LOCK = st.write_lock({**{arm: str(p) for arm, p in FINAL.items()}, 'train_csv': str(train_csv),"):
        assert line in lines_of(src) or line in src, line
    assert "pm.dev_unseen_numbers(P5_DEV_PARTS[1], CAT_DEV, P5_DEV_CAL, 'P5 (crop_b1 + disease_clip)')" in src
    assert "pm.dev_unseen_numbers(DEV_PARTS[1], CAT_DEV, DEV_CAL, 'P6 (four models)')" in src
    assert "**{f'p5_{name}': str(path) for name, path in P5_FILES.items()}, 'p5_crop_b1_model_local': str(LOCAL_CROP_B1),\n" in src
    same_csv = ("for split in DRIVE_TEST_CSVS:\n"
                "    assert LOCK['files'][f'test_csv_{split}']['sha256'] == LOCK['files'][f'drive_test_csv_{split}']['sha256'], (\n")
    assert same_csv in src and src.index(same_csv) > src.index("st.write_lock(") and src.index(same_csv) < src.index("save_results(")
    assert "'flip': FLIP, 'gamma': GAMMA,\n" in src and "'gamma_hm': GAMMA_HM, 'r_hm': R_HM,\n" in src
    assert "'gamma': GAMMA_HM" not in src and "'gamma_hm': GAMMA," not in src
    assert "OUT / 'lock.json')" in src and "'p5_sha256': P5_SHA}" in src


def test_official_test_wiring_of_cell_11(cells):
    src = cell_src(cells, "# 11. Official test")
    flip = "LOGP_NEW[arm, split] = st.predict(model, df, IMAGE_DIR, CROP_IDS, DISEASE_IDS, DEVICE, amp=AMP, workers=WORKERS, flip=FLIP)\n"
    noflip = "LOGP_NEW_NOFLIP[arm, split] = st.predict(model, df, IMAGE_DIR, CROP_IDS, DISEASE_IDS, DEVICE, amp=AMP, workers=WORKERS)\n"
    assert src.count(flip) == 1 and src.count(noflip) == 1 and src.count("st.predict(") == 3
    assert src.count("LOGP_NEW[") == 1 and src.count("LOGP_NEW_NOFLIP[") == 1
    row_predict = ("for split, df in test.items():\n"
                   "    ROW_CHECK_NOW[split] = st.predict(row_model, df, IMAGE_DIR, CROP_IDS, DISEASE_IDS, DEVICE, amp=AMP, workers=WORKERS, flip=FLIP)\n")
    assert src.count(row_predict) == 1
    assert "ROW_CHECK = pm.check_rows_match_saved(ROW_CHECK_NOW, {split: P5_TEST[pm.P5_ARMS['crop'], split] for split in pm.SPLITS})\n" in src
    assert "(RESULTS / 'p5_row_check.json').write_text(json.dumps(ROW_CHECK, indent=1))\n" in src
    assert "row_model = st.load_trained(LOCAL_CROP_B1, DEVICE)\nROW_CHECK_NOW = {}\n" in src
    before, after = src.split("STARTED.write_text(", 1)
    for name in ("P5_FILES", "DRIVE_DIR", "P5_RESULTS", "read_csv", "verify_lock"):
        assert name not in after, f"{name} is used after the marker: no Drive read of a P5 file or the lock after the marker"
    assert "verify_lock" in before and "P5_SAVED_TABLE = pm.check_results_layout(" in before
    assert "for arm, path in FINAL.items():\n    model = st.load_trained(path, DEVICE)\n" in src
    assert "TEST_PARTS = pm.p6_parts(P5_TEST, LOGP_NEW, (test['seen'], test['unseen']))\n" in src
    assert "HEADS = pm.parts_heads(TEST_PARTS)\n" in src
    assert "METRICS, results = pm.official_table(HEADS, test, CROP_IDS, DISEASE_IDS, ALLOWED_333)\n" in src
    assert "_, P5_TABLE = pm.official_table(pm.member_heads(P5_TEST, pm.P5_ARMS), test, CROP_IDS, DISEASE_IDS, ALLOWED_333)\n" in src
    assert "np.savez_compressed(RESULTS / 'official_test_logprobs_p6_new.npz', **{f'{a}_{s}': v for (a, s), v in LOGP_NEW.items()})\n" in src
    assert ("np.savez_compressed(RESULTS / 'official_test_logprobs_p6_new_noflip.npz', **{f'{a}_{s}': v for (a, s), v in LOGP_NEW_NOFLIP.items()})\n"
            in src)
    assert "**{f'{head}_{s}': HEADS[s][i] for s in pm.SPLITS for i, head in enumerate(('crop', 'disease'))})" in src
    assert "METRICS" in cell_src(cells, "# 14.")
    first_save = src.index("save_results(OUT, DRIVE_RESULTS, model_folders=('stage2',), retries=10, wait=30)")
    assert src.index("results.to_csv(RESULTS / 'official_test_results.csv'") < first_save < src.index(noflip)
    assert src.index("official_test_logprobs_p6_ensemble.npz") < first_save
    assert src.rindex("save_results(") > src.index("official_test_logprobs_p6_new_noflip.npz")


def test_topline_uses_rule_1_gamma_and_rule_2_gamma_hm_and_r_hm_with_seen_parts_first(cells):
    src = cell_src(cells, "# 11b.")
    assert "verify_lock" not in src and "P5_FILES" not in src and "DRIVE_DIR" not in src
    assert "LOCK_INFO = st.verify_lock(OUT / 'lock.json')['info']" in cell_src(cells, "# 11. Official test")
    assert ("pg.topline_rows(TEST_PARTS[0], TEST_PARTS[1], load_catalogue(CAT_FINAL_PATH),\n"
            "                                       LOCK_INFO['gamma'], gamma_hm=LOCK_INFO['gamma_hm'], r_hm=LOCK_INFO['r_hm'])") in src
    assert "topline.to_csv(RESULTS / 'official_test_raw_and_calibrated.csv', index=False)\nsave_results(OUT, DRIVE_RESULTS, model_folders=('stage2',), retries=10, wait=30)\n" in src
    assert find(cells, "# 11b.") > find(cells, "# 11. Official test")


def test_comparison_table_is_p5_then_p6_under_the_all_pairs_rule(cells):
    src = cell_src(cells, "# 12.")
    assert "vs = pm.compare_table(P5_TABLE, results, B1_TEST)\n" in src and "pair_list" not in src
    assert "vs.to_csv(RESULTS / 'p6_vs_p5_vs_b1.csv', index=False)\nsave_results(OUT, DRIVE_RESULTS, model_folders=('stage2',), retries=10, wait=30)\n" in src


def test_unseen_pair_table_and_confusion_use_the_four_model_heads(cells):
    assert "per_pair = pm.unseen_pair_table(HEADS['unseen'][0], HEADS['unseen'][1], test['unseen'], CROP_IDS, DISEASE_IDS)\n" in cell_src(cells, "# 13.")
    assert ("conf, rows_pairs, pairs = pm.seen_confusion(HEADS['seen'][0], HEADS['seen'][1], test['seen'], catalogue, CROP_IDS, DISEASE_IDS, ALLOWED_333)\n"
            in cell_src(cells, "# 14."))


def test_stage_2_final_models_are_the_ones_the_lock_and_the_test_load(cells):
    stage2 = cell_src(cells, "# 9. Stage 2")
    assert "STAGE2 = OUT / 'stage2'" in stage2
    assert "'--stage1', str(STAGE1)" in stage2 and "'--out', str(STAGE2)" in stage2
    assert "FINAL = {arm: STAGE2 / arm / 'best_train_acc.pt' for arm in FINAL_ARMS}" in cell_src(cells, "# 10. Calibration")
    assert "stage1" not in cell_src(cells, "# 11. Official test").lower().replace("stage1_", "")


def test_b1_constants_equal_the_p5_notebook_constants(cells):
    assert P5_NOTEBOOK.is_file(), P5_NOTEBOOK
    p5_cells = ["".join(c["source"]) for c in json.loads(P5_NOTEBOOK.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    owners = [src for src in p5_cells if "B1_TEST = {" in src]
    assert len(owners) == 1

    def b1(src):
        assigns = [n for n in ast.parse(src).body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "B1_TEST" for t in n.targets)]
        assert len(assigns) == 1
        return ast.literal_eval(assigns[0].value)

    assert b1(cell_src(cells, "# 1. Settings")) == b1(owners[0])
    assert b1(owners[0]) == {"Crop": (89.30, 73.07, 96.20, 84.74), "Disease": (66.25, 11.12, 88.45, 26.73),
                             "Composition": (63.65, 6.56, 84.00, 18.85)}
