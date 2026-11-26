import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cdccp import final_model as fm
from cdccp import single_task as st
from cdccp.config import REPO

GENERATOR = REPO / "scripts" / "make_final_model_notebook.py"
P6_GENERATOR = REPO / "scripts" / "make_p6_colab_notebook.py"
P5_TABLE = Path(r"D:\dev\uni\AML\runs\p5_colab_L4\p5_results\results\official_test_results.csv")
P6_TABLE = Path(r"D:\dev\uni\AML\runs\p6_colab_A100\p6_results\results\official_test_results.csv")
DUMMY_SHA = "cd" * 32


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return load(GENERATOR, "make_final_model_notebook")


@pytest.fixture(scope="module")
def notebook(gen, tmp_path_factory):
    out = tmp_path_factory.mktemp("nb") / "New_approach.ipynb"
    gen.build(out, DUMMY_SHA)
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cells(notebook):
    return [(c["cell_type"], "".join(c["source"])) for c in notebook["cells"]]


def code(cells):
    return [src for kind, src in cells if kind == "code"]


def index_of(cells, needle):
    hits = [i for i, (kind, src) in enumerate(cells) if kind == "code" and needle in src]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


def frames(n_seen=6, n_unseen=4, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for split, n in (("seen", n_seen), ("unseen", n_unseen)):
        crop, disease = rng.integers(0, 55, n), rng.integers(0, 175, n)
        out[split] = pd.DataFrame({"filename": [f"{split}_{i}.jpg" for i in range(n)], "crop_id": crop, "disease_id": disease,
                                   "composition_id": crop * 175 + disease, "composition_name": [f"p{c}_{d}" for c, d in zip(crop, disease)]})
    return out


def random_logp(dfs, seed=1):
    rng = np.random.default_rng(seed)
    out = {}
    for arm in fm.ALL_MEMBERS:
        classes = 55 if st.ARMS[arm].task == "crop" else 175
        for split, df in dfs.items():
            x = rng.normal(size=(len(df), classes)) * 3
            out[arm, split] = (x - np.log(np.exp(x).sum(1, keepdims=True))).astype(np.float32)
    return out


# ---- module -------------------------------------------------------------------------------------------------------------

def lsm64(x):
    x = np.asarray(x, dtype=np.float64)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def test_phases_are_p1_p2_p3_and_five_models():
    assert fm.PHASES == {"P1": {"crop": ("crop_b1",), "disease": ("disease_b1",)},
                         "P2": {"crop": ("crop_b1",), "disease": ("disease_clip",)},
                         "P3": {"crop": ("crop_b1", "crop_clip"), "disease": ("disease_clip", "disease_small_reg")}}
    assert fm.ALL_MEMBERS == ("crop_b1", "disease_b1", "disease_clip", "crop_clip", "disease_small_reg")
    for tasks in fm.PHASES.values():
        assert len(tasks["crop"]) == len(tasks["disease"])
        for task, arms in tasks.items():
            assert all(st.ARMS[arm].task == task for arm in arms)
    backbones = {phase: {task: [st.ARMS[a].backbone for a in arms] for task, arms in tasks.items()} for phase, tasks in fm.PHASES.items()}
    assert backbones["P1"] == {"crop": ["convnext_tiny"], "disease": ["convnext_tiny"]}
    assert backbones["P2"]["disease"] == ["clip_vit_b16"] and backbones["P3"]["crop"] == ["convnext_tiny", "clip_vit_b16"]


def test_phase_parts_single_models_and_the_p3_mean():
    dfs = frames()
    logp = random_logp(dfs)
    parts = {phase: fm.phase_parts(logp, (dfs["seen"], dfs["unseen"]), phase) for phase in fm.PHASES}
    for i, split in enumerate(("seen", "unseen")):
        assert np.allclose(parts["P1"][i]["crop_logits"], lsm64(logp["crop_b1", split]), rtol=0, atol=1e-6)
        assert np.allclose(parts["P1"][i]["disease_logits"], lsm64(logp["disease_b1", split]), rtol=0, atol=1e-6)
        assert np.allclose(parts["P2"][i]["crop_logits"], parts["P1"][i]["crop_logits"], rtol=0, atol=0)
        assert np.allclose(parts["P2"][i]["disease_logits"], lsm64(logp["disease_clip", split]), rtol=0, atol=1e-6)
        for head, (a, b) in (("crop_logits", fm.PHASES["P3"]["crop"]), ("disease_logits", fm.PHASES["P3"]["disease"])):
            want = 0.5 * lsm64(logp[a, split]) + 0.5 * lsm64(logp[b, split])
            assert np.allclose(parts["P3"][i][head], want, rtol=0, atol=1e-6)
            assert not np.allclose(parts["P3"][i][head], lsm64(logp[a, split]), atol=1e-3)
        for phase in fm.PHASES:
            assert (parts[phase][i]["crop_id"] == dfs[split].crop_id.to_numpy()).all()
            assert (parts[phase][i]["fn"] == dfs[split].filename.to_numpy()).all()


def test_phase_parts_refuses_a_missing_model_only_for_the_phases_that_use_it():
    dfs = frames()
    logp = random_logp(dfs)
    del logp["crop_clip", "unseen"]
    with pytest.raises(ValueError, match="crop_clip"):
        fm.phase_parts(logp, (dfs["seen"], dfs["unseen"]), "P3")
    fm.phase_parts(logp, (dfs["seen"], dfs["unseen"]), "P2")
    with pytest.raises(KeyError):
        fm.phase_parts(logp, (dfs["seen"], dfs["unseen"]), "P4")


def test_check_jobs_needs_each_member_once():
    five = {(0, 0): ["disease_small_reg", "crop_b1"], (0, 1): ["disease_clip", "crop_clip", "disease_b1"]}
    assert sorted(fm.check_jobs(five)) == sorted(fm.ALL_MEMBERS)
    with pytest.raises(ValueError):
        fm.check_jobs({(0, 0): ["disease_small_reg", "crop_b1"], (0, 1): ["disease_clip", "crop_clip"]})
    with pytest.raises(ValueError):
        fm.check_jobs({(0, 0): ["disease_small_reg", "crop_b1", "crop_b1"], (0, 1): ["disease_clip", "crop_clip", "disease_b1"]})


def test_job_commands_match_train_job_arguments():
    one = fm.job_command("stage1", "crop_b1", splits="S", images="I", out="O", epochs=25, patience=5, workers=4, device="cuda")
    assert one[1:] == ["scripts/train_job.py", "stage1", "--arm", "crop_b1", "--images", "I", "--out", "O", "--epochs", "25",
                       "--workers", "4", "--device", "cuda", "--splits", "S", "--patience", "5"]
    smoke = fm.job_command("stage1", "crop_b1", splits="S", images="I", out="O", epochs=1, patience=1, workers=0, device="cpu",
                           amp=False, pretrained=False, max_batches=2)
    assert smoke[-4:] == ["--no-pretrained", "--max-train-batches", "2", "--no-amp"]
    two = fm.job_command("stage2", "disease_clip", stage1="S1", train_csv="T", images="I", out="O2", epochs=10, workers=4, device="cuda")
    assert two[2:] == ["stage2", "--arm", "disease_clip", "--images", "I", "--out", "O2", "--epochs", "10", "--workers", "4",
                       "--device", "cuda", "--stage1", "S1", "--train-csv", "T"]
    assert "--no-pretrained" not in fm.job_command("stage2", "crop_b1", stage1="S1", train_csv="T", images="I", out="O", epochs=1,
                                                   workers=0, device="cpu", pretrained=False)
    with pytest.raises(ValueError):
        fm.job_command("stage3", "crop_b1", images="I", out="O", epochs=1, workers=0, device="cpu")
    train_job = load(REPO / "scripts" / "train_job.py", "train_job")
    parser_flags = (REPO / "scripts" / "train_job.py").read_text()
    for flag in ("--splits", "--patience", "--stage1", "--train-csv", "--no-pretrained", "--max-train-batches", "--no-amp"):
        assert flag in parser_flags
    assert callable(train_job.main)


def test_member_settings_come_from_the_training_code():
    table = fm.member_settings().set_index("model")
    assert list(table.index) == list(fm.ALL_MEMBERS)
    assert table.loc["crop_b1", "weight decay"] == 0.001 and table.loc["crop_b1", "stage 1 lr (backbone / head)"] == "1e-05 / 0.001"
    assert table.loc["disease_small_reg", "label smoothing"] == 0.1 and table.loc["disease_small_reg", "stochastic depth"] == 0.4
    assert table.loc["disease_small_reg", "augmentation"] == "strong" and table.loc["disease_clip", "weight decay"] == 0.1
    assert table.loc["crop_clip", "stage 2 lr (backbone / head)"] == "1e-06 / 0.0001"
    assert table.loc["crop_b1", "head"] == "dropout 0.3 + linear" and table.loc["disease_small_reg", "head"] == "dropout 0.2 + linear"
    assert table.loc["disease_b1", "head"] == "dropout 0.2 + linear" and table.loc["disease_b1", "weight decay"] == 0.001
    assert table.loc["disease_b1", "augmentation"] == "b1" and table.loc["disease_b1", "label smoothing"] == 0.0
    # ConvNeXt-Tiny gets no explicit value, so the torchvision default (0.1) is what the network really has
    assert np.isclose(table.loc["crop_b1", "stochastic depth"], 0.1) and np.isclose(table.loc["disease_small_reg", "stochastic depth"], 0.4)
    assert np.isclose(table.loc["disease_b1", "stochastic depth"], 0.1)
    assert table.loc["crop_clip", "stochastic depth"] == 0.0 and table.loc["disease_clip", "stochastic depth"] == 0.0


class _FakeWeights:
    class IMAGENET1K_V1:
        downloads = []

        @classmethod
        def get_state_dict(cls, progress=False):
            cls.downloads.append(progress)
            return {}


def _fake_backbone(value):
    import torch

    class Net(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 4), value)

    return lambda weights=None: Net()


def _preflight(monkeypatch, members, amp, pretrained, value=1.0, zero_shot=50.0):
    import torch
    monkeypatch.setattr(st, "BACKBONES", {"convnext_tiny": (_fake_backbone(value), _FakeWeights)})
    calls = []
    monkeypatch.setattr(st, "clip_text_check", lambda task, names, pretrained: calls.append(("text", task)))
    monkeypatch.setattr(st, "clip_zero_shot", lambda task, df, image_dir, names, device, n, pretrained: zero_shot)
    monkeypatch.setattr(st, "clip_fp16_probe", lambda device, pretrained: calls.append(("fp16",)) or (1, 8))
    names = {"crop": [f"c{i}" for i in range(55)], "disease": [f"d{i}" for i in range(175)]}
    fm.preflight(members, torch.device("cpu"), amp, pretrained, pd.DataFrame({"x": range(3)}), Path("."), names)
    return calls


def test_preflight_downloads_weights_and_passes_finite_half_precision(monkeypatch, capsys):
    _FakeWeights.IMAGENET1K_V1.downloads.clear()
    calls = _preflight(monkeypatch, ["crop_b1", "crop_clip"], amp=True, pretrained=True)
    out = capsys.readouterr().out
    assert _FakeWeights.IMAGENET1K_V1.downloads == [False] and "half precision OK for convnext_tiny" in out
    assert ("text", "crop") in calls and ("fp16",) in calls and "CLIP zero-shot crop" in out


def test_preflight_stops_on_nan_half_precision(monkeypatch):
    with pytest.raises(RuntimeError, match="NaN/inf"):
        _preflight(monkeypatch, ["crop_b1"], amp=True, pretrained=False, value=float("nan"))


def test_preflight_stops_when_clip_zero_shot_is_near_chance(monkeypatch):
    with pytest.raises(RuntimeError, match="not above 2x chance"):
        _preflight(monkeypatch, ["crop_clip"], amp=False, pretrained=True, zero_shot=3.0)   # 2x chance of 55 crops = 3.64%
    _preflight(monkeypatch, ["crop_clip"], amp=False, pretrained=False, zero_shot=3.0)       # random weights: no guard


BACKBONE_NAMES = {"convnext_tiny": "ConvNeXt-Tiny", "convnext_small": "ConvNeXt-Small", "clip_vit_b16": "CLIP ViT-B/16"}


def test_markdown_phase_table_matches_the_phases_and_names_no_old_runs(cells):
    text = "\n".join(src for kind, src in cells if kind == "markdown")
    rows = {}
    for line in text.splitlines():
        cols = [c.strip().replace("**", "") for c in line.split("|")[1:-1]]
        if cols and cols[0] in fm.PHASES:
            rows[cols[0]] = (cols[1], cols[2])
    assert set(rows) == set(fm.PHASES)
    for phase, tasks in fm.PHASES.items():
        want = tuple(" + ".join(BACKBONE_NAMES[st.ARMS[a].backbone] for a in tasks[task]) for task in ("crop", "disease"))
        assert rows[phase] == want, phase
    for old in ("P5", "P6", "Final model", "repeat", "first run", "SupCon", "prototype"):
        assert old not in text, old


def test_no_cell_and_no_message_on_the_notebook_path_names_p4_p5_or_p6(cells):
    import re
    old = re.compile(r"\b[pP][4-6](?:_|-|\b)")
    hits = [(i, m.group(0)) for i, (_, src) in enumerate(cells) for m in old.finditer(src)]
    assert not hits, hits
    import inspect
    from cdccp import calibration, colab_io, ensemble, gpu_runner
    modules = {"fm": fm, "cb": calibration, "ens": ensemble, "st": st}
    called = {(alias, name) for src in code(cells) for alias, name in re.findall(r"\b(fm|cb|ens|st)\.(\w+)\(", src)}
    functions = [getattr(modules[alias], name) for alias, name in sorted(called)]
    functions += [gpu_runner.run_on_gpus, colab_io.copy_tree, colab_io.check_images, colab_io.save_results]
    assert len(functions) > 20
    for function in functions:
        messages = re.findall(r"(?:print|RuntimeError|ValueError|description=)\(?(.*)", inspect.getsource(function))
        assert not [m for m in messages if old.search(m)], function.__name__
        assert not old.search(inspect.getdoc(function) or ""), function.__name__   # what Colab shows on hover
    for module in (fm, calibration, ensemble, gpu_runner, colab_io, st):
        assert not old.search(inspect.getdoc(module) or ""), module.__name__
    # modules that run whole on the notebook path (training jobs, GPU lanes, Drive copies): no old name anywhere in them
    for path in (REPO / "cdccp" / "single_task.py", REPO / "cdccp" / "gpu_runner.py", REPO / "cdccp" / "colab_io.py",
                 REPO / "cdccp" / "final_model.py", REPO / "scripts" / "train_job.py"):
        assert not old.findall(path.read_text(encoding="utf-8")), path.name


def test_markdown_settings_match_the_training_code(cells):
    text = "\n".join(src for kind, src in cells if kind == "markdown")
    table = fm.member_settings().set_index("model")
    wd = [table.loc[a, "weight decay"] for a in ("crop_b1", "disease_small_reg", "crop_clip")]
    assert table.loc["disease_b1", "weight decay"] == wd[0] and table.loc["disease_clip", "weight decay"] == wd[2]
    assert f"| Weight decay | {wd[0]:g} | {wd[1]:g} | {wd[2]:g} |" in text
    assert "| 0.1 (torchvision default) | 0.4 | none |" in text
    lrs = {table.loc[a, "stage 1 lr (backbone / head)"] for a in fm.ALL_MEMBERS} | {table.loc[a, "stage 2 lr (backbone / head)"] for a in fm.ALL_MEMBERS}
    assert lrs == {"1e-05 / 0.001", "1e-06 / 0.0001"} and "stage 1: 1e-5 / 1e-3; stage 2: 1e-6 / 1e-4" in text
    drop = st.B1["dropout"]
    assert f"dropout ({drop['crop']:g} crop, {drop['disease']:g} disease) + linear | dropout {drop['disease']:g} + linear" in text
    assert f"| AdamW, {st.B1['batch_size']} images |" in text
    assert all(len(p) == 4 for p in st.PROMPT_TEMPLATES.values()) and "(4 per class)" in text
    smooth = {table.loc[a, "label smoothing"] for a in fm.ALL_MEMBERS if a != "disease_small_reg"}
    assert smooth == {0.0} and table.loc["disease_small_reg", "label smoothing"] == 0.1 and "label smoothing 0.1 |" in text
    scope = {}
    exec(code(cells)[0], scope)
    assert f"at most {scope['STAGE1_EPOCHS']} epochs" in text and f"stop after {scope['PATIENCE']} epochs" in text
    assert f"{scope['STAGE2_EPOCHS']} epochs, cosine (T_max 10)" in text
    assert f"{st.DEV_SIZES[0]:,} dev-train images" in text


def test_earlier_runs_hold_only_baseline_1_equal_to_the_b1_constants():
    p6_gen = load(P6_GENERATOR, "make_p6_colab_notebook")
    settings = next(src for kind, src in p6_gen.CELLS if kind == "code" and "B1_TEST" in src)
    b1 = next(node for node in ast.walk(ast.parse(settings)) if isinstance(node, ast.Assign)
              and any(getattr(t, "id", None) == "B1_TEST" for t in node.targets))
    assert list(fm.EARLIER_RUNS) == ["Baseline 1"]
    assert fm.EARLIER_RUNS["Baseline 1"] == ast.literal_eval(b1.value)


def test_compare_runs_one_row_per_phase_after_baseline_1():
    p2, p3 = pd.read_csv(P5_TABLE), pd.read_csv(P6_TABLE)   # real saved tables as stand-ins for two phases
    vs = fm.compare_runs({"P1": p2, "P2": p2, "P3": p3})
    assert list(vs.index) == ["Baseline 1", "P1", "P2", "P3"]
    assert np.allclose(vs.loc["P1"].to_numpy(dtype=float), vs.loc["P2"].to_numpy(dtype=float))
    assert abs(vs.loc["P3", "pair HM"] - 17.55055631565838) < 1e-9 and abs(vs.loc["P3", "pair seen"] - 70.95) < 1e-9
    assert abs(vs.loc["Baseline 1", "pair HM"] - 11.89) < 0.01 and abs(vs.loc["P2", "pair HM"] - 15.406816882327018) < 1e-9
    assert abs(vs.loc["P3", "pair Top-5 HM"] - 45.065277664042306) < 1e-9 and abs(vs.loc["P3", "disease HM"] - 23.536995090099207) < 1e-9
    with pytest.raises(ValueError):
        fm.compare_runs({"P1": p3[p3.Task != "Disease"]})


def test_code_fingerprint_is_the_p6_rule(tmp_path):
    p6_gen = load(P6_GENERATOR, "make_p6_colab_notebook")
    (tmp_path / "cdccp" / "__pycache__").mkdir(parents=True)
    (tmp_path / "cdccp" / "a.py").write_text("x = 1")
    (tmp_path / "cdccp" / "__pycache__" / "a.pyc").write_bytes(b"junk")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "b.csv").write_text("1,2")
    assert fm.code_fingerprint(tmp_path) == p6_gen.code_sha_of_dir(tmp_path)
    before = fm.code_fingerprint(tmp_path)
    (tmp_path / "data" / "b.csv").write_text("1,3")
    assert fm.code_fingerprint(tmp_path) != before


def test_check_inputs_accepts_the_real_files_and_refuses_a_changed_split(tmp_path):
    splits = REPO / "data" / "splits_v2"
    train_csv = Path(r"D:\dev\uni\AML\data\DPD_Small\DPD_Small_train.csv")
    fm.check_inputs(train_csv, splits)
    for name in fm.SPLIT_SHA:
        (tmp_path / name).write_bytes((splits / name).read_bytes())
    (tmp_path / "val_seen.csv").write_bytes((splits / "val_seen.csv").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="val_seen.csv"):
        fm.check_inputs(train_csv, tmp_path)


def test_stage2_summary_reads_done_json(tmp_path):
    for arm, (source, best, acc) in zip(fm.ALL_MEMBERS, ((10, 10, 0.99), (12, 8, 0.97), (13, 9, 0.9996), (5, 6, 1.0), (21, 10, 0.922))):
        (tmp_path / arm).mkdir()
        (tmp_path / arm / "done.json").write_text(json.dumps({"source_epoch": source, "best_epoch": best, "best_train_acc": acc}))
    table = fm.stage2_summary(tmp_path).set_index("model")
    assert table.loc["disease_small_reg", "started from stage-1 epoch"] == 21 and table.loc["crop_clip", "kept stage-2 epoch"] == 6
    assert abs(table.loc["disease_small_reg", "train accuracy (%)"] - 92.2) < 1e-9


# ---- notebook -----------------------------------------------------------------------------------------------------------

def test_notebook_shape_ids_image_and_no_placeholders(notebook, cells):
    assert notebook["nbformat"] == 4 and all(c["id"] for c in notebook["cells"])
    assert len({c["id"] for c in notebook["cells"]}) == len(notebook["cells"])
    text = "\n".join(src for _, src in cells)
    assert "@@" not in text and DUMMY_SHA in text
    assert "data:image" not in text
    for src in code(cells):
        ast.parse(src)


def test_every_name_a_cell_uses_is_defined_by_an_earlier_cell_or_a_builtin(cells):
    import builtins
    defined = set(dir(builtins)) | {"display"}
    problems = []
    for number, src in enumerate(code(cells)):
        tree = ast.parse(src)
        local = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                local.add(node.name)
                local |= {a.arg for a in node.args.args + node.args.kwonlyargs}
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                local.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                local |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
            elif isinstance(node, ast.comprehension):
                local |= {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        problems += [(number, name) for name in sorted(used - defined - local)]
        defined |= local
    assert not problems, problems


def test_settings_cell_is_the_final_model_design(cells):
    scope = {}
    exec(code(cells)[0], scope)
    assert (scope["STAGE1_EPOCHS"], scope["PATIENCE"], scope["STAGE2_EPOCHS"]) == (25, 5, 10)
    assert scope["AMP"] is True and scope["FLIP"] is True and scope["PRETRAINED"] is True and scope["MAX_TRAIN_BATCHES"] is None
    assert sorted(a for arms in scope["GPU_JOBS"].values() for a in arms) == sorted(fm.ALL_MEMBERS)
    assert scope["EXPECT_TEST"] == (2000, 1448) and scope["TEST_NOT_IN_TRAIN"] is True and scope["EXPECT_IMAGES"] == 11448
    assert scope["DRIVE_RESULTS"].name == "final_model_results" and scope["EXPECTED_CODE_SHA"] == DUMMY_SHA
    assert "p5_results" not in "\n".join(code(cells)) and "p6_results" not in "\n".join(code(cells))


def test_test_csvs_are_read_only_after_gamma_is_saved(cells):
    sources = code(cells)
    calibration = index_of(cells, "ens.dev_calibration(")
    test_cell = index_of(cells, "st.read_labelled_csv(INPUT / f'DPD_Small_test_{split}.csv')")
    assert calibration < test_cell
    cal_src = cells[calibration][1]
    assert cal_src.index("ens.dev_calibration(") < cal_src.index("to_csv(RESULTS / 'dev_calibration.csv'") < cal_src.index("save_results(")
    for i, (kind, src) in enumerate(cells):
        if kind != "code" or i == test_cell:
            continue
        tree = ast.parse(src)
        reads = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", getattr(n.func, "id", None)) in {"read_labelled_csv", "read_csv", "read_text", "read_bytes"}
                 and "test" in ast.unparse(n).lower()]
        assert not reads, (i, reads)
    assert sum("DPD_Small_test" in src for src in sources) == 2   # copied in the data cell, read in the test cell


def test_dev_calibration_uses_dev_outputs_and_the_test_uses_stage2_models(cells):
    cal = cells[index_of(cells, "ens.dev_calibration(")][1]
    assert ("ens.dev_calibration(*fm.phase_parts(DEV_LOGP, (val_seen, val_unseen), phase), CAT_DEV) for phase in fm.PHASES" in cal)
    assert "CAT_DEV = load_catalogue(CODE_DIR / 'data' / 'mappings' / 'catalogue_dev.json')" in cal
    dev = cells[index_of(cells, "DEV_LOGP = fm.predict_members(")][1]
    assert "STAGE1 / arm / 'best.pt'" in dev and "FLIP" in dev and "for arm in TRAINED" in dev
    test = cells[index_of(cells, "TEST_LOGP = fm.predict_members(")][1]
    assert "STAGE2 / arm / 'best_train_acc.pt'" in test and "FLIP" in test and "for arm in TRAINED" in test
    assert "fm.phase_parts(TEST_LOGP, (test['seen'], test['unseen']), phase)" in test and "for phase in fm.PHASES:" in test
    assert "ens.check_test_frames(test" in test and "disjoint=TEST_NOT_IN_TRAIN" in test
    assert sum("fm.predict_members(" in src for src in code(cells)) == 2   # dev once, test once: each model is tested once


def test_each_phase_is_calibrated_with_its_own_dev_gamma(cells):
    top = cells[index_of(cells, "cb.topline_rows(")][1]
    assert ("cb.topline_rows(*TEST_PARTS[phase], cat_final, CAL[phase]['gamma_keep'],\n"
            "                                                  gamma_hm=CAL[phase]['gamma_hm'], r_hm=CAL[phase]['r_hm'])") in top
    assert "for phase in fm.PHASES" in top


def test_raw_results_come_before_calibrated_rows(cells):
    raw = index_of(cells, "ens.official_table(HEADS")
    calibrated = index_of(cells, "cb.topline_rows(")
    assert raw < calibrated
    assert index_of(cells, "fm.compare_runs(TABLES)") > calibrated


def test_no_gpu_stops_before_drive_folders_are_renamed(cells):
    setup = cells[index_of(cells, "drive.mount(")][1]
    check = "assert DEVICE_NAME != 'cuda' or torch.cuda.is_available()"
    assert check in setup and setup.index(check) < setup.index("drive.mount(") < setup.index("folder.rename(")
    assert index_of(cells, "drive.mount(") < index_of(cells, "copy_tree(")


def test_stage_commands_use_the_settings(cells):
    s1 = cells[index_of(cells, "def stage1_command")][1]
    for part in ("epochs=STAGE1_EPOCHS", "patience=PATIENCE", "splits=SPLITS", "pretrained=PRETRAINED", "amp=AMP", "max_batches=MAX_TRAIN_BATCHES"):
        assert part in s1
    s2 = cells[index_of(cells, "def stage2_command")][1]
    for part in ("epochs=STAGE2_EPOCHS", "stage1=STAGE1", "train_csv=TRAIN_CSV", "amp=AMP"):
        assert part in s2


def test_code_sha_of_zip_matches_the_unpacked_fingerprint(gen, tmp_path):
    import zipfile
    root = tmp_path / "src" / "cdc-cp"
    (root / "cdccp").mkdir(parents=True)
    (root / "cdccp" / "final_model.py").write_text("y = 2")
    zip_path = tmp_path / "code.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(root / "cdccp" / "final_model.py", "cdc-cp/cdccp/final_model.py")
    assert gen.code_sha_of_zip(zip_path) == fm.code_fingerprint(root)


def test_build_refuses_a_bad_sha(gen, tmp_path):
    with pytest.raises(ValueError):
        gen.build(tmp_path / "x.ipynb", "nothex")

