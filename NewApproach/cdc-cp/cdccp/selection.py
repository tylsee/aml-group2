"""Manual checkpoint selection rule and budget check (plan section 4)."""
import numpy as np


def moving_average(h):
    h = np.asarray(h, dtype=float)
    return np.array([h[max(0, i - 1):i + 2].mean() for i in range(len(h))])


def pick_epoch(h):
    """Index of the highest 3-epoch moving average of H. The ends use the epochs available."""
    return int(np.argmax(moving_average(h)))


def late_gain(h):
    """Mean H of the last 3 epochs minus mean H of the 3 middle epochs."""
    h = np.asarray(h, dtype=float)
    middle = len(h) // 2
    return float(h[-3:].mean() - h[middle - 2:middle + 1].mean())


def seed_delta(h_a, h_b):
    """Noise floor for the gate, from two seeds of one configuration."""
    return max(1.0, abs(float(moving_average(h_a).max() - moving_average(h_b).max())))


def budget_check(h_a, h_b=None):
    """True when the run looks underfit, so U should be doubled once. h_b is None if E3 was cut."""
    delta = 3.0 if h_b is None else seed_delta(h_a, h_b)
    runs = [h_a] if h_b is None else [h_a, h_b]
    return all(late_gain(h) >= delta for h in runs)


def pick_backbone(board, backbones, seeds=(42, 43, 44), head="h256", n_images=400):
    """Rank backbones by mean unseen disease Top-1 over the given seeds.

    board: calib_board.csv rows with columns run, backbone, 'unseen disease', 'H keep', 'seen disease'.
    Exactly one scored run named sweep_<backbone>_<head>_seed<seed> must exist for every planned backbone
    and seed, so no backbone wins on fewer runs, a missing backbone, or a run of another head.
    The leader ties with the runner-up when the gap is no more than two standard errors of the difference.
    That error adds the seed spread and the image noise of n_images unseen images per run
    (binomial, unpaired, so it is on the safe side).
    Returns (table sorted best first, winner, runner-up, tie flag, the gap a win must beat).
    """
    seeds = tuple(seeds)
    if len(set(seeds)) < 3:
        raise ValueError(f"need at least 3 distinct seeds, got {list(seeds)}")
    if len(set(backbones)) < 2:
        raise ValueError(f"need at least 2 backbones to compare, got {list(backbones)}")
    df = board.assign(seed=board["run"].str.extract(r"_seed(\d+)$")[0].astype(float))
    df = df[df["backbone"].isin(backbones) & df["seed"].isin(seeds)].astype({"seed": int})
    wrong = sorted(r for r, b, s in zip(df["run"], df["backbone"], df["seed"]) if r != f"sweep_{b}_{head}_seed{s}")
    if wrong:
        raise ValueError(f"runs that are not sweep_<backbone>_{head}_seed<seed>: {wrong}")
    unscored = sorted(df.loc[df["unseen disease"].isna(), "run"])
    if unscored:
        raise ValueError(f"runs without an unseen disease score: {unscored}")
    doubled = sorted(df.loc[df.duplicated(["backbone", "seed"], keep=False), "run"].unique())
    if doubled:
        raise ValueError(f"more than one row for: {doubled}")
    have = set(zip(df["backbone"], df["seed"]))
    missing = sorted(f"sweep_{b}_{head}_seed{s}" for b in backbones for s in seeds if (b, s) not in have)
    if missing:
        raise ValueError(f"planned runs missing from the board: {missing}")
    order = {s: i for i, s in enumerate(seeds)}
    df = df.assign(order=df["seed"].map(order)).sort_values(["backbone", "order"])
    table = df.groupby("backbone").agg(
        unseen_disease=("unseen disease", "mean"), sd=("unseen disease", "std"), n=("seed", "nunique"),
        per_seed=("unseen disease", lambda s: " / ".join(f"{v:.1f}" for v in s)),
        H_keep=("H keep", "mean"), seen_disease=("seen disease", "mean"),
    ).sort_values("unseen_disease", ascending=False)
    a, b = table.iloc[0], table.iloc[1]
    pa, pb = a.unseen_disease / 100, b.unseen_disease / 100
    image_var = 100 ** 2 * (pa * (1 - pa) + pb * (1 - pb)) / n_images
    limit = 2 * float(np.sqrt(a.sd ** 2 / a.n + b.sd ** 2 / b.n + image_var))
    tie = bool(a.unseen_disease - b.unseen_disease <= limit)
    return table, table.index[0], table.index[1], tie, limit


def run_scores(history):
    """Scores of one run from its history.csv, early-stopped or not.

    The score is the highest full 3-epoch average of unseen disease Top-1, centred on pick_epoch, so one lucky epoch
    counts less than in the single best epoch, which is shown for the record. Unlike pick_epoch, the first and last
    epochs are not centres when the run has 3 or more epochs, so every window holds 3 epochs. Seen disease is read at
    the same epoch.
    """
    h = history.sort_values("epoch")
    if h.empty:
        raise ValueError("history has no epochs")
    if h[["unseen_disease_top1", "seen_disease_top1"]].isna().any().any():
        raise ValueError("history has NaN disease scores")
    smooth = moving_average(h["unseen_disease_top1"])
    lo, hi = (1, len(h) - 1) if len(h) >= 3 else (0, len(h))
    i = lo + int(np.argmax(smooth[lo:hi]))
    best = int(np.argmax(h["unseen_disease_top1"].to_numpy()))
    return {"unseen_disease": float(smooth[i]), "seen_disease": float(moving_average(h["seen_disease_top1"])[i]),
            "pick_epoch": int(h["epoch"].iloc[i]), "best_epoch": int(h["epoch"].iloc[best]),
            "best_unseen_disease": float(h["unseen_disease_top1"].iloc[best]), "epochs": int(h["epoch"].iloc[-1])}


def compare_arms(scores, control, treatment, seeds=(42, 43, 44), n_images=400, z=2.0):
    """Paired comparison of two configurations run on the same seeds, by unseen disease Top-1.

    scores: rows with columns arm, seed, unseen_disease, seen_disease (from run_scores). Exactly one row must exist
    for every planned arm and seed. diff = treatment - control per seed. The verdict is "better" or "worse" when the
    mean diff is more than z standard errors: the paired seed spread plus the binomial noise of n_images unseen images
    per arm (unpaired, so it is on the safe side). Raise z when several treatments are compared with one control
    (2.4 for three), so that one of them clearing the bar by luck stays about as rare as for a single comparison. Early stopping and the score both pick on the same validation images,
    and that optimism is not in the error, so this is a development comparison of two training recipes, not a
    calibrated significance test or an estimate of the official test.
    Returns (table per arm, per-seed diffs, mean diff, limit, verdict "better", "worse" or "no clear change").
    """
    seeds = tuple(seeds)
    if not z > 0:
        raise ValueError(f"z must be positive, got {z}")
    if len(set(seeds)) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError(f"need at least 3 distinct seeds, each once, got {list(seeds)}")
    if control == treatment:
        raise ValueError("control and treatment must differ")
    df = scores[scores["arm"].isin([control, treatment]) & scores["seed"].isin(seeds)]
    if df["unseen_disease"].isna().any():
        raise ValueError("runs without an unseen disease score")
    doubled = df[df.duplicated(["arm", "seed"], keep=False)]
    if len(doubled):
        raise ValueError(f"more than one row for: {sorted(set(zip(doubled['arm'], doubled['seed'])))}")
    have = set(zip(df["arm"], df["seed"]))
    missing = [f"{a}_seed{s}" for a in (control, treatment) for s in seeds if (a, s) not in have]
    if missing:
        raise ValueError(f"planned runs missing: {missing}")
    wide = df.pivot(index="seed", columns="arm", values="unseen_disease").loc[list(seeds)]
    diffs = wide[treatment] - wide[control]
    table = df.groupby("arm").agg(unseen_disease=("unseen_disease", "mean"), sd=("unseen_disease", "std"),
                                  seen_disease=("seen_disease", "mean")).loc[[control, treatment]]
    pc, pt = table["unseen_disease"] / 100
    image_var = 100 ** 2 * (pc * (1 - pc) + pt * (1 - pt)) / n_images
    limit = z * float(np.sqrt(diffs.var(ddof=1) / len(diffs) + image_var))
    mean = float(diffs.mean())
    verdict = "better" if mean > limit else "worse" if mean < -limit else "no clear change"
    return table, diffs, mean, limit, verdict
