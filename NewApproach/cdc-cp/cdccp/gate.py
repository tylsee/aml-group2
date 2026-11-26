"""The one written gate every optional change must pass (plan section 4)."""
import numpy as np

from .evaluate import bootstrap_delta, harmonic


def subgroup_masks(df, train_df):
    """Masks on df: images from a source lacking the disease, and images whose disease has more than one crop."""
    source = lambda frame: frame.filename.str.split("_").str[0]
    diseases_by_source = train_df.groupby(source(train_df)).disease_id.apply(set).to_dict()
    crops_per_disease = train_df.groupby("disease_id").crop_id.nunique()
    lacks = np.array([d not in diseases_by_source.get(s, set()) for s, d in zip(source(df), df.disease_id)])
    multi = df.disease_id.map(crops_per_disease).fillna(0).values > 1
    return {"source_lacks_disease": lacks, "multi_crop": multi}


def accept(variant, control, delta, seed=42):
    """Each argument is {seen_hits, seen_pair, unseen_hits, unseen_pair, subgroups}. Returns a verdict record."""
    h = lambda r: harmonic(100 * r["seen_hits"].mean(), 100 * r["unseen_hits"].mean())
    gain = h(variant) - h(control)
    draws = bootstrap_delta(
        control["seen_hits"], variant["seen_hits"], control["seen_pair"],
        control["unseen_hits"], variant["unseen_hits"], control["unseen_pair"], seed=seed,
    )
    p10 = float(np.percentile(draws, 10))
    # An empty subgroup has nothing to fall, so it is reported as NaN and does not fail clause (c).
    falls = {
        name: 100 * (control["unseen_hits"][mask].mean() - variant["unseen_hits"][mask].mean()) if mask.any() else float("nan")
        for name, mask in control["subgroups"].items()
    }
    clauses = {"a": gain >= delta, "b": p10 > 0, "c": all(np.isnan(fall) or fall <= delta for fall in falls.values())}
    return {"accept": all(clauses.values()), "clauses": clauses, "delta_H": gain, "p10": p10, "subgroup_falls": falls, "delta": delta}
