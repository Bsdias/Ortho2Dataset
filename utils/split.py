"""Splits generated tiles into train/val/test partitions."""
import random


def parse_split_config(split_config):
    """Validates a `split` config mapping (train/val/test ratios + optional seed).

    Returns None if `split_config` is falsy (no splitting requested). `test`
    may be omitted (treated as 0, i.e. no test partition). Ratios must sum to
    1.0 (within floating point tolerance).
    """
    if not split_config:
        return None

    ratios = {
        "train": float(split_config.get("train", 0)),
        "val": float(split_config.get("val", 0)),
        "test": float(split_config.get("test", 0)),
    }
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total} ({ratios})")

    return {
        "ratios": {name: r for name, r in ratios.items() if r > 0},
        "seed": split_config.get("seed", 42),
    }


def assign_splits(ids, split):
    """Randomly assigns each id in `ids` to a split name ("train"/"val"/"test"),
    matching the configured ratios as closely as possible.

    Returns a dict {id: split_name}.
    """
    ratios = split["ratios"]
    names = list(ratios.keys())

    shuffled = list(ids)
    random.Random(split["seed"]).shuffle(shuffled)

    n = len(shuffled)
    assignment = {}
    start = 0
    for i, name in enumerate(names):
        end = n if i == len(names) - 1 else start + round(ratios[name] * n)
        for item_id in shuffled[start:end]:
            assignment[item_id] = name
        start = end
    return assignment
