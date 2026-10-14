"""Degree-preserving wiring scramble: a null-model control for Stonkfly.

Same neurons, same number of incoming and outgoing connections per neuron,
same outgoing synapse counts and signs per neuron -- but WHO connects to WHOM
is randomized. If the real fly beats this scrambled fly, the specific MaleCNS
wiring matters. If they perform alike, the result is not about fly wiring.

Held fixed (never scrambled), so the rest of Stonkfly still works and the
comparison isolates the brain's internal wiring:

* Photoreceptor outputs (R1-R6, R7, R8). The R8 screen position is inferred
  from these edges, so scrambling them would also scramble the "eye" and mix
  two different manipulations.
* The memory circuit Stonkfly's learning rule needs: KC -> MBON07/MBON11 and
  PAM11/PPL101 -> MBON07/MBON11.
* Any extra presynaptic cell types passed with --keep-type.

Scrambled edges are also never allowed to create new KC/DAN -> MBON07/11
connections, so Stonkfly's plastic edge set and dopamine gains are identical
to the real fly's.

Method: the targets of all non-protected edges are randomly permuted among
themselves (configuration-model shuffle), then duplicate pre->post pairs are
removed by further random target swaps. Every swap moves two targets, so each
neuron's in-degree and out-degree stay exactly the same. Each edge keeps its
presynaptic neuron, synapse count and sign, so total outgoing strength per
neuron is unchanged; total incoming strength per neuron is NOT preserved.

By default excitatory and inhibitory edges are shuffled separately, so every
neuron also keeps its number of excitatory and of inhibitory inputs. Without
this, a neuron that was mostly inhibited could become mostly excited, and a
scrambled brain might lose only because its activity went silent or ran away.
--mix-signs turns this off (plain degree-preserving shuffle).

Usage:
    python -m stonkfly.neural.scramble --seed 1
    STONKFLY_DATA=data-scrambled-seed1 python -m stonkfly run --out runs/scrambled-seed1
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

PHOTORECEPTORS = r"^(R1-R6|R7|R8)"
MEMORY_OUTPUTS = ["MBON07", "MBON11"]
MEMORY_DANS = ["PAM11", "PPL101"]


def digest(array):
    """Same array hash Stonkfly uses for its locks (sha256 of raw bytes)."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def protected_edges(pre, post, cell_types, keep_types=()):
    """Boolean mask of edges that must keep their original target."""
    import pandas as pd

    types = pd.Series(cell_types, dtype=object).fillna("").astype(str)
    photo = types.str.match(PHOTORECEPTORS).to_numpy()
    kc = types.str.startswith("KC").to_numpy()
    mbon = types.isin(MEMORY_OUTPUTS).to_numpy()
    dan = types.isin(MEMORY_DANS).to_numpy()
    keep = types.isin(list(keep_types)).to_numpy()
    memory_pre = kc | dan
    mask = photo[pre] | keep[pre] | (mbon[post] & memory_pre[pre])
    # Scrambling must not invent new KC/DAN -> MBON07/11 edges either, or
    # Stonkfly would pick them up as extra plastic edges or DAN gains.
    forbidden = (~mask) & memory_pre[pre]
    return mask, forbidden, mbon, {
        "photoreceptor_cells": int(photo.sum()),
        "kc_cells": int(kc.sum()),
        "memory_mbon_cells": int(mbon.sum()),
        "memory_dan_cells": int(dan.sum()),
        "extra_keep_types": sorted(keep_types),
        "extra_keep_cells": int(keep.sum()),
    }


def duplicate_free_edges(pre, post, n, free_flag):
    """Free edges that repeat an existing pre->post pair, plus all sorted pairs.

    Within a group of identical pairs, a protected edge (if any) is kept and
    the remaining free copies are returned for repair.
    """
    key = pre.astype(np.int64) * n + post.astype(np.int64)
    # Sort by pair, protected edges (free_flag 0) first inside each pair.
    order = np.lexsort((free_flag, key))
    sorted_key = key[order]
    repeat = np.r_[False, sorted_key[1:] == sorted_key[:-1]]
    bad = order[repeat]
    if np.any(free_flag[bad] == 0):
        raise ValueError("Two protected edges share one pre->post pair.")
    return bad, sorted_key


def contains(sorted_key, keys):
    at = np.minimum(np.searchsorted(sorted_key, keys), len(sorted_key) - 1)
    return sorted_key[at] == keys


def scramble(
    ptr, post, cell_types, seed, keep_types=(), signs=None, max_rounds=200, log=print
):
    """Return a new post array plus a statistics report. ptr is not changed.

    signs: optional per-edge weights (or signs). When given, negative and
    non-negative edges are shuffled as two separate pools, so every neuron
    keeps its inhibitory and excitatory in-degree.
    """
    ptr = np.asarray(ptr, dtype=np.int64)
    post = np.asarray(post, dtype=np.int32)
    n = len(ptr) - 1
    if len(cell_types) != n:
        raise ValueError("Cell types must align with graph neurons.")
    if signs is not None and len(signs) != len(post):
        raise ValueError("Edge signs must align with graph edges.")
    pre = np.repeat(np.arange(n, dtype=np.int32), np.diff(ptr))
    protect, forbidden, mbon, protect_report = protected_edges(
        pre, post, cell_types, keep_types
    )
    free = np.flatnonzero(~protect)
    free_flag = (~protect).astype(np.uint8)
    rng = np.random.default_rng(seed)
    # Pool 0: excitatory (or every edge when signs are mixed); pool 1: inhibitory.
    group = (
        np.zeros(len(post), dtype=np.uint8)
        if signs is None
        else (np.asarray(signs) < 0).astype(np.uint8)
    )
    pools = [free[group[free] == g] for g in (0, 1)]

    original_duplicates = len(duplicate_free_edges(pre, post, n, free_flag)[0])
    new = post.copy()
    for pool in pools:
        new[pool] = post[pool][rng.permutation(len(pool))]

    def find_bad(targets):
        duplicates, sorted_key = duplicate_free_edges(pre, targets, n, free_flag)
        invented = np.flatnonzero(forbidden & mbon[targets])
        return np.union1d(duplicates, invented), sorted_key

    history = []
    bad, sorted_key = find_bad(new)
    for round_number in range(max_rounds):
        history.append(int(len(bad)))
        if not len(bad):
            break
        # When few edges are left, give each one several candidate partners
        # per round so stubborn hub-to-hub duplicates still get repaired.
        tries = int(min(64, max(1, 200_000 // len(bad))))
        a = np.repeat(bad, tries)
        b = np.empty_like(a)
        for g, pool in enumerate(pools):
            pick = group[a] == g
            if pick.any():
                b[pick] = rng.choice(pool, size=int(pick.sum()))
        # Each bad edge a (row pa, target ta) trades targets with partner b
        # (row pb, target tb) from the same sign pool. Accept only swaps whose
        # two new pairs pa->tb and pb->ta do not exist yet and do not invent
        # memory-circuit edges.
        ta, tb = new[a].astype(np.int64), new[b].astype(np.int64)
        k1 = pre[a].astype(np.int64) * n + tb
        k2 = pre[b].astype(np.int64) * n + ta
        valid = ~contains(sorted_key, k1) & ~contains(sorted_key, k2)
        valid &= ~(forbidden[a] & mbon[tb]) & ~(forbidden[b] & mbon[ta])
        # Keep only each bad edge's first valid candidate.
        candidate = np.flatnonzero(valid)
        candidate = candidate[np.unique(a[candidate], return_index=True)[1]]
        a, b, k1, k2 = a[candidate], b[candidate], k1[candidate], k2[candidate]
        valid = np.ones(len(a), dtype=bool)
        # Within one round, every edge and every new pair may be used only once.
        for ids in (np.r_[a, b], np.r_[k1, k2]):
            _, first, counts = np.unique(ids, return_index=True, return_counts=True)
            once = np.zeros(len(ids), dtype=bool)
            once[first[counts == 1]] = True
            valid &= once[: len(a)] & once[len(a) :]
        a, b = a[valid], b[valid]
        new[a], new[b] = new[b].copy(), new[a].copy()
        bad, sorted_key = find_bad(new)
        log(f"  repair round {round_number + 1}: {len(bad)} edges left to repair")
    remaining = int(len(bad))

    out_before = np.diff(ptr)
    in_before = np.bincount(post, minlength=n)
    in_after = np.bincount(new, minlength=n)
    if not np.array_equal(in_before, in_after):
        raise AssertionError("In-degree changed; scramble is not degree-preserving.")
    inhibitory = group == 1
    if not np.array_equal(
        np.bincount(post[inhibitory], minlength=n), np.bincount(new[inhibitory], minlength=n)
    ):
        raise AssertionError("Inhibitory in-degree changed within a sign pool.")
    if not np.array_equal(new[protect], post[protect]):
        raise AssertionError("A protected edge changed target.")
    if remaining:
        raise RuntimeError(
            f"{remaining} edges could not be repaired; try another seed or more rounds."
        )

    changed = int(np.count_nonzero(new[free] != post[free]))
    by_sign = signs is not None
    report = {
        "method": "configuration-model target permutation of non-protected edges, then random target swaps to remove duplicate pairs",
        "sign_pools": "excitatory and inhibitory edges shuffled separately"
        if by_sign
        else "mixed: edges of both signs shuffled together",
        "seed": int(seed),
        "neurons": int(n),
        "edges": int(len(post)),
        "protected_edges": int(protect.sum()),
        "scrambled_edges": int(len(free)),
        "scrambled_inhibitory_edges": int(len(pools[1])),
        "scrambled_edges_with_new_target": changed,
        "scrambled_fraction_with_new_target": changed / max(len(free), 1),
        "protected": protect_report,
        "preserved_exactly": [
            "every neuron's out-degree",
            "every neuron's in-degree",
            *(["every neuron's excitatory and inhibitory in-degree"] if by_sign else []),
            "every edge's presynaptic neuron, synapse count and sign",
            "every protected edge",
        ],
        "not_preserved": [
            "who connects to whom among scrambled edges",
            "each neuron's total incoming synaptic strength",
            "self-connection count",
        ],
        "self_edges_before": int(np.count_nonzero(pre == post)),
        "self_edges_after": int(np.count_nonzero(pre == new)),
        "duplicate_pairs_before": int(original_duplicates),
        "duplicate_pairs_after": int(len(duplicate_free_edges(pre, new, n, free_flag)[0])),
        "memory_circuit_edges_invented": int(np.count_nonzero(forbidden & mbon[new])),
        "repair_rounds": history,
        "out_degree_unchanged": True,
        "in_degree_unchanged": True,
        "max_out_degree": int(out_before.max()),
        "max_in_degree": int(in_before.max()),
    }
    return new, report


def write_scrambled_dataset(
    source, target, seed, keep_types=(), force=False, mix_signs=False
):
    import pyarrow.feather as feather

    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError("Write the scrambled graph to a separate data directory.")
    if (target / "graph.npz").exists() and not force:
        raise SystemExit(f"{target} already has a graph; pass --force to replace it.")
    if (source / "scramble.lock.json").exists():
        raise SystemExit("The source directory is already scrambled; use the real data.")
    for needed in ["graph.npz", "annotations.feather", "normalized/neurons.feather"]:
        if not (source / needed).exists():
            raise SystemExit(
                f"Missing {source / needed}. Run `python -m stonkfly prepare` first."
            )
    started = time.time()
    with np.load(source / "graph.npz", allow_pickle=False) as g:
        arrays = {k: g[k] for k in g.files}
    nodes = feather.read_table(source / "normalized/neurons.feather").to_pandas()
    if not np.array_equal(nodes.source_id.to_numpy(), arrays["ids"]):
        raise SystemExit("neurons.feather order does not match graph.npz.")
    print(f"Scrambling {len(arrays['post']):,} edges with seed {seed} ...", flush=True)
    new_post, report = scramble(
        arrays["ptr"],
        arrays["post"],
        nodes.cell_type.to_numpy(),
        seed,
        keep_types,
        signs=None if mix_signs else arrays["weight"],
    )
    original = {k: digest(arrays[k]) for k in ["ptr", "post", "weight"]}
    arrays["post"] = new_post.astype(np.int32)

    target.mkdir(parents=True, exist_ok=True)
    (target / "normalized").mkdir(exist_ok=True)
    for name in ["annotations.feather", "normalized/neurons.feather"]:
        link = target / name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(source / name, link)
    partial = target / "graph.partial.npz"
    np.savez(partial, **arrays)
    partial.replace(target / "graph.npz")
    lock = {
        "source_data": str(source),
        "original_arrays": original,
        "scrambled_arrays": {k: digest(arrays[k]) for k in ["ptr", "post", "weight"]},
        "summary": report,
        "seconds": round(time.time() - started, 1),
    }
    (target / "scramble.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, required=True, help="Random seed; use several")
    parser.add_argument("--source", type=Path, default=Path(os.environ.get("STONKFLY_DATA", "data")))
    parser.add_argument("--out", type=Path, help="Default: data-scrambled-seed<SEED>")
    parser.add_argument(
        "--keep-type",
        action="append",
        default=[],
        help="Also keep all outgoing edges of this cell type (repeatable)",
    )
    parser.add_argument(
        "--mix-signs",
        action="store_true",
        help="Shuffle excitatory and inhibitory edges together (weaker control)",
    )
    parser.add_argument("--force", action="store_true")
    a = parser.parse_args()
    out = a.out or Path(f"data-scrambled-seed{a.seed}")
    lock = write_scrambled_dataset(a.source, out, a.seed, a.keep_type, a.force, a.mix_signs)
    s = lock["summary"]
    print(
        json.dumps(
            {
                "out": str(out),
                "seed": s["seed"],
                "sign_pools": s["sign_pools"],
                "scrambled_edges": s["scrambled_edges"],
                "protected_edges": s["protected_edges"],
                "new_target_fraction": round(s["scrambled_fraction_with_new_target"], 4),
                "duplicate_pairs_after": s["duplicate_pairs_after"],
                "self_edges_before_after": [s["self_edges_before"], s["self_edges_after"]],
            },
            indent=2,
        )
    )
    print(f"Run it with: STONKFLY_DATA={out} python -m stonkfly run --out runs/scrambled-seed{a.seed}")


if __name__ == "__main__":
    main()
