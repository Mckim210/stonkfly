"""Synthetic-graph checks for the degree-preserving scramble control."""

import json

import numpy as np
import pytest

from stonkfly.neural.scramble import digest, scramble, write_scrambled_dataset


def synthetic(n=3000, edges=60000, seed=0):
    rng = np.random.default_rng(seed)
    types = np.array(["other"] * n, dtype=object)
    types[:100] = "R1-R6"
    types[100:130] = "R8p"
    types[130:430] = "KCg-m"
    types[430:434] = "MBON07"
    types[434:436] = "MBON11"
    types[436:451] = "PAM11"
    types[451:453] = "PPL101"
    types[453:460] = "DNp20"
    pairs = set()
    # Memory circuit edges the learning rule depends on.
    for kc in range(130, 430):
        pairs.add((kc, int(rng.integers(430, 436))))
    for dan in range(436, 453):
        pairs.update({(dan, 430), (dan, 434)})
    # Heavy-tailed degrees so duplicate repair actually has work to do.
    weights = rng.pareto(1.5, n) + 1
    weights /= weights.sum()
    while len(pairs) < edges:
        i, j = rng.choice(n, size=2, p=weights)
        pairs.add((int(i), int(j)))
    pairs = np.array(sorted(pairs), dtype=np.int64)
    pre, post = pairs[:, 0], pairs[:, 1]
    ptr = np.r_[0, np.cumsum(np.bincount(pre, minlength=n))].astype(np.int64)
    weight = (rng.integers(1, 50, len(post)) * rng.choice([-1, 1], len(post)) * 0.275).astype(np.float32)
    return ptr, post.astype(np.int32), weight, types


def edge_pre(ptr):
    return np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))


def test_degrees_preserved_and_wiring_changed():
    ptr, post, _, types = synthetic()
    new, report = scramble(ptr, post, types, seed=1, log=lambda *_: None)
    n = len(ptr) - 1
    assert np.array_equal(np.bincount(new, minlength=n), np.bincount(post, minlength=n))
    assert report["scrambled_fraction_with_new_target"] > 0.9
    assert report["duplicate_pairs_after"] == 0
    key = edge_pre(ptr) * n + new
    assert len(np.unique(key)) == len(key)
    assert new.dtype == np.int32 and new.min() >= 0 and new.max() < n


def test_protected_edges_unchanged():
    ptr, post, _, types = synthetic()
    new, report = scramble(ptr, post, types, seed=2, log=lambda *_: None)
    pre = edge_pre(ptr)
    photo = np.isin(types, ["R1-R6", "R8p"])
    kc = np.char.startswith(types.astype(str), "KC")
    mbon = np.isin(types, ["MBON07", "MBON11"])
    dan = np.isin(types, ["PAM11", "PPL101"])
    keep = photo[pre] | (mbon[post] & (kc[pre] | dan[pre]))
    assert keep.sum() == report["protected_edges"]
    assert np.array_equal(new[keep], post[keep])
    # Stonkfly's plastic edges are found by edge position; positions must match.
    plastic_before = np.flatnonzero(mbon[post] & kc[pre])
    plastic_after = np.flatnonzero(mbon[new] & kc[pre])
    assert np.array_equal(plastic_before, plastic_after)


def test_sign_pools_keep_excitatory_and_inhibitory_in_degree():
    ptr, post, weight, types = synthetic()
    n = len(ptr) - 1
    inhibitory = weight < 0
    new, report = scramble(ptr, post, types, seed=5, signs=weight, log=lambda *_: None)
    for pool in (inhibitory, ~inhibitory):
        assert np.array_equal(
            np.bincount(post[pool], minlength=n), np.bincount(new[pool], minlength=n)
        )
    assert report["sign_pools"].startswith("excitatory and inhibitory")
    assert report["duplicate_pairs_after"] == 0
    assert report["scrambled_fraction_with_new_target"] > 0.9
    # The weaker mixed control does move inhibitory inputs between neurons.
    mixed, mixed_report = scramble(ptr, post, types, seed=5, log=lambda *_: None)
    assert mixed_report["sign_pools"].startswith("mixed")
    assert not np.array_equal(
        np.bincount(post[inhibitory], minlength=n),
        np.bincount(mixed[inhibitory], minlength=n),
    )


def test_keep_type_and_seed_behaviour():
    ptr, post, _, types = synthetic()
    pre = edge_pre(ptr)
    kept, _ = scramble(ptr, post, types, seed=3, keep_types=["DNp20"], log=lambda *_: None)
    dn = types[pre] == "DNp20"
    assert np.array_equal(kept[dn], post[dn])
    a, _ = scramble(ptr, post, types, seed=3, log=lambda *_: None)
    b, _ = scramble(ptr, post, types, seed=3, log=lambda *_: None)
    c, _ = scramble(ptr, post, types, seed=4, log=lambda *_: None)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_dataset_written_and_accepted_by_verify_lock(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    feather = pytest.importorskip("pyarrow.feather")
    ptr, post, weight, types = synthetic(n=800, edges=8000)
    src = tmp_path / "data"
    (src / "normalized").mkdir(parents=True)
    ids = np.arange(1, len(ptr), dtype=np.int64) * 10
    np.savez(src / "graph.npz", ptr=ptr, post=post, weight=weight, ids=ids,
             retina=np.arange(5, dtype=np.int32))
    feather.write_feather(pd.DataFrame({"source_id": ids, "cell_type": types}),
                          src / "normalized/neurons.feather")
    feather.write_feather(pd.DataFrame({"bodyId": ids}), src / "annotations.feather")

    out = tmp_path / "scrambled"
    lock = write_scrambled_dataset(src, out, seed=7)
    assert lock["summary"]["sign_pools"].startswith("excitatory and inhibitory")
    with np.load(out / "graph.npz") as g:
        assert np.array_equal(g["ptr"], ptr) and np.array_equal(g["weight"], weight)
        assert not np.array_equal(g["post"], post)
        assert digest(g["post"]) == lock["scrambled_arrays"]["post"]
        inhibitory = weight < 0
        assert np.array_equal(
            np.bincount(g["post"][inhibitory], minlength=len(ptr) - 1),
            np.bincount(post[inhibitory], minlength=len(ptr) - 1),
        )
    assert (out / "annotations.feather").is_symlink()
    assert (out / "normalized/neurons.feather").is_symlink()

    import stonkfly.data as data

    monkeypatch.setattr(data, "DATA", out)
    official = {k: digest(a) for k, a in [("ptr", ptr), ("post", post), ("weight", weight)]}
    expected = dict(official)
    accepted = data.scramble_lock(expected)
    assert accepted["summary"]["seed"] == 7
    assert expected["post"] == lock["scrambled_arrays"]["post"]

    # A scrambled graph built from some other, unlocked graph is rejected.
    tampered = dict(official, post="0" * 64)
    with pytest.raises(RuntimeError):
        data.scramble_lock(tampered)

    # Refuses to scramble an already-scrambled directory.
    with pytest.raises(SystemExit):
        write_scrambled_dataset(out, tmp_path / "again", seed=1)


def test_no_lock_means_no_change(tmp_path, monkeypatch):
    import stonkfly.data as data

    monkeypatch.setattr(data, "DATA", tmp_path)
    expected = {"post": "abc"}
    assert data.scramble_lock(expected) is None
    assert expected == {"post": "abc"}
