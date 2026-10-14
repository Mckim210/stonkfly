"""Random-trader control: matched proposals, resumability, outcomes and p-values."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stonkfly.config import D, Settings
from stonkfly.control import (
    RandomController,
    compare,
    matched_proposals,
    outcome,
    parse_seeds,
    summary,
)
from stonkfly.ledger import Ledger
from stonkfly.replay import file_sha256

SIDES = ["BUY"] * 7 + ["SELL"] * 5 + ["HOLD"] * 28
ROOT = str(Path(__file__).resolve().parents[1])


def cli_env():
    """Subprocesses import this checkout even when it is not pip-installed."""
    return {**os.environ, "PYTHONPATH": os.pathsep.join(
        x for x in [ROOT, os.environ.get("PYTHONPATH")] if x)}


def fake_fly_run(path, tape, sides=SIDES, policy="fly connectome"):
    """A finished fly run's records (the connectome itself is not needed here)."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "provenance.json").write_text(json.dumps({
        "policy": policy,
        "settings": {"products": ["BTC-USDC"]},
        "tape": {"path": str(tape), "sha256": file_sha256(tape)},
    }))
    with (path / "events.jsonl").open("w") as f:
        for side in sides:
            f.write(json.dumps({"neural": {"side": side}}) + "\n")
    return path


def test_shuffle_keeps_exact_proposal_counts():
    a = RandomController(1, sequence=SIDES)
    b = RandomController(1, sequence=SIDES)
    c = RandomController(2, sequence=SIDES)
    assert sorted(a.sequence) == sorted(SIDES)
    assert a.sequence == b.sequence and a.sequence != c.sequence
    sides = [a.observe(None, "none")["side"] for _ in SIDES]
    assert sides == a.sequence and a.done()
    with pytest.raises(RuntimeError):
        a.observe(None, "none")


def test_rates_and_validation():
    r = RandomController(3, rates=(0.3, 0.2))
    sides = [r.observe(None, "reward")["side"] for _ in range(20000)]
    assert abs(sides.count("BUY") / 20000 - 0.3) < 0.02
    assert abs(sides.count("SELL") / 20000 - 0.2) < 0.02
    for bad in [dict(), dict(sequence=SIDES, rates=(0.1, 0.1)), dict(rates=(0.7, 0.4)),
                dict(rates=(-0.1, 0.2))]:
        with pytest.raises(ValueError):
            RandomController(1, **bad)


@pytest.mark.parametrize("kind", ["sequence", "rates"])
def test_checkpoint_resumes_identically(tmp_path, kind):
    args = {"sequence": SIDES} if kind == "sequence" else {"rates": (0.25, 0.25)}
    full = RandomController(9, **args)
    expected = [full.observe(None, "none")["side"] for _ in range(30)]
    first = RandomController(9, **args)
    got = [first.observe(None, "none")["side"] for _ in range(12)]
    first.save(tmp_path / "brain-0.npz")
    resumed = RandomController(9, **args)
    resumed.restore(tmp_path / "brain-0.npz")
    got += [resumed.observe(None, "none")["side"] for _ in range(18)]
    assert got == expected
    with pytest.raises(ValueError):
        RandomController(10, **args).restore(tmp_path / "brain-0.npz")


def test_matched_proposals_need_same_tape_and_a_fly(tmp_path, tape):
    run = fake_fly_run(tmp_path / "fly", tape)
    assert matched_proposals(run, file_sha256(tape)) == SIDES
    with pytest.raises(SystemExit):
        matched_proposals(run, "0" * 64)
    with pytest.raises(SystemExit):
        matched_proposals(fake_fly_run(tmp_path / "r", tape, policy="RANDOM control"))


def test_compare_percentile_and_p_value():
    r = compare(1.0, [-1, 0, 0.5, 1.0, 2.0])
    assert r["percentile"] == 60.0
    assert r["random_at_least_as_good"] == 2
    assert r["p_value"] == pytest.approx(3 / 6)
    assert compare(9, np.zeros(999))["p_value"] == pytest.approx(1 / 1000)


def test_outcome_marks_holdings_to_last_bid(tmp_path):
    run = tmp_path / "run"
    s = Settings()
    ledger = Ledger(run / "ledger.sqlite", s, "paper")
    plan = ledger.reserve({"product": "BTC-USDC", "side": "BUY", "base_size": "0.001",
                           "limit_price": "10000", "fee_ceiling": "1"}, 0)
    ledger.settle(plan["client_order_id"], "0.001", "9", "0.054")
    ledger.close()
    with (run / "events.jsonl").open("w") as f:
        for bid, side, status in [("9000", "BUY", "FILLED"), ("9500", "HOLD", "HOLD")]:
            f.write(json.dumps({"product": "BTC-USDC", "quote": {"bid": bid},
                                "neural": {"side": side}, "execution": {"status": status}}) + "\n")
    r = outcome(run)
    expected = D(100) - D(9) - D("0.054") + D("0.001") * D(9500)
    assert r["final_equity"] == pytest.approx(float(expected))
    assert r["pnl"] == pytest.approx(float(expected - 100))
    assert (r["observations"], r["buy_proposals"], r["fills"]) == (2, 1, 1)


def test_parse_seeds():
    assert parse_seeds("1-3,7") == [1, 2, 3, 7]
    with pytest.raises(SystemExit):
        parse_seeds("1-3,2")


def test_cli_keeps_controls_off_live_and_needs_a_tape(tmp_path):
    base = [sys.executable, "-m", "stonkfly", "run", "--out", str(tmp_path / "x")]
    for extra in (["--live", "--replay", "t"], ["--live", "--policy", "random", "--seed", "1"],
                  ["--policy", "random", "--buy-rate", ".1", "--sell-rate", ".1"],
                  ["--replay", "t", "--fixture"], ["--match", "somewhere"]):
        r = subprocess.run(base + extra, capture_output=True, text=True, cwd=tmp_path,
                           env=cli_env())
        assert r.returncode == 2, (extra, r.stderr)
    assert not (tmp_path / "x").exists()


def test_end_to_end_random_controls_on_a_tape(tmp_path, tape, monkeypatch):
    """Full run loop with paper fills; skipped where AgentKit is not installed."""
    pytest.importorskip("coinbase_agentkit")
    from stonkfly.control import batch

    monkeypatch.setenv("PYTHONPATH", cli_env()["PYTHONPATH"])
    fly = fake_fly_run(tmp_path / "fly", tape, sides=(SIDES * 3)[:100])
    cmd = [sys.executable, "-m", "stonkfly", "run", "--replay", str(tape), "--policy",
           "random", "--match", str(fly), "--seed", "4", "--out", str(tmp_path / "one")]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, env=cli_env())
    assert r.returncode == 0, r.stderr
    rows = [json.loads(x) for x in (tmp_path / "one/events.jsonl").read_text().splitlines()]
    assert len(rows) == 100
    sides = [x["neural"]["side"] for x in rows]
    assert sorted(sides) == sorted((SIDES * 3)[:100])
    # Market time spaces observations 60 s apart, so cooldown never blocks a
    # proposal here even though the replay runs in well under a minute.
    assert not any("cooldown" in x["execution"].get("reason", "") for x in rows)
    assert sum(x["execution"]["status"] == "FILLED" for x in rows) > 0
    prov = json.loads((tmp_path / "one/provenance.json").read_text())
    assert prov["policy"].startswith("RANDOM control") and prov["dataset"] is None
    assert prov["tape"]["sha256"] == file_sha256(tape) and prov["feed"] == "replay"
    # Running again resumes a finished control without adding observations.
    again = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, env=cli_env())
    assert again.returncode == 0, again.stderr
    assert len((tmp_path / "one/events.jsonl").read_text().splitlines()) == 100

    batch(fly, [1, 2, 3], tmp_path / "random", jobs=2)
    report = summary([tmp_path / "one"], tmp_path / "random")
    assert report["same_tape"] and report["random"]["count"] == 3
    assert {r["observations"] for r in report["runs"]} == {100}
    assert 0 < report["runs"][0]["vs_random"]["p_value"] <= 1
