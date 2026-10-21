"""Learning controls: shuffled reinforcement, final memory record, day summary, report."""

import hashlib
import json
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest

from stonkfly.control import ShuffledReinforcement, delivered_signals, events
from stonkfly.learning import day, interval, report
from stonkfly.replay import file_sha256

from test_control import cli_env

SIGNALS = ["none"] * 30 + ["reward"] * 6 + ["aversive"] * 9


def test_shuffle_keeps_signal_counts_and_follows_the_seed():
    a = ShuffledReinforcement("runs/learn", SIGNALS, 1)
    assert Counter(a.sequence) == Counter(SIGNALS) and a.sequence != SIGNALS
    assert a.sequence == ShuffledReinforcement("x", SIGNALS, 1).sequence
    assert a.sequence != ShuffledReinforcement("x", SIGNALS, 2).sequence
    assert a.counts() == {"none": 30, "reward": 6, "aversive": 9}
    assert [a.signal(i) for i in range(len(SIGNALS))] == a.sequence
    assert not a.done(len(SIGNALS) - 1) and a.done(len(SIGNALS))
    a.check([{"tick": i + 1, "reinforcement_delivered": s} for i, s in enumerate(a.sequence)])
    wrong = [{"tick": i + 1, "reinforcement_delivered": s} for i, s in enumerate(SIGNALS)]
    with pytest.raises(RuntimeError):
        a.check(wrong)


def fake_run(path, tape, signals, policy="fly connectome"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "provenance.json").write_text(json.dumps({
        "policy": policy, "tape": {"path": str(tape), "sha256": file_sha256(tape)}}))
    with (path / "events.jsonl").open("w") as f:
        for s in signals:
            f.write(json.dumps({"reinforcement_delivered": s, "neural": {"side": "HOLD"}}) + "\n")
    return path


def test_delivered_signals_need_a_recorded_fly_run_on_the_same_tape(tmp_path, tape):
    run = fake_run(tmp_path / "learn", tape, SIGNALS)
    assert delivered_signals(run, file_sha256(tape)) == SIGNALS
    with pytest.raises(SystemExit):
        delivered_signals(run, "0" * 64)
    with pytest.raises(SystemExit):
        delivered_signals(fake_run(tmp_path / "r", tape, SIGNALS, "RANDOM control"))
    with pytest.raises(SystemExit):
        delivered_signals(fake_run(tmp_path / "old", tape, [None] * 3))


def test_cli_rejects_unsafe_or_incomplete_shuffle_flags(tmp_path):
    base = [sys.executable, "-m", "stonkfly", "run", "--out", str(tmp_path / "x")]
    for extra in (["--live", "--shuffle-reinforcement", "a", "--shuffle-seed", "1"],
                  ["--replay", "t", "--shuffle-reinforcement", "a"],
                  ["--replay", "t", "--shuffle-seed", "1"],
                  ["--shuffle-reinforcement", "a", "--shuffle-seed", "1"],
                  ["--replay", "t", "--policy", "random", "--seed", "1", "--match", "m",
                   "--shuffle-reinforcement", "a", "--shuffle-seed", "1"]):
        r = subprocess.run(base + extra, capture_output=True, text=True, cwd=tmp_path,
                           env=cli_env())
        assert r.returncode == 2, (extra, r.stderr)
    assert not (tmp_path / "x").exists()


class StubBrain:
    def __init__(self):
        self.w = np.zeros(4)
        self.sim_ms = 0.0
        self.circuit = {"report": {"stub": "no connectome"}}
        self.visual_report = {"stub": "no connectome"}

    def memory(self):
        return {"plastic_edges": 4, "changed_edges": int(np.count_nonzero(self.w)),
                "mean_efficacy": float(1 + self.w.mean()),
                "sha256": hashlib.sha256(self.w.tobytes()).hexdigest()}


class StubFly:
    """Deterministic stand-in for FlyController: pulses change memory, memory changes proposals."""

    ignores_freeze = False

    def __init__(self, settings):
        self.s = settings
        self.brain = StubBrain()

    def observe(self, rgb, reinforcement):
        b = self.brain
        b.sim_ms += self.s.neural_ms
        if reinforcement != "none" and (self.s.learning or self.ignores_freeze):
            b.w[0 if reinforcement == "reward" else 1] += 0.01
        h = hashlib.sha256(np.asarray(rgb).tobytes() + b.w.tobytes()).digest()
        return {"side": ("BUY", "SELL", "HOLD", "HOLD")[h[0] % 4], "brain_ms": b.sim_ms,
                "total_spikes": 100 + h[1], "KC_spikes": h[2] % 20, "spike_sha256": h.hex(),
                "stimulus": reinforcement, "memory": b.memory()}

    def save(self, path):
        with open(path, "wb") as f:
            np.savez(f, w=self.brain.w, ms=self.brain.sim_ms)

    def restore(self, path):
        with np.load(path) as a:
            self.brain.w[:] = a["w"]
            self.brain.sim_ms = float(a["ms"])


class CheatingFly(StubFly):
    ignores_freeze = True


@pytest.fixture
def stub_cli(monkeypatch, tmp_path):
    pytest.importorskip("coinbase_agentkit")
    import stonkfly.data
    import stonkfly.neural.controller as controller
    from stonkfly import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stonkfly.data, "verify", lambda: {"stub": "no connectome"})

    def run(*args, fly=StubFly):
        monkeypatch.setattr(controller, "FlyController", fly)
        monkeypatch.setattr(sys, "argv", ["stonkfly", "run", *map(str, args)])
        cli.main()

    return run


def test_learn_frozen_shuffled_and_day_summary(tmp_path, tape, stub_cli, monkeypatch):
    from stonkfly.control import batch

    root = tmp_path / "runs/day"
    stub_cli("--replay", tape, "--out", root / "learn")
    stub_cli("--replay", tape, "--frozen", "--out", root / "frozen")
    stub_cli("--replay", tape, "--out", root / "shuffled",
             "--shuffle-reinforcement", root / "learn", "--shuffle-seed", 1)
    learn, frozen, shuffled = (events(root / arm) for arm in ("learn", "frozen", "shuffled"))
    assert len(learn) == len(frozen) == len(shuffled) == 120
    # Normal runs deliver the signal their own P&L earned, and it reaches the brain.
    for rows in (learn, frozen):
        assert all(r["reinforcement_delivered"] == r["reinforcement_earned"]
                   == r["neural"]["stimulus"] for r in rows)
    sent = [r["reinforcement_delivered"] for r in learn]
    assert {"reward", "aversive"} <= set(sent)
    # The shuffled run gets learn's exact signals in seeded order, not its own P&L.
    expected = ShuffledReinforcement(root / "learn", sent, 1).sequence
    got = [r["reinforcement_delivered"] for r in shuffled]
    assert got == expected and Counter(got) == Counter(sent) and got != sent
    assert [r["neural"]["stimulus"] for r in shuffled] == got
    prov = json.loads((root / "shuffled/provenance.json").read_text())
    assert prov["reinforcement"]["seed"] == 1
    final = {arm: json.loads((root / arm / "final.json").read_text())
             for arm in ("learn", "frozen", "shuffled")}
    assert final["frozen"]["memory"]["changed_edges"] == 0
    assert final["learn"]["memory"]["changed_edges"] > 0
    assert final["learn"]["memory"] == learn[-1]["neural"]["memory"]
    assert final["shuffled"]["signals_delivered"] == final["learn"]["signals_delivered"]
    # Rerunning a finished run adds nothing and rewrites the same final record.
    stub_cli("--replay", tape, "--out", root / "learn")
    assert len(events(root / "learn")) == 120
    assert json.loads((root / "learn/final.json").read_text()) == final["learn"]

    monkeypatch.setenv("PYTHONPATH", cli_env()["PYTHONPATH"])
    batch(root / "learn", [1, 2, 3], root / "random", jobs=2)
    summary = day(root, tmp_path / "results/day.json")
    assert all(summary["checks"].values()), summary["checks"]
    assert summary["random"]["count"] == 3 and summary["random"]["seeds"] == [1, 2, 3]
    assert 1 <= summary["learn_vs_random"]["rank"] <= 4
    assert summary["learn_minus_frozen"] == pytest.approx(
        summary["runs"]["learn"]["pnl"] - summary["runs"]["frozen"]["pnl"])
    assert summary["runs"]["learn"]["first_pulse_observation"] >= 2
    # Learning on or off is identical until the first pulse can change memory.
    assert summary["divergence_from_learn"]["frozen"]["first_spike_difference_observation"] \
        >= summary["runs"]["learn"]["first_pulse_observation"]
    assert json.loads((tmp_path / "results/day.json").read_text())["date"] == "day"


def test_frozen_run_that_changes_memory_stops(tmp_path, tape, stub_cli):
    with pytest.raises(SystemExit):
        stub_cli("--replay", tape, "--frozen", "--steps", 30, "--out", tmp_path / "f",
                 fly=CheatingFly)
    error = json.loads((tmp_path / "f/error.json").read_text())
    assert error["reason"] == "Frozen run changed plastic synapses"
    assert not (tmp_path / "f/final.json").exists()


def fake_day(date, learn, frozen, shuffled, rank):
    def arm(pnl):
        return {"pnl": pnl, "buy_proposals": 5, "sell_proposals": 2, "hold_proposals": 173,
                "fills": 7, "vetoes": 0, "mean_total_spikes": 1e5, "mean_kc_spikes": 12.0,
                "signals_delivered": {"none": 170, "reward": 4, "aversive": 6},
                "memory": {"changed_edges": 0, "mean_efficacy": 1.0},
                "first_pulse_observation": 2, "first_memory_change_observation": 3}
    divergence = {"first_spike_difference_observation": 3,
                  "first_proposal_difference_observation": None,
                  "observations_with_different_proposal": 0}
    return {"date": date, "runs": {"learn": arm(learn), "frozen": arm(frozen),
                                   "shuffled": arm(shuffled)},
            "learn_minus_frozen": learn - frozen, "learn_minus_shuffled": learn - shuffled,
            "divergence_from_learn": {"frozen": divergence, "shuffled": divergence},
            "random": {"mean_pnl": -0.1, "p5_pnl": -0.2, "p95_pnl": 0.0},
            "learn_vs_random": {"p_value": rank / 1001, "rank": rank, "of": 1001},
            "checks": {"same_tape": True}}


def test_interval_uses_t_with_nine_degrees_of_freedom():
    r = interval(range(1, 11))
    assert r["mean"] == 5.5 and r["sd"] == pytest.approx(3.0276504)
    assert r["ci95"] == pytest.approx([5.5 - 2.1658506, 5.5 + 2.1658506])
    assert (r["days_better"], r["days_equal"], r["days_worse"]) == (10, 0, 0)


def write_days(tmp_path, frozen_diffs, shuffled_diffs):
    files = []
    for i, (f, s) in enumerate(zip(frozen_diffs, shuffled_diffs)):
        d = fake_day(f"2026-10-{12 + i}", 0.0, -f, -s, rank=100 + i)
        files.append(tmp_path / f"{d['date']}.json")
        files[-1].write_text(json.dumps(d))
    return files


def test_report_verdict_needs_both_intervals_above_zero(tmp_path):
    up = [0.1, 0.2, 0.15, 0.1, 0.3, 0.2, 0.1, 0.25, 0.2, 0.1]
    mixed = [0.1, -0.2, 0.15, -0.1, 0.3, -0.2, 0.1, -0.25, 0.2, -0.1]
    r = report(write_days(tmp_path, up, up), md=tmp_path / "summary.md")
    assert r["verdict"] == "학습이 성과를 개선했다"
    assert r["learn_vs_random"]["mean_rank"] == pytest.approx(104.5)
    assert set(r["exploratory_fills"]) == {"frozen", "shuffled"}
    assert "학습이 성과를 개선했다" in (tmp_path / "summary.md").read_text()
    r = report(write_days(tmp_path, up, mixed))
    assert r["verdict"] == "개선 증거 없음" and r["learn_minus_frozen"]["ci95"][0] > 0
    zeros = [0.0] * 10
    r = report(write_days(tmp_path, zeros, zeros))
    assert r["verdict"] == "개선 증거 없음" and r["exploratory_fills"] == {}
    assert r["learn_minus_frozen"]["days_equal"] == 10
    with pytest.raises(SystemExit):
        report(write_days(tmp_path, up, up)[:9])
