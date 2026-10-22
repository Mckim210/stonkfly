"""Trade-level reinforcement: FIFO realized P&L, signals, buy-chart replay, pilot report."""

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from stonkfly.config import D, Settings
from stonkfly.control import events
from stonkfly.display import market_frame
from stonkfly.ledger import Ledger
from stonkfly.roundtrip import (DESCRIPTION, OWN, buy_frame, jaccard, new_state, pilot, signal,
                                sync)

from test_control import cli_env
from test_learning import StubFly, stub_cli  # noqa: F401  (stub_cli is a fixture)


@pytest.fixture
def ledger(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite", Settings(), "paper")
    yield ledger
    ledger.close()


def fill(ledger, side, base, price, fee, observation=None):
    plan = ledger.reserve({"product": "BTC-USDC", "side": side, "base_size": base,
                           "limit_price": price, "fee_ceiling": "1",
                           "neural_observation": observation}, 0)
    ledger.settle(plan["client_order_id"], base, D(base) * D(price), fee)
    return plan["client_order_id"]


def test_fifo_realized_pnl_counts_both_fees_across_lots(ledger):
    state = new_state()
    a = fill(ledger, "BUY", "0.0001", "50000", "0.03")
    state, sale = sync(ledger.db, state, 1)
    assert sale is None and signal(sale) == "none"
    b = fill(ledger, "BUY", "0.0002", "40000", "0.048")
    state, _ = sync(ledger.db, state, 3)
    fill(ledger, "SELL", "0.00015", "60000", "0.054")
    state, sale = sync(ledger.db, state, 5)
    # Lot a whole (cost 5, fee .03) plus a quarter of lot b (cost 2, fee .012).
    assert D(sale["before_fees"]) == D("2")
    assert D(sale["after_fees"]) == D("2") - D(".03") - D(".012") - D(".054")
    assert [(m["buy_order"], m["buy_observation"], D(m["base"])) for m in sale["matched"]] \
        == [(a, 1, D("0.0001")), (b, 3, D("0.00005"))]
    assert sale["sell_observation"] == 5 and signal(sale) == "reward"
    assert sync(ledger.db, state, 6) == (state, None)  # nothing new: no second signal
    fill(ledger, "SELL", "0.00015", "39000", "0.0351")
    state, sale = sync(ledger.db, state, 7)
    assert D(sale["before_fees"]) == D("5.85") - D("6")
    assert D(sale["after_fees"]) == D("-0.15") - D(".036") - D(".0351")
    assert sale["matched"][0]["buy_observation"] == 3 and signal(sale) == "aversive"
    assert state["lots"]["BTC-USDC"] == []
    assert signal({"after_fees": "0"}) == "none"


def test_sync_skips_rejections_and_refuses_unclear_books(ledger):
    plan = ledger.reserve({"product": "BTC-USDC", "side": "BUY", "base_size": "1",
                           "limit_price": "1", "fee_ceiling": "1"}, 0)
    ledger.mark(plan["client_order_id"], "REJECTED")
    state, sale = sync(ledger.db, new_state(), 1)
    assert state["orders_seen"] == 1 and sale is None and state["lots"] == {}
    ledger.reserve({"product": "BTC-USDC", "side": "BUY", "base_size": "1",
                    "limit_price": "1", "fee_ceiling": "1"}, 0)
    with pytest.raises(RuntimeError, match="Unreconciled"):
        sync(ledger.db, state, 2)
    other = Ledger(ledger.path.with_name("other.sqlite"), Settings(), "paper")
    fill(other, "BUY", "0.0001", "50000", "0.03")
    fill(other, "BUY", "0.0001", "50000", "0.03")
    with pytest.raises(RuntimeError, match="More than one"):
        sync(other.db, new_state(), 1)
    third = Ledger(ledger.path.with_name("third.sqlite"), Settings(), "paper")
    third.put("positions", {"BTC-USDC": "1"})  # holdings the FIFO book never saw bought
    fill(third, "SELL", "0.0001", "50000", "0.03")
    with pytest.raises(RuntimeError, match="exceeds"):
        sync(third.db, new_state(), 1)
    other.close()
    third.close()


def test_buy_frame_is_the_chart_seen_at_the_buy(ledger):
    history = [60000.0 + i for i in range(120)]
    frame = market_frame("BTC-USDC", history, D("60100.5"), D("60101.5"))
    observation = {"product": "BTC-USDC", "quote": {"bid": "60100.5", "ask": "60101.5"},
                   "market_history": {"BTC-USDC": history},
                   "neural": {"input_sha256": hashlib.sha256(frame.tobytes()).hexdigest()}}
    good = fill(ledger, "BUY", "0.0001", "50000", "0.03", observation)
    assert np.array_equal(buy_frame(ledger.db, good), frame)
    observation["neural"]["input_sha256"] = "0" * 64
    bad = fill(ledger, "BUY", "0.0001", "50000", "0.03", observation)
    with pytest.raises(RuntimeError, match="reproduce"):
        buy_frame(ledger.db, bad)


def test_jaccard():
    assert jaccard([1, 2, 3], [2, 3, 4]) == 0.5
    assert jaccard([1], [1]) == 1.0 and jaccard([1], [2]) == 0.0
    assert jaccard([], []) is None


def test_cli_keeps_roundtrip_reinforcement_a_paper_fly_experiment(tmp_path):
    base = [sys.executable, "-m", "stonkfly", "run", "--out", str(tmp_path / "x")]
    for extra in (["--live", "--reinforcement", "roundtrip"],
                  ["--reinforcement", "roundtrip-replay", "--replay", "t", "--policy", "random",
                   "--seed", "1", "--match", "m"],
                  ["--reinforcement", "roundtrip", "--replay", "t",
                   "--shuffle-reinforcement", "a", "--shuffle-seed", "1"],
                  ["--reinforcement", "trade"]):
        r = subprocess.run(base + extra, capture_output=True, text=True, cwd=tmp_path,
                           env=cli_env())
        assert r.returncode == 2, (extra, r.stderr)
    assert not (tmp_path / "x").exists()


class TradeFly(StubFly):
    """StubFly plus the replay step; its KC set depends only on the frame shown."""

    def observe(self, rgb, reinforcement):
        self.frame = np.asarray(rgb)
        r = super().observe(rgb, reinforcement)
        r["input_sha256"] = hashlib.sha256(self.frame.tobytes()).hexdigest()
        return r

    def replay(self, rgb, reinforcement):
        r = self.observe(rgb, reinforcement)
        del r["side"]
        return r

    def kc_active(self):
        h = hashlib.sha256(self.frame.tobytes()).digest()
        return sorted(set(h[:12]))

    def plastic_weights(self):
        return self.brain.w.copy()


def fifo_from_executions(rows):
    """Independent FIFO over the paper fills recorded in the event log."""
    lots, sales = [], {}
    for r in rows:
        x = r["execution"]
        if x["status"] != "FILLED":
            continue
        base, quote, fee = D(x["base"]), D(x["quote"]), D(x["fee"])
        if r["neural"]["side"] == "BUY":
            lots.append([r["tick"], base, quote, fee, base])
            continue
        left, paid, first = base, D(0), lots[0][0]
        while left:
            lot = lots[0]
            take = min(left, lot[4])
            paid += (lot[2] + lot[3]) * take / lot[1]
            lot[4] -= take
            left -= take
            if not lot[4]:
                lots.pop(0)
        sales[r["tick"]] = (quote - fee - paid, first)
    return sales


def test_roundtrip_signals_follow_realized_pnl_and_replay_the_first_buy(tmp_path, tape, stub_cli):
    root = tmp_path / "runs"
    stub_cli("--replay", tape, "--reinforcement", "roundtrip-replay", "--out", root / "rr",
             fly=TradeFly)
    rows = events(root / "rr")
    assert len(rows) == 120  # the replay step adds no observation or proposal
    sales = fifo_from_executions(rows)
    assert len(sales) >= 3
    by_tick = {r["tick"]: r for r in rows}
    replays = 0
    for r in rows:
        rt, previous = r["roundtrip"], r["tick"] - 1
        assert r["reinforcement_delivered"] == r["reinforcement_earned"] == r["neural"]["stimulus"]
        if previous not in sales:
            assert rt["sale"] is None and rt["replay"] is None
            assert r["reinforcement_delivered"] == "none"
            continue
        pnl, first = sales[previous]
        replays += 1
        assert rt["sale"]["sell_observation"] == previous
        assert abs(D(rt["realized_pnl_after_fees"]) - pnl) < D("1e-20")
        assert D(rt["realized_pnl_before_fees"]) > D(rt["realized_pnl_after_fees"])
        assert r["reinforcement_delivered"] == ("reward" if pnl > 0 else "aversive")
        assert rt["buy_observation"] == rt["replay"]["buy_observation"] == first
        assert rt["replay"]["input_sha256"] == by_tick[first]["neural"]["input_sha256"]
        assert rt["replay"]["stimulus"] == r["reinforcement_delivered"]
        assert "side" not in rt["replay"]
        assert rt["replay"]["kc_active"] == by_tick[first]["roundtrip"]["kc_active"]
    assert rows[-1]["neural"]["brain_ms"] == 500 * (120 + replays)
    final = json.loads((root / "rr/final.json").read_text())["roundtrip"]
    delivered = [r["reinforcement_delivered"] for r in rows]
    assert final["trade_signals"] == {s: delivered.count(s) for s in ("reward", "aversive")}
    assert final["trade_pulses_including_replay"] == 2 * replays
    assert sum(final["sells_by_realized_sign"].values()) == len(sales)
    assert final["replay_kc_jaccard"]["mean"] == 1.0
    assert final["replay_kc_jaccard"]["count"] == replays
    # Plain roundtrip: same signals rule, no replay step.
    stub_cli("--replay", tape, "--reinforcement", "roundtrip", "--out", root / "rt", fly=TradeFly)
    plain = events(root / "rt")
    assert not any(r["roundtrip"]["replay"] for r in plain)
    assert plain[-1]["neural"]["brain_ms"] == 500 * 120


def test_roundtrip_resume_matches_an_uninterrupted_run(tmp_path, tape, stub_cli):
    stub_cli("--replay", tape, "--reinforcement", "roundtrip-replay", "--out", tmp_path / "a",
             fly=TradeFly)
    for steps in (37, 0):
        stub_cli("--replay", tape, "--reinforcement", "roundtrip-replay", "--steps", steps,
                 "--out", tmp_path / "b", fly=TradeFly)

    def strip(x):
        # Wall time and random client order IDs differ between any two runs.
        if isinstance(x, list):
            return [strip(v) for v in x]
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items()
                    if k not in ("wall_time", "sell_order", "buy_order")}
        return x

    assert strip(events(tmp_path / "a")) == strip(events(tmp_path / "b"))


def test_default_portfolio_run_is_unchanged_and_pilot_report(tmp_path, tape, stub_cli):
    root = tmp_path / "pilot"
    stub_cli("--replay", tape, "--out", root / "portfolio", fly=TradeFly)
    stub_cli("--replay", tape, "--frozen", "--out", root / "frozen", fly=TradeFly)
    for mode in ("roundtrip", "roundtrip-replay"):
        stub_cli("--replay", tape, "--reinforcement", mode, "--out", root / mode, fly=TradeFly)
    rows = events(root / "portfolio")
    assert not any("roundtrip" in r for r in rows)
    assert all(r["reinforcement_delivered"] == r["reinforcement_earned"] for r in rows)
    assert set(json.loads((root / "portfolio/final.json").read_text())) == {
        "tick", "learning", "memory", "signals_delivered"}
    prov = json.loads((root / "portfolio/provenance.json").read_text())
    assert prov["reinforcement"] == OWN
    prov = json.loads((root / "roundtrip-replay/provenance.json").read_text())
    assert prov["reinforcement"] == DESCRIPTION["roundtrip-replay"]

    report = pilot(root, tmp_path / "r.json", tmp_path / "r.md")
    assert all(report["checks"].values()), report["checks"]
    c = report["criteria"]
    assert c["3_replay_kc_jaccard"]["value"] == 1.0 and c["3_replay_kc_jaccard"]["pass"]
    signals = report["roundtrip_summaries"]["roundtrip-replay"]["trade_signals"]
    assert c["1_trade_signals"]["value"] == sum(signals.values())
    assert c["1_trade_signals"]["pass"] == (sum(signals.values()) >= 5)
    assert report["verdict"] == ("정식 실험 가능" if all(x["pass"] for x in c.values())
                                 else "정식 실험 안 함")
    assert len(report["failure_reasons"]) == sum(not x["pass"] for x in c.values())
    assert "## 판정" in (tmp_path / "r.md").read_text()
