"""Market tapes: recording, candle rebuilds and paper replay with a market clock."""

import json

import pytest

from stonkfly.config import D, Settings
from stonkfly.ledger import Ledger
from stonkfly.market import Quote
from stonkfly.replay import ReplayMarket, TapeEnd, from_candles, load, record
from stonkfly.risk import Guard, Veto

PRODUCT = {
    "product_id": "BTC-USDC",
    "product_type": "SPOT",
    "quote_currency_id": "USDC",
    "base_increment": ".00000001",
    "quote_increment": ".01",
    "quote_min_size": "1",
    "base_min_size": ".00000001",
}


class Candles:
    """Public-candle double: one candle per minute except a missing minute."""

    def __init__(self, missing=()):
        self.missing = set(missing)
        self.calls = []

    def get_public_candles(self, product, start, end, granularity, limit):
        self.calls.append((int(start), int(end), limit))
        a, b = int(start), int(end)
        rows = [
            {"start": str(t), "close": str(100 + (t // 60) % 7)}
            for t in range(a, b, 60)
            if t not in self.missing
        ]
        # Out-of-window rows must be ignored, never used.
        rows.append({"start": str(b), "close": "999999"})
        return {"candles": rows[::-1]}

    def get_public_product(self, product):
        return dict(PRODUCT, product_id=product)


def test_candle_tape_has_no_future_and_skips_gaps(tmp_path):
    start, end = 1_790_000_040 // 60 * 60, 1_790_000_040 // 60 * 60 + 600 * 60
    client = Candles(missing={start + 5 * 60})
    n = from_candles(("BTC-USDC",), start, end, tmp_path / "t.jsonl", "0.0001", client)
    header, rows = load(tmp_path / "t.jsonl")
    assert n == len(rows) == 599
    assert header["skipped_minutes"] == 1 and header["half_spread"] == "0.0001"
    assert len(header["seed_history"]["BTC-USDC"]) == 120
    assert all(lim <= 300 for _, _, lim in client.calls)
    first = rows[0]
    assert first["time"] == start + 60  # observed at the first candle's close
    q = first["quotes"]["BTC-USDC"]
    close = D(100 + (start // 60) % 7)
    assert D(q["bid"]) == close * D(".9999") and D(q["ask"]) == close * D("1.0001")
    assert all(r["quotes"]["BTC-USDC"]["bid"] != "999999" for r in rows)
    assert start + 5 * 60 + 60 not in [r["time"] for r in rows]
    with pytest.raises(SystemExit):
        from_candles(("BTC-USDC",), start, end, tmp_path / "t.jsonl", "0.0001", client)
    with pytest.raises(ValueError):
        from_candles(("BTC-USDC",), start, end, tmp_path / "u.jsonl", "0.01", client)


class FlakyMarket:
    def __init__(self):
        self.history = {"BTC-USDC": [1.0, 2.0]}
        self.calls = 0

    def snapshot(self):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("network")
        return {"BTC-USDC": Quote("BTC-USDC", D(100), D("100.1"), 50.0 + self.calls,
                                  D(".00000001"), D(".01"), D(".01"), D(1), D(".00000001"))}


def test_live_record_never_fabricates_missing_observations(tmp_path):
    clock = iter(range(1000, 2000, 7)).__next__
    waits = []
    n = record(("BTC-USDC",), tmp_path / "live.jsonl", 3, 60, FlakyMarket(),
               sleep=waits.append, clock=lambda: float(clock()))
    header, rows = load(tmp_path / "live.jsonl")
    assert n == 3 and len(rows) == 3
    assert header["seed_history"] == {"BTC-USDC": [1.0, 2.0]}
    assert [r["quotes"]["BTC-USDC"]["timestamp"] for r in rows] == [51.0, 53.0, 54.0]
    assert all(0 <= w <= 60 for w in waits)


def test_replay_sequence_clock_and_end(tape):
    m = ReplayMarket(tape, ("BTC-USDC",))
    header, rows = load(tape)
    assert m.identity["observations"] == len(rows) and len(m.identity["sha256"]) == 64
    assert m.history["BTC-USDC"] == header["seed_history"]["BTC-USDC"]
    with pytest.raises(RuntimeError):
        m.clock()
    first = m.snapshot()
    assert m.clock() == rows[0]["time"]
    # Execution re-reads the same observation; it never advances the tape.
    assert m.execution_snapshot() == first and m.tick == 1
    for _ in rows[1:]:
        m.snapshot()
    with pytest.raises(TapeEnd):
        m.snapshot()
    with pytest.raises(SystemExit):
        ReplayMarket(tape, ("ETH-USDC",))


def test_unordered_tape_rejected(tape, tmp_path):
    lines = open(tape).read().splitlines()
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join([lines[0], lines[2], lines[1]]) + "\n")
    with pytest.raises(ValueError):
        load(bad)


def test_guard_uses_market_time_not_wall_time(tape, tmp_path):
    s = Settings()
    m = ReplayMarket(tape, ("BTC-USDC",))
    ledger = Ledger(tmp_path / "ledger.sqlite", s, "paper")
    guard = Guard(s, ledger, tmp_path / "STOP", clock=m.clock)
    try:
        quotes = m.snapshot()
        plan = guard.plan("BTC-USDC", "BUY", quotes)
        ledger.reserve(plan, guard.clock())
        ledger.mark(ledger.pending()[0]["id"], "REJECTED")
        # Next observation is 60 s later in market time, however fast we replay.
        quotes = m.snapshot()
        assert guard.plan("BTC-USDC", "BUY", quotes)["quote_timestamp"] == m.now - 0.4
        # Same market instant again: cooldown applies in market time.
        ledger.reserve(guard.plan("BTC-USDC", "BUY", quotes), guard.clock())
        ledger.mark(ledger.pending()[0]["id"], "REJECTED")
        with pytest.raises(Veto, match="cooldown"):
            guard.plan("BTC-USDC", "BUY", quotes)
    finally:
        ledger.close()


def test_default_guard_still_uses_wall_clock(tmp_path, monkeypatch):
    import stonkfly.risk as risk

    s = Settings()
    ledger = Ledger(tmp_path / "ledger.sqlite", s, "paper")
    try:
        monkeypatch.setattr(risk.time, "time", lambda: 12345.0)
        assert Guard(s, ledger, tmp_path / "STOP").clock() == 12345.0
    finally:
        ledger.close()


def test_tape_lines_are_plain_json(tape):
    for line in open(tape):
        json.loads(line)
