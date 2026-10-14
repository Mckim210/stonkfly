"""Market tapes: observe once, replay the identical sequence to every run.

A tape is JSON Lines. Line 1 is a header; each later line is one observation:
{"time": <observation time>, "quotes": {product: Quote.json()}}.

The header holds the seed history (the past one-minute closes drawn in the
first chart) and how the tape was made. Replays are paper-only and use the
recorded observation time as the clock, so cooldowns, daily limits and
quote-age checks follow market time, not how fast the computer replays.

Two sources:
* live: polls the public Coinbase book exactly like a normal run (real
  bid/ask), one observation per interval. Takes real time.
* candles: rebuilds a past period from public one-minute candles. Candles have
  no order book, so bid/ask = close -/+ a declared half-spread. Instant, and
  lets you replay any past window or several held-out days.
"""

import hashlib
import json
import math
import sys
import time
from pathlib import Path

from .config import D
from .market import CoinbaseMarket, Quote, unwrap

TAPE_VERSION = 1
SEED_MINUTES = 120


class TapeEnd(Exception):
    """The replayed tape has no more observations."""


def quote_from_json(d):
    return Quote(
        d["product"],
        D(d["bid"]),
        D(d["ask"]),
        float(d["timestamp"]),
        D(d["base_increment"]),
        D(d["quote_increment"]),
        D(d["price_increment"]),
        D(d["minimum_quote"]),
        D(d["minimum_base"]),
    )


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _open_new(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return path.open("x")
    except FileExistsError:
        raise SystemExit(f"{path} already exists; tapes are never overwritten.")


def _line(f, value):
    f.write(json.dumps(value, allow_nan=False) + "\n")
    f.flush()


def record(products, path, observations, interval=60, market=None,
           sleep=time.sleep, clock=time.time, max_failures=10):
    """Poll the public book like a normal run and write every observation."""
    if observations < 1 or not math.isfinite(interval) or interval <= 0:
        raise ValueError("Positive observation count and interval required")
    market = market or CoinbaseMarket(products)
    written = failures = 0
    with _open_new(path) as f:
        while written < observations:
            started = clock()
            try:
                quotes = market.snapshot()
            except Exception as e:  # never fabricate a missing observation
                failures += 1
                print(f"observation skipped: {type(e).__name__}", file=sys.stderr, flush=True)
                if failures >= max_failures:
                    raise RuntimeError("Too many consecutive failed observations") from e
            else:
                failures = 0
                if written == 0:
                    _line(f, {
                        "tape": TAPE_VERSION,
                        "source": "live public Coinbase order book",
                        "products": list(products),
                        "interval_seconds": interval,
                        "seed_history": {p: market.history[p] for p in products},
                    })
                _line(f, {"time": clock(), "quotes": {p: q.json() for p, q in quotes.items()}})
                written += 1
                print(f"recorded {written}/{observations}", flush=True)
            if written < observations:
                sleep(max(0.0, started + interval - clock()))
    return written


def _product_quote_fields(client, product):
    m = unwrap(client.get_public_product(product))
    if (
        m.get("product_id") != product
        or m.get("product_type") != "SPOT"
        or m.get("quote_currency_id") != "USDC"
    ):
        raise RuntimeError("Unexpected product")
    return {
        "base_increment": m["base_increment"],
        "quote_increment": m["quote_increment"],
        "price_increment": m.get("price_increment", m["quote_increment"]),
        "minimum_quote": m["quote_min_size"],
        "minimum_base": m["base_min_size"],
    }


def _closes(client, product, start, end, chunk=300):
    """Completed one-minute closes keyed by candle start, start <= t < end."""
    closes = {}
    for a in range(start, end, chunk * 60):
        b = min(end, a + chunk * 60)
        candles = unwrap(
            client.get_public_candles(product, str(a), str(b), "ONE_MINUTE", limit=chunk)
        ).get("candles", [])
        for c in candles:
            t = int(c["start"])
            if a <= t < b and t % 60 == 0:
                value = D(c["close"])
                if not value.is_finite() or value <= 0:
                    raise RuntimeError("Invalid historical price")
                closes[t] = value
    return closes


def from_candles(products, start, end, path, half_spread, client=None):
    """Rebuild [start, end) from one-minute candles.

    Each observation is taken at a candle's close (start + 60 s). Minutes
    without a candle for every product are skipped, never interpolated.
    """
    start, end = int(start) // 60 * 60, int(end) // 60 * 60
    half_spread = D(half_spread)
    if end <= start:
        raise ValueError("End must be after start")
    # Stay inside the guard's 0.5% spread limit with margin.
    if not D(0) < half_spread <= D(".002"):
        raise ValueError("Half-spread must be in (0, 0.2%]")
    if client is None:
        from coinbase.rest import RESTClient

        client = RESTClient(api_key=None, api_secret=None, timeout=10)
    fields = {p: _product_quote_fields(client, p) for p in products}
    seed = {}
    for p in products:
        past = _closes(client, p, start - SEED_MINUTES * 60, start)
        if not past:
            raise RuntimeError("No candles before the start time for the first chart")
        seed[p] = [float(past[t]) for t in sorted(past)]
    closes = {p: _closes(client, p, start, end) for p in products}
    minutes = sorted(set.intersection(*(set(c) for c in closes.values())))
    if not minutes:
        raise RuntimeError("No candles in the requested window")
    with _open_new(path) as f:
        _line(f, {
            "tape": TAPE_VERSION,
            "source": "public Coinbase one-minute candles; bid/ask = close -/+ half_spread",
            "half_spread": str(half_spread),
            "products": list(products),
            "interval_seconds": 60,
            "window": [start, end],
            "skipped_minutes": (end - start) // 60 - len(minutes),
            "seed_history": seed,
        })
        for t in minutes:
            now = float(t + 60)
            quotes = {}
            for p in products:
                close = closes[p][t]
                quotes[p] = Quote(
                    p, close * (1 - half_spread), close * (1 + half_spread), now,
                    **{k: D(v) for k, v in fields[p].items()},
                ).json()
            _line(f, {"time": now, "quotes": quotes})
    return len(minutes)


def load(path):
    lines = Path(path).read_text().splitlines()
    if not lines:
        raise ValueError("Empty tape")
    header = json.loads(lines[0])
    if header.get("tape") != TAPE_VERSION:
        raise ValueError("Unknown tape format")
    rows = [json.loads(x) for x in lines[1:] if x.strip()]
    times = [r["time"] for r in rows]
    if not rows or any(not math.isfinite(t) for t in times) or times != sorted(times):
        raise ValueError("Tape needs observations in time order")
    return header, rows


class ReplayMarket:
    """Serves a tape in order. Paper only; never used with a live broker."""

    def __init__(self, path, products):
        self.path = Path(path)
        self.header, self.rows = load(self.path)
        missing = set(products) - set(self.header["products"])
        if missing:
            raise SystemExit(f"Tape lacks products: {sorted(missing)}")
        self.products = products
        self.history = {p: list(self.header["seed_history"][p]) for p in products}
        self.tick = 0
        self.now = None
        self.current = None
        self.identity = {
            "path": str(self.path),
            "sha256": file_sha256(self.path),
            "source": self.header["source"],
            "observations": len(self.rows),
            "first_time": self.rows[0]["time"],
            "last_time": self.rows[-1]["time"],
            **({"half_spread": self.header["half_spread"]} if "half_spread" in self.header else {}),
        }

    def snapshot(self):
        if self.tick >= len(self.rows):
            raise TapeEnd()
        row = self.rows[self.tick]
        self.tick += 1
        self.now = float(row["time"])
        self.current = {p: quote_from_json(row["quotes"][p]) for p in self.products}
        return dict(self.current)

    def execution_snapshot(self):
        # A live run fetches a fresh book after neural compute. In a replay no
        # market time passes, so execution sees the same recorded quote.
        if self.current is None:
            raise RuntimeError("No observation yet")
        return dict(self.current)

    def clock(self):
        if self.now is None:
            raise RuntimeError("Replay clock starts at the first observation")
        return self.now

    record = CoinbaseMarket.record
