import json

import numpy as np
import pytest

from stonkfly.config import D
from stonkfly.market import Quote


def write_tape(path, n=120, start=1_790_000_000.0, interval=60.0, seed=0,
               products=("BTC-USDC",)):
    """Synthetic random-walk tape in the recorded format (test input only)."""
    rng = np.random.default_rng(seed)
    prices = {p: 60000 * np.exp(np.cumsum(rng.normal(0, 0.002, n))) for p in products}
    with open(path, "w") as f:
        f.write(json.dumps({
            "tape": 1,
            "source": "synthetic test tape",
            "products": list(products),
            "interval_seconds": interval,
            "seed_history": {p: [60000.0 + i for i in range(120)] for p in products},
        }) + "\n")
        for i in range(n):
            now = start + i * interval
            quotes = {}
            for p in products:
                mid = D(f"{prices[p][i]:.2f}")
                quotes[p] = Quote(p, mid - D(".01"), mid + D(".01"), now - 0.4,
                                  D(".00000001"), D(".01"), D(".01"), D("1"),
                                  D(".00000001")).json()
            f.write(json.dumps({"time": now, "quotes": quotes}) + "\n")
    return path


@pytest.fixture
def tape(tmp_path):
    return write_tape(tmp_path / "tape.jsonl")
