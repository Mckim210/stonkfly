"""Random-trader control and result summary for skill-vs-luck comparisons.

The random trader replaces only the decision. Market tape, chart, risk guard,
paper fills, fees and accounting are the same code a fly run uses.

    # 1. one fly run (real or scrambled wiring) on a recorded tape
    python -m stonkfly run --replay tapes/day.jsonl --out runs/fly
    # 2. many random traders that make exactly the fly's proposals, shuffled in time
    python -m stonkfly.control batch --match runs/fly --seeds 1-200 --out runs/random
    # 3. where does the fly fall in the random distribution?
    python -m stonkfly.control summary runs/fly runs/scrambled-seed1 --random runs/random
"""

import argparse
import json
import sqlite3
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .config import D

SIDES = ("BUY", "SELL", "HOLD")


def events(run):
    path = Path(run) / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def provenance(run):
    path = Path(run) / "provenance.json"
    return json.loads(path.read_text()) if path.exists() else {}


def tape_sha(prov):
    return (prov.get("tape") or {}).get("sha256")


def matched_proposals(run, tape_sha256=None):
    """The exact BUY/SELL/HOLD proposal sequence of a finished fly run."""
    prov = provenance(run)
    if not str(prov.get("policy", "")).startswith("fly"):
        raise SystemExit(f"{run} is not a fly run made with this version; nothing to match.")
    if tape_sha256 is not None and tape_sha(prov) != tape_sha256:
        raise SystemExit(f"{run} used a different tape; compare runs on the same tape.")
    sides = [row["neural"]["side"] for row in events(run)]
    if not sides or set(sides) - set(SIDES):
        raise SystemExit(f"{run} has no usable proposals.")
    return sides


class RandomController:
    """Coin-flip decisions; no neural network, no learning.

    match: proposals of a fly run, shuffled with the seed, so the random trader
    makes exactly as many BUY and SELL proposals as the fly, at random times.
    rates: (buy, sell) probabilities per observation instead.
    """

    policy = "random"

    def __init__(self, seed, sequence=None, rates=None):
        if (sequence is None) == (rates is None):
            raise ValueError("Give a matched sequence or (buy, sell) rates")
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.index = 0
        self.rates = None
        self.sequence = None
        if sequence is not None:
            self.sequence = [str(x) for x in self.rng.permutation(np.asarray(sequence))]
        else:
            buy, sell = map(float, rates)
            if min(buy, sell) < 0 or buy + sell > 1:
                raise ValueError("Rates must be >= 0 and sum to <= 1")
            self.rates = (buy, sell)

    def describe(self):
        if self.sequence is not None:
            counts = {s: self.sequence.count(s) for s in SIDES}
            how = f"shuffled proposals of a matched fly run {counts}"
        else:
            how = f"independent draws, P(BUY)={self.rates[0]}, P(SELL)={self.rates[1]}"
        return f"RANDOM control, no neural network: {how}; seed {self.seed}"

    def done(self):
        return self.sequence is not None and self.index >= len(self.sequence)

    def observe(self, frame, reinforcement):
        if self.sequence is not None:
            if self.index >= len(self.sequence):
                raise RuntimeError("More observations than the matched run")
            side = self.sequence[self.index]
        else:
            u = self.rng.random()
            side = "BUY" if u < self.rates[0] else "SELL" if u < sum(self.rates) else "HOLD"
        self.index += 1
        return {"side": side, "policy": "random", "seed": self.seed, "draw": self.index,
                "stimulus_ignored": reinforcement}

    def save(self, path):
        path = Path(path)
        state = {"seed": self.seed, "index": self.index, "rates": self.rates,
                 "sequence": self.sequence, "rng": self.rng.bit_generator.state}
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as handle:
            np.savez(handle, metadata=json.dumps(state))
        temporary.replace(path)

    def restore(self, path):
        with np.load(path, allow_pickle=False) as a:
            state = json.loads(str(a["metadata"]))
        rates = tuple(state["rates"]) if state["rates"] else None
        if (state["seed"], rates, state["sequence"]) != (self.seed, self.rates, self.sequence):
            raise ValueError("Checkpoint belongs to a different random control")
        self.index = state["index"]
        self.rng.bit_generator.state = state["rng"]


def outcome(run):
    """Final paper result of one run, marked to the last observed bid."""
    run = Path(run)
    db = sqlite3.connect(f"file:{run / 'ledger.sqlite'}?mode=ro", uri=True)
    try:
        meta = {k: json.loads(v) for k, v in db.execute("SELECT key,value FROM meta")}
    finally:
        db.close()
    rows = events(run)
    bids = {}
    for row in rows:
        bids[row["product"]] = D(row["quote"]["bid"])
    positions = {p: D(v) for p, v in meta["positions"].items()}
    if set(p for p, v in positions.items() if v) - set(bids):
        raise RuntimeError(f"{run}: holding a product never observed")
    equity = D(meta["cash"]) + sum((v * bids[p] for p, v in positions.items() if v), D(0))
    prov = provenance(run)
    proposals = [r["neural"]["side"] for r in rows]
    spikes = [r["neural"]["total_spikes"] for r in rows if "total_spikes" in r["neural"]]
    return {
        "run": str(run),
        "policy": prov.get("policy", "unknown"),
        "wiring": (prov.get("dataset") or {}).get("wiring", "real MaleCNS")
        if str(prov.get("policy", "")).startswith("fly") else "-",
        "tape_sha256": tape_sha(prov),
        "observations": len(rows),
        "buy_proposals": proposals.count("BUY"),
        "sell_proposals": proposals.count("SELL"),
        "fills": sum(r["execution"].get("status") == "FILLED" for r in rows),
        "vetoes": sum(r["execution"].get("status") == "VETO" for r in rows),
        "halted": meta.get("halted"),
        "mean_total_spikes": statistics.fmean(spikes) if spikes else None,
        "final_equity": float(equity),
        "pnl": float(equity - D(meta["initial_cash"])),
    }


def compare(value, random_pnls):
    """Share of random traders that did at least as well, and a one-sided p."""
    r = np.asarray(random_pnls, dtype=float)
    at_least = int(np.count_nonzero(r >= value))
    return {
        "percentile": float(np.mean(r < value) * 100),
        "p_value": (1 + at_least) / (1 + len(r)),
        "random_at_least_as_good": at_least,
    }


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        seeds += list(range(int(a), int(b or a) + 1))
    if not seeds or len(set(seeds)) != len(seeds):
        raise SystemExit("Seeds like 1-200 or 1,2,5 without repeats")
    return seeds


def batch(match, seeds, out, jobs=1, python=sys.executable):
    prov = provenance(match)
    tape = (prov.get("tape") or {}).get("path")
    if not tape or not Path(tape).exists():
        raise SystemExit(f"{match} has no replay tape on disk; random controls need the same tape.")
    matched_proposals(match, tape_sha(prov))
    products = prov["settings"]["products"]
    out = Path(out)

    def one(seed):
        # Rerunning a finished or interrupted seed resumes it; nothing is redone.
        target = out / f"seed-{seed}"
        cmd = [python, "-m", "stonkfly", "run", "--replay", tape, "--policy", "random",
               "--match", str(match), "--seed", str(seed), "--out", str(target),
               "--products", *products]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return seed, "done" if result.returncode == 0 else f"exit {result.returncode}"

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for seed, status in pool.map(one, seeds):
            print(f"seed {seed}: {status}", flush=True)


def summary(runs, random_dir=None, as_json=None):
    rows = [outcome(r) for r in runs]
    random_rows = []
    if random_dir:
        random_rows = [outcome(p.parent) for p in sorted(Path(random_dir).glob("*/ledger.sqlite"))]
    tapes = {r["tape_sha256"] for r in rows + random_rows}
    report = {"runs": rows, "random": None, "same_tape": len(tapes) == 1 and None not in tapes}
    if random_rows:
        pnls = [r["pnl"] for r in random_rows]
        report["random"] = {
            "count": len(pnls),
            "mean_pnl": statistics.fmean(pnls),
            "sd_pnl": statistics.pstdev(pnls),
            "p5_pnl": float(np.percentile(pnls, 5)),
            "median_pnl": float(np.median(pnls)),
            "p95_pnl": float(np.percentile(pnls, 95)),
            "halted": sum(bool(r["halted"]) for r in random_rows),
        }
        for r in rows:
            r["vs_random"] = compare(r["pnl"], pnls)
    if as_json:
        Path(as_json).write_text(json.dumps(report, indent=2) + "\n")
    return report


def print_summary(report):
    if not report["same_tape"]:
        print("WARNING: runs did not all use the same replay tape; results are not comparable.\n")
    header = f"{'run':<34}{'wiring':<12}{'obs':>5}{'buy':>5}{'sell':>5}{'fills':>6}{'P&L $':>10}{'spikes':>9}"
    print(header + ("   vs random" if report["random"] else ""))
    for r in report["runs"]:
        wiring = "scrambled" if "SCRAMBLED" in r["wiring"] else "real" if r["wiring"] != "-" else "-"
        spikes = f"{r['mean_total_spikes']:.0f}" if r["mean_total_spikes"] is not None else "-"
        line = (f"{Path(r['run']).name[:33]:<34}{wiring:<12}{r['observations']:>5}{r['buy_proposals']:>5}"
                f"{r['sell_proposals']:>5}{r['fills']:>6}{r['pnl']:>10.4f}{spikes:>9}")
        if "vs_random" in r:
            v = r["vs_random"]
            line += f"   top {100 - v['percentile']:.1f}%  p={v['p_value']:.3f}"
        if r["halted"]:
            line += f"   halted: {r['halted']}"
        print(line)
    s = report["random"]
    if s:
        print(f"\nrandom traders: n={s['count']}, mean {s['mean_pnl']:.4f}, sd {s['sd_pnl']:.4f}, "
              f"5-95% [{s['p5_pnl']:.4f}, {s['p95_pnl']:.4f}], halted {s['halted']}")
        print("p = chance a random trader with the same proposals does at least as well.")


def main():
    parser = argparse.ArgumentParser(prog="python -m stonkfly.control")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("batch", help="Run many random controls matched to a fly run")
    b.add_argument("--match", type=Path, required=True)
    b.add_argument("--seeds", default="1-200")
    b.add_argument("--out", type=Path, default=Path("runs/random"))
    b.add_argument("--jobs", type=int, default=1)
    s = sub.add_parser("summary", help="Compare runs against the random distribution")
    s.add_argument("runs", nargs="+", type=Path)
    s.add_argument("--random", type=Path)
    s.add_argument("--json", type=Path)
    a = parser.parse_args()
    if a.command == "batch":
        batch(a.match, parse_seeds(a.seeds), a.out, a.jobs)
    else:
        print_summary(summary(a.runs, a.random, a.json))


if __name__ == "__main__":
    main()
