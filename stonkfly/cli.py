"""Single-worker run loop. Default execution is paper; live must be explicit."""

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

from .config import D, Settings


def main():
    p = argparse.ArgumentParser(prog="stonkfly")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--reuse-doomfly", type=Path)
    sub.add_parser("verify")
    run = sub.add_parser("run")
    run.add_argument("--live", action="store_true")
    run.add_argument(
        "--preflight-only",
        action="store_true",
        help="Read-only exchange checks; never submit an order",
    )
    run.add_argument(
        "--resume-reviewed",
        action="store_true",
        help="After manual review, clear a transient halt only after successful reconciliation",
    )
    run.add_argument(
        "--fixture",
        action="store_true",
        help="Synthetic offline market input; paper only",
    )
    run.add_argument("--steps", type=int, default=0, help="0 keeps running")
    run.add_argument(
        "--fast",
        action="store_true",
        help="Skip waiting in paper mode; execution cooldown still applies",
    )
    run.add_argument(
        "--frozen",
        action="store_true",
        help="Freeze all memory efficacies for a control run",
    )
    run.add_argument("--out", type=Path)
    run.add_argument(
        "--products",
        nargs="+",
        default=["BTC-USDC"],
        choices=["BTC-USDC", "ETH-USDC", "SOL-USDC"],
    )
    run.add_argument("--neural-ms", type=float, default=500)
    run.add_argument(
        "--replay",
        type=Path,
        help="Paper-only: replay a recorded market tape; its time is the clock",
    )
    run.add_argument(
        "--policy",
        choices=["fly", "random"],
        default="fly",
        help="random = coin-flip control with no neural network (needs --replay)",
    )
    run.add_argument(
        "--match",
        type=Path,
        help="random: shuffle this fly run's exact BUY/SELL/HOLD proposals",
    )
    run.add_argument("--buy-rate", type=float, help="random: P(BUY) per observation")
    run.add_argument("--sell-rate", type=float, help="random: P(SELL) per observation")
    run.add_argument("--seed", type=int, default=0, help="random: seed")
    run.add_argument(
        "--shuffle-reinforcement",
        type=Path,
        metavar="RUN",
        help="Control: deliver this fly run's recorded reinforcement in shuffled order, ignoring own P&L (needs --replay)",
    )
    run.add_argument("--shuffle-seed", type=int, help="Seed for --shuffle-reinforcement")
    run.add_argument(
        "--reinforcement",
        choices=["portfolio", "roundtrip", "roundtrip-replay"],
        default="portfolio",
        help="portfolio = marked-to-bid P&L each observation (default). Paper experiments: "
        "roundtrip = realized FIFO P&L of each filled sell; roundtrip-replay also replays "
        "the first matched buy's chart with the pulse (docs/roundtrip.md)",
    )
    tape = sub.add_parser("tape", help="Record a market tape for replay")
    tape.add_argument("--out", type=Path, required=True)
    tape.add_argument(
        "--products",
        nargs="+",
        default=["BTC-USDC"],
        choices=["BTC-USDC", "ETH-USDC", "SOL-USDC"],
    )
    tape.add_argument("--minutes", type=int, help="Live: record this many 1-minute observations")
    tape.add_argument("--start", help="Candles: UTC start, e.g. 2026-10-01T00:00Z")
    tape.add_argument("--end", help="Candles: UTC end (exclusive)")
    tape.add_argument(
        "--half-spread",
        default="0.0001",
        help="Candles only: bid/ask = close -/+ this fraction (no order book in candles)",
    )
    status = sub.add_parser("status")
    status.add_argument("--out", type=Path, default=Path("runs/paper"))
    a = p.parse_args()
    from dotenv import load_dotenv

    # Never search parent projects for unrelated account credentials.
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    if a.command in ("prepare", "verify"):
        from .data import prepare, verify

        if a.command == "prepare":
            prepare(a.reuse_doomfly)
        else:
            print(json.dumps(verify()))
        return
    if a.command == "tape":
        from .market import utc_timestamp
        from .replay import from_candles, record

        if (a.minutes is None) == (a.start is None and a.end is None):
            p.error("Use --minutes (live) or --start/--end (past candles)")
        if a.minutes is not None:
            n = record(tuple(a.products), a.out, a.minutes, interval=60)
        else:
            if not a.start or not a.end:
                p.error("Candles need both --start and --end")
            n = from_candles(
                tuple(a.products),
                utc_timestamp(a.start),
                utc_timestamp(a.end),
                a.out,
                a.half_spread,
            )
        print(json.dumps({"tape": str(a.out), "observations": n}))
        return
    if a.command == "status":
        import sqlite3

        db = sqlite3.connect(f"file:{a.out / 'ledger.sqlite'}?mode=ro", uri=True)
        meta = {k: json.loads(v) for k, v in db.execute("SELECT key,value FROM meta")}
        print(
            json.dumps(
                {
                    k: meta.get(k)
                    for k in [
                        "mode",
                        "tick",
                        "cash",
                        "positions",
                        "initial_cash",
                        "anchor",
                        "halted",
                    ]
                },
                indent=2,
            )
        )
        return
    if a.live and (
        a.fixture or a.fast or a.replay or a.policy != "fly" or a.shuffle_reinforcement
    ):
        p.error("Live mode forbids fixtures, fast replay, tapes and control policies")
    if (a.shuffle_reinforcement is None) != (a.shuffle_seed is None):
        p.error("Use --shuffle-reinforcement RUN together with --shuffle-seed N")
    if a.shuffle_reinforcement and (a.policy != "fly" or not a.replay):
        p.error("Shuffled reinforcement is a fly control; it needs --replay of the same tape")
    if a.reinforcement != "portfolio" and (
        a.live or a.policy != "fly" or a.shuffle_reinforcement
    ):
        p.error(
            "Round-trip reinforcement is a paper fly experiment: not with --live, "
            "a random policy or shuffled reinforcement"
        )
    if a.fixture and a.replay:
        p.error("Choose --fixture or --replay")
    if a.policy == "random":
        if not a.replay:
            p.error("A random control needs --replay so it sees the same market tape")
        if (a.match is None) == (a.buy_rate is None and a.sell_rate is None):
            p.error("Random control: give --match RUN or --buy-rate/--sell-rate")
        if a.match is None and (a.buy_rate is None or a.sell_rate is None):
            p.error("Give both --buy-rate and --sell-rate")
    elif a.match or a.buy_rate is not None or a.sell_rate is not None:
        p.error("--match and rates apply only to --policy random")
    if a.steps < 0:
        p.error("steps cannot be negative")
    settings = Settings(
        products=tuple(a.products),
        learning=not a.frozen,
        neural_ms=a.neural_ms,
        pulse_ms=min(200, a.neural_ms),
    )
    out = a.out or Path("runs/live" if a.live else "runs/paper")
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "worker.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A worker already owns this run directory")
    from .broker import CoinbaseBroker, PaperBroker
    from .ledger import Ledger

    ledger = Ledger(out / "ledger.sqlite", settings, "live" if a.live else "paper")
    try:
        broker = (
            CoinbaseBroker.from_env(settings, ledger)
            if a.live
            else PaperBroker(settings, ledger)
        )
        result = broker.preflight()
        print(json.dumps(result), flush=True)
        if a.resume_reviewed:
            if (out / "STOP").exists() or ledger.pending():
                raise RuntimeError(
                    "Remove STOP only after review; unresolved orders cannot resume"
                )
            reason = ledger.get("halted")
            if reason and ("Loss stop" in reason or "fee exceeded" in reason):
                raise RuntimeError("A financial stop cannot be cleared by this flag")
            ledger.put("halted", None)
        if a.preflight_only:
            return
        verified = None
        if a.policy == "fly":
            from .data import verify

            verified = verify()
            if a.live and "scramble" in verified:
                raise SystemExit("A scrambled-wiring control graph is paper-only.")
        from PIL import Image

        from .actions import StonkflyActions
        from .display import market_frame
        from .market import CoinbaseMarket, FixtureMarket
        from .reinforcement import reinforcement
        from .risk import Guard, Veto

        if a.replay:
            from .replay import ReplayMarket, TapeEnd

            market = ReplayMarket(a.replay, settings.products)
        else:
            TapeEnd = ()
            market = (
                FixtureMarket(settings.products)
                if a.fixture
                else CoinbaseMarket(settings.products)
            )
        previous = ledger.get("observation")
        if previous:
            market.history = previous["market_history"]
            if a.fixture or a.replay:
                market.tick = previous["fixture_tick"]
        if a.policy == "random":
            from .control import RandomController, matched_proposals

            controller = RandomController(
                a.seed,
                sequence=matched_proposals(a.match, market.identity["sha256"])
                if a.match
                else None,
                rates=None if a.match else (a.buy_rate, a.sell_rate),
            )
        else:
            from .neural.controller import FlyController

            controller = FlyController(settings)
        shuffle = None
        if a.shuffle_reinforcement:
            from .control import ShuffledReinforcement, delivered_signals

            shuffle = ShuffledReinforcement(
                a.shuffle_reinforcement,
                delivered_signals(a.shuffle_reinforcement, market.identity["sha256"]),
                a.shuffle_seed,
            )
        trade_level = a.reinforcement != "portfolio"
        if trade_level:
            from . import roundtrip

            # FIFO lots are committed with each tick, so a restart rebooks
            # exactly the orders settled after the last committed observation.
            trades = (previous or {}).get("roundtrip_book") or roundtrip.new_state()
        cp = ledger.get("checkpoint")
        if cp:
            path = out / cp["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != cp["sha256"]:
                raise RuntimeError("Checkpoint integrity mismatch")
            controller.restore(path)
        fly = a.policy == "fly"
        provenance = {
            "settings": dataclasses.asdict(settings),
            "policy": "fly connectome" if fly else controller.describe(),
            "dataset": verified,
            "circuit": controller.brain.circuit["report"] if fly else None,
            "vision": controller.brain.visual_report if fly else None,
            "mode": broker.mode,
            "feed": "replay" if a.replay else "fixture" if a.fixture else "coinbase-public",
            "tape": market.identity if a.replay else None,
            "decoder": "DNp20 mean R-L: buy/sell; DNpe017 spike gate; otherwise hold. Engineered fixed mapping."
            if fly
            else None,
            "reinforcement": shuffle.describe()
            if shuffle
            else roundtrip.DESCRIPTION[a.reinforcement]
            if trade_level
            else "own marked-to-bid portfolio P&L"
            if fly
            else None,
            "learning_validated": False,
            "pain_receptors_modeled": False,
            "timing": "Each observation advances configured neural_ms regardless of wall-market time; no claim of real-time fly physiology.",
            "source_sha256": {
                str(path.relative_to(Path(__file__).parent)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in Path(__file__).parent.rglob("*")
                if path.suffix in (".py", ".cpp")
            },
        }
        signature = hashlib.sha256(
            json.dumps(provenance, sort_keys=True).encode()
        ).hexdigest()
        if ledger.get("provenance_sha256") not in (None, signature):
            raise RuntimeError(
                "Run source/protocol changed; use a separate paper run or explicitly review migration"
            )
        ledger.put("provenance_sha256", signature)
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        guard = Guard(
            settings, ledger, out / "STOP", clock=market.clock if a.replay else None
        )
        provider = StonkflyActions(guard, broker)
        action = provider.get_actions()[0]
        count = 0
        while not a.steps or count < a.steps:
            started = time.monotonic()
            if (out / "STOP").exists() or ledger.get("halted"):
                break
            if getattr(controller, "done", lambda: False)():
                print("Matched proposal sequence finished.", flush=True)
                break
            if shuffle and shuffle.done(ledger.get("tick")):
                print("Shuffled reinforcement sequence finished.", flush=True)
                break
            broker.reconcile()
            broker.verify_balances()
            try:
                quotes = market.snapshot()
            except TapeEnd:
                print("Replay tape finished.", flush=True)
                break
            guard.check(quotes, guard.clock())
            market.record(quotes)
            product = settings.products[ledger.get("tick") % len(settings.products)]
            q = quotes[product]
            equity = ledger.equity(quotes)
            kind, delta = reinforcement(
                equity, ledger.get("anchor"), settings.reward_deadband
            )
            # A shuffled-reinforcement control ignores its own P&L signal.
            delivered = shuffle.signal(ledger.get("tick")) if shuffle else kind
            frame = market_frame(product, market.history[product], q.bid, q.ask)
            if trade_level:
                # Only the previous observation's sell fill, if any, earns a signal.
                memory_before = controller.brain.memory()
                trades, sale = roundtrip.sync(ledger.db, trades, ledger.get("tick"))
                earned = delivered = roundtrip.signal(sale)
                replayed = None
                if sale and a.reinforcement == "roundtrip-replay":
                    replayed = roundtrip.replay(controller, ledger.db, sale, delivered)
                weights = controller.plastic_weights()
            neural = controller.observe(frame, delivered)
            signals = {
                "reinforcement_earned": earned if trade_level else kind,
                "reinforcement_delivered": delivered if fly else None,
            }
            record = {}
            if trade_level:
                record["roundtrip"] = {
                    "portfolio_signal": kind,
                    "sale": sale,
                    "realized_pnl_before_fees": sale and sale["before_fees"],
                    "realized_pnl_after_fees": sale and sale["after_fees"],
                    "replay": replayed,
                    "buy_observation": replayed and replayed["buy_observation"],
                    "memory_before": memory_before,
                    "kc_active": controller.kc_active(),
                    "edges_changed_in_step": roundtrip.edges_changed(
                        weights, controller.plastic_weights()
                    ),
                }
            # Checkpoint + accounting anchor are committed before any trade.
            # Two slots keep the last committed snapshot safe during a crash.
            slot = ledger.get("tick") % 2
            checkpoint = out / f"brain-{slot}.npz"
            controller.save(checkpoint)
            checkpoint_info = {
                "file": checkpoint.name,
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
            observation = {
                "neural": neural,
                "product": product,
                "quote": q.json(),
                "pnl_delta_usdc": str(delta),
                **signals,
                "market_history": market.history,
                "fixture_tick": getattr(market, "tick", None),
            }
            if trade_level:
                observation["roundtrip_book"] = trades
            ledger.commit_tick(equity, checkpoint_info, observation)
            order = {"status": "HOLD"}
            if neural["side"] != "HOLD":
                try:
                    # Neural integration can be slow; use a fresh execution book.
                    fresh = (
                        market.execution_snapshot() if a.replay else market.snapshot()
                    )
                    latest = fresh[product]
                    if abs(latest.bid - q.bid) / q.bid > D(settings.slippage):
                        raise Veto("Price moved beyond neural observation tolerance")
                    provider.quotes = fresh
                    order = action.invoke({"product": product, "side": neural["side"]})
                except Veto as e:
                    order = {"status": "VETO", "reason": str(e)}
            row = {
                "tick": ledger.get("tick"),
                "wall_time": time.time(),
                "market_time": guard.clock(),
                "product": product,
                "mode": broker.mode,
                "quote": q.json(),
                "equity_usdc": str(equity),
                "pnl_delta_usdc": str(delta),
                **signals,
                **record,
                "neural": neural,
                "execution": order,
            }
            with (out / "events.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            Image.fromarray(frame).save(out / "latest-input.png")
            (out / "latest.json").write_text(json.dumps(row, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "tick": row["tick"],
                        "side": neural["side"],
                        "execution": order["status"],
                        "equity": str(equity),
                        "stimulus": delivered,
                        "plastic_edges_changed": neural.get("memory", {}).get(
                            "changed_edges"
                        ),
                    }
                ),
                flush=True,
            )
            count += 1
            if not (a.fast or a.replay) and (not a.steps or count < a.steps):
                until = started + settings.interval_seconds
                while time.monotonic() < until and not (out / "STOP").exists():
                    time.sleep(min(1, until - time.monotonic()))
        if fly:
            from .control import SIGNALS, events

            rows = events(out)
            memory = controller.brain.memory()
            if not settings.learning and memory["changed_edges"]:
                raise RuntimeError("Frozen run changed plastic synapses")
            if shuffle:
                shuffle.check(rows)
            sent = [r["reinforcement_delivered"] for r in rows]
            final = {
                "tick": ledger.get("tick"),
                "learning": settings.learning,
                "memory": memory,
                "signals_delivered": {s: sent.count(s) for s in SIGNALS},
            }
            if trade_level:
                # A sell at the last observation has no next observation to signal.
                _, last = roundtrip.sync(ledger.db, trades, ledger.get("tick"))
                final["reinforcement"] = a.reinforcement
                final["roundtrip"] = roundtrip.summary(rows, last)
            (out / "final.json").write_text(json.dumps(final, indent=2) + "\n")
            print(json.dumps({"final": final}), flush=True)
    except KeyboardInterrupt:
        print("Stopped; run state preserved.", flush=True)
    except Exception as e:
        # Never print SDK exception text: it may contain account/request details.
        if not ledger.get("halted"):
            ledger.halt(type(e).__name__)
        frames = traceback.extract_tb(e.__traceback__)
        origin = frames[-1] if frames else None
        internal = origin and Path(origin.filename).is_relative_to(
            Path(__file__).parent
        )
        diagnostic = {
            "type": type(e).__name__,
            "reason": str(e)
            if internal
            else "External dependency error; review connection and account state.",
            "locations": [
                f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in frames
            ],
        }
        (out / "error.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        print(
            f"Stopped safely: {type(e).__name__}. Inspect local state and reconcile before restarting.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    finally:
        ledger.close()
        lock.close()


if __name__ == "__main__":
    main()
