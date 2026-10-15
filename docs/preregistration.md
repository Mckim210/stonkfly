# Preregistration: skill or luck?

Registered 2026-10-15, committed **before** any test-day tape was recorded and before any run on a test day. The procedure is [the control experiment](experiment.md); this file fixes the choices it leaves open.

## Question

On held-out market tapes, does the real MaleCNS fly's trade timing do better than (a) degree-preserving scrambled-wiring flies and (b) random traders that make the fly's exact proposals in shuffled order?

## Data

| Item | Fixed choice |
| --- | --- |
| Test days | 2026-10-02 through 2026-10-11 (10 days) |
| Window | Each day 00:00–03:00 UTC, end exclusive: up to 180 one-minute observations |
| Source | Public Coinbase one-minute candles, BTC-USDC, default half-spread 0.0001 |
| Tape command | `python -m stonkfly tape --start 2026-10-DDT00:00Z --end 2026-10-DDT03:00Z --out tapes/2026-10-DD.jsonl` |

Excluded: 2026-10-01. Its 00:00–02:00 window was the timing pilot (5 observations; the fly's proposals were seen). No other market window was looked at for this purpose.

## Runs per test day

| Run | Count | Command |
| --- | --- | --- |
| Real fly | 1 | `python -m stonkfly run --replay tapes/2026-10-DD.jsonl --out runs/2026-10-DD/fly` |
| Scrambled fly | seeds 1–5 | `STONKFLY_DATA=data-scrambled-seedN python -m stonkfly run --replay tapes/2026-10-DD.jsonl --out runs/2026-10-DD/scrambled-seedN` |
| Random trader | seeds 1–1000 | `python -m stonkfly.control batch --match runs/2026-10-DD/fly --seeds 1-1000 --jobs 4 --out runs/2026-10-DD/random` |

- Scrambled graphs are made once with `python -m stonkfly.neural.scramble --seed N` (default: sign-preserving) and reused for every day.
- All settings are the defaults: $100 paper balance, learning on, 500 ms neural time per observation, 24 orders per UTC day. Each run starts fresh. **Paper only; `--live` is never used.**
- Up to 4 processes run at once. Run order does not change results; each run is independent and resumable.

## Analysis

Per day: `python -m stonkfly.control summary runs/2026-10-DD/fly runs/2026-10-DD/scrambled-seed* --random runs/2026-10-DD/random --json results/2026-10-DD.json`. Each day's result is committed when that day finishes.

- Outcome: paper P&L marked to the last observed bid, after modeled fees.
- For each fly run (real and scrambled): one-sided `p = (1 + #random with P&L ≥ run) / (1 + 1000)`.
- Report for every run: proposals, fills, vetoes, halts, mean spikes per observation. Compare real and scrambled spike counts before attributing a difference to wiring.
- All 10 days are reported, whatever the outcome. No day, seed or run is dropped or rerun to change a result. An interrupted run is resumed with the same command, and that is reported.

**Not pre-specified:** pooling the 10 days into one test, and multiple-comparison correction. Any such analysis is labelled exploratory.

## Pre-run checks

Before the main experiment:

1. Determinism: rerun the real fly on the pilot tape for 5 steps in a new folder; `spike_sha256` must match the pilot for every step. If it does not, one real-fly run per day is not a sufficient sample, and this plan is revised before any test-day run.
2. Scramble seed 1 on the real MaleCNS graph, recording time and peak memory.

Results (2026-10-15, still before any test-day tape):

1. Deterministic. A fresh 5-step real-fly run on the pilot tape matched the pilot's `spike_sha256`, proposal and equity at every step. One real-fly run per day stands.
2. Scramble seed 1: 53 s, 1.9 GB peak memory, 1,571 edges needing a second repair round, zero duplicate pairs, accepted by `verify`. A 3-step smoke run on the pilot tape: about 540,000–551,000 spikes per observation (real fly: 373,000–388,000), 1,500–1,900 KC spikes (real fly: 10–16), zero DNpe017 gate spikes, so every proposal was HOLD. If that persists, a scrambled fly's P&L reflects never trading, not worse timing; spike counts are reported to show this.

## Known limitations, stated in advance

- The fly proposed a trade on 5 of 5 pilot observations. If that continues, the 24-orders-per-day limit is reached within the first hour or so of each 3-hour tape and later proposals are vetoed. This applies to every policy, but the result then mostly reflects early timing plus holding.
- The 0.6% modeled fee costs about $0.06 per $10 fill.
- Candle tapes have no order book; bid/ask is close ∓ 0.01%.
- Ten 3-hour windows of one asset cover a narrow range of market conditions. A small `p` would show that the fly beat shuffled timing on these tapes, not general trading skill.
