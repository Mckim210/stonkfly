# Skill or luck? Control experiment

A fly run that ends in profit does not show skill. Rising prices can make any buyer look good, and one run is one draw. This procedure compares a fly run with controls that see **the identical market tape** and go through the same risk guard, paper fills, fees and accounting.

| Run | Decision source | Question it answers |
| --- | --- | --- |
| Real fly | MaleCNS connectome | The result being tested |
| Scrambled fly (several seeds) | Same neurons and degrees, randomized wiring ([scramble](validation.md#wiring-control-scrambled-connectome)) | Does the *specific* fly wiring matter? |
| Random traders (hundreds of seeds) | The real fly's own BUY/SELL/HOLD proposals, shuffled in time | Could the same trading activity at random times do as well? |

## 1. Record a tape

```sh
# a past window from public one-minute candles (instant)
python -m stonkfly tape --start 2026-10-01T00:00Z --end 2026-10-02T00:00Z --out tapes/oct01.jsonl
# or the live public order book, one observation per minute (takes real time)
python -m stonkfly tape --minutes 1440 --out tapes/live-day.jsonl
```

Candles carry no order book, so a candle tape sets bid/ask = close ∓ `--half-spread` (default 0.01%), recorded in the tape header. Minutes without a candle are skipped, never interpolated. Tapes are never overwritten.

## 2. Run every brain on the same tape

```sh
python -m stonkfly run --replay tapes/oct01.jsonl --out runs/fly
python -m stonkfly.neural.scramble --seed 1
STONKFLY_DATA=data-scrambled-seed1 python -m stonkfly run --replay tapes/oct01.jsonl --out runs/scrambled-seed1
```

A replay is paper-only and runs as fast as the computer allows. Its clock is the tape's recorded observation time, so the 60-second order cooldown, the 24-orders-per-UTC-day limit and quote-age checks behave as they would in market time. Each run records the tape's SHA-256 in `provenance.json`.

## 3. Random traders matched to the fly

```sh
python -m stonkfly.control batch --match runs/fly --seeds 1-200 --jobs 4 --out runs/random
```

Each seed shuffles the fly's exact proposal sequence, so every random trader makes exactly as many BUY and SELL proposals as the fly did. The run is labelled `RANDOM control, no neural network` and needs no connectome data. A random trader on one day of one-minute data takes about 15 seconds. `--buy-rate/--sell-rate` with `run --policy random` gives unmatched independent draws instead.

## 4. Compare

```sh
python -m stonkfly.control summary runs/fly runs/scrambled-seed* --random runs/random --json results.json
```

For each run: proposals, fills, P&L marked to the last observed bid, mean spikes per observation, and where its P&L falls in the random distribution. `p` is the share of random traders (plus one) that did at least as well: (1 + #random ≥ run) / (1 + N). The summary warns if runs used different tapes.

## Reading the result

- A small `p` for the real fly says its timing beat shuffled timing *on this tape*. Repeat on several held-out tapes before calling it skill; do not pick the tape after seeing results.
- Real ≈ scrambled says the effect is not about fly wiring. Compare mean spikes too: a scrambled brain that goes silent or saturates loses for a different reason.
- With learning on, the fly's later proposals depend on its own P&L feedback; the shuffled control removes that link by design.
- The daily order limit vetoes most proposals in a busy run. It applies equally to every policy, but report fills, not only proposals.
