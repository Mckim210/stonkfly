# Validation status

Recorded during implementation on 2026-09-09. All exchange-order tests use an in-memory SDK double; **no real orders or funded-account checks were performed**.

Final local result: **42 tests passed**, including the opt-in full-connectome test. An offline fixture run also resumed from its saved ledger/checkpoint with balances and neural time preserved. Four upstream AgentKit/Pydantic deprecation warnings remain; they did not fail the tests.

| Check | Observed result | What it does not establish |
| --- | --- | --- |
| Rebuild from checksum-verified released files | 166,700 neurons, 25,582,938 connections and 124,177,617 contacts; every compiled array matches its lock | Completeness or physiological accuracy of the reconstruction |
| Execution/unit tests | Budget, stale data, spread, inventory, permissions, preview fees, STOP, cooldown, duplicate settlement, unknown submission and read-only public-client isolation pass | Successful execution against a particular live account, region or fee tier |
| Full-network sensory/feedback test | White RGB input activates KCs; reward and aversive pulses cause spikes in identified DAN cells; eligible synapses change | Accurate fly retinal responses or an acquired trading association |
| Same-checkpoint controls | Reward changes weights differently from no external reward; frozen memory stays exactly unchanged; checkpoint restoration works | Improved decisions, retention of a useful strategy, or conditioning equivalent to a biological experiment |
| Six accelerated observations of actual Coinbase public BTC-USDC data | One neural BUY filled in paper mode; five subsequent BUY proposals vetoed by cooldown | A realistic trading return or unbiased strategy |

In that six-observation run, the first paper fill used 9.754680320 USDC plus 0.058528081920 USDC in modeled fees. The next observation scheduled a **200 ms aversive pulse**, with **55 spikes across the two PPL101 cells**. KCs produced 11–16 spikes per observation; five candidate edges differed from baseline. Those five edges had already changed before the external loss pulse: endogenous dopamine activity can also drive the rule. Do not attribute every weight change to P&L.

The repeated BUY proposals are an important limitation. A fixed directional decoder can turn circuit bias into one-sided exposure. This run demonstrated neither successful learning nor a profitable policy. White-field neural tests activate substantially more KCs than the actual chart, so visual representation remains a major open modeling question.

The former dark chart produced no KC spikes in an early three-step probe. A light-background display restored some activity without altering the neural parameters. That is a disclosed sensory-adapter change, not evidence that we found biologically correct vision.

## Reproduce

```sh
python -m pytest -q
OPENBLAS_NUM_THREADS=1 STONKFLY_FULL_TEST=1 python -m pytest -q
python -m stonkfly verify
python -m stonkfly run --fixture --fast --steps 6 --out runs/check-fixture
python -m stonkfly run --fast --steps 6 --out runs/check-public
```

The last command reads current public prices and simulates fills. Prices, signals and paper outcomes will differ. All runtime evidence stays local in `runs/`; it is not uploaded with this report. A normal `run` omits `--fast` and samples at the configured wall interval.

Before claiming learned performance, implement the held-out replay, shuffled reinforcement, exposure baselines, retention and memory-reset comparisons described in [the model](model.md). This repository currently provides a functioning experimental loop, not that empirical result.

## Wiring control (scrambled connectome)

`python -m stonkfly.neural.scramble --seed N` writes a degree-preserving null-model graph to `data-scrambled-seedN/`. Same neurons, same per-edge synapse counts and signs, same in- and out-degree per neuron, and by default the same excitatory and inhibitory in-degree per neuron; only *who connects to whom* is randomized. Photoreceptor outputs and the KC/PAM11/PPL101 → MBON07/MBON11 memory edges are held fixed, so the eye, the plastic edge set and the dopamine gains match the real fly. `--mix-signs` gives the weaker plain degree-preserving shuffle.

```sh
python -m stonkfly.neural.scramble --seed 1
STONKFLY_DATA=data-scrambled-seed1 python -m stonkfly run --out runs/scrambled-seed1
```

`verify` accepts the graph only if `scramble.lock.json` derives it from the locked MaleCNS arrays, and records `"wiring": "SCRAMBLED degree-preserving control"` in the run provenance. Scrambled graphs refuse `--live`.

Checked on 2026-10-14 against a synthetic MaleCNS-sized graph (166,700 neurons, 25,582,938 edges, lognormal hubs up to ~40,000 partners): 20 s, 2.6 GB peak memory, 5 repair rounds, zero duplicate pairs, all invariants held. **Not yet run on the real MaleCNS graph**, whose hub structure may need more repair rounds.

What it does not establish: incoming synaptic strength per neuron is not preserved, so compare total spike counts between real and scrambled runs before attributing a performance difference to wiring. Use several seeds, and expose real and scrambled runs to the same market data.

## Replay and random-trader controls

Checked on 2026-10-15 with synthetic tapes; no connectome, exchange or network was used. The full run loop replayed a 1,440-observation tape with a shuffled-proposal random trader in 14 s. Market-time cooldowns held, fills were paper only, rerunning a finished control added nothing, and a 20-seed batch plus `summary` produced the comparison table. Coinbase AgentKit could not be installed in that environment, so the loop ran with a no-network stand-in for its `ActionProvider` base class; `test_end_to_end_random_controls_on_a_tape` runs automatically wherever AgentKit is installed. Candle tapes and live recording were exercised against public-API doubles, not the real Coinbase API.
