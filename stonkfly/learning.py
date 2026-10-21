"""Learning control: does dopamine memory improve paper P&L on held-out days?

Every arm sees the identical tape and starts from a fresh, unlearned brain
(docs/preregistration-learning.md):

    learn     python -m stonkfly run --replay tapes/D.jsonl --out runs/D/learn
    frozen    ... --frozen --out runs/D/frozen
    shuffled  ... --shuffle-reinforcement runs/D/learn --shuffle-seed 1 --out runs/D/shuffled
    random    python -m stonkfly.control batch --match runs/D/learn --seeds 1-1000 --out runs/D/random

    python -m stonkfly.learning day runs/2026-10-12 --json results/learning/2026-10-12.json
    python -m stonkfly.learning report results/learning/2026-10-*.json --md results/learning/summary.md
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from .control import compare, events, outcome, provenance

ARMS = ("learn", "frozen", "shuffled")
OWN = "own marked-to-bid portfolio P&L"
DAYS = 10
# Two-sided 95% Student t critical value for 9 degrees of freedom (10 days).
T_975_DF9 = 2.262157162740992


def first(rows, test):
    """1-based number of the first observation passing test, or None."""
    return next((i + 1 for i, r in enumerate(rows) if test(r)), None)


def timeline(rows, neural_ms):
    kc = [r["neural"]["KC_spikes"] for r in rows if "KC_spikes" in r["neural"]]
    return {
        "first_pulse_observation": first(rows, lambda r: r["reinforcement_delivered"] != "none"),
        "first_memory_change_observation": first(
            rows, lambda r: r["neural"]["memory"]["changed_edges"] > 0),
        "mean_kc_spikes": statistics.fmean(kc) if kc else None,
        # A brain restored from an earlier run would not start at neural time 0.
        "fresh_start": bool(rows) and rows[0]["neural"]["brain_ms"] == neural_ms,
    }


def divergence(a, b):
    """Where two runs on one tape first differ in spikes and in proposals."""
    n = min(len(a), len(b))
    pairs = list(zip(a[:n], b[:n]))
    return {
        "first_spike_difference_observation": first(
            pairs, lambda p: p[0]["neural"]["spike_sha256"] != p[1]["neural"]["spike_sha256"]),
        "first_proposal_difference_observation": first(
            pairs, lambda p: p[0]["neural"]["side"] != p[1]["neural"]["side"]),
        "observations_with_different_proposal": sum(
            x["neural"]["side"] != y["neural"]["side"] for x, y in pairs),
        "compared_observations": n,
    }


def day(root, as_json=None):
    """One test day: the three fly arms, the random distribution and checks."""
    root = Path(root)
    runs = {arm: outcome(root / arm) for arm in ARMS}
    rows = {arm: events(root / arm) for arm in ARMS}
    for arm in ARMS:
        runs[arm].update(timeline(rows[arm], provenance(root / arm)["settings"]["neural_ms"]))
    random_runs = sorted((root / "random").glob("seed-*/ledger.sqlite"),
                         key=lambda p: int(p.parent.name.removeprefix("seed-")))
    random_rows = [outcome(p.parent) for p in random_runs]
    learn = runs["learn"]
    tape = provenance(root / "learn").get("tape") or {}
    shuffle = runs["shuffled"]["reinforcement"]
    tapes = {r["tape_sha256"] for r in [*runs.values(), *random_rows]}
    checks = {
        "same_tape": len(tapes) == 1 and None not in tapes,
        "full_tape": all(r["observations"] == tape.get("observations") for r in runs.values()),
        "finished_with_memory_record": all(r["memory"] is not None for r in runs.values()),
        "fresh_start": all(r["fresh_start"] for r in runs.values()),
        "learning_on_off": (learn["learning"], runs["frozen"]["learning"],
                            runs["shuffled"]["learning"]) == (True, False, True),
        "own_pnl_reinforcement": learn["reinforcement"] == OWN
        and runs["frozen"]["reinforcement"] == OWN,
        "shuffled_from_learn": isinstance(shuffle, dict)
        and Path(shuffle["source"]).resolve() == (root / "learn").resolve(),
        "shuffled_signal_counts_match_learn": runs["shuffled"]["signals_delivered"]
        == learn["signals_delivered"],
        "frozen_memory_unchanged": (runs["frozen"]["memory"] or {}).get("changed_edges") == 0,
        "random_matched_to_learn": bool(random_rows) and all(
            (r["buy_proposals"], r["sell_proposals"], r["hold_proposals"])
            == (learn["buy_proposals"], learn["sell_proposals"], learn["hold_proposals"])
            for r in random_rows),
        "random_full_tape": bool(random_rows) and all(
            r["observations"] == tape.get("observations") for r in random_rows),
        "no_halts":not any(r["halted"] for r in [*runs.values(), *random_rows]),
    }
    report = {
        "date": root.name,
        "tape": tape,
        "runs": runs,
        "learn_minus_frozen": learn["pnl"] - runs["frozen"]["pnl"],
        "learn_minus_shuffled": learn["pnl"] - runs["shuffled"]["pnl"],
        "divergence_from_learn": {arm: divergence(rows["learn"], rows[arm])
                                  for arm in ("frozen", "shuffled")},
        "random": None,
        "learn_vs_random": None,
        "checks": checks,
    }
    if random_rows:
        pnls = [r["pnl"] for r in random_rows]
        fills = [r["fills"] for r in random_rows]
        report["random"] = {
            "count": len(pnls),
            "mean_pnl": statistics.fmean(pnls),
            "sd_pnl": statistics.pstdev(pnls),
            "p5_pnl": float(np.percentile(pnls, 5)),
            "median_pnl": float(np.median(pnls)),
            "p95_pnl": float(np.percentile(pnls, 95)),
            "mean_fills": statistics.fmean(fills),
            # Per-seed raw outcomes: run folders stay local, so keep them here.
            "seeds": [int(p.parent.name.removeprefix("seed-")) for p in random_runs],
            "pnls": pnls,
            "fills": fills,
        }
        v = compare(learn["pnl"], pnls)
        # Rank 1 = best of learn + N random; ties count against learn.
        report["learn_vs_random"] = {**v, "rank": 1 + v["random_at_least_as_good"],
                                     "of": 1 + len(pnls)}
    if as_json:
        Path(as_json).parent.mkdir(parents=True, exist_ok=True)
        Path(as_json).write_text(json.dumps(report, indent=2) + "\n")
    return report


def interval(values, t=T_975_DF9):
    """Mean difference with a two-sided 95% t interval (days are the unit)."""
    values = [float(x) for x in values]
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    half = t * sd / math.sqrt(len(values))
    return {
        "mean": mean,
        "sd": sd,
        "ci95": [mean - half, mean + half],
        "t": mean / (sd / math.sqrt(len(values))) if sd else None,
        "days_better": sum(x > 0 for x in values),
        "days_equal": sum(x == 0 for x in values),
        "days_worse": sum(x < 0 for x in values),
    }


def report(files, as_json=None, md=None):
    """Preregistered analyses 1-3 over the ten test days."""
    days = sorted((json.loads(Path(f).read_text()) for f in files), key=lambda d: d["date"])
    dates = [d["date"] for d in days]
    if len(days) != DAYS or len(set(dates)) != DAYS:
        raise SystemExit(f"The preregistered analysis needs exactly {DAYS} distinct days")
    if any(d["learn_vs_random"] is None for d in days):
        raise SystemExit("Every day needs its random-trader distribution")
    frozen = interval([d["learn_minus_frozen"] for d in days])
    shuffled = interval([d["learn_minus_shuffled"] for d in days])
    ranks = [d["learn_vs_random"]["rank"] for d in days]
    improved = frozen["ci95"][0] > 0 and shuffled["ci95"][0] > 0
    result = {
        "dates": dates,
        "learn_minus_frozen": frozen,
        "learn_minus_shuffled": shuffled,
        "learn_vs_random": {
            "p_values": [d["learn_vs_random"]["p_value"] for d in days],
            "ranks": ranks,
            "mean_rank": statistics.fmean(ranks),
            "of": days[0]["learn_vs_random"]["of"],
        },
        "verdict": "학습이 성과를 개선했다" if improved else "개선 증거 없음",
        "failed_checks": {d["date"]: [k for k, ok in d["checks"].items() if not ok]
                          for d in days if not all(d["checks"].values())},
        "exploratory_fills": {
            arm: fills_note(days, arm) for arm, s in (("frozen", frozen), ("shuffled", shuffled))
            if s["mean"] > 0
        },
    }
    if as_json:
        Path(as_json).write_text(json.dumps(result, indent=2) + "\n")
    if md:
        Path(md).write_text(markdown(days, result))
    return result


def fills_note(days, arm):
    """Exploratory: is a learn advantage over arm explained by different fill counts?"""
    rows = [{"date": d["date"], "pnl_difference": d[f"learn_minus_{arm}"],
             "fill_difference": d["runs"]["learn"]["fills"] - d["runs"][arm]["fills"]}
            for d in days]
    same = [r["pnl_difference"] for r in rows if r["fill_difference"] == 0]
    return {"days": rows, "days_with_equal_fills": len(same),
            "mean_pnl_difference_equal_fills": statistics.fmean(same) if same else None}


def money(x):
    return f"{x:+.4f}"


def markdown(days, r):
    f, s, v = r["learn_minus_frozen"], r["learn_minus_shuffled"], r["learn_vs_random"]
    out = [
        "# 학습 대조 실험 결과: 사전 지정 분석",
        "",
        "사전등록: [docs/preregistration-learning.md](../../docs/preregistration-learning.md). "
        "날짜별 원자료: 이 폴더의 `2026-10-DD.json`. 손익은 수수료를 반영한 모의 손익(마지막 매수호가 기준, USDC). "
        "모의거래만 했고 실제 주문은 없다.",
        "",
        f"## 판정: {r['verdict']}",
        "",
        "사전 기준: ①과 ②의 95% 신뢰구간 하한이 모두 0보다 클 때만 \"학습이 성과를 개선했다\".",
        "",
        "| 비교 | 10일 평균 차이 ($) | 95% 신뢰구간 (t, 자유도 9) | learn이 나은 날 / 같은 날 / 못한 날 |",
        "|---|---|---|---|",
    ]
    for name, x in (("① learn − frozen", f), ("② learn − shuffled", s)):
        out.append(f"| {name} | {money(x['mean'])} | [{money(x['ci95'][0])}, {money(x['ci95'][1])}] "
                   f"| {x['days_better']} / {x['days_equal']} / {x['days_worse']} |")
    out += ["", f"③ learn의 무작위 투자자 대비 평균 순위: **{v['mean_rank']:.1f} / {v['of']}** "
            "(1 = 최고, 동점은 learn에 불리하게).", "", "## 날짜별 손익", "",
            "| 날짜 | learn | frozen | shuffled | ① learn − frozen | ② learn − shuffled "
            "| 무작위 평균 [5%, 95%] | ③ p | 순위 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for d in days:
        runs, rv, lr = d["runs"], d["random"], d["learn_vs_random"]
        out.append(f"| {d['date']} | {money(runs['learn']['pnl'])} | {money(runs['frozen']['pnl'])} "
                   f"| {money(runs['shuffled']['pnl'])} | {money(d['learn_minus_frozen'])} "
                   f"| {money(d['learn_minus_shuffled'])} | {money(rv['mean_pnl'])} "
                   f"[{money(rv['p5_pnl'])}, {money(rv['p95_pnl'])}] | {lr['p_value']:.3f} "
                   f"| {lr['rank']} |")
    out += ["", "p = (1 + 손익이 learn 이상인 무작위 수) / 1001. 무작위 투자자는 learn의 제안을 순서만 섞었다.",
            "", "## 함께 보고: 제안, 체결, 기억, 스파이크", "",
            "| 날짜 | 실행 | 제안 BUY/SELL/HOLD | 체결 | 거절 | 관측당 스파이크 | 관측당 KC 스파이크 "
            "| 보상/혐오 신호 | 바뀐 시냅스 | 평균 효율 | 첫 신호 → 첫 기억 변화 (관측) "
            "| learn과 처음 다른 스파이크 / 제안 (관측) | 다른 제안 수 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in days:
        for arm in ARMS:
            x = d["runs"][arm]
            m = x["memory"] or {}
            sig = x["signals_delivered"]
            dv = d["divergence_from_learn"].get(arm)
            div = (f"{dv['first_spike_difference_observation'] or '-'} / "
                   f"{dv['first_proposal_difference_observation'] or '-'}") if dv else "-"
            diff = dv["observations_with_different_proposal"] if dv else "-"
            out.append(
                f"| {d['date']} | {arm} | {x['buy_proposals']}/{x['sell_proposals']}/{x['hold_proposals']} "
                f"| {x['fills']} | {x['vetoes']} | {x['mean_total_spikes']:,.0f} "
                f"| {x['mean_kc_spikes']:.1f} | {sig['reward']}/{sig['aversive']} "
                f"| {m.get('changed_edges', '-')} | {m.get('mean_efficacy', float('nan')):.6f} "
                f"| {x['first_pulse_observation'] or '-'} → {x['first_memory_change_observation'] or '-'} "
                f"| {div} | {diff} |")
    if r["exploratory_fills"]:
        out += ["", "## 탐색 분석 (사전 판정에 쓰지 않음): 체결 수 차이", ""]
        for arm, note in r["exploratory_fills"].items():
            mean = note["mean_pnl_difference_equal_fills"]
            out += [f"learn − {arm} 평균이 0보다 커서 사전등록대로 확인했다. 체결 수가 같은 날 "
                    f"{note['days_with_equal_fills']}일의 평균 손익 차이: "
                    f"{money(mean) if mean is not None else '해당 없음'}.", "",
                    "| 날짜 | 손익 차이 ($) | 체결 수 차이 |", "|---|---|---|"]
            out += [f"| {x['date']} | {money(x['pnl_difference'])} | {x['fill_difference']:+d} |"
                    for x in note["days"]]
            out.append("")
    out += ["", "## 점검", ""]
    if r["failed_checks"]:
        out += [f"- {date}: 실패한 점검 {', '.join(k)}" for date, k in r["failed_checks"].items()]
    else:
        out.append("10일 모두 날짜별 점검(같은 테이프, 끝까지 실행, 새 뇌로 시작, 학습 켬/끔, "
                   "shuffled 신호 개수 = learn, frozen 시냅스 변화 0, 무작위 제안 수 = learn, 무작위도 끝까지 실행, 정지 없음)을 통과했다.")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(prog="python -m stonkfly.learning")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("day", help="Summarize one test day's learn/frozen/shuffled/random runs")
    d.add_argument("root", type=Path)
    d.add_argument("--json", type=Path)
    r = sub.add_parser("report", help="Preregistered analyses over the ten day summaries")
    r.add_argument("files", nargs="+", type=Path)
    r.add_argument("--json", type=Path)
    r.add_argument("--md", type=Path)
    a = parser.parse_args()
    if a.command == "day":
        result = day(a.root, a.json)
        print(json.dumps({k: result[k] for k in
                          ("date", "learn_minus_frozen", "learn_minus_shuffled",
                           "learn_vs_random", "checks")}, indent=2))
    else:
        result = report(a.files, a.json, a.md)
        print(json.dumps({k: result[k] for k in
                          ("learn_minus_frozen", "learn_minus_shuffled", "learn_vs_random",
                           "verdict", "failed_checks")}, indent=2))


if __name__ == "__main__":
    main()
