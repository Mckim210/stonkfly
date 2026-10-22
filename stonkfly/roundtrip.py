"""Trade-level reinforcement: an experimental alternative to portfolio P&L.

Default runs are unchanged: dopamine pulses follow marked-to-bid portfolio P&L
(reinforcement.py). With `--reinforcement roundtrip` a pulse follows only the
realized P&L of a filled sell, matched FIFO to the buys it closes, after both
buy and sell fees. It is delivered at the next observation, like portfolio P&L.
`roundtrip-replay` also runs one extra neural step right after the fill: the
chart the fly saw at the first matched buy, with the same pulse. That step is
never decoded, so it cannot propose a trade. See docs/roundtrip.md.

    python -m stonkfly run --replay T --reinforcement roundtrip-replay --out runs/x
    python -m stonkfly.roundtrip pilot runs/pilot-roundtrip --json r.json --md r.md
"""

import argparse
import copy
import hashlib
import json
import sqlite3
import statistics
from pathlib import Path

import numpy as np

from .config import D
from .display import market_frame

MODES = ("portfolio", "roundtrip", "roundtrip-replay")
OWN = "own marked-to-bid portfolio P&L"
DESCRIPTION = {
    "roundtrip": "own realized round-trip P&L: each filled sell, FIFO cost incl. buy and sell fees, signed at the next observation",
    "roundtrip-replay": "own realized round-trip P&L: each filled sell, FIFO cost incl. buy and sell fees, signed at the next observation, plus one undecoded replay step of the first FIFO buy's chart with the same pulse",
}


def orders(db):
    """Every order intent in creation order, with its final settlement if any."""
    return [
        {"id": r[0], "status": r[1], "plan": json.loads(r[2]),
         "settlement": json.loads(r[3]) if r[3] else None}
        for r in db.execute("SELECT id,status,plan,settlement FROM orders ORDER BY rowid")
    ]


def new_state():
    return {"orders_seen": 0, "lots": {}}


def book(lots, order, observation):
    """Apply one settled order to FIFO lots; return the sale a filled sell realizes."""
    p, s = order["plan"], order["settlement"]
    base, quote, fee = D(s["base"]), D(s["quote"]), D(s["fee"])
    if base == 0:
        return None
    held = lots.setdefault(p["product"], [])
    if p["side"] == "BUY":
        held.append({"order": order["id"], "observation": observation, "base": str(base),
                     "quote": str(quote), "fee": str(fee), "open": str(base)})
        return None
    remaining, cost, buy_fees, matched = base, D(0), D(0), []
    while remaining > 0:
        if not held:
            raise RuntimeError("Sell exceeds the FIFO-booked buys")
        lot = held[0]
        take = min(remaining, D(lot["open"]))
        cost += D(lot["quote"]) * take / D(lot["base"])
        buy_fees += D(lot["fee"]) * take / D(lot["base"])
        matched.append({"buy_order": lot["order"], "buy_observation": lot["observation"],
                        "base": str(take)})
        lot["open"] = str(D(lot["open"]) - take)
        remaining -= take
        if D(lot["open"]) == 0:
            held.pop(0)
    return {
        "sell_order": order["id"],
        "sell_observation": observation,
        "product": p["product"],
        "base": str(base),
        "proceeds": str(quote),
        "cost": str(cost),
        "buy_fees": str(buy_fees),
        "sell_fee": str(fee),
        "before_fees": str(quote - cost),
        "after_fees": str(quote - cost - buy_fees - fee),
        "matched": matched,
    }


def sync(db, state, observation):
    """Book the orders settled since state. Each loop iteration places at most
    one order after its tick is committed, so new orders belong to `observation`,
    the last committed tick. Returns (new state, sale or None)."""
    state = copy.deepcopy(state)
    sale = None
    filled = 0
    for o in orders(db)[state["orders_seen"]:]:
        if o["status"] not in ("SETTLED", "REJECTED"):
            raise RuntimeError("Unreconciled order; round-trip accounting needs final fills")
        state["orders_seen"] += 1
        if o["status"] == "SETTLED":
            filled += 1
            sale = book(state["lots"], o, observation) or sale
    if filled > 1:
        raise RuntimeError("More than one fill since the last observation")
    return state, sale


def signal(sale):
    if sale is None:
        return "none"
    x = D(sale["after_fees"])
    return "reward" if x > 0 else "aversive" if x < 0 else "none"


def buy_frame(db, order_id):
    """Rebuild, from the buy's immutable order intent, the chart the fly saw."""
    plan = json.loads(db.execute("SELECT plan FROM orders WHERE id=?", (order_id,)).fetchone()[0])
    obs = plan["neural_observation"]
    product = obs["product"]
    frame = market_frame(product, obs["market_history"][product],
                         D(obs["quote"]["bid"]), D(obs["quote"]["ask"]))
    if hashlib.sha256(frame.tobytes()).hexdigest() != obs["neural"]["input_sha256"]:
        raise RuntimeError("Cannot reproduce the chart the fly saw at the buy")
    return frame


def edges_changed(before, after):
    return int(np.count_nonzero(before != after))


def replay(controller, db, sale, kind):
    """One extra, undecoded neural step: the first FIFO buy's chart with this sale's pulse."""
    first = sale["matched"][0]
    frame = buy_frame(db, first["buy_order"])
    before = controller.plastic_weights()
    neural = controller.replay(frame, kind)
    return {
        "buy_observation": first["buy_observation"],
        "buy_order": first["buy_order"],
        **neural,
        "kc_active": controller.kc_active(),
        "edges_changed_in_step": edges_changed(before, controller.plastic_weights()),
    }


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else None


def sign_counts(sales):
    x = [D(s["after_fees"]) for s in sales]
    return {"positive": sum(v > 0 for v in x), "negative": sum(v < 0 for v in x),
            "zero": sum(v == 0 for v in x)}


def summary(rows, undelivered=None):
    """Run summary: trade-level signals, sells by realized sign, replay KC overlap."""
    by_tick = {r["tick"]: r for r in rows}
    sold = [r for r in rows if r["roundtrip"]["sale"]]
    pulsed = [r for r in sold if r["reinforcement_delivered"] != "none"]
    overlaps = []
    for r in rows:
        x = r["roundtrip"]["replay"]
        if not x:
            continue
        buy = by_tick.get(x["buy_observation"])
        overlaps.append({
            "observation": r["tick"],
            "buy_observation": x["buy_observation"],
            "replay_kcs": len(x["kc_active"]),
            "buy_kcs": len(buy["roundtrip"]["kc_active"]) if buy else None,
            "jaccard": jaccard(x["kc_active"], buy["roundtrip"]["kc_active"]) if buy else None,
        })
    values = [x["jaccard"] for x in overlaps if x["jaccard"] is not None]
    sales = [r["roundtrip"]["sale"] for r in sold] + ([undelivered] if undelivered else [])
    return {
        "trade_signals": {s: sum(r["reinforcement_delivered"] == s for r in sold)
                          for s in ("reward", "aversive")},
        "trade_pulses_including_replay": sum(
            1 + bool(r["roundtrip"]["replay"]) for r in pulsed),
        "sells_by_realized_sign": sign_counts(sales),
        "undelivered_final_sale": undelivered,
        "pulse_memory": [{
            "observation": r["tick"],
            "signal": r["reinforcement_delivered"],
            "changed_edges_before": r["roundtrip"]["memory_before"]["changed_edges"],
            "changed_edges_after_replay": (r["roundtrip"]["replay"] or {}).get(
                "memory", {}).get("changed_edges"),
            "changed_edges_after": r["neural"]["memory"]["changed_edges"],
            "mean_efficacy_before": r["roundtrip"]["memory_before"]["mean_efficacy"],
            "mean_efficacy_after": r["neural"]["memory"]["mean_efficacy"],
        } for r in pulsed],
        "replay_kc_jaccard": {
            "mean": statistics.fmean(values) if values else None,
            "count": len(values),
            "undefined": len(overlaps) - len(values),
            "steps": overlaps,
        },
    }


# --- Pilot report (docs/pilot-roundtrip.md) -------------------------------------

ARMS = ("portfolio", "frozen", "roundtrip", "roundtrip-replay")
LEARNING = {"portfolio": True, "frozen": False, "roundtrip": True, "roundtrip-replay": True}
REINFORCEMENT = {"portfolio": OWN, "frozen": OWN, **DESCRIPTION}


def ledger_sales(run):
    """FIFO-realized sells of any finished run, recomputed from its ledger."""
    db = sqlite3.connect(f"file:{Path(run) / 'ledger.sqlite'}?mode=ro", uri=True)
    try:
        lots, sales = {}, []
        for o in orders(db):
            if o["status"] == "SETTLED":
                sale = book(lots, o, None)
                if sale:
                    sales.append(sale)
        return sales
    finally:
        db.close()


def criteria(rows, frozen, run_summary):
    """The four preregistered go/no-go criteria for the roundtrip-replay run."""
    pulsed = [r for r in rows if r["roundtrip"]["sale"] and r["reinforcement_delivered"] != "none"]
    memory = [{"observation": r["tick"],
               "before": r["roundtrip"]["memory_before"]["changed_edges"],
               "after": r["neural"]["memory"]["changed_edges"]} for r in pulsed]
    jac = run_summary["replay_kc_jaccard"]["mean"]
    start = pulsed[0]["tick"] if pulsed else None
    other = {r["tick"]: r["neural"]["side"] for r in frozen}
    differ = [r["tick"] for r in rows
              if r["tick"] in other and r["neural"]["side"] != other[r["tick"]]]
    after = [t for t in differ if start is not None and t >= start]
    return {
        "1_trade_signals": {"value": len(pulsed), "threshold": 5, "pass": len(pulsed) >= 5},
        "2_memory_changes_after_pulse": {
            "pulses": len(memory),
            "pulses_with_increase": sum(m["after"] > m["before"] for m in memory),
            "steps": memory,
            "pass": any(m["after"] > m["before"] for m in memory),
        },
        "3_replay_kc_jaccard": {"value": jac, "threshold": 0.5,
                                "pass": jac is not None and jac >= 0.5},
        "4_differs_from_frozen_after_first_pulse": {
            "first_pulse_observation": start,
            "different_after": len(after),
            "first_different_after": after[0] if after else None,
            "different_before": len(differ) - len(after),
            "first_different_overall": differ[0] if differ else None,
            "pass": bool(after),
        },
    }


REASONS = {
    "1_trade_signals": "① 3시간 동안 거래 단위 보상+혐오 신호가 5번 미만",
    "2_memory_changes_after_pulse": "② 거래 단위 펄스 뒤 changed_edges가 늘어난 적이 없음",
    "3_replay_kc_jaccard": "③ replay 단계와 원래 매수 시점의 KC 집합 평균 Jaccard가 0.5 미만(또는 계산 불가)",
    "4_differs_from_frozen_after_first_pulse": "④ 첫 거래 단위 펄스 뒤 frozen과 제안이 다른 관측이 없음",
}


def pilot(root, as_json=None, md=None):
    from .control import events, outcome, provenance
    from .learning import timeline

    root = Path(root)
    runs, rows, finals = {}, {}, {}
    for arm in ARMS:
        runs[arm] = outcome(root / arm)
        rows[arm] = events(root / arm)
        final = root / arm / "final.json"
        finals[arm] = json.loads(final.read_text()) if final.exists() else {}
        prov = provenance(root / arm)
        runs[arm].update(timeline(rows[arm], prov["settings"]["neural_ms"]))
        runs[arm]["sells_by_realized_sign"] = sign_counts(ledger_sales(root / arm))
    tape = provenance(root / "portfolio").get("tape") or {}
    rr = rows["roundtrip-replay"]
    summaries = {arm: finals[arm].get("roundtrip") for arm in ("roundtrip", "roundtrip-replay")}
    result = criteria(rr, rows["frozen"], summaries["roundtrip-replay"])
    reference = criteria(rows["roundtrip"], rows["frozen"], summaries["roundtrip"])
    by_tick = {r["tick"]: r for r in rr}
    replays = [r for r in rr if r["roundtrip"]["replay"]]
    checks = {
        "same_tape": len({r["tape_sha256"] for r in runs.values()}) == 1
        and tape.get("sha256") is not None,
        "full_tape": all(r["observations"] == tape.get("observations") for r in runs.values()),
        "fresh_start": all(r["fresh_start"] for r in runs.values()),
        "learning_on_off": all(runs[a]["learning"] is LEARNING[a] for a in ARMS),
        "reinforcement_modes": all(runs[a]["reinforcement"] == REINFORCEMENT[a] for a in ARMS),
        "finished_with_memory_record": all(runs[a]["memory"] is not None for a in ARMS),
        "frozen_memory_unchanged": (runs["frozen"]["memory"] or {}).get("changed_edges") == 0,
        "no_halts": not any(r["halted"] for r in runs.values()),
        "signal_is_sign_of_realized_pnl": all(
            r["reinforcement_delivered"] == r["reinforcement_earned"] == signal(r["roundtrip"]["sale"])
            for a in ("roundtrip", "roundtrip-replay") for r in rows[a]),
        "fifo_matches_ledger": all(
            [r["roundtrip"]["sale"]["after_fees"] for r in rows[a] if r["roundtrip"]["sale"]]
            + ([finals[a]["roundtrip"]["undelivered_final_sale"]["after_fees"]]
               if (finals[a].get("roundtrip") or {}).get("undelivered_final_sale") else [])
            == [s["after_fees"] for s in ledger_sales(root / a)]
            for a in ("roundtrip", "roundtrip-replay")),
        "replay_only_in_replay_arm": not any(r["roundtrip"]["replay"] for r in rows["roundtrip"])
        and all(bool(r["roundtrip"]["replay"]) == bool(r["roundtrip"]["sale"]) for r in rr),
        "replay_shows_first_fifo_buy_chart": all(
            r["roundtrip"]["replay"]["buy_observation"]
            == r["roundtrip"]["sale"]["matched"][0]["buy_observation"]
            and r["roundtrip"]["replay"]["input_sha256"]
            == by_tick[r["roundtrip"]["replay"]["buy_observation"]]["neural"]["input_sha256"]
            for r in replays),
    }
    failed = [k for k, v in result.items() if not v["pass"]]
    report = {
        "tape": tape,
        "runs": runs,
        "roundtrip_summaries": summaries,
        "criteria": result,
        "verdict": "정식 실험 가능" if not failed else "정식 실험 안 함",
        "failure_reasons": [REASONS[k] for k in failed],
        "reference_roundtrip_arm_criteria": reference,
        "checks": checks,
    }
    if as_json:
        Path(as_json).parent.mkdir(parents=True, exist_ok=True)
        Path(as_json).write_text(json.dumps(report, indent=2) + "\n")
    if md:
        Path(md).write_text(markdown(report))
    return report


def money(x):
    return f"{x:+.4f}"


def ok(x):
    return "통과" if x else "실패"


def markdown(r):
    c = r["criteria"]
    c1, c2 = c["1_trade_signals"], c["2_memory_changes_after_pulse"]
    c3, c4 = c["3_replay_kc_jaccard"], c["4_differs_from_frozen_after_first_pulse"]
    t = r["tape"]
    jac = "계산 불가" if c3["value"] is None else f"{c3['value']:.3f}"
    out = [
        "# 파일럿 결과: 거래 단위(실현 손익) 보상",
        "",
        "계획: [docs/pilot-roundtrip.md](../docs/pilot-roundtrip.md). 원자료 요약: "
        "[pilot-roundtrip.json](pilot-roundtrip.json). 모의거래만 했고 실제 주문은 없다.",
        "",
        f"테이프: `{Path(t.get('path', '')).name}`, 관측 {t.get('observations')}개, "
        f"SHA-256 `{t.get('sha256')}`.",
        "",
        f"## 판정: {r['verdict']}",
        "",
        "| 기준 (roundtrip-replay) | 측정값 | 기준값 | 결과 |",
        "|---|---|---|---|",
        f"| ① 거래 단위 보상+혐오 신호 수 | {c1['value']} | ≥ 5 | {ok(c1['pass'])} |",
        f"| ② 펄스 뒤 changed_edges 증가 | 펄스 {c2['pulses']}번 중 {c2['pulses_with_increase']}번 증가 "
        f"| 1번 이상 | {ok(c2['pass'])} |",
        f"| ③ replay KC 집합 vs 원래 매수 시점 KC 집합 평균 Jaccard | {jac} | ≥ 0.5 | {ok(c3['pass'])} |",
        f"| ④ 첫 거래 단위 펄스(관측 {c4['first_pulse_observation'] or '-'}) 뒤 frozen과 다른 제안 "
        f"| {c4['different_after']}관측 | 1관측 이상 | {ok(c4['pass'])} |",
        "",
    ]
    if r["failure_reasons"]:
        out += ["실패 이유:", ""] + [f"- {x}" for x in r["failure_reasons"]] + [""]
    out += [
        "## 실행별 요약 (손익은 참고용, 판정에 쓰지 않음)",
        "",
        "| 실행 | 학습 | 제안 BUY/SELL/HOLD | 체결 | 거절 | 전달 신호 보상/혐오 | 실현 손익 +/−/0 매도 "
        "| 바뀐 시냅스 | 평균 효율 | 관측당 KC 스파이크 | 손익 ($) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        x = r["runs"][arm]
        m = x["memory"] or {}
        s = x["signals_delivered"]
        z = x["sells_by_realized_sign"]
        kc = "-" if x["mean_kc_spikes"] is None else f"{x['mean_kc_spikes']:.1f}"
        out.append(
            f"| {arm} | {'켬' if x['learning'] else '끔'} | {x['buy_proposals']}/{x['sell_proposals']}/"
            f"{x['hold_proposals']} | {x['fills']} | {x['vetoes']} | {s['reward']}/{s['aversive']} "
            f"| {z['positive']}/{z['negative']}/{z['zero']} | {m.get('changed_edges', '-')} "
            f"| {m.get('mean_efficacy', float('nan')):.6f} | {kc} | {money(x['pnl'])} |")
    out += ["", "## 거래 단위 신호 (roundtrip 두 실행)", "",
            "| 실행 | 보상 | 혐오 | 펄스 수(replay 포함) | 실현 손익 +/−/0 매도 | 전달 못 한 마지막 매도 |",
            "|---|---|---|---|---|---|"]
    for arm, s in r["roundtrip_summaries"].items():
        if not s:
            continue
        z = s["sells_by_realized_sign"]
        out.append(f"| {arm} | {s['trade_signals']['reward']} | {s['trade_signals']['aversive']} "
                   f"| {s['trade_pulses_including_replay']} | {z['positive']}/{z['negative']}/{z['zero']} "
                   f"| {'있음' if s['undelivered_final_sale'] else '없음'} |")
    rr = r["roundtrip_summaries"].get("roundtrip-replay") or {}
    steps = {x["observation"]: x for x in rr.get("replay_kc_jaccard", {}).get("steps", [])}
    out += ["", "## roundtrip-replay: 펄스마다 기억과 KC 겹침", "",
            "| 관측 | 신호 | 매수 관측 | changed_edges 전 → replay 뒤 → 관측 끝 | 평균 효율 전 → 끝 "
            "| replay KC / 매수 KC | Jaccard |",
            "|---|---|---|---|---|---|---|"]
    for p in rr.get("pulse_memory", []):
        j = steps.get(p["observation"], {})
        jv = j.get("jaccard")
        out.append(
            f"| {p['observation']} | {p['signal']} | {j.get('buy_observation', '-')} "
            f"| {p['changed_edges_before']} → {p['changed_edges_after_replay']} → {p['changed_edges_after']} "
            f"| {p['mean_efficacy_before']:.6f} → {p['mean_efficacy_after']:.6f} "
            f"| {j.get('replay_kcs', '-')} / {j.get('buy_kcs', '-')} "
            f"| {'-' if jv is None else f'{jv:.3f}'} |")
    ref = r["reference_roundtrip_arm_criteria"]["4_differs_from_frozen_after_first_pulse"]
    out += ["", "## ④ 보충", "",
            f"roundtrip-replay와 frozen의 제안이 처음 달라진 관측: {c4['first_different_overall'] or '없음'}. "
            f"첫 거래 단위 펄스 전에 이미 다른 관측: {c4['different_before']}개, 뒤: {c4['different_after']}개. "
            f"참고로 roundtrip 실행은 첫 거래 단위 펄스(관측 {ref['first_pulse_observation'] or '-'}) 뒤 "
            f"frozen과 {ref['different_after']}관측에서 달랐다.",
            "", "## 점검", "",
            "| 점검 | 결과 |", "|---|---|"]
    out += [f"| {k} | {ok(v)} |" for k, v in r["checks"].items()]
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(prog="python -m stonkfly.roundtrip")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pilot", help="Summarize the four pilot runs and apply the go/no-go criteria")
    p.add_argument("root", type=Path)
    p.add_argument("--json", type=Path)
    p.add_argument("--md", type=Path)
    a = parser.parse_args()
    r = pilot(a.root, a.json, a.md)
    print(json.dumps({k: r[k] for k in ("verdict", "failure_reasons", "checks")}, indent=2,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
