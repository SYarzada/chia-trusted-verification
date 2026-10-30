#!/usr/bin/env python3

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("runs/rtl_formal_gemmini")
SEEDS = {200, 201, 202}


def load_json(path):
    with path.open() as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def get_formal_status(row):
    return (
        row.get("feedback", {}).get("formal_status")
        or row.get("metrics", {}).get("formal_status")
        or row.get("formal_status")
        or row.get("status")
        or "unknown"
    )


def metric_true(metrics, key):
    value = metrics.get(key, 0)
    try:
        return float(value or 0) > 0
    except (TypeError, ValueError):
        return bool(value)


runs = []

for final_path in sorted(ROOT.glob("*/seed2*/llm_agent/final_trusted_result.json")):
    intent = final_path.parts[-4]
    seed_dir = final_path.parts[-3]

    try:
        seed = int(seed_dir.removeprefix("seed"))
    except ValueError:
        continue

    if seed not in SEEDS:
        continue

    final = load_json(final_path)
    metrics = final.get("metrics", {})
    feedback = final.get("feedback", {})

    trusted_results_path = final_path.parents[1] / "trusted_results.jsonl"
    all_rows = load_jsonl(trusted_results_path)

    proposals = [r for r in all_rows if r.get("role") == "proposal"]
    proposals.sort(key=lambda r: r.get("evaluation_sequence", 10**9))

    runs.append(
        {
            "intent": intent,
            "seed": seed,
            "final": final,
            "metrics": metrics,
            "feedback": feedback,
            "formal_status": get_formal_status(final),
            "compile_ok": metric_true(metrics, "compile_ok"),
            "proved": metric_true(metrics, "proved"),
            "proposals": proposals,
        }
    )


if not runs:
    raise SystemExit(
        "No final Backend II runs found under runs/rtl_formal_gemmini/. "
        "Run this script from the repository root."
    )


print("=" * 72)
print("BACKEND II FINAL ANALYSIS")
print("=" * 72)
print(f"runs found: {len(runs)}")
print()


# ---------------------------------------------------------------------
# Overall final results
# ---------------------------------------------------------------------

print("=" * 72)
print("OVERALL FINAL RESULTS")
print("=" * 72)

overall = Counter(r["formal_status"] for r in runs)

preferred_order = [
    "proved",
    "counterexample",
    "timeout",
    "compile_error",
    "tool_error",
    "not_run",
    "unknown",
]

seen = set()

for status in preferred_order:
    if status in overall:
        count = overall[status]
        print(
            f"{status:20s} {count:3d} / {len(runs):3d} "
            f"= {100.0 * count / len(runs):5.1f}%"
        )
        seen.add(status)

for status in sorted(set(overall) - seen):
    count = overall[status]
    print(
        f"{status:20s} {count:3d} / {len(runs):3d} "
        f"= {100.0 * count / len(runs):5.1f}%"
    )

compile_ok = sum(r["compile_ok"] for r in runs)
proved = sum(r["proved"] for r in runs)

print()
print(
    f"compile success: {compile_ok}/{len(runs)} "
    f"= {100.0 * compile_ok / len(runs):.1f}%"
)
print(
    f"proved:          {proved}/{len(runs)} "
    f"= {100.0 * proved / len(runs):.1f}%"
)

if compile_ok:
    print(
        f"proved | compiled: {proved}/{compile_ok} "
        f"= {100.0 * proved / compile_ok:.1f}%"
    )


# ---------------------------------------------------------------------
# Per-intent results
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("PER-INTENT RESULTS")
print("=" * 72)

by_intent = defaultdict(list)

for r in runs:
    by_intent[r["intent"]].append(r)

for intent in sorted(by_intent):
    intent_runs = sorted(by_intent[intent], key=lambda r: r["seed"])
    counts = Counter(r["formal_status"] for r in intent_runs)

    seed_summary = ", ".join(
        f"{r['seed']}:{r['formal_status']}" for r in intent_runs
    )

    print(
        f"{intent:48s} "
        f"proved={counts['proved']} "
        f"cex={counts['counterexample']} "
        f"timeout={counts['timeout']} "
        f"compile_error={counts['compile_error']}"
    )
    print(f"  {seed_summary}")


# ---------------------------------------------------------------------
# Proposal accounting
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("PROPOSAL ACCOUNTING")
print("=" * 72)

proposal_counts = [len(r["proposals"]) for r in runs]
total_proposals = sum(proposal_counts)

print(f"proposal rows found: {total_proposals}")
print(f"expected maximum:    {len(runs) * 4}")
print(f"runs with 4 proposals: {sum(n == 4 for n in proposal_counts)}/{len(runs)}")

if proposal_counts:
    print(
        f"proposals/run: min={min(proposal_counts)} "
        f"max={max(proposal_counts)} "
        f"mean={sum(proposal_counts)/len(proposal_counts):.2f}"
    )


# ---------------------------------------------------------------------
# First proposal -> final result
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("FIRST PROPOSAL -> FINAL RESULT")
print("=" * 72)

transitions = Counter()

first_compile = 0
final_compile = 0
first_proved = 0
final_proved = 0
runs_with_first = 0

for r in runs:
    proposals = r["proposals"]

    if proposals:
        first = proposals[0]
        first_status = get_formal_status(first)
        first_metrics = first.get("metrics", {})
        runs_with_first += 1

        first_compile += metric_true(first_metrics, "compile_ok")
        first_proved += metric_true(first_metrics, "proved")
    else:
        first_status = "missing"

    final_status = r["formal_status"]

    transitions[(first_status, final_status)] += 1
    final_compile += r["compile_ok"]
    final_proved += r["proved"]

for (first_status, final_status), count in sorted(
    transitions.items(),
    key=lambda item: (-item[1], item[0][0], item[0][1]),
):
    print(f"{count:3d}  {first_status:20s} -> {final_status}")

print()

if runs_with_first:
    print(
        "First-proposal compile success: "
        f"{first_compile}/{runs_with_first} "
        f"({100.0 * first_compile / runs_with_first:.1f}%)"
    )
    print(
        "First-proposal proved:          "
        f"{first_proved}/{runs_with_first} "
        f"({100.0 * first_proved / runs_with_first:.1f}%)"
    )

print(
    "Final compile success:          "
    f"{final_compile}/{len(runs)} "
    f"({100.0 * final_compile / len(runs):.1f}%)"
)
print(
    "Final proved:                   "
    f"{final_proved}/{len(runs)} "
    f"({100.0 * final_proved / len(runs):.1f}%)"
)


# ---------------------------------------------------------------------
# Within-run repair analysis
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("WITHIN-RUN REPAIR")
print("=" * 72)

compile_failure_to_compiled_final = 0
first_cex_to_proved_final = 0
first_timeout_to_proved_final = 0
first_compile_error_to_proved_final = 0
any_proved_proposal = 0
proved_after_first_attempt = 0

for r in runs:
    proposals = r["proposals"]
    if not proposals:
        continue

    first = proposals[0]
    first_status = get_formal_status(first)
    first_metrics = first.get("metrics", {})

    final_proved_flag = r["proved"]
    final_compile_flag = r["compile_ok"]

    if not metric_true(first_metrics, "compile_ok") and final_compile_flag:
        compile_failure_to_compiled_final += 1

    if first_status == "counterexample" and final_proved_flag:
        first_cex_to_proved_final += 1

    if first_status == "timeout" and final_proved_flag:
        first_timeout_to_proved_final += 1

    if first_status == "compile_error" and final_proved_flag:
        first_compile_error_to_proved_final += 1

    proved_indices = [
        idx
        for idx, proposal in enumerate(proposals, start=1)
        if metric_true(proposal.get("metrics", {}), "proved")
    ]

    if proved_indices:
        any_proved_proposal += 1
        if min(proved_indices) > 1:
            proved_after_first_attempt += 1

print(
    "first compile failure -> compiled final: "
    f"{compile_failure_to_compiled_final}"
)
print(
    "first counterexample -> proved final:    "
    f"{first_cex_to_proved_final}"
)
print(
    "first timeout -> proved final:           "
    f"{first_timeout_to_proved_final}"
)
print(
    "first compile_error -> proved final:     "
    f"{first_compile_error_to_proved_final}"
)
print(
    "runs containing any proved proposal:    "
    f"{any_proved_proposal}/{len(runs)}"
)
print(
    "runs first proving after attempt 1:      "
    f"{proved_after_first_attempt}/{len(runs)}"
)


# ---------------------------------------------------------------------
# Attempt-by-attempt outcome distribution
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("OUTCOME BY PROPOSAL ATTEMPT")
print("=" * 72)

attempt_rows = defaultdict(list)

for r in runs:
    for idx, proposal in enumerate(r["proposals"], start=1):
        attempt_rows[idx].append(proposal)

for attempt in sorted(attempt_rows):
    rows = attempt_rows[attempt]
    counts = Counter(get_formal_status(row) for row in rows)
    proved_count = sum(
        metric_true(row.get("metrics", {}), "proved")
        for row in rows
    )
    compile_count = sum(
        metric_true(row.get("metrics", {}), "compile_ok")
        for row in rows
    )

    print(f"attempt {attempt}: n={len(rows)}")
    print(
        f"  compile success = {compile_count}/{len(rows)} "
        f"({100.0 * compile_count / len(rows):.1f}%)"
    )
    print(
        f"  proved          = {proved_count}/{len(rows)} "
        f"({100.0 * proved_count / len(rows):.1f}%)"
    )

    for status, count in sorted(counts.items()):
        print(f"  {status:18s} {count:3d}")


# ---------------------------------------------------------------------
# Intents with at least one proof
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("INTENT-LEVEL PROOF COVERAGE")
print("=" * 72)

intents_with_proof = 0
all_three_proved = 0

for intent, intent_runs in sorted(by_intent.items()):
    num_proved = sum(r["formal_status"] == "proved" for r in intent_runs)

    if num_proved > 0:
        intents_with_proof += 1
    if num_proved == len(intent_runs) == 3:
        all_three_proved += 1

print(
    f"intents with >=1 proved seed: {intents_with_proof}/{len(by_intent)} "
    f"({100.0 * intents_with_proof / len(by_intent):.1f}%)"
)
print(
    f"intents proved in all 3 seeds: {all_three_proved}/{len(by_intent)} "
    f"({100.0 * all_three_proved / len(by_intent):.1f}%)"
)


# ---------------------------------------------------------------------
# Infrastructure sanity check
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("INFRASTRUCTURE SANITY CHECK")
print("=" * 72)

infra_bad = []

for r in runs:
    if r["formal_status"] in {"tool_error", "infrastructure_error"}:
        infra_bad.append(
            (r["intent"], r["seed"], r["formal_status"])
        )

if infra_bad:
    print("WARNING: infrastructure-related final results found:")
    for item in infra_bad:
        print("  ", item)
else:
    print("No final tool_error/infrastructure_error results found.")

print()
print("Analysis complete.")
