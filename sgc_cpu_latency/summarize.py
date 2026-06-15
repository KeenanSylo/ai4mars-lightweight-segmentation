"""Aggregate the per-config JSON files in results/ into a single table."""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

ORDER = [
    "v2_R11_MNv4-S_512_baseline",
    "v2_R11_MNv4-S_1024_baseline",
]

rows = []
for name in ORDER:
    p = RESULTS_DIR / f"{name}_cpu_latency.json"
    if not p.is_file():
        rows.append((name, "MISSING", "-", "-", "-", "-", "-"))
        continue
    d = json.loads(p.read_text())
    rows.append((
        name,
        d["mode"],
        f"{d['sgc2_latency_avg_ms']:.1f}",
        f"{d['sgc2_latency_p95_ms']:.1f}",
        f"{d['sgc2_latency_max_ms']:.1f}",
        "PASS" if d["sgc2_latency_pass"] else "FAIL",
        f"{d['wall_clock_seconds']:.0f}",
    ))

header = ("run_name", "mode", "avg_ms", "p95_ms", "max_ms", "@500ms", "wall_s")
widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]

def fmt(row):
    return "  ".join(str(r).ljust(w) for r, w in zip(row, widths))

print(fmt(header))
print("  ".join("-" * w for w in widths))
for r in rows:
    print(fmt(r))
