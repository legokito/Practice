#!/usr/bin/env python3
"""Render benchmark/results.csv into assets/benchmark.png.

A horizontal bar chart of GFLOPs per kernel, sorted by the progression, with
each bar labeled by its % of the cuBLAS baseline. This is the visual embedded
in the README -- run it after every benchmark to refresh the chart.

    python benchmark/plot.py [results.csv] [assets/benchmark.png]
"""
import sys
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
IN = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "benchmark" / "results.csv"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "assets" / "benchmark.png"

rows = []
with open(IN) as f:
    for r in csv.DictReader(f):
        rows.append(r)

# cuBLAS (id 0) is the baseline / ceiling.
baseline = next((float(r["gflops"]) for r in rows if r["kernel_id"] == "0"), None)
size = rows[0]["size"] if rows else "?"

# Plot kernels in id order, cuBLAS last as the ceiling bar.
rows.sort(key=lambda r: int(r["kernel_id"]))
labels = [r["name"] for r in rows]
values = [float(r["gflops"]) for r in rows]
is_cublas = [r["kernel_id"] == "0" for r in rows]

plt.rcParams.update({"font.size": 11, "figure.facecolor": "white"})
fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1.6))

# Placeholder / illustrative data gets a visible banner so nobody mistakes it
# for a real measurement. plot.py stamps real runs automatically (no banner).
example = any("example" in r.get("status", "").lower() for r in rows) or "example" in IN.name

colors = ["#9aa0a6" if c else "#4285f4" for c in is_cublas]
bars = ax.barh(range(len(rows)), values, color=colors)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("GFLOPs (higher is better)")
ax.set_title(f"SGEMM kernel progression  ·  {size}x{size} FP32", fontweight="bold")

for i, (bar, val, cub) in enumerate(zip(bars, values, is_cublas)):
    pct = f"{100*val/baseline:.0f}% of cuBLAS" if baseline and not cub else "baseline"
    ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}  ({pct})", va="center", fontsize=9)

ax.set_xlim(0, max(values) * 1.25)
ax.spines[["top", "right"]].set_visible(False)

if example:
    fig.text(0.5, 0.5, "ILLUSTRATIVE — replace with your own run", ha="center",
             va="center", fontsize=20, color="#d93025", alpha=0.25, rotation=15,
             fontweight="bold")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
