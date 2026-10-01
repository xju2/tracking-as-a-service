#!/usr/bin/env python
"""Plot a histogram comparing FRNN and libFRNN latency from a frnn-eval metrics CSV."""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot(csv_path: Path, out_path: Path):
    df = pd.read_csv(csv_path)
    a, b = df["frnn_latency_ms"], df["libfrnn_latency_ms"]
    print(df[["frnn_latency_ms", "libfrnn_latency_ms"]].describe())

    fig, ax = plt.subplots(figsize=(8, 5))
    lo, hi = min(a.min(), b.min()) * 0.9, max(a.max(), b.max()) * 1.1
    bins = np.linspace(lo, hi, 40)
    for s, color, name in [(a, "#2a78d6", "FRNN"), (b, "#eb6834", "libFRNN")]:
        ax.hist(s, bins=bins, color=color, alpha=0.6, edgecolor="white", linewidth=1,
                label=f"{name}  (median {s.median():.1f} ms, mean {s.mean():.1f} ms)")
        ax.axvline(s.median(), color=color, linestyle="--", linewidth=2)
    ax.set_xlabel("Latency [ms]")
    ax.set_ylabel("Requests")
    ax.set_title(f"FRNN vs libFRNN latency  (N={len(df)}, "
                 f"speedup median {np.median(a / b):.2f}×)", loc="left")
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="eval_metrics_*.csv from frnn-eval")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output image (default: latency_hist_<csv stem>.png next to the CSV)")
    args = parser.parse_args()
    out = args.output or args.csv.with_name(
        f"latency_hist_{args.csv.stem.removeprefix('eval_metrics_')}.png")
    plot(args.csv, out)
