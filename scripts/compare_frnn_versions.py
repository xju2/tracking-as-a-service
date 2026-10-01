#!/usr/bin/env python
"""Compare the original FRNN with several libFRNN versions from frnn-eval metrics CSVs.

Each CSV comes from one libFRNN version run over the same events. FRNN is timed in
every run, so its latency is the per-event mean over all runs.

    python scripts/compare_frnn_versions.py \\
        v1.0=frnn_eval_outputs/eval_metrics_v1.0.csv \\
        v1.1=frnn_eval_outputs/eval_metrics_v1.1.csv
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed order: FRNN first, then one hue per libFRNN version.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def load(runs: list[str], skip: int) -> tuple[np.ndarray, dict[str, pd.DataFrame]]:
    tables = {}
    for run in runs:
        label, _, path = run.partition("=")
        if not path:
            raise SystemExit(f"expected label=path, got {run!r}")
        tables[label] = pd.read_csv(path).iloc[skip:].reset_index(drop=True)
    points = next(iter(tables.values())).num_space_points.values
    for label, df in tables.items():
        if not np.array_equal(df.num_space_points.values, points):
            raise SystemExit(f"{label}: events differ from the first CSV; runs are not comparable")
    return points, tables


def summarize(points, tables):
    frnn = np.mean([df.frnn_latency_ms.values for df in tables.values()], axis=0)
    series = [("FRNN", frnn)] + [(f"libFRNN {k}", df.libfrnn_latency_ms.values)
                                 for k, df in tables.items()]
    print(f"{len(points)} events")
    print(f"{'':16}{'median':>9}{'mean':>9}{'p90':>9}{'speedup':>9}{'ms/1k pts':>11}")
    for name, y in series:
        print(f"{name:16}{np.median(y):9.1f}{y.mean():9.1f}{np.quantile(y, 0.9):9.1f}"
              f"{np.median(frnn / y):8.2f}x{np.polyfit(points / 1e3, y, 1)[0]:11.2f}")
    for label, df in tables.items():
        if "libfrnn_peak_mem_mb" in df and df.libfrnn_peak_mem_mb.notna().all():
            print(f"peak GPU memory, run {label}: FRNN median {df.frnn_peak_mem_mb.median():.0f} MB, "
                  f"libFRNN {label} median {df.libfrnn_peak_mem_mb.median():.0f} MB")
        else:
            print(f"peak GPU memory, run {label}: not recorded")
    return series


def plot(points, series, skip, out_path: Path):
    n = points / 1e3
    fig, (hist, scatter) = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(min(y.min() for _, y in series) * 0.9,
                       max(y.max() for _, y in series) * 1.05, 45)
    for (name, y), color in zip(series, COLORS):
        if name == "FRNN" and len(series) > 2:
            name = "FRNN (mean of all runs)"
        hist.hist(y, bins=bins, color=color, alpha=0.6, edgecolor="white", linewidth=1,
                  label=f"{name}: median {np.median(y):.0f} ms")
        hist.axvline(np.median(y), color=color, linestyle="--", linewidth=2)
        scatter.scatter(n, y, s=36, color=color, edgecolor="white", linewidth=1,
                        label=name, zorder=3)
        slope, offset = np.polyfit(n, y, 1)
        xs = np.array([n.min(), n.max()])
        scatter.plot(xs, slope * xs + offset, color=color, linewidth=2, zorder=2)

    excluded = f" (first {skip} excluded)" if skip else ""
    hist.set(xlabel="Latency [ms]", ylabel="Requests")
    hist.set_title(f"Latency, {len(points)} events{excluded}", loc="left")
    hist.set_ylim(top=hist.get_ylim()[1] * 1.1)
    scatter.set(xlabel="Space points [thousands]", ylabel="Latency [ms]", ylim=(0, None))
    scatter.set_title("Latency vs event size", loc="left")
    for ax in (hist, scatter):
        ax.grid(axis="y", color="#e0e0e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="+", help="label=eval_metrics.csv, one per libFRNN version")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output image (default: latency_compare_<labels>.png next to the first CSV)")
    parser.add_argument("--skip", type=int, default=1,
                        help="leading requests to drop as CUDA warm-up (default: 1)")
    args = parser.parse_args()
    if len(args.runs) > len(COLORS) - 1:
        raise SystemExit(f"at most {len(COLORS) - 1} versions per plot")

    points, tables = load(args.runs, args.skip)
    series = summarize(points, tables)
    out = args.output or Path(args.runs[0].partition("=")[2]).with_name(
        f"latency_compare_frnn_{'_'.join(tables)}.png")
    plot(points, series, args.skip, out)
