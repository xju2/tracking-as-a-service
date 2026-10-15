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
from matplotlib.ticker import MaxNLocator  # noqa: E402

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


def style(ax):
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot(points, series, prefix: Path):
    """Save the latency histogram and latency vs event size as two standalone PDFs.

    No titles: in a paper the caption carries them, including that FRNN is the
    per-event mean over all runs.
    """
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10})  # TrueType fonts for journals
    n = points / 1e3
    labels = [name for name, _ in series]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bins = np.linspace(min(y.min() for _, y in series) * 0.9,
                       max(y.max() for _, y in series) * 1.05, 45)
    for (_, y), label, color in zip(series, labels, COLORS):
        ax.hist(y, bins=bins, color=color, alpha=0.6, edgecolor="white", linewidth=0.8,
                label=f"{label} (median {np.median(y):.0f} ms)")
        ax.axvline(np.median(y), ymax=0.76, color=color, linestyle="--", linewidth=1.5)
    ax.set(xlabel="Latency [ms]", ylabel="Events")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)  # room for the legend
    style(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{prefix}_hist.pdf")
    print(f"saved {prefix}_hist.pdf")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for (_, y), label, color in zip(series, labels, COLORS):
        slope, offset = np.polyfit(n, y, 1)
        ax.scatter(n, y, s=20, color=color, edgecolor="white", linewidth=0.6,
                   label=f"{label}: {slope:.2f} ms per 1k points", zorder=3)
        xs = np.array([n.min(), n.max()])
        ax.plot(xs, slope * xs + offset, color=color, linewidth=1.5, zorder=2)
    ax.set(xlabel="Space points [thousands]", ylabel="Latency [ms]", ylim=(0, None))
    style(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{prefix}_vs_size.pdf")
    print(f"saved {prefix}_vs_size.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="+", help="label=eval_metrics.csv, one per libFRNN version")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output prefix; writes <prefix>_hist.pdf and <prefix>_vs_size.pdf "
                        "(default: latency_compare_frnn_<labels> next to the first CSV)")
    parser.add_argument("--skip", type=int, default=1,
                        help="leading requests to drop as CUDA warm-up (default: 1)")
    args = parser.parse_args()
    if len(args.runs) > len(COLORS) - 1:
        raise SystemExit(f"at most {len(COLORS) - 1} versions per plot")

    points, tables = load(args.runs, args.skip)
    series = summarize(points, tables)
    prefix = args.output or Path(args.runs[0].partition("=")[2]).with_name(
        f"latency_compare_frnn_{'_'.join(tables)}")
    plot(points, series, prefix)
