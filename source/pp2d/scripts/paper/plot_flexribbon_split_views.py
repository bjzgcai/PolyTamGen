#!/usr/bin/env python3
"""Plot split FlexRibbon views from the aligned per-instance T2 table."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_ORDER = ["Reference ligand", "PocketXMol", "PolyTamGen-DPO"]
PALETTE = {
    "Reference ligand": "#8a8a8a",
    "PocketXMol": "#d98c2b",
    "PolyTamGen-DPO": "#2b6cb0",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def metric_limits(values: pd.Series) -> tuple[float, float]:
    lo = float(values.quantile(0.005))
    hi = float(values.quantile(0.995))
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def save(fig, output: Path, png_output: Path | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if png_output is not None:
        png_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ecdf(df: pd.DataFrame, metric: str, title: str, output: Path, png_output: Path | None) -> None:
    setup_style()
    fig, ax = plt.subplots(1, 1, figsize=(4.25, 2.9))
    for method in METHOD_ORDER:
        vals = np.sort(df.loc[df["method"] == method, metric].to_numpy())
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", color=PALETTE[method], linewidth=1.8, label=method, alpha=0.95)
        ax.plot([float(np.median(vals))], [0.5], marker="o", markersize=3.5,
                color=PALETTE[method], markeredgecolor="white", markeredgewidth=0.5)
    ax.axvline(0, color="#505050", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(0.5, color="#cbd5e1", linewidth=0.7, linestyle=":", alpha=0.9)
    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xlabel("FlexRibbon predicted affinity")
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlim(*metric_limits(df[metric]))
    ax.set_ylim(0, 1)
    ax.grid(axis="both", color="#e5e7eb", linewidth=0.7)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), handlelength=2.2)
    fig.tight_layout(rect=(0, 0.1, 1, 1), pad=0.9)
    save(fig, output, png_output)


def kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.1
    bandwidth = max(1.06 * std * (len(values) ** (-1.0 / 5.0)), 0.05)
    density = np.zeros_like(grid, dtype=float)
    norm = len(values) * bandwidth * np.sqrt(2 * np.pi)
    for start in range(0, len(values), 5000):
        chunk = values[start : start + 5000]
        z = (grid[:, None] - chunk[None, :]) / bandwidth
        density += np.exp(-0.5 * z * z).sum(axis=1)
    return density / norm


def plot_density(df: pd.DataFrame, metric: str, title: str | None, output: Path, png_output: Path | None) -> None:
    setup_style()
    limits = metric_limits(df[metric])
    grid = np.linspace(limits[0], limits[1], 400)
    fig, ax = plt.subplots(1, 1, figsize=(4.25, 2.9))
    for method in METHOD_ORDER:
        vals = df.loc[df["method"] == method, metric].to_numpy()
        density = kde(vals, grid)
        ax.plot(grid, density, color=PALETTE[method], linewidth=1.9, label=method)
        ax.fill_between(grid, 0, density, color=PALETTE[method], alpha=0.11)
        ax.axvline(float(np.median(vals)), color=PALETTE[method], linewidth=0.8, linestyle=":", alpha=0.85)
    ax.axvline(0, color="#505050", linewidth=0.8, linestyle="--", alpha=0.7)
    if title:
        ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xlabel("FlexRibbon predicted affinity")
    ax.set_ylabel("Density")
    ax.set_xlim(*limits)
    ax.grid(axis="both", color="#e5e7eb", linewidth=0.7)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), handlelength=2.2)
    fig.tight_layout(rect=(0, 0.1, 1, 1), pad=0.9)
    save(fig, output, png_output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/supplement/flexribbon/t2_rdkit_aligned_per_instance.csv"))
    parser.add_argument("--avg-ecdf-output", type=Path, default=Path("outputs/paper_rebuild/figures/flexribbon_affinity_avg_ecdf.pdf"))
    parser.add_argument("--avg-ecdf-png-output", type=Path, default=None)
    parser.add_argument("--avg-density-output", type=Path, default=Path("outputs/paper_rebuild/figures/flexribbon_affinity_avg_density.pdf"))
    parser.add_argument("--avg-density-png-output", type=Path, default=None)
    parser.add_argument("--weakest-ecdf-output", type=Path, default=Path("outputs/paper_rebuild/figures/flexribbon_affinity_weakest_ecdf.pdf"))
    parser.add_argument("--weakest-ecdf-png-output", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    plot_ecdf(df, "avg_affinity", "Average over intended targets", args.avg_ecdf_output, args.avg_ecdf_png_output)
    plot_density(df, "avg_affinity", None, args.avg_density_output, args.avg_density_png_output)
    plot_ecdf(df, "worst_affinity", "Weakest intended target", args.weakest_ecdf_output, args.weakest_ecdf_png_output)
    print(f"wrote {args.avg_ecdf_output}")
    print(f"wrote {args.avg_density_output}")
    print(f"wrote {args.weakest_ecdf_output}")


if __name__ == "__main__":
    main()
