#!/usr/bin/env python3
"""Plot per-pair min-max normalized joint docking distributions for T2."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

def load_per_molecule(path: Path) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(path)
    try: return pd.read_json(path, orient="records")
    except: return pd.DataFrame(np.load(path, allow_pickle=True))

def _pair_cols(df): return ["ref_smiles"] + sorted([c for c in df.columns if c.startswith("p") and c[1:].isdigit()], key=lambda c: int(c[1:]))
def _dock_cols(df): return sorted([c for c in df.columns if re.match(r"^dock_p\d+$", c)], key=lambda c: int(c[6:]))

def build_df_all(df, method_map):
    pair_cols, dock_cols = _pair_cols(df), _dock_cols(df)
    rows = []
    for label, method_name in method_map.items():
        sub = df[df["method"] == method_name].copy()
        if sub.empty: sub = df[df["method"].astype(str).str.contains(method_name, regex=False)].copy()
        if sub.empty: continue
        sub["method_label"] = label
        sub = sub[pair_cols + ["method_label"] + dock_cols]
        rows.append(sub)
    if not rows: return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={dock_cols[0]: "score_t1", dock_cols[1]: "score_t2"})
    out["pair_id"] = out[pair_cols].astype(str).agg("|".join, axis=1)
    return out[["pair_id", "method_label", "score_t1", "score_t2"]]

def vectorized_min_max_normalize(df):
    grouped = df.groupby("pair_id")
    v_min_t1, v_max_t1 = grouped["score_t1"].transform("min"), grouped["score_t1"].transform("max")
    v_min_t2, v_max_t2 = grouped["score_t2"].transform("min"), grouped["score_t2"].transform("max")
    eps = 1e-8
    df = df.copy()
    df["norm_t1"] = (v_max_t1 - df["score_t1"]) / (v_max_t1 - v_min_t1 + eps)
    df["norm_t2"] = (v_max_t2 - df["score_t2"]) / (v_max_t2 - v_min_t2 + eps)
    return df

def plot_panels(df_norm: pd.DataFrame, out_path: Path) -> None:
    try:
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError(
            "seaborn is required for joint distribution KDE plots. "
            "Install it with `pip install seaborn` or install requirements.txt."
        ) from exc

    sns.set_theme(style="ticks", font_scale=1.2)
    
    c_p1 = "#D62728"      # Deep Red
    c_p2 = "#FF7F0E"      # Deep Orange
    c_ours = "#1F77B4"    # Deep Blue
    c_ideal = "#FFD700"   # Gold

    baselines = ["Single", "TamGen", "PocketXMol"]
    titles = [
        "(a) vs. Single Model (Ablation)",
        "(b) vs. TamGen (Sequence SOTA)",
        "(c) vs. PocketXMol (Structure SOTA)",
    ]

    df_ours_full = df_norm[df_norm["method_label"] == "PolyTamGen_DPO"]
    ours_centroid = (df_ours_full["norm_t1"].mean(), df_ours_full["norm_t2"].mean())
    
    MAX_POINTS = 50000 
    if len(df_ours_full) > MAX_POINTS:
        df_ours_plot = df_ours_full.sample(n=MAX_POINTS, random_state=42)
    else:
        df_ours_plot = df_ours_full

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharex=True, sharey=True)
    
    KDE_THRESH = 0.2  
    KDE_LEVELS = 6
    
    for i, base_name in enumerate(baselines):
        ax = axes[i]
        
        if base_name == "Single":
            p1_label, p2_label = "Single_P1", "Single_P2"
        elif base_name == "TamGen":
            p1_label, p2_label = "TamGen_P1", "TamGen_P2"
        else:
            p1_label, p2_label = "PocketXMol_P1", "PocketXMol_P2"
            
        df_p1 = df_norm[df_norm["method_label"] == p1_label]
        df_p2 = df_norm[df_norm["method_label"] == p2_label]
        
        p1_centroid = (df_p1["norm_t1"].mean(), df_p1["norm_t2"].mean())
        p2_centroid = (df_p2["norm_t1"].mean(), df_p2["norm_t2"].mean())

        if len(df_p1) > MAX_POINTS: df_p1 = df_p1.sample(n=MAX_POINTS, random_state=42)
        if len(df_p2) > MAX_POINTS: df_p2 = df_p2.sample(n=MAX_POINTS, random_state=42)

        print(f"Panel {i+1}: {base_name} vs Ours (Thresh={KDE_THRESH})...")

        rect = patches.Rectangle((0.7, 0.7), 0.4, 0.4, linewidth=0, 
                                 edgecolor='none', facecolor=c_ideal, alpha=0.1, zorder=0)
        ax.add_patch(rect)
        ax.grid(True, linestyle=':', alpha=0.4, zorder=0)

        # Plot distributions with a fixed density threshold.
        
        # Baseline P1 (Red)
        if not df_p1.empty:
            sns.kdeplot(data=df_p1, x="norm_t1", y="norm_t2",
                        color=c_p1, fill=True, alpha=0.25, 
                        levels=KDE_LEVELS, thresh=KDE_THRESH, clip=(0,1), ax=ax, zorder=2)
            sns.kdeplot(data=df_p1, x="norm_t1", y="norm_t2",
                        color=c_p1, fill=False, linewidths=1.5, 
                        levels=KDE_LEVELS, thresh=KDE_THRESH, clip=(0,1), ax=ax, zorder=3)

        # Baseline P2 (Orange)
        if not df_p2.empty:
            sns.kdeplot(data=df_p2, x="norm_t1", y="norm_t2",
                        color=c_p2, fill=True, alpha=0.25, 
                        levels=KDE_LEVELS, thresh=KDE_THRESH, clip=(0,1), ax=ax, zorder=2)
            sns.kdeplot(data=df_p2, x="norm_t1", y="norm_t2",
                        color=c_p2, fill=False, linewidths=1.5, 
                        levels=KDE_LEVELS, thresh=KDE_THRESH, clip=(0,1), ax=ax, zorder=3)

        if not df_ours_plot.empty:
            sns.kdeplot(data=df_ours_plot, x="norm_t1", y="norm_t2",
                        color=c_ours, fill=True, alpha=0.1, 
                        levels=8, thresh=KDE_THRESH, clip=(0,1), ax=ax, zorder=4)
            sns.kdeplot(data=df_ours_plot, x="norm_t1", y="norm_t2",
                        color=c_ours, fill=False, linewidths=2.5, 
                        levels=8, thresh=KDE_THRESH, clip=(0,1), ax=ax, zorder=5)

        # Centroids.
        if not df_p1.empty:
            ax.scatter(p1_centroid[0], p1_centroid[1], color=c_p1, marker='X', s=80, edgecolor='white', lw=1, zorder=6)
        if not df_p2.empty:
            ax.scatter(p2_centroid[0], p2_centroid[1], color=c_p2, marker='X', s=80, edgecolor='white', lw=1, zorder=6)
        
        ax.scatter(ours_centroid[0], ours_centroid[1], color=c_ours, marker='*', s=200, edgecolor='white', lw=1, zorder=7)

        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1.5, alpha=0.5, zorder=1)
        ax.set_title(titles[i], fontsize=14, pad=12, fontweight='bold')
        ax.set_xlabel("Normalized Score ($T_1$)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Normalized Score ($T_2$)", fontsize=12)
        
        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(0.0, 1.05)

        if i == 0:
            legend_elements = [
                Line2D([0], [0], color=c_p1, lw=2, alpha=0.8, label="Baseline (Source P1)"),
                Line2D([0], [0], color=c_p2, lw=2, alpha=0.8, label="Baseline (Source P2)"),
                Line2D([0], [0], color=c_ours, lw=3, label="PolyTamGen (Joint)"),
                Line2D([0], [0], color='grey', marker='X', linestyle='None', markersize=8, label="Baseline Means"),
                Line2D([0], [0], color=c_ours, marker='*', linestyle='None', markersize=12, label="Ours Mean"),
                patches.Patch(facecolor=c_ideal, alpha=0.2, label='Pareto Frontier Zone')
            ]
            ax.legend(handles=legend_elements, loc="upper left", frameon=True, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t2_path", default="results/scored/T2/per_molecule.json")
    p.add_argument("--output", default="outputs/paper_rebuild/figures/joint_dist_panels.pdf")
    args = p.parse_args()

    print("Loading data...")
    df = load_per_molecule(Path(args.t2_path))
    method_map = {
        "PolyTamGen_DPO": "PolyTamGen-DPO",
        "Single_P1": "PolyTamGen-Single-p1",
        "Single_P2": "PolyTamGen-Single-p2",
        "TamGen_P1": "TamGen-p1",
        "TamGen_P2": "TamGen-p2",
        "PocketXMol_P1": "PocketXMol-p1",
        "PocketXMol_P2": "PocketXMol-p2",
    }
    df_all = build_df_all(df, method_map)
    df_norm = vectorized_min_max_normalize(df_all)
    plot_panels(df_norm, Path(args.output))

if __name__ == "__main__":
    main()
