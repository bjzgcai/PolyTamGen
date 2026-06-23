#!/usr/bin/env python3
"""Generate molecule-level consistency LaTeX tables from per_molecule.json files."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_per_molecule(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    # Prefer pandas fast JSON reader, fallback to json.load.
    try:
        df = pd.read_json(path, orient="records")
    except Exception:
        with path.open("r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of records: {path}")
        df = pd.DataFrame(data)
    if df.empty:
        raise ValueError(f"Empty data: {path}")
    return df


def pocket_cols_from_df(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("p") and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def pair_level_stats(
    df: pd.DataFrame,
    method: str,
    pair_cols: List[str],
    strict_use_qed_sa: bool,
    strict_threshold: float = -7.0,
) -> pd.DataFrame:
    sub = df[df["method"] == method].copy()
    if sub.empty:
        return pd.DataFrame()

    # Ensure optional columns exist to avoid KeyError on datasets without properties.
    for col in ["validity", "diversity", "qed", "sa_norm"]:
        if col not in sub.columns:
            sub[col] = np.nan

    dock_avg = sub["dock_avg"]
    strict = dock_avg < strict_threshold
    if strict_use_qed_sa:
        strict = strict & (sub["qed"] > 0.25) & (sub["sa_norm"] > 0.59)
    strict = strict.where(np.isfinite(dock_avg), np.nan)
    sub["strict"] = strict.astype(float)

    agg = sub.groupby(pair_cols).agg(
        avgdock=("dock_avg", "mean"),
        sr_strict=("strict", "mean"),
        validity=("validity", "mean"),
        diversity=("diversity", "mean"),
        qed=("qed", "mean"),
        sa=("sa_norm", "mean"),
        mean_worst=("dock_worst", "mean"),
    )
    return agg


def mean_std(series: pd.Series) -> Tuple[float, float]:
    vals = series.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))


def fmt_mean_std(val: Tuple[float, float], decimals: int = 3) -> str:
    m, s = val
    if not np.isfinite(m):
        return "NA"
    return f"{m:.{decimals}f} $\\pm$ {s:.{decimals}f}"


def table1_t2(df: pd.DataFrame, out_path: Path, decimals: int = 3, num_workers: int = 1) -> None:
    pair_cols = ["ref_smiles"] + pocket_cols_from_df(df)

    rows = [
        ("TamGen", "TamGen"),
        ("PocketXMol", "PocketXMol"),
        ("\\ourM{}-Single", "PolyTamGen-Single"),
        ("\\ourM{}-SFT", "PolyTamGen-SFT"),
        ("\\ourM{}-DPO", "PolyTamGen-DPO"),
    ]

    jobs = [(method, pair_cols, True) for _, method in rows]
    stats_map = dict(_parallel_map(df, jobs, _pair_stats_worker, num_workers, "T2 stats"))

    lines = []
    for label, method in rows:
        agg = stats_map.get(method, pd.DataFrame())
        if agg.empty:
            lines.append([label] + ["NA"] * 6)
            continue
        line = [
            label,
            fmt_mean_std(mean_std(agg["avgdock"]), decimals),
            fmt_mean_std(mean_std(agg["sr_strict"]), decimals),
            fmt_mean_std(mean_std(agg["validity"]), decimals),
            fmt_mean_std(mean_std(agg["diversity"]), decimals),
            fmt_mean_std(mean_std(agg["qed"]), decimals),
            fmt_mean_std(mean_std(agg["sa"]), decimals),
        ]
        lines.append(line)

    header = [
        "Method",
        "AvgDock",
        "SR\\_strict",
        "Validity",
        "Diversity",
        "QED",
        "SA",
    ]
    tex = render_table(header, lines)
    out_path.write_text(tex)


def _init_worker(df: pd.DataFrame) -> None:
    global _GLOBAL_DF
    _GLOBAL_DF = df


def _pair_stats_worker(args):
    method, pair_cols, strict_use_qed_sa = args
    return method, pair_level_stats(_GLOBAL_DF, method, pair_cols, strict_use_qed_sa)


def _parallel_map(df: pd.DataFrame, jobs: List[Tuple], worker, num_workers: int, desc: str):
    if num_workers <= 1:
        _init_worker(df)
        return [worker(j) for j in tqdm(jobs, desc=desc)]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=num_workers, initializer=_init_worker, initargs=(df,)) as pool:
        return list(tqdm(pool.imap(worker, jobs, chunksize=1), total=len(jobs), desc=desc))


def table3_t3(df: pd.DataFrame, out_path: Path, decimals: int = 3, num_workers: int = 1) -> None:
    pair_cols = ["ref_smiles"] + pocket_cols_from_df(df)

    rows = [
        ("TamGen", "TamGen"),
        ("PocketXMol", "PocketXMol"),
        ("\\ourM{}-SFT", "PolyTamGen-SFT"),
        ("\\ourM{}-DPO", "PolyTamGen-DPO"),
    ]

    jobs = [(method, pair_cols, True) for _, method in rows]
    stats_map = dict(_parallel_map(df, jobs, _pair_stats_worker, num_workers, "T3 stats"))

    lines = []
    for label, method in rows:
        agg = stats_map.get(method, pd.DataFrame())
        if agg.empty:
            lines.append([label] + ["NA"] * 6)
            continue
        line = [
            label,
            fmt_mean_std(mean_std(agg["avgdock"]), decimals),
            fmt_mean_std(mean_std(agg["sr_strict"]), decimals),
            fmt_mean_std(mean_std(agg["validity"]), decimals),
            fmt_mean_std(mean_std(agg["diversity"]), decimals),
            fmt_mean_std(mean_std(agg["qed"]), decimals),
            fmt_mean_std(mean_std(agg["sa"]), decimals),
        ]
        lines.append(line)

    header = ["Method", "AvgDock", "SR\\_strict", "Validity", "Diversity", "QED", "SA"]
    tex = render_table(header, lines)
    out_path.write_text(tex)


def render_table(header: List[str], rows: List[List[str]]) -> str:
    cols = "l" + "c" * (len(header) - 1)
    lines = []
    lines.append("\\begin{tabular}{" + cols + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--t2_path", default="results/scored/T2/per_molecule.json")
    p.add_argument("--t3_path", default="results/scored/T3/per_molecule.json")
    p.add_argument("--output_dir", default="outputs/paper_rebuild/consistency_tables")
    p.add_argument("--decimals", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_t2 = load_per_molecule(Path(args.t2_path))
    df_t3 = load_per_molecule(Path(args.t3_path))

    table1_t2(df_t2, out_dir / "table1_t2_main.tex", args.decimals, args.num_workers)
    table3_t3(df_t3, out_dir / "table3_t3_stress.tex", args.decimals, args.num_workers)

    print(f"Saved molecule-level consistency LaTeX tables to {out_dir}")


if __name__ == "__main__":
    main()
