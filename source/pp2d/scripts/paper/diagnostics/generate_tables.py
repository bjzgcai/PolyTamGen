#!/usr/bin/env python3
"""Generate a battery of *candidate* paper tables as CSV.

This script intentionally does NOT plot or write LaTeX.
Workflow:
1) Generate many tables (CSV) from per_target.csv.
2) Manually/automatically inspect which metrics are informative.
3) Then decide what to plot and what to include in the paper.

Inputs
- results/scored/T2/per_target.csv

Outputs
- <out-dir>/*.csv

- Some extended metrics (success_all_*, hit_topk_*) are supported by
  scripts/benchmark/benchmark_multitarget_docking.py but may be missing from the current
  per_target.csv on disk. This script will emit tables only for columns that
  exist.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


METHOD_LABELS: Dict[str, str] = {
    "PolyTamGen-SFT": "PolyTamGen-SFT",
    "PolyTamGen-Single": "PolyTamGen-Single",
    "PolyTamGen-Single-sampled-100-x10": "PolyTamGen-Single-sampled-100-x10",
    "TamGen": "TamGen",
    "TamGen-sampled-100-x10": "TamGen-sampled-100-x10",
    "PocketXMol": "PocketXMol",
    "PocketXMol-sampled-100-x10": "PocketXMol-sampled-100-x10",
    "Reference": "Reference",
}

DEFAULT_METHODS = [
    "PolyTamGen-SFT",
    "PolyTamGen-Single-sampled-100-x10",
    "TamGen-sampled-100-x10",
    "PocketXMol-sampled-100-x10",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _label_method(m: str) -> str:
    return METHOD_LABELS.get(m, m)


def _bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.nanmean,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    n = values.size
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        stats[i] = float(statistic(sample))

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(stats, alpha))
    hi = float(np.quantile(stats, 1.0 - alpha))
    center = float(statistic(values))
    return center, lo, hi


def _bootstrap_ci_delta(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size == 0:
        return (float("nan"), float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    n = a.size
    diffs = a - b

    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = float(np.nanmean(diffs[idx]))

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(stats, alpha))
    hi = float(np.quantile(stats, 1.0 - alpha))
    center = float(np.nanmean(diffs))
    return center, lo, hi


def load_per_target(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ref_smiles", "p1", "p2", "method", "sim_category", "max_sim"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    df = df.copy()
    df["pair_id"] = df["ref_smiles"].astype(str) + "|" + df["p1"].astype(str) + "|" + df["p2"].astype(str)
    df["method_label"] = df["method"].astype(str).map(_label_method)
    return df


def add_derived_metrics(df: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    df = df.copy()

    # Pair-level hit derived from best_worst/best_avg when explicit hit_* columns are absent.
    # This matches the benchmark script definition: at least one molecule succeeds.
    for t in thresholds:
        col = f"hit_worst@{t}"  # keep float formatting consistent with per_target.csv
        if col not in df.columns and "best_worst" in df.columns:
            df[col] = (df["best_worst"].astype(float) <= float(t)).astype(float)

        col2 = f"hit_avg@{t}"
        if col2 not in df.columns and "best_avg" in df.columns:
            df[col2] = (df["best_avg"].astype(float) <= float(t)).astype(float)

    return df


@dataclass(frozen=True)
class MetricSpec:
    name: str
    higher_is_better: bool
    kind: str  # "rate" or "score" or "property"


def available_metric_specs(df: pd.DataFrame, thresholds: Sequence[float]) -> List[MetricSpec]:
    specs: List[MetricSpec] = []

    # continuous docking aggregations
    for k in ["mean_worst", "mean_avg", "best_worst", "best_avg"]:
        if k in df.columns:
            specs.append(MetricSpec(name=k, higher_is_better=False, kind="score"))

    # top-k mean (proxy for budget)
    for k in [1, 5, 10, 20, 50, 100]:
        col = f"top{k}_mean_worst"
        if col in df.columns:
            specs.append(MetricSpec(name=col, higher_is_better=False, kind="score"))
        col_avg = f"top{k}_mean_avg"
        if col_avg in df.columns:
            specs.append(MetricSpec(name=col_avg, higher_is_better=False, kind="score"))

    # success/hit-like rates at thresholds
    for t in thresholds:
        for base in [
            "success_worst",
            "success_avg",
            "success_all_worst",
            "success_all_avg",
            "success_rate_strict_worst",
            "hit_worst",
            "hit_avg",
        ]:
            col = f"{base}@{t}"
            if col in df.columns:
                specs.append(MetricSpec(name=col, higher_is_better=True, kind="rate"))

        for k in [1, 5, 10, 20, 50, 100]:
            col = f"hit_top{k}_worst@{t}"
            if col in df.columns:
                specs.append(MetricSpec(name=col, higher_is_better=True, kind="rate"))

    # chemistry sanity columns
    for k in ["validity", "diversity", "qed", "lipinski", "sas", "sa_norm", "logp", "mw", "molwt"]:
        if k in df.columns:
            specs.append(MetricSpec(name=k, higher_is_better=True, kind="property"))

    return specs


def table_overall_summary(
    df: pd.DataFrame,
    methods: Sequence[str],
    metric_specs: Sequence[MetricSpec],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for m in methods:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        row: Dict[str, object] = {
            "method": m,
            "method_label": _label_method(m),
            "n_pairs": int(sub["pair_id"].nunique()),
        }
        for ms in metric_specs:
            if ms.name not in sub.columns:
                continue
            center, lo, hi = _bootstrap_ci(sub[ms.name].to_numpy(dtype=float), n_boot=n_boot, seed=seed)
            row[f"{ms.name}_mean"] = center
            row[f"{ms.name}_lo"] = lo
            row[f"{ms.name}_hi"] = hi
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("method_label")


def table_threshold_curve(
    df: pd.DataFrame,
    methods: Sequence[str],
    prefixes: Sequence[str],
    thresholds: Sequence[float],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for m in methods:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        for p in prefixes:
            for t in thresholds:
                col = f"{p}@{t}"
                if col not in sub.columns:
                    continue
                rows.append(
                    {
                        "method": m,
                        "method_label": _label_method(m),
                        "metric": p,
                        "threshold": float(t),
                        "mean": float(np.nanmean(sub[col].to_numpy(dtype=float))),
                    }
                )
    return pd.DataFrame(rows)


def table_by_group_ci(
    df: pd.DataFrame,
    methods: Sequence[str],
    metric: str,
    group_col: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for gval, gdf in df.groupby(group_col):
        for m in methods:
            sub = gdf[gdf["method"] == m]
            if sub.empty or metric not in sub.columns:
                continue
            center, lo, hi = _bootstrap_ci(sub[metric].to_numpy(dtype=float), n_boot=n_boot, seed=seed)
            rows.append(
                {
                    group_col: gval,
                    "method": m,
                    "method_label": _label_method(m),
                    "n_pairs": int(sub["pair_id"].nunique()),
                    "mean": center,
                    "lo": lo,
                    "hi": hi,
                }
            )
    return pd.DataFrame(rows)


def table_by_cutoff_ci(
    df: pd.DataFrame,
    methods: Sequence[str],
    metric: str,
    cutoffs: Sequence[float],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    max_sim = df["max_sim"].astype(float)
    for c in cutoffs:
        subc = df[max_sim <= float(c)]
        for m in methods:
            subm = subc[subc["method"] == m]
            if subm.empty or metric not in subm.columns:
                continue
            center, lo, hi = _bootstrap_ci(subm[metric].to_numpy(dtype=float), n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "cutoff": float(c),
                    "method": m,
                    "method_label": _label_method(m),
                    "n_pairs": int(subm["pair_id"].nunique()),
                    "mean": center,
                    "lo": lo,
                    "hi": hi,
                }
            )
    return pd.DataFrame(rows)


def table_by_bins_ci(
    df: pd.DataFrame,
    methods: Sequence[str],
    metric: str,
    bins: Sequence[float],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    cats = pd.cut(df["max_sim"].astype(float), bins=bins, include_lowest=True)
    for bin_label, gbin in df.groupby(cats, observed=False):
        if bin_label is None:
            continue
        if not hasattr(bin_label, "left") or not hasattr(bin_label, "right"):
            continue
        left = float(getattr(bin_label, "left"))
        right = float(getattr(bin_label, "right"))
        mid = float((left + right) / 2.0)
        for m in methods:
            subm = gbin[gbin["method"] == m]
            if subm.empty or metric not in subm.columns:
                continue
            center, lo, hi = _bootstrap_ci(subm[metric].to_numpy(dtype=float), n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "sim_bin": str(bin_label),
                    "bin_left": left,
                    "bin_right": right,
                    "bin_mid": mid,
                    "method": m,
                    "method_label": _label_method(m),
                    "n_pairs": int(subm["pair_id"].nunique()),
                    "mean": center,
                    "lo": lo,
                    "hi": hi,
                }
            )
    return pd.DataFrame(rows).sort_values(["bin_mid", "method_label"]) if rows else pd.DataFrame(rows)


def table_reference_relative(df: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    # These columns exist when 'gt' present in benchmark output.
    needed = {"delta_best_worst_vs_gt", "delta_best_avg_vs_gt"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for m in methods:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        dw = sub["delta_best_worst_vs_gt"].to_numpy(dtype=float)
        da = sub["delta_best_avg_vs_gt"].to_numpy(dtype=float)

        def _summ(v: np.ndarray) -> Dict[str, float]:
            v = v[np.isfinite(v)]
            if v.size == 0:
                return {
                    "mean": float("nan"),
                    "median": float("nan"),
                    "p10": float("nan"),
                    "p25": float("nan"),
                    "p75": float("nan"),
                    "p90": float("nan"),
                    "win_rate": float("nan"),
                }
            return {
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "p10": float(np.quantile(v, 0.10)),
                "p25": float(np.quantile(v, 0.25)),
                "p75": float(np.quantile(v, 0.75)),
                "p90": float(np.quantile(v, 0.90)),
                "win_rate": float(np.mean((v <= 0).astype(float))),
            }

        sw = _summ(dw)
        sa = _summ(da)
        rows.append(
            {
                "method": m,
                "method_label": _label_method(m),
                "n_pairs": int(sub["pair_id"].nunique()),
                "win_worst": sw["win_rate"],
                "delta_best_worst_mean": sw["mean"],
                "delta_best_worst_median": sw["median"],
                "delta_best_worst_p10": sw["p10"],
                "delta_best_worst_p90": sw["p90"],
                "win_avg": sa["win_rate"],
                "delta_best_avg_mean": sa["mean"],
                "delta_best_avg_median": sa["median"],
                "delta_best_avg_p10": sa["p10"],
                "delta_best_avg_p90": sa["p90"],
            }
        )

    return pd.DataFrame(rows).sort_values("method_label") if rows else pd.DataFrame(rows)


def table_paired_comparisons(
    df: pd.DataFrame,
    method_a: str,
    others: Sequence[str],
    metrics: Sequence[MetricSpec],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for b in others:
        a_sub = df[df["method"] == method_a]
        b_sub = df[df["method"] == b]
        if a_sub.empty or b_sub.empty:
            continue
        for ms in metrics:
            if ms.name not in df.columns:
                continue
            a = a_sub[["pair_id", ms.name]].set_index("pair_id")
            bb = b_sub[["pair_id", ms.name]].set_index("pair_id")
            joined = a.join(bb, how="inner", lsuffix="_a", rsuffix="_b")
            va = joined[f"{ms.name}_a"].to_numpy(dtype=float)
            vb = joined[f"{ms.name}_b"].to_numpy(dtype=float)

            mask = np.isfinite(va) & np.isfinite(vb)
            va = va[mask]
            vb = vb[mask]
            if va.size == 0:
                continue

            delta, dlo, dhi = _bootstrap_ci_delta(va, vb, n_boot=n_boot, seed=seed)

            # win rate (ties = 0.5)
            if ms.higher_is_better:
                d = va - vb
                wins = np.sum(d > 0)
                ties = np.sum(d == 0)
            else:
                d = vb - va  # smaller is better
                wins = np.sum(d > 0)
                ties = np.sum(d == 0)
            win_rate = float((wins + 0.5 * ties) / d.size)

            rows.append(
                {
                    "metric": ms.name,
                    "kind": ms.kind,
                    "higher_is_better": ms.higher_is_better,
                    "A": method_a,
                    "A_label": _label_method(method_a),
                    "B": b,
                    "B_label": _label_method(b),
                    "n_pairs": int(d.size),
                    "delta_mean(A-B)": float(delta),
                    "delta_lo": float(dlo),
                    "delta_hi": float(dhi),
                    "win_rate_A": win_rate,
                    "mean_A": float(np.mean(va)),
                    "mean_B": float(np.mean(vb)),
                }
            )
    return pd.DataFrame(rows)


def table_metric_diagnostics(df: pd.DataFrame, metric_specs: Sequence[MetricSpec]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for ms in metric_specs:
        if ms.name not in df.columns:
            continue
        v = df[ms.name].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        row: Dict[str, object] = {
            "metric": ms.name,
            "kind": ms.kind,
            "higher_is_better": ms.higher_is_better,
            "n": int(v.size),
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "p1": float(np.quantile(v, 0.01)),
            "p5": float(np.quantile(v, 0.05)),
            "p50": float(np.quantile(v, 0.50)),
            "p95": float(np.quantile(v, 0.95)),
            "p99": float(np.quantile(v, 0.99)),
            "range": float(np.max(v) - np.min(v)),
        }
        if ms.kind == "rate":
            row["frac_ge_0.99"] = float(np.mean((v >= 0.99).astype(float)))
            row["frac_le_0.01"] = float(np.mean((v <= 0.01).astype(float)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["kind", "metric"]) if rows else pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate candidate paper tables as CSV")
    ap.add_argument("--per-target", default="results/scored/T2/per_target.csv")
    ap.add_argument("--out-dir", default="outputs/paper_rebuild/tables/diagnostics")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--thresholds", default="-6.0,-7.0,-8.0")
    ap.add_argument("--cutoffs", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--bins", default="0.0,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    cutoffs = [float(x) for x in args.cutoffs.split(",") if x.strip()]
    bins = [float(x) for x in args.bins.split(",") if x.strip()]

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    _ensure_dir(args.out_dir)

    df = load_per_target(args.per_target)
    df = df[df["method"].isin(methods)].copy()
    df = add_derived_metrics(df, thresholds=thresholds)

    metric_specs = available_metric_specs(df, thresholds=thresholds)

    # 0) inventory
    inv = {
        "per_target": args.per_target,
        "n_rows": int(len(df)),
        "n_pairs": int(df["pair_id"].nunique()),
        "methods": methods,
        "method_labels": {m: _label_method(m) for m in methods},
        "available_columns": list(df.columns),
        "available_metrics": [ms.name for ms in metric_specs],
    }
    with open(os.path.join(args.out_dir, "inventory.json"), "w", encoding="utf-8") as f:
        json.dump(inv, f, indent=2)

    # 1) overall summary (mean + bootstrap CI)
    overall = table_overall_summary(df, methods=methods, metric_specs=metric_specs, n_boot=args.n_boot, seed=args.seed)
    overall.to_csv(os.path.join(args.out_dir, "overall_summary.csv"), index=False)

    # 2) metric diagnostics (helps decide what not to plot)
    diag = table_metric_diagnostics(df, metric_specs)
    diag.to_csv(os.path.join(args.out_dir, "metric_diagnostics.csv"), index=False)

    # 3) threshold curves (raw means across thresholds)
    thr = table_threshold_curve(
        df,
        methods=methods,
        prefixes=[
            "success_worst",
            "success_avg",
            "hit_worst",
            "hit_avg",
            "success_all_worst",
            "success_all_avg",
            "success_rate_strict_worst",
        ],
        thresholds=thresholds,
    )
    thr.to_csv(os.path.join(args.out_dir, "threshold_curve.csv"), index=False)

    # 4) reference-relative summary if available
    ref = table_reference_relative(df, methods=methods)
    if not ref.empty:
        ref.to_csv(os.path.join(args.out_dir, "reference_relative.csv"), index=False)

    # Decide which metrics to stratify (category/cutoff/bin)
    # - all score metrics (docking objective; usually most informative)
    # - rate metrics at the primary threshold (-7.0 by default)
    # - optionally chemistry properties (kept overall/paired; stratification can be noisy)
    primary_t = -7.0
    stratify_metrics: List[str] = []
    for ms in metric_specs:
        if ms.kind == "score":
            stratify_metrics.append(ms.name)
        elif ms.kind == "rate" and ms.name.endswith(f"@{primary_t}"):
            stratify_metrics.append(ms.name)

    # 5) category stratifications
    for metric in stratify_metrics:
        if metric not in df.columns:
            continue
        out = table_by_group_ci(
            df,
            methods=methods,
            metric=metric,
            group_col="sim_category",
            n_boot=args.n_boot,
            seed=args.seed,
        )
        out.to_csv(os.path.join(args.out_dir, f"by_category__{metric}.csv"), index=False)

    # 6) similarity cutoff trends (cumulative)
    for metric in stratify_metrics:
        if metric not in df.columns:
            continue
        out = table_by_cutoff_ci(
            df,
            methods=methods,
            metric=metric,
            cutoffs=cutoffs,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        out.to_csv(os.path.join(args.out_dir, f"by_cutoff__{metric}.csv"), index=False)

    # 7) similarity binned (non-cumulative) + counts
    counts = pd.cut(df["max_sim"].astype(float), bins=bins, include_lowest=True)
    pd.DataFrame({"sim_bin": counts.astype(str)}).value_counts().rename("n").reset_index().to_csv(
        os.path.join(args.out_dir, "bin_counts.csv"), index=False
    )

    for metric in stratify_metrics:
        if metric not in df.columns:
            continue
        out = table_by_bins_ci(df, methods=methods, metric=metric, bins=bins, n_boot=args.n_boot, seed=args.seed)
        out.to_csv(os.path.join(args.out_dir, f"by_bin__{metric}.csv"), index=False)

    # 8) paired comparisons (OurM vs baselines) for all available metrics
    if "PolyTamGen-SFT" in methods:
        others = [m for m in methods if m != "PolyTamGen-SFT"]
        pairs = table_paired_comparisons(
            df,
            method_a="PolyTamGen-SFT",
            others=others,
            metrics=metric_specs,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        pairs.to_csv(os.path.join(args.out_dir, "paired_comparisons.csv"), index=False)


if __name__ == "__main__":
    main()
