#!/usr/bin/env python3
"""Assemble paper-ready table diagnostics.

Input: a directory produced by scripts/paper/diagnostics/generate_tables.py
Output: a paper-result directory with CSVs, diagnostics, and a manifest.

Heuristics:
- Prefer continuous score metrics (mean_worst, topk_mean_worst, ...).
- Prefer success_* over hit_* when hit saturates.
- Use paired comparisons (OurM vs baselines) to rank metrics by effect size
  and win-rate.
- Down-rank binned generalization when bin counts are tiny.

This script does not create plots/LaTeX; it only curates CSVs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


OURM_LABEL = "OurM"


@dataclass
class MetricDecision:
    metric: str
    kind: str
    reason: str
    tier: str  # main/appendix/exclude


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _safe_copy(src: str, dst: str) -> None:
    _ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def _metric_family(metric: str) -> str:
    if metric.startswith("hit_") or metric.startswith("hit_top"):
        return "hit"
    if metric.startswith("success_"):
        return "success"
    if metric.startswith("mean_") or metric.startswith("best_") or metric.startswith("top"):
        return "score"
    return "other"


def _primary_threshold(metrics: List[str]) -> Optional[str]:
    # prefer -7.0 if present
    for t in ["-7.0", "-7", "-7.00"]:
        if any(m.endswith(f"@{t}") for m in metrics):
            return t
    # else pick first threshold found
    m = re.search(r"@(-?\d+(?:\.\d+)?)", " ".join(metrics))
    return m.group(1) if m else None


def _load_inputs(in_dir: str) -> Dict[str, object]:
    paths = {
        "overall": os.path.join(in_dir, "overall_summary.csv"),
        "paired": os.path.join(in_dir, "paired_comparisons.csv"),
        "diag": os.path.join(in_dir, "metric_diagnostics.csv"),
        "bins": os.path.join(in_dir, "bin_counts.csv"),
        "thr": os.path.join(in_dir, "threshold_curve.csv"),
        "ref": os.path.join(in_dir, "reference_relative.csv"),
    }

    out: Dict[str, object] = {"paths": paths}
    for k, p in paths.items():
        if os.path.exists(p):
            out[k] = _read_csv(p)
        else:
            out[k] = None

    # collect stratified files
    def glob_prefix(prefix: str) -> List[str]:
        return sorted([os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.startswith(prefix) and f.endswith(".csv")])

    out["by_category_files"] = glob_prefix("by_category__")
    out["by_cutoff_files"] = glob_prefix("by_cutoff__")
    out["by_bin_files"] = glob_prefix("by_bin__")
    return out


def _rate_saturation(diag: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if diag is None or diag.empty:
        return {}
    sat = {}
    for _, r in diag.iterrows():
        metric = str(r.get("metric"))
        if str(r.get("kind")) != "rate":
            continue
        sat[metric] = {
            "frac_ge_0.99": float(r.get("frac_ge_0.99")) if pd.notna(r.get("frac_ge_0.99")) else float("nan"),
            "frac_le_0.01": float(r.get("frac_le_0.01")) if pd.notna(r.get("frac_le_0.01")) else float("nan"),
            "std": float(r.get("std")) if pd.notna(r.get("std")) else float("nan"),
        }
    return sat


def _paired_rank_table(paired: pd.DataFrame, a_label: str = OURM_LABEL) -> pd.DataFrame:
    if paired is None or paired.empty:
        return pd.DataFrame()

    df = paired.copy()
    df = df[df["A_label"] == a_label]
    if df.empty:
        return df

    # Score metrics: more negative delta means A better (since delta = mean(A-B) and lower is better).
    # We will use |delta| plus win_rate as ranking signals.
    df["abs_delta"] = df["delta_mean(A-B)"].abs()
    df["ci_width"] = (df["delta_hi"] - df["delta_lo"]).abs()
    df["rank_score"] = df["abs_delta"] / (df["ci_width"] + 1e-9) * (0.5 + df["win_rate_A"])
    return df.sort_values(["rank_score"], ascending=False)


def _rank_metrics_for_paper(
    paired: pd.DataFrame,
    diag: pd.DataFrame,
) -> Tuple[List[MetricDecision], Dict[str, object]]:
    decisions: List[MetricDecision] = []

    sat = _rate_saturation(diag)
    ranked = _paired_rank_table(paired)

    all_metrics = sorted(set(ranked["metric"].tolist())) if not ranked.empty else []
    threshold = _primary_threshold(all_metrics)

    meta: Dict[str, object] = {
        "primary_threshold": threshold,
        "n_metrics_paired": int(len(all_metrics)),
    }

    # First: choose a primary continuous score metric.
    score_candidates = ranked[ranked["kind"] == "score"] if not ranked.empty else pd.DataFrame()
    primary_score = None
    if not score_candidates.empty:
        # prefer mean_worst if it's competitive; otherwise pick top ranked score
        if (score_candidates["metric"] == "mean_worst").any():
            # if mean_worst is near the top 3, keep it (interpretability)
            top3 = score_candidates.head(3)["metric"].tolist()
            if "mean_worst" in top3:
                primary_score = "mean_worst"
            else:
                primary_score = score_candidates.iloc[0]["metric"]
        else:
            primary_score = score_candidates.iloc[0]["metric"]

    if primary_score:
        decisions.append(MetricDecision(metric=primary_score, kind="score", tier="main", reason="Primary docking objective (continuous) with strong paired effect."))

    # Second: choose one budget-proxy (topk) if available and distinct from primary
    budget_score = None
    if not score_candidates.empty:
        topk = [m for m in score_candidates["metric"].tolist() if m.startswith("top") and m.endswith("_mean_worst")]
        preferred = ["top10_mean_worst", "top5_mean_worst", "top1_mean_worst"]
        for p in preferred:
            if p in topk and p != primary_score:
                budget_score = p
                break
        if budget_score is None:
            for m in topk:
                if m != primary_score:
                    budget_score = m
                    break

    if budget_score:
        decisions.append(MetricDecision(metric=budget_score, kind="score", tier="main", reason="Search-budget proxy (top-k mean worst) with clear paired separation."))

    # Third: choose success@threshold as main (prefer worst over hit)
    success_metric = None
    if threshold:
        for cand in [f"success_worst@{threshold}", f"success_avg@{threshold}"]:
            if cand in all_metrics:
                success_metric = cand
                break

    if success_metric:
        decisions.append(MetricDecision(metric=success_metric, kind="rate", tier="main", reason=f"Main success rate at threshold @{threshold} (less saturated than hit)."))

    # Decide about hit@threshold: include only if not saturated
    hit_metric = None
    if threshold:
        for cand in [f"hit_worst@{threshold}", f"hit_avg@{threshold}"]:
            if cand in all_metrics:
                hit_metric = cand
                break

    if hit_metric:
        frac = sat.get(hit_metric, {}).get("frac_ge_0.99", float("nan"))
        if np.isfinite(frac) and frac >= 0.5:
            decisions.append(MetricDecision(metric=hit_metric, kind="rate", tier="appendix", reason=f"Hit is saturated (frac>=0.99 is {frac:.3f}); keep as sanity/appendix."))
        else:
            decisions.append(MetricDecision(metric=hit_metric, kind="rate", tier="main", reason="Hit not saturated; can be included."))

    # Add threshold sensitivity table as appendix anchor (not a metric)

    # Add chemistry properties as appendix if present in paired
    chem = ["validity", "diversity", "qed", "lipinski", "logp"]
    for m in chem:
        if m in all_metrics:
            decisions.append(MetricDecision(metric=m, kind="property", tier="appendix", reason="Chemical quality/sanity metric; typically appendix."))

    # Exclude: low-information rates (very saturated) and redundant metrics.
    # Mark the rest for appendix if they have decent effect; else exclude.
    chosen = {d.metric for d in decisions}
    for m in all_metrics:
        if m in chosen:
            continue
        kind = str(ranked[ranked["metric"] == m].iloc[0]["kind"]) if not ranked.empty and (ranked["metric"] == m).any() else "other"
        fam = _metric_family(m)
        if fam == "hit":
            frac = sat.get(m, {}).get("frac_ge_0.99", float("nan"))
            if np.isfinite(frac) and frac >= 0.7:
                decisions.append(MetricDecision(metric=m, kind=kind, tier="exclude", reason=f"Highly saturated hit (frac>=0.99 is {frac:.3f})."))
            else:
                decisions.append(MetricDecision(metric=m, kind=kind, tier="appendix", reason="Hit-like metric; keep only if needed."))
        elif fam == "success":
            # keep a couple as appendix (other thresholds)
            decisions.append(MetricDecision(metric=m, kind=kind, tier="appendix", reason="Additional threshold/definition variant; appendix."))
        elif fam == "score":
            decisions.append(MetricDecision(metric=m, kind=kind, tier="appendix", reason="Additional score aggregation; appendix."))
        else:
            decisions.append(MetricDecision(metric=m, kind=kind, tier="exclude", reason="Not used for paper narrative."))

    return decisions, meta


def _pick_files_for_metric(files: List[str], metric: str) -> List[str]:
    out = []
    for p in files:
        # filename pattern: by_category__<metric>.csv (metric may include '@' and '.')
        base = os.path.basename(p)
        if base == f"by_category__{metric}.csv" or base == f"by_cutoff__{metric}.csv" or base == f"by_bin__{metric}.csv":
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="paper_tables")
    ap.add_argument("--out-dir", default="paper_table_bundle")
    ap.add_argument("--a-label", default=OURM_LABEL)
    args = ap.parse_args()

    inputs = _load_inputs(args.in_dir)

    overall: Optional[pd.DataFrame] = inputs.get("overall")  # type: ignore
    paired: Optional[pd.DataFrame] = inputs.get("paired")  # type: ignore
    diag: Optional[pd.DataFrame] = inputs.get("diag")  # type: ignore

    if paired is None or paired.empty:
        raise SystemExit("paired_comparisons.csv missing/empty; cannot rank paper metrics.")

    decisions, meta = _rank_metrics_for_paper(paired=paired, diag=diag)

    # Organize outputs
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    _ensure_dir(args.out_dir)

    # Always copy these inventory-like tables
    always = [
        "inventory.json",
        "metric_diagnostics.csv",
        "overall_summary.csv",
        "paired_comparisons.csv",
        "threshold_curve.csv",
        "reference_relative.csv",
        "bin_counts.csv",
    ]
    for f in always:
        src = os.path.join(args.in_dir, f)
        if os.path.exists(src):
            _safe_copy(src, os.path.join(args.out_dir, "all", f))

    # Copy metric-specific stratified tables.
    by_category_files: List[str] = inputs.get("by_category_files", [])  # type: ignore
    by_cutoff_files: List[str] = inputs.get("by_cutoff_files", [])  # type: ignore
    by_bin_files: List[str] = inputs.get("by_bin_files", [])  # type: ignore

    main_metrics = [d.metric for d in decisions if d.tier == "main"]
    appendix_metrics = [d.metric for d in decisions if d.tier == "appendix"]

    for metric in main_metrics:
        for p in _pick_files_for_metric(by_category_files, metric):
            _safe_copy(p, os.path.join(args.out_dir, "main", os.path.basename(p)))
        for p in _pick_files_for_metric(by_cutoff_files, metric):
            _safe_copy(p, os.path.join(args.out_dir, "main", os.path.basename(p)))
        for p in _pick_files_for_metric(by_bin_files, metric):
            _safe_copy(p, os.path.join(args.out_dir, "main", os.path.basename(p)))

    for metric in appendix_metrics:
        for p in _pick_files_for_metric(by_category_files, metric):
            _safe_copy(p, os.path.join(args.out_dir, "appendix", os.path.basename(p)))
        for p in _pick_files_for_metric(by_cutoff_files, metric):
            _safe_copy(p, os.path.join(args.out_dir, "appendix", os.path.basename(p)))
        for p in _pick_files_for_metric(by_bin_files, metric):
            _safe_copy(p, os.path.join(args.out_dir, "appendix", os.path.basename(p)))

    # Report
    ranked = _paired_rank_table(paired, a_label=args.a_label)
    report_lines: List[str] = []
    report_lines.append("# Paper table bundle report\n")
    report_lines.append(f"Input dir: {args.in_dir}")
    report_lines.append(f"Output dir: {args.out_dir}\n")
    report_lines.append(f"Primary threshold: {meta.get('primary_threshold')}\n")

    report_lines.append("## Main Tables\n")
    for d in decisions:
        if d.tier == "main":
            report_lines.append(f"- {d.metric}: {d.reason}")

    report_lines.append("\n## Appendix Tables\n")
    for d in decisions:
        if d.tier == "appendix":
            report_lines.append(f"- {d.metric}: {d.reason}")

    report_lines.append("\n## Excluded\n")
    for d in decisions:
        if d.tier == "exclude":
            report_lines.append(f"- {d.metric}: {d.reason}")

    report_lines.append("\n## Top paired effects (OurM vs baselines)\n")
    # show a small top list
    top = ranked.head(15)
    for _, r in top.iterrows():
        report_lines.append(
            f"- {r['metric']} vs {r['B_label']}: Δ={r['delta_mean(A-B)']:.4f} [{r['delta_lo']:.4f},{r['delta_hi']:.4f}], win={r['win_rate_A']:.3f}"
        )

    with open(os.path.join(args.out_dir, "table_bundle_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    # Machine-readable manifest
    manifest = {
        "meta": meta,
        "main_metrics": main_metrics,
        "appendix_metrics": appendix_metrics,
        "excluded_metrics": [d.metric for d in decisions if d.tier == "exclude"],
    }
    with open(os.path.join(args.out_dir, "paper_table_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
