#!/usr/bin/env python3
"""Reproduce every number, table and figure of the study with one command.

    python run_pipeline.py            # full study: 5 seeds + sweeps (~5 min on 4 cores)
    python run_pipeline.py --quick    # 1 seed, no sweeps (~1 min) for a smoke test

Outputs
  data/      synthetic dataset of the reference seed (logs, hidden labels, campaigns)
  results/   metrics CSVs, summary.md, figures/*.png
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import PipelineConfig  # noqa: E402
import pipeline as P  # noqa: E402
import plots  # noqa: E402


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Minimal DataFrame -> GitHub-markdown table (avoids a tabulate dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="one seed, skip sweeps")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    args = ap.parse_args()

    pcfg = PipelineConfig()
    seeds = (0,) if args.quick else pcfg.seeds_for_variance
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    t0 = time.time()

    metrics, per_type, fps, abl, budget, lofk, stats = [], [], [], [], [], [], []
    ref = None
    for seed in seeds:
        res = P.run_seed(seed, pcfg, data_dir=args.data if seed == seeds[0] else None)
        metrics.append(res["metrics"])
        per_type.append(res["per_type"])
        fps.append(res["fp"])
        lab, test = res["labels"], res["test"]
        stats.append(dict(seed=seed, events=len(lab), anomalous_events=int(lab.is_anomaly.sum()),
                          anomaly_rate=lab.is_anomaly.mean(), campaigns=lab.attack_id.nunique() - 1,
                          fit_events=int(res["fit"].sum()), calib_events=int(res["calib"].sum()),
                          test_events=int(test.sum()),
                          calib_anomaly_rate=lab.is_anomaly[res["calib"]].mean(),
                          test_anomaly_rate=lab.is_anomaly[test].mean(),
                          test_campaigns=lab.attack_id[test].nunique() - 1,
                          **{f"fit_seconds_{k}": v for k, v in res["fit_time"].items()}))
        if not args.quick:
            abl.append(P.ablation(res, pcfg))
            budget.append(P.budget_sweep(res, pcfg))
            lofk.append(P.lof_k_sweep(res, pcfg))
        if seed == seeds[0]:
            ref = res
        else:
            del res

    metrics = pd.concat(metrics, ignore_index=True)
    per_type = pd.concat(per_type, ignore_index=True)
    fps = pd.concat(fps, ignore_index=True)
    stats = pd.DataFrame(stats)

    # ---------------- tables ----------------
    metrics.to_csv(os.path.join(args.out, "metrics_per_seed.csv"), index=False)
    cols = ["roc_auc", "avg_precision", "precision", "recall", "f1", "best_f1", "alert_rate"]
    summ = metrics.groupby("method", sort=False)[cols].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(os.path.join(args.out, "metrics_summary.csv"), index=False)

    per_type.to_csv(os.path.join(args.out, "per_attack_type_per_seed.csv"), index=False)
    per_type["detected_campaigns"] = per_type.campaign_detection * per_type.n_campaigns
    pt = per_type.groupby(["method", "attack_type"], sort=False).agg(
        type_auc=("type_auc", "mean"), event_recall=("event_recall", "mean"),
        detected_campaigns=("detected_campaigns", "sum"), n_campaigns=("n_campaigns", "sum"))
    pt["campaign_detection_pooled"] = pt.detected_campaigns / pt.n_campaigns
    pt = pt.reset_index()
    pt.to_csv(os.path.join(args.out, "per_attack_type_summary.csv"), index=False)

    fp = fps.groupby(["method", "source"], sort=False)[["n_events", "false_positives"]].sum()
    fp["fp_rate"] = fp.false_positives / fp.n_events
    fp = fp.reset_index()
    fp.to_csv(os.path.join(args.out, "false_positive_breakdown.csv"), index=False)
    stats.to_csv(os.path.join(args.out, "dataset_stats.csv"), index=False)

    # ---------------- figures ----------------
    methods = P.METHODS
    plots.fig_data_overview(ref, pcfg, fig_dir)
    plots.fig_features(ref, fig_dir)
    plots.fig_scores(ref, methods, fig_dir)
    # The timeline shows the detector with the best mean average precision;
    # this label-informed choice is for illustration only.
    best = summ["avg_precision_mean"].drop("Rules").idxmax()
    plots.fig_timeline(ref, best, fig_dir)
    plots.fig_confusion(ref, methods, fig_dir)
    plots.fig_roc_pr(ref, methods, fig_dir)
    plots.fig_by_type(per_type, methods, fig_dir)
    if not args.quick:
        abl, budget, lofk = (pd.concat(x, ignore_index=True) for x in (abl, budget, lofk))
        abl.to_csv(os.path.join(args.out, "ablation_feature_groups.csv"), index=False)
        budget.to_csv(os.path.join(args.out, "budget_sweep.csv"), index=False)
        lofk.to_csv(os.path.join(args.out, "lof_k_sweep.csv"), index=False)
        plots.fig_sensitivity(budget, abl, lofk, fig_dir)

    # ---------------- human-readable summary ----------------
    fmt = summ.reset_index()
    pretty = pd.DataFrame({"method": fmt.method})
    for c in cols:
        pretty[c] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(fmt[f"{c}_mean"], fmt[f"{c}_std"])]
    lines = [f"# Results summary ({len(seeds)} seed(s): {list(seeds)})", "",
             "Test period only; threshold = 2 % alert budget on the calibration period.", "",
             "## Overall metrics (mean ± std over seeds)", "", md_table(pretty), "",
             "## Dataset statistics", "", md_table(stats.round(4)), "",
             "## Per attack type (ROC-AUC & event recall: mean over seeds; "
             "campaign detection: pooled over seeds)", "",
             md_table(pt.round(3)), "",
             "## False-positive sources (summed over seeds)", "", md_table(fp.round(4), "{:.4f}"), ""]
    if not args.quick:
        a = abl.groupby("removed_group", sort=False)[["roc_auc", "avg_precision", "f1"]].agg(
            ["mean", "std"]).round(3)
        a.columns = [f"{x}_{y}" for x, y in a.columns]
        b = budget.groupby(["method", "budget"], sort=False)[["precision", "recall", "f1",
                                                              "alert_rate"]].mean().round(3)
        k = lofk.groupby("k").mean(numeric_only=True).drop(columns="seed").round(3)
        lines += ["## Feature-group ablation (Isolation Forest)", "", md_table(a.reset_index()), "",
                  "## Alert-budget sweep (mean over seeds)", "", md_table(b.reset_index()), "",
                  "## LOF neighbourhood-size sweep (mean over seeds)", "",
                  md_table(k.reset_index()), ""]
    lines.append(f"Total runtime: {time.time() - t0:.0f} s")
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    print(f"\nDone in {time.time() - t0:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
