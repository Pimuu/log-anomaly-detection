"""End-to-end experiment: generate -> featurise -> fit -> score -> evaluate.

``run_seed`` executes one complete, independent replicate (new synthetic
organisation, new attacks, new model seeds).  ``run_all`` repeats it for
several seeds, runs the feature-group ablation and writes every table and
figure under ``results/``.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from config import GeneratorConfig, PipelineConfig
from evaluate import (binary_metrics, false_positive_breakdown, per_type_metrics,
                      threshold_from_budget)
from features import FEATURE_GROUPS, FEATURES, build_features, rule_inputs
from generate import generate
from models import (AutoencoderDetector, IForestDetector, LOFDetector, RankEnsemble,
                    RuleBaseline)

METHODS = ["IsolationForest", "LOF", "Autoencoder", "Ensemble-mean", "Ensemble-max", "Rules"]
LEARNED = ["IsolationForest", "LOF", "Autoencoder"]


def split_by_day(logs: pd.DataFrame, pcfg: PipelineConfig):
    """Chronological split -> boolean masks (fit, calibration, test).

    Splitting by whole days (not by random events) prevents near-duplicate
    events of one session from landing on both sides of a split.  Scores on
    the calibration days are out-of-sample for every detector, which matters
    for LOF/autoencoder whose in-sample scores are optimistically low.
    """
    ts = pd.to_datetime(logs["timestamp"])
    day = (ts - ts.iloc[0].normalize()).dt.days.to_numpy()
    b, f, c = pcfg.burnin_days, pcfg.fit_days, pcfg.calib_days
    fit = (day >= b) & (day < b + f)
    calib = (day >= b + f) & (day < b + f + c)
    test = day >= b + f + c
    return fit, calib, test


def run_seed(seed: int, pcfg: PipelineConfig, data_dir: str | None = None,
             verbose: bool = True) -> dict:
    t0 = time.time()
    logs, labels, campaigns = generate(GeneratorConfig(seed=seed))
    if data_dir:  # persist the dataset of the reference seed
        os.makedirs(data_dir, exist_ok=True)
        logs.to_csv(os.path.join(data_dir, "logs.csv.gz"), index=False)
        labels.to_csv(os.path.join(data_dir, "labels.csv.gz"), index=False)
        campaigns.to_csv(os.path.join(data_dir, "campaigns.csv"), index=False)

    feats = build_features(logs)
    X = feats.to_numpy()
    fit, calib, test = split_by_day(logs, pcfg)
    y = labels["is_anomaly"].to_numpy()

    # ---- fit the learned detectors on the unlabelled fit period ------------
    detectors = {
        "IsolationForest": IForestDetector(pcfg.iforest_estimators, seed),
        "LOF": LOFDetector(pcfg.lof_neighbors, pcfg.lof_max_train, seed),
        "Autoencoder": AutoencoderDetector(pcfg.ae_hidden, pcfg.ae_max_train, seed),
    }
    scores, fit_time = {}, {}
    for name, det in detectors.items():
        s = time.time()
        det.fit(X[fit])
        scores[name] = det.score(X)
        fit_time[name] = time.time() - s
    for how in ("mean", "max"):
        ens = RankEnsemble(LEARNED, how).fit_from_scores({m: scores[m][calib] for m in LEARNED})
        scores[ens.name] = ens.combine(scores)
    scores["Rules"] = RuleBaseline.score_rules(rule_inputs(feats))

    # ---- thresholds: label-free alert budget on the calibration period -----
    thresholds = {m: threshold_from_budget(scores[m][calib], pcfg.alert_budget)
                  for m in METHODS if m != "Rules"}
    thresholds["Rules"] = 1.0  # any rule firing raises an alert

    rows, per_type, fp_rows = [], [], []
    for m in METHODS:
        met = binary_metrics(y[test], scores[m][test], thresholds[m])
        rows.append(dict(seed=seed, method=m, threshold=thresholds[m], **met))
        pred = scores[m][test] >= thresholds[m]
        pt = per_type_metrics(labels[test].reset_index(drop=True), scores[m][test], pred)
        per_type.append(pt.assign(seed=seed, method=m))
        fp = false_positive_breakdown(labels[test].reset_index(drop=True),
                                      logs["user"][test], pred)
        fp_rows.append(fp.assign(seed=seed, method=m))

    if verbose:
        msg = ", ".join(f"{r['method']}: AUC={r['roc_auc']:.3f} F1={r['f1']:.3f}" for r in rows)
        print(f"[seed {seed}] {len(logs):,} events ({y.mean():.2%} anomalous) "
              f"in {time.time() - t0:.1f}s -> {msg}")

    return dict(seed=seed, logs=logs, labels=labels, campaigns=campaigns, feats=feats,
                fit=fit, calib=calib, test=test, scores=scores, thresholds=thresholds,
                metrics=pd.DataFrame(rows), per_type=pd.concat(per_type),
                fp=pd.concat(fp_rows), fit_time=fit_time)


def ablation(res: dict, pcfg: PipelineConfig) -> pd.DataFrame:
    """Isolation Forest with one feature group removed at a time."""
    feats, y = res["feats"], res["labels"]["is_anomaly"].to_numpy()
    fit, calib, test = res["fit"], res["calib"], res["test"]
    rows = []
    for group in ["none"] + list(FEATURE_GROUPS):
        cols = [c for c in FEATURES if group == "none" or c not in FEATURE_GROUPS[group]]
        X = feats[cols].to_numpy()
        det = IForestDetector(pcfg.iforest_estimators, res["seed"]).fit(X[fit])
        s = det.score(X)
        thr = threshold_from_budget(s[calib], pcfg.alert_budget)
        met = binary_metrics(y[test], s[test], thr)
        rows.append(dict(seed=res["seed"], removed_group=group, n_features=len(cols),
                         roc_auc=met["roc_auc"], avg_precision=met["avg_precision"],
                         f1=met["f1"], recall=met["recall"], precision=met["precision"]))
    return pd.DataFrame(rows)


def budget_sweep(res: dict, pcfg: PipelineConfig) -> pd.DataFrame:
    """Precision / recall / F1 on the test period for several alert budgets."""
    y = res["labels"]["is_anomaly"].to_numpy()
    calib, test = res["calib"], res["test"]
    rows = []
    for m in METHODS:
        if m == "Rules":
            continue  # rules have a fixed operating point
        for b in pcfg.budget_grid:
            thr = threshold_from_budget(res["scores"][m][calib], b)
            met = binary_metrics(y[test], res["scores"][m][test], thr)
            rows.append(dict(seed=res["seed"], method=m, budget=b, precision=met["precision"],
                             recall=met["recall"], f1=met["f1"], alert_rate=met["alert_rate"]))
    return pd.DataFrame(rows)


def lof_k_sweep(res: dict, pcfg: PipelineConfig) -> pd.DataFrame:
    """LOF sensitivity to the neighbourhood size k (masking by attack bursts)."""
    X, y = res["feats"].to_numpy(), res["labels"]["is_anomaly"].to_numpy()
    fit, calib, test = res["fit"], res["calib"], res["test"]
    rows = []
    for k in pcfg.lof_k_grid:
        s = LOFDetector(k, pcfg.lof_max_train, res["seed"]).fit(X[fit]).score(X)
        met = binary_metrics(y[test], s[test], threshold_from_budget(s[calib], pcfg.alert_budget))
        pt = per_type_metrics(res["labels"][test].reset_index(drop=True), s[test],
                              s[test] >= threshold_from_budget(s[calib], pcfg.alert_budget))
        row = dict(seed=res["seed"], k=k, roc_auc=met["roc_auc"],
                   avg_precision=met["avg_precision"], f1=met["f1"])
        row.update({f"auc_{a}": v for a, v in zip(pt.attack_type, pt.type_auc)})
        rows.append(row)
    return pd.DataFrame(rows)
