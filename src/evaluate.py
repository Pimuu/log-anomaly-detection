"""Evaluation against the hidden ground truth.

This is the **only** module that reads ``labels.csv``.  Metrics:

* threshold-free: ROC-AUC and average precision (area under the PR curve);
* at the operating threshold (alert budget): precision, recall, F1, alert rate;
* ``best_f1``: F1 at the oracle threshold that maximises it - an *upper bound*
  that a real unsupervised deployment cannot attain, reported for context;
* per attack type: event-level recall and campaign-level detection rate
  (a campaign counts as detected if at least one of its events is alerted,
  because a single SOC alert is enough to open an investigation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score)


def threshold_from_budget(calib_scores: np.ndarray, budget: float) -> float:
    """Label-free threshold: the (1 - budget) quantile of calibration scores.

    This mimics how a SOC sizes an alert queue: "we can triage about 2 % of
    events", rather than assuming the (unknown) true contamination rate.
    """
    return float(np.quantile(calib_scores, 1.0 - budget))


def binary_metrics(y: np.ndarray, score: np.ndarray, thr: float) -> dict:
    pred = score >= thr
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    prec, rec, _ = precision_recall_curve(y, score)
    f_all = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(roc_auc=roc_auc_score(y, score), avg_precision=average_precision_score(y, score),
                precision=p, recall=r, f1=f, best_f1=float(f_all.max()),
                alert_rate=float(pred.mean()), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def per_type_metrics(labels: pd.DataFrame, score: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """Per attack type: ROC-AUC vs. all normal events, event recall and
    campaign detection rate at the operating threshold."""
    d = labels.assign(pred=pred, score=score)
    att = d[d.is_anomaly == 1]
    normal_scores = d.loc[d.is_anomaly == 0, "score"].to_numpy()
    auc = {k: roc_auc_score(np.r_[np.zeros(len(normal_scores)), np.ones(len(g))],
                            np.r_[normal_scores, g["score"].to_numpy()])
           for k, g in att.groupby("attack_type")}
    ev = att.groupby("attack_type")["pred"].mean().rename("event_recall")
    camp = att.groupby(["attack_type", "attack_id"])["pred"].max().groupby("attack_type")
    out = pd.concat([pd.Series(auc, name="type_auc"), ev,
                     camp.mean().rename("campaign_detection"),
                     camp.size().rename("n_campaigns")], axis=1)
    return out.rename_axis("attack_type").reset_index()


def false_positive_breakdown(labels: pd.DataFrame, users: pd.Series,
                             pred: np.ndarray) -> pd.DataFrame:
    """Where do false alarms come from?  (benign confounders vs. other)."""
    d = labels.assign(pred=pred, user=users.values)
    normal = d[d.is_anomaly == 0].copy()
    normal["source"] = np.where(normal.benign_tag.fillna("") != "", normal.benign_tag,
                                np.where(normal.user.str.startswith("svc_"),
                                         "service_account", "ordinary_human"))
    g = normal.groupby("source")
    return pd.DataFrame({"n_events": g.size(), "false_positives": g["pred"].sum(),
                         "fp_rate": g["pred"].mean()}).reset_index()
