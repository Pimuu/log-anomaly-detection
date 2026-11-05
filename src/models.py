"""Unsupervised detectors with a common ``fit(X) / score(X)`` interface.

All detectors return a score where **higher = more anomalous**.  None of them
ever sees a label: they are fitted on the (contaminated) early period and
applied to the later test period, exactly as a deployed detector would be.

* ``IForestDetector``     - Isolation Forest (Liu et al., 2008): anomalies are
                            isolated by fewer random axis-parallel splits.
* ``LOFDetector``         - Local Outlier Factor (Breunig et al., 2000) in
                            novelty mode: density relative to k neighbours.
* ``AutoencoderDetector`` - an under-complete MLP autoencoder trained to
                            reconstruct normal-looking feature vectors; the
                            reconstruction error is the anomaly score.
* ``RuleBaseline``        - a hand-written SIEM correlation-rule set, the
                            practical baseline a SOC would already have.
* ``RankEnsemble``        - combines the calibration-relative percentile ranks
                            of the three learned detectors by their mean
                            ("Ensemble-mean") or maximum ("Ensemble-max").
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class IForestDetector:
    name = "IsolationForest"

    def __init__(self, n_estimators=300, seed=0):
        # max_samples=256 is the value recommended by the original paper;
        # contamination is irrelevant because we threshold scores ourselves.
        self.model = IsolationForest(n_estimators=n_estimators, max_samples=256,
                                     random_state=seed, n_jobs=-1)

    def fit(self, X):
        self.model.fit(X)
        return self

    def score(self, X):
        return -self.model.score_samples(X)


class LOFDetector:
    name = "LOF"

    def __init__(self, n_neighbors=35, max_train=25000, seed=0):
        self.scaler = StandardScaler()
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, n_jobs=-1)
        self.max_train, self.rng = max_train, np.random.default_rng(seed)

    def fit(self, X):
        Xs = self.scaler.fit_transform(X)
        if len(Xs) > self.max_train:                     # LOF fit is ~O(n^2)
            Xs = Xs[self.rng.choice(len(Xs), self.max_train, replace=False)]
        # Many events are exact duplicates (e.g. identical page views);
        # tiny jitter avoids zero reachability distances and infinite LOF.
        Xs = Xs + self.rng.normal(0, 1e-3, Xs.shape)
        self.model.fit(Xs)
        return self

    def score(self, X):
        return -self.model.score_samples(self.scaler.transform(X))


class AutoencoderDetector:
    name = "Autoencoder"

    def __init__(self, hidden=(24, 8, 24), max_train=30000, seed=0):
        self.scaler = StandardScaler()
        self.model = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                                  alpha=1e-4, learning_rate_init=1e-3, max_iter=200,
                                  early_stopping=True, n_iter_no_change=10,
                                  random_state=seed)
        self.max_train, self.rng = max_train, np.random.default_rng(seed)

    def fit(self, X):
        Xs = self.scaler.fit_transform(X)
        if len(Xs) > self.max_train:
            Xs = Xs[self.rng.choice(len(Xs), self.max_train, replace=False)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(Xs, Xs)                       # learn identity through bottleneck
        return self

    def score(self, X):
        Xs = self.scaler.transform(X)
        return np.mean((self.model.predict(Xs) - Xs) ** 2, axis=1)


class RuleBaseline:
    """Classic threshold rules; score = number of rules that fire (0-5).

    R1  >= 10 failed logins from one IP within 5 min      (brute force)
    R2  >=  5 failed logins on one account within 10 min  (brute force)
    R3  >=  3 HTTP 403 for one account within 1 h          (priv. probing)
    R4  successful login from an IP never used by the account before
    R5  >= 10 distinct accounts from one IP within 30 min  (spraying)
    """
    name = "Rules"

    def fit(self, X):
        return self

    @staticmethod
    def score_rules(r):
        return ((r["ip_fail_5m"] >= 10).astype(int) + (r["user_fail_10m"] >= 5).astype(int)
                + (r["user_forbidden_1h"] >= 3).astype(int) + r["new_ip_login"].astype(int)
                + (r["ip_users_30m"] >= 10).astype(int)).to_numpy(dtype=float)


class RankEnsemble:
    """Combine detectors via percentile ranks relative to calibration scores.

    Raw scores of different detectors live on incomparable scales, so each is
    first mapped to its empirical CDF on the calibration period.  ``mean``
    rewards events that *all* detectors find unusual (consensus); ``max``
    rewards events that *any* detector finds extreme, which suits detectors
    with complementary blind spots (Aggarwal & Sathe, 2015).
    """

    def __init__(self, members, how="mean"):
        self.members, self.how = members, how
        self.name = f"Ensemble-{how}"

    def fit_from_scores(self, calib_scores: dict):
        self.ref = {m: np.sort(calib_scores[m]) for m in self.members}
        return self

    def combine(self, scores: dict):
        ranks = [np.searchsorted(self.ref[m], scores[m], side="right") / len(self.ref[m])
                 for m in self.members]
        return np.mean(ranks, axis=0) if self.how == "mean" else np.max(ranks, axis=0)
