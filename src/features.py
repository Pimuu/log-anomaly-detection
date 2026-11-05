"""Feature engineering: raw log rows -> one numeric vector per event.

Design principles
-----------------
* **Event-level, causal features.**  Every feature of an event at time *t* is
  computed only from events at times <= *t*.  A detector built on these
  features could run online in a SIEM, and no information leaks from the
  future (e.g. "this IP will go on to make 200 requests").
* **Context from two entities.**  Authentication attacks are visible either
  from the *source IP* (brute force, spraying) or from the *account*
  (odd hours, new IP, privilege probing), so sliding-window aggregates are
  computed for both, plus long-term "profile" features describing how
  familiar the (user, IP), (user, hour) and (user, endpoint) combinations are.
* **Heavy tails are log-compressed** (``log1p``) so that distance-based
  methods (LOF, autoencoder) are not dominated by a single huge count.

No label information is touched here.
"""
from __future__ import annotations

from collections import Counter, deque

import numpy as np
import pandas as pd

ADMIN_PREFIX = "/admin"

# Feature groups (used for documentation and for the ablation study).
FEATURE_GROUPS = {
    "event": ["is_login", "is_fail_login", "is_forbidden", "is_admin_ep", "ep_rarity"],
    "time": ["hour_sin", "hour_cos", "is_weekend", "user_hour_rarity"],
    "ip_window": ["ip_events_1m", "ip_fail_5m", "ip_fail_ratio_1h", "ip_users_30m"],
    "user_window": ["user_fail_10m", "user_events_5m", "user_forbidden_1h",
                    "user_ips_24h", "user_gap_log"],
    "profile": ["ip_events_7d", "ip_users_7d", "user_events_7d", "user_ip_share_7d",
                "user_ip_age", "user_ep_share_7d"],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]


# ---------------------------------------------------------------------------
# Generic causal sliding-window helpers
# ---------------------------------------------------------------------------
def _window_sum(keys: pd.Series, t: np.ndarray, window_s: float,
                values: np.ndarray | None = None) -> np.ndarray:
    """For every row, sum ``values`` over rows with the same key in (t-w, t].

    Fully vectorised: rows are sorted by (key, time) and a composite
    ``key_code * BIG + t`` array is binary-searched, so all groups are
    processed in a single ``searchsorted`` call.
    """
    if values is None:
        values = np.ones(len(t))
    codes = pd.factorize(keys)[0].astype(np.float64)
    big = 1e8                                       # > time span in seconds
    order = np.lexsort((np.arange(len(t)), t, codes))  # stable within ties
    comp = codes[order] * big + t[order]
    csum = np.concatenate([[0.0], np.cumsum(values[order])])
    lo = np.searchsorted(comp, comp - window_s, side="right")
    hi = np.arange(len(t)) + 1                      # include the current row
    out = np.empty(len(t))
    out[order] = csum[hi] - csum[lo]
    return out


def _window_distinct(keys: pd.Series, others: pd.Series, t: np.ndarray,
                     window_s: float) -> np.ndarray:
    """Distinct ``others`` per ``keys`` within the causal window (t-w, t]."""
    out = np.zeros(len(t))
    df = pd.DataFrame({"k": keys.values, "o": others.values, "t": t,
                       "i": np.arange(len(t))})
    for _, g in df.groupby("k", sort=False):
        q: deque = deque()
        cnt: Counter = Counter()
        for o, tt, i in zip(g["o"].values, g["t"].values, g["i"].values):
            while q and q[0][1] <= tt - window_s:
                old, _ = q.popleft()
                cnt[old] -= 1
                if cnt[old] == 0:
                    del cnt[old]
            q.append((o, tt))
            cnt[o] += 1
            out[i] = len(cnt)
    return out


def _prior_count(df: pd.DataFrame, cols) -> np.ndarray:
    """Number of *earlier* events sharing the values of ``cols`` (cumcount)."""
    return df.groupby(cols, sort=False).cumcount().to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------
def build_features(logs: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame (same row order as ``logs``) with columns FEATURES.

    ``logs`` must be sorted by timestamp (the generator guarantees this).
    """
    df = logs.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    t = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    f = pd.DataFrame(index=df.index)

    is_login = (df["endpoint"] == "/login").to_numpy()
    fail = is_login & (df["status"] == 401).to_numpy()
    forbidden = (df["status"] == 403).to_numpy()
    admin_ep = df["endpoint"].str.startswith(ADMIN_PREFIX).to_numpy()
    ep_key = df["method"] + " " + df["endpoint"]

    # --- event intrinsics -------------------------------------------------
    f["is_login"] = is_login.astype(float)
    f["is_fail_login"] = fail.astype(float)
    f["is_forbidden"] = forbidden.astype(float)
    f["is_admin_ep"] = admin_ep.astype(float)
    # Surprisal of the (method, endpoint) pair given all events seen so far.
    n_prior = np.arange(len(df), dtype=float)
    ep_prior = _prior_count(df.assign(ep=ep_key), ["ep"])
    f["ep_rarity"] = -np.log((ep_prior + 1.0) / (n_prior + 20.0))

    # --- time ---------------------------------------------------------------
    hour = (df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0).to_numpy()
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(float).to_numpy()
    # How unusual is this hour *for this user*?  Smoothed surprisal of the
    # user's past activity in the +/-1 h bucket around the current hour.
    hb = df["timestamp"].dt.hour.to_numpy()
    onehot = np.zeros((len(df), 24))
    onehot[np.arange(len(df)), hb] = 1.0
    cum = pd.DataFrame(onehot).groupby(df["user"].values).cumsum().to_numpy() - onehot
    near = cum[np.arange(len(df)), hb] + cum[np.arange(len(df)), (hb - 1) % 24] \
        + cum[np.arange(len(df)), (hb + 1) % 24]
    user_prior = cum.sum(axis=1)
    f["user_hour_rarity"] = -np.log((near + 1.0) / (user_prior + 8.0))

    # --- source-IP sliding windows (rate / failure / fan-out) --------------
    ip = df["src_ip"]
    ip_ev_1h = _window_sum(ip, t, 3600)
    f["ip_events_1m"] = np.log1p(_window_sum(ip, t, 60))
    f["ip_fail_5m"] = np.log1p(_window_sum(ip, t, 300, fail.astype(float)))
    f["ip_fail_ratio_1h"] = _window_sum(ip, t, 3600, fail.astype(float)) / ip_ev_1h
    f["ip_users_30m"] = np.log1p(_window_distinct(ip, df["user"], t, 1800))

    # --- account sliding windows -------------------------------------------
    user = df["user"]
    f["user_fail_10m"] = np.log1p(_window_sum(user, t, 600, fail.astype(float)))
    f["user_events_5m"] = np.log1p(_window_sum(user, t, 300))
    f["user_forbidden_1h"] = np.log1p(_window_sum(user, t, 3600, forbidden.astype(float)))
    f["user_ips_24h"] = np.log1p(_window_distinct(user, ip, t, 86400))
    gap = pd.Series(t).groupby(user.values).diff().fillna(7 * 86400).to_numpy()
    f["user_gap_log"] = np.log1p(gap)

    # --- long-term profile / novelty (7-day lookback) ------------------------
    # Profiles use a fixed 7-day lookback instead of all-time cumulative
    # counts: cumulative counts grow without bound, so the test period would
    # contain values never seen at fit time (non-stationarity) and distance /
    # reconstruction-based detectors would flag *everything late* as odd.
    week = 7 * 86400.0
    pair = df["user"] + "|" + ip
    user_ev_7d = _window_sum(user, t, week)
    f["ip_events_7d"] = np.log1p(_window_sum(ip, t, week))
    f["ip_users_7d"] = np.log1p(_window_distinct(ip, user, t, week))
    f["user_events_7d"] = np.log1p(user_ev_7d)
    f["user_ip_share_7d"] = _window_sum(pair, t, week) / user_ev_7d
    # Age of the (user, IP) relationship, capped at the lookback length.
    # The group minimum is causal because it is always <= the current time.
    first_seen = pd.Series(t).groupby(pair.values).transform("min").to_numpy()
    f["user_ip_age"] = np.log1p(np.minimum(t - first_seen, week))
    f["user_ep_share_7d"] = _window_sum(df["user"] + "|" + ep_key, t, week) / user_ev_7d

    return f[FEATURES]


# Raw (non-log) counts used by the rule-based SIEM baseline.
def rule_inputs(feats: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "ip_fail_5m": np.expm1(feats["ip_fail_5m"]),
        "user_fail_10m": np.expm1(feats["user_fail_10m"]),
        "user_forbidden_1h": np.expm1(feats["user_forbidden_1h"]),
        # first successful login of an account from an IP (pair age == 0)
        # for an account that already has history in the lookback window
        "new_ip_login": ((feats["user_ip_age"] == 0) & (feats["user_events_7d"] > np.log(2) + 1e-9)
                         & (feats["is_login"] == 1) & (feats["is_fail_login"] == 0)),
        "ip_users_30m": np.expm1(feats["ip_users_30m"]),
    })
