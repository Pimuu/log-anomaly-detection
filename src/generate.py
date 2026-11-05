"""Synthetic authentication / access-log generator.

The generator simulates a mid-sized organisation's web application that is
fronted by a login endpoint.  It produces two files:

* ``logs.csv``   - what a SIEM would see: one row per HTTP request
                   (timestamp, user, src_ip, method, endpoint, status).
* ``labels.csv`` - hidden ground truth keyed by ``event_id``.  It is used
                   **only** by the evaluation step, never by the detectors.

Normal behaviour
----------------
* Human users have personal working hours, a session rate, a preferred mix of
  endpoints and a small set of source IPs (shared office NAT, a VPN pool or a
  home address).  Weekends are quiet.  ~6 % of users are administrators who
  legitimately use ``/admin/*``.
* Service accounts run scheduled jobs 24/7 from fixed internal addresses, so
  "night-time activity" alone is *not* a sufficient anomaly signal.
* Benign confounders are injected on purpose: password typos, forgotten
  passwords (short failure streaks), business travel (new IP for a day) and
  occasional legitimate late-night work.  They are tagged in ``labels.csv``
  (``benign_tag``) for error analysis but are labelled *normal*.

Attack campaigns (labelled anomalous)
-------------------------------------
``brute_force``          single external IP hammers one account (30-150 tries)
``password_spray``       single external IP tries 1-2 passwords on many users
``odd_hour_login``       stolen credentials used between 01:00 and 04:59
``rare_ip_takeover``     stolen credentials used from a never-seen IP
``privilege_escalation`` non-admin insider probes ``/admin/*`` (403s), then
                         abuses a misconfiguration to gain admin access

Every event emitted by an attacker is labelled anomalous, including
ordinary-looking page views inside a hijacked session; this makes the
event-level task deliberately harder and more honest.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from config import GeneratorConfig

# ---------------------------------------------------------------------------
# Static application model
# ---------------------------------------------------------------------------
REGULAR_ENDPOINTS = [
    "/dashboard", "/api/orders", "/api/customers", "/api/reports",
    "/api/inventory", "/profile", "/api/search", "/files/download", "/api/export",
]
# Baseline popularity of each regular endpoint (export / download are rare).
REGULAR_POPULARITY = np.array([0.22, 0.17, 0.12, 0.10, 0.12, 0.05, 0.15, 0.05, 0.02])
ADMIN_ENDPOINTS = ["/admin/users", "/admin/roles", "/admin/audit-log", "/admin/settings"]
SERVICE_ENDPOINTS = ["/api/export", "/api/inventory", "/api/orders", "/api/reports"]

# Column order of the emitted log (event_id is added after sorting).
LOG_COLUMNS = ["timestamp", "user", "src_ip", "method", "endpoint", "status"]


@dataclass
class User:
    """Behavioural profile of one human account."""

    name: str
    is_admin: bool
    work_start: float          # hour of day the working window opens
    work_len: float            # length of the working window (hours)
    sessions_per_day: float    # Poisson mean of sessions on an active day
    weekend_prob: float        # P(active on a Saturday/Sunday)
    office_prob: float         # P(working from the office on a given day)
    uses_vpn: bool             # remote work goes through the VPN pool
    home_ip: str
    endpoint_pref: np.ndarray = field(repr=False)


class Emitter:
    """Collects log rows together with their hidden labels."""

    def __init__(self) -> None:
        self.rows: List[tuple] = []

    def emit(self, ts: float, user: str, ip: str, method: str, endpoint: str,
             status: int, label: int = 0, attack_type: str = "normal",
             attack_id: int = -1, benign_tag: str = "") -> None:
        self.rows.append((ts, user, ip, method, endpoint, status,
                          label, attack_type, attack_id, benign_tag))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def random_public_ip(rng: np.random.Generator) -> str:
    """Random routable-looking IPv4 address (avoids private / reserved /8s)."""
    first = rng.choice([a for a in range(11, 224) if a not in (100, 127, 169, 172, 192, 198, 203)])
    return f"{first}.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}"


def make_users(cfg: GeneratorConfig, rng: np.random.Generator) -> List[User]:
    users = []
    n_admin = max(2, int(round(cfg.admin_fraction * cfg.n_users)))
    admin_idx = set(rng.choice(cfg.n_users, n_admin, replace=False).tolist())
    for i in range(cfg.n_users):
        pref = rng.dirichlet(REGULAR_POPULARITY * 40)  # user-specific endpoint mix
        users.append(User(
            name=f"user{i:03d}",
            is_admin=i in admin_idx,
            work_start=float(np.clip(rng.normal(8.4, 0.9), 6.0, 11.0)),
            work_len=float(np.clip(rng.normal(8.5, 1.0), 6.0, 11.0)),
            sessions_per_day=float(np.clip(rng.lognormal(np.log(2.2), 0.4), 0.8, 7)),
            weekend_prob=float(rng.beta(1.2, 10)),
            office_prob=float(rng.beta(4, 2)),
            uses_vpn=bool(rng.random() < 0.35),
            home_ip=random_public_ip(rng),
            endpoint_pref=pref,
        ))
    return users


def emit_login(em: Emitter, rng, t: float, user: str, ip: str, n_fail: int,
               success: bool = True, gap=(4, 20), **lab) -> float:
    """Emit ``n_fail`` failed logins followed by an optional successful one."""
    for _ in range(n_fail):
        em.emit(t, user, ip, "POST", "/login", 401, **lab)
        t += rng.uniform(*gap)
    if success:
        em.emit(t, user, ip, "POST", "/login", 200, **lab)
    return t


def emit_session_body(em: Emitter, rng, t: float, user: str, ip: str,
                      n_req: int, pref: np.ndarray, admin_share: float,
                      think_mean: float = 35.0, logout_p: float = 0.6, **lab) -> float:
    """Emit the page views of one authenticated session."""
    for _ in range(n_req):
        t += rng.exponential(think_mean) + 1.0
        if rng.random() < admin_share:
            ep, method = ADMIN_ENDPOINTS[rng.integers(len(ADMIN_ENDPOINTS))], "GET"
            status = 200
        else:
            ep = REGULAR_ENDPOINTS[rng.choice(len(REGULAR_ENDPOINTS), p=pref)]
            method = "GET" if ep != "/api/orders" or rng.random() < 0.8 else "POST"
            status = 404 if rng.random() < 0.01 else 200
        em.emit(t, user, ip, method, ep, status, **lab)
    if rng.random() < logout_p:
        t += rng.exponential(10) + 1
        em.emit(t, user, ip, "POST", "/logout", 200, **lab)
    return t


# ---------------------------------------------------------------------------
# Normal traffic
# ---------------------------------------------------------------------------
def simulate_normal(cfg: GeneratorConfig, rng, users: List[User], em: Emitter,
                    office_ips, vpn_ips) -> None:
    day_s = 86400.0
    for d in range(cfg.n_days):
        weekend = (d % 7) >= 5
        for u in users:
            p_active = u.weekend_prob if weekend else 0.92
            if rng.random() >= p_active:
                continue
            # Where is the user working from today?
            travel = rng.random() < cfg.p_travel_day
            if travel:
                day_ip, tag = random_public_ip(rng), "travel"
            elif not weekend and rng.random() < u.office_prob:
                day_ip, tag = office_ips[rng.integers(len(office_ips))], ""
            else:
                day_ip = vpn_ips[rng.integers(len(vpn_ips))] if u.uses_vpn else u.home_ip
                tag = ""
            n_sess = max(1, rng.poisson(u.sessions_per_day * (0.5 if weekend else 1.0)))
            starts = np.sort(rng.uniform(u.work_start, u.work_start + u.work_len, n_sess))
            for h in starts:
                ip = day_ip
                if u.uses_vpn and not travel and ip in vpn_ips:
                    ip = vpn_ips[rng.integers(len(vpn_ips))]   # VPN address rotates
                t = d * day_s + h * 3600.0
                _benign_session(cfg, rng, em, u, t, ip, tag)
            # Rare legitimate late-night work (confounder for odd-hour attacks).
            if rng.random() < cfg.p_late_night_day:
                h = rng.uniform(22.0, 26.5)  # may spill past midnight
                t = d * day_s + h * 3600.0
                ip = vpn_ips[rng.integers(len(vpn_ips))] if u.uses_vpn else u.home_ip
                _benign_session(cfg, rng, em, u, t, ip, "late_night")


def _benign_session(cfg, rng, em, u: User, t: float, ip: str, tag: str) -> None:
    lab = dict(benign_tag=tag)
    r = rng.random()
    if r < cfg.p_forgot_password:
        # Forgotten password: a short failure streak, sometimes giving up.
        n_fail = int(rng.integers(4, 9))
        ok = rng.random() < 0.6
        t = emit_login(em, rng, t, u.name, ip, n_fail, ok, gap=(8, 40),
                       benign_tag="forgot_password")
        if not ok:
            return
    else:
        n_fail = int(rng.integers(1, 4)) if r < cfg.p_typo else 0
        t = emit_login(em, rng, t, u.name, ip, n_fail, True, **lab)
    n_req = int(np.clip(rng.lognormal(np.log(8), 0.6), 1, 60))
    admin_share = 0.18 if u.is_admin else 0.0
    t = emit_session_body(em, rng, t, u.name, ip, n_req, u.endpoint_pref,
                          admin_share, **lab)
    # A regular user occasionally clicks a stale admin link -> one 403.
    if not u.is_admin and rng.random() < 0.004:
        em.emit(t + rng.uniform(2, 30), u.name, ip, "GET",
                ADMIN_ENDPOINTS[rng.integers(len(ADMIN_ENDPOINTS))], 403, **lab)


def simulate_service_accounts(cfg: GeneratorConfig, rng, em: Emitter) -> None:
    """Scheduled machine jobs: periodic, 24/7, fixed internal IP, few endpoints."""
    names = ["svc_backup", "svc_etl", "svc_reporting", "svc_monitor", "svc_sync",
             "svc_billing", "svc_crm", "svc_search", "svc_audit", "svc_mail",
             "svc_inventory", "svc_ci"]
    horizon = cfg.n_days * 86400.0
    for i in range(cfg.n_service_accounts):
        name = names[i % len(names)] + ("" if i < len(names) else str(i))
        ip = f"10.20.{rng.integers(0, 8)}.{rng.integers(2, 250)}"
        period = float(rng.choice([1800, 3600, 3600, 7200]))
        eps = rng.choice(SERVICE_ENDPOINTS, size=int(rng.integers(1, 3)), replace=False)
        t = rng.uniform(0, period)
        while t < horizon:
            ts = t + rng.normal(0, 20)  # scheduler jitter
            em.emit(ts, name, ip, "POST", "/login", 200)
            for ep in eps:
                ts += rng.uniform(0.5, 3)
                em.emit(ts, name, ip, "GET", ep, 200)
            t += period


# ---------------------------------------------------------------------------
# Attack campaigns
# ---------------------------------------------------------------------------
def inject_attacks(cfg: GeneratorConfig, rng, users: List[User], em: Emitter,
                   vpn_ips) -> pd.DataFrame:
    """Inject labelled attack campaigns; returns a per-campaign summary table."""
    horizon_days = cfg.n_days
    attack_id = 0
    summary = []
    admins = [u for u in users if u.is_admin]
    regulars = [u for u in users if not u.is_admin]
    invalid_names = ["admin", "root", "test", "administrator", "guest", "oracle",
                     "support", "info", "backup", "user"]

    def lab(kind):
        return dict(label=1, attack_type=kind, attack_id=attack_id)

    for kind, count in cfg.attack_counts.items():
        for _ in range(count):
            day = int(rng.integers(0, horizon_days))
            base = day * 86400.0
            n_before = len(em.rows)
            if kind == "brute_force":
                pool = admins if rng.random() < 0.4 else users
                victim = pool[rng.integers(len(pool))]
                ip = random_public_ip(rng)
                t = base + rng.uniform(0, 24) * 3600
                n_try = int(rng.integers(30, 151))
                gap = rng.uniform(0.5, 6.0)            # tool speed differs per campaign
                for _k in range(n_try):
                    em.emit(t, victim.name, ip, "POST", "/login", 401, **lab(kind))
                    t += rng.exponential(gap) + 0.2
                if rng.random() < 0.3:                 # the guess eventually works
                    em.emit(t, victim.name, ip, "POST", "/login", 200, **lab(kind))
                    emit_session_body(em, rng, t, victim.name, ip, int(rng.integers(5, 16)),
                                      victim.endpoint_pref, 0.3 if victim.is_admin else 0.0,
                                      think_mean=8, **lab(kind))
                target = victim.name
            elif kind == "password_spray":
                ip = random_public_ip(rng)
                t = base + rng.uniform(0, 24) * 3600
                n_targets = int(rng.integers(25, 71))
                pool = [u.name for u in users] + invalid_names
                targets = rng.choice(pool, n_targets, replace=False)
                for name in targets:
                    for _k in range(int(rng.integers(1, 3))):
                        em.emit(t, str(name), ip, "POST", "/login", 401, **lab(kind))
                        t += rng.uniform(20, 90)       # low-and-slow
                target = f"{n_targets} accounts"
            elif kind == "odd_hour_login":
                victim = users[rng.integers(len(users))]
                ip = vpn_ips[rng.integers(len(vpn_ips))] if rng.random() < 0.5 else victim.home_ip
                t = base + rng.uniform(1.0, 5.0) * 3600
                t = emit_login(em, rng, t, victim.name, ip, 0, True, **lab(kind))
                pref = victim.endpoint_pref.copy()
                pref[7:] += 0.25                         # skew towards download/export
                pref /= pref.sum()
                emit_session_body(em, rng, t, victim.name, ip, int(rng.integers(8, 31)),
                                  pref, 0.0, think_mean=15, logout_p=0.2, **lab(kind))
                target = victim.name
            elif kind == "rare_ip_takeover":
                victim = users[rng.integers(len(users))]
                ip = random_public_ip(rng)
                h = victim.work_start + rng.uniform(0, victim.work_len)  # blends in
                t = base + h * 3600
                n_fail = int(rng.integers(0, 3))
                t = emit_login(em, rng, t, victim.name, ip, n_fail, True, **lab(kind))
                pref = victim.endpoint_pref.copy()
                pref[[2, 7, 8]] += 0.2                   # customers / download / export
                pref /= pref.sum()
                emit_session_body(em, rng, t, victim.name, ip, int(rng.integers(10, 41)),
                                  pref, 0.0, think_mean=12, logout_p=0.3, **lab(kind))
                target = victim.name
            elif kind == "privilege_escalation":
                victim = regulars[rng.integers(len(regulars))]
                ip = victim.home_ip if rng.random() < 0.5 else vpn_ips[rng.integers(len(vpn_ips))]
                h = victim.work_start + rng.uniform(0, victim.work_len)
                t = base + h * 3600
                t = emit_login(em, rng, t, victim.name, ip, 0, True, **lab(kind))
                t = emit_session_body(em, rng, t, victim.name, ip, int(rng.integers(2, 6)),
                                      victim.endpoint_pref, 0.0, logout_p=0.0, **lab(kind))
                for _k in range(int(rng.integers(5, 26))):   # probing -> 403s
                    t += rng.exponential(6) + 0.5
                    em.emit(t, victim.name, ip, "GET",
                            ADMIN_ENDPOINTS[rng.integers(len(ADMIN_ENDPOINTS))], 403, **lab(kind))
                if rng.random() < 0.6:                   # misconfiguration exploited
                    t += rng.uniform(5, 60)
                    em.emit(t, victim.name, ip, "PUT", "/admin/roles", 200, **lab(kind))
                    for _k in range(int(rng.integers(3, 12))):
                        t += rng.exponential(10) + 1
                        em.emit(t, victim.name, ip, "GET",
                                ADMIN_ENDPOINTS[rng.integers(len(ADMIN_ENDPOINTS))], 200, **lab(kind))
                target = victim.name
            summary.append(dict(attack_id=attack_id, attack_type=kind, target=target,
                                src_ip=ip, day=day, n_events=len(em.rows) - n_before))
            attack_id += 1
    return pd.DataFrame(summary)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def generate(cfg: GeneratorConfig):
    """Generate the dataset; returns (logs, labels, campaigns) DataFrames."""
    rng = np.random.default_rng(cfg.seed)
    office_ips = [f"203.0.113.{10 + i}" for i in range(cfg.n_office_ips)]   # RFC 5737
    vpn_ips = [f"198.51.100.{20 + i}" for i in range(cfg.n_vpn_ips)]        # RFC 5737
    users = make_users(cfg, rng)
    em = Emitter()
    simulate_normal(cfg, rng, users, em, office_ips, vpn_ips)
    simulate_service_accounts(cfg, rng, em)
    campaigns = inject_attacks(cfg, rng, users, em, vpn_ips)

    df = pd.DataFrame(em.rows, columns=LOG_COLUMNS + ["is_anomaly", "attack_type",
                                                      "attack_id", "benign_tag"])
    horizon = cfg.n_days * 86400.0
    df = df[(df["timestamp"] >= 0) & (df["timestamp"] < horizon)]
    t0 = pd.Timestamp(cfg.start_date)
    df["timestamp"] = t0 + pd.to_timedelta(df["timestamp"].round(3), unit="s")
    df = df.sort_values(["timestamp", "user"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "event_id", np.arange(len(df)))
    logs = df[["event_id"] + LOG_COLUMNS].copy()
    labels = df[["event_id", "is_anomaly", "attack_type", "attack_id", "benign_tag"]].copy()
    # Campaign table: recompute event counts after clipping to the horizon.
    counts = labels[labels.attack_id >= 0].groupby("attack_id").size()
    campaigns["n_events"] = campaigns["attack_id"].map(counts).fillna(0).astype(int)
    campaigns = campaigns[campaigns.n_events > 0].reset_index(drop=True)
    return logs, labels, campaigns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    cfg = GeneratorConfig(seed=args.seed)
    logs, labels, campaigns = generate(cfg)
    os.makedirs(args.out, exist_ok=True)
    logs.to_csv(os.path.join(args.out, "logs.csv.gz"), index=False)
    labels.to_csv(os.path.join(args.out, "labels.csv.gz"), index=False)
    campaigns.to_csv(os.path.join(args.out, "campaigns.csv"), index=False)
    print(f"{len(logs):,} events, {labels.is_anomaly.mean():.2%} anomalous, "
          f"{len(campaigns)} campaigns -> {args.out}/")


if __name__ == "__main__":
    main()
