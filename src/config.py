"""Central configuration for the log anomaly-detection pipeline.

Every tunable used by the generator, feature extractor, detectors and the
evaluation lives here so that a single file documents the experiment and a
single seed makes it reproducible.
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class GeneratorConfig:
    """Parameters of the synthetic authentication/access-log generator."""

    seed: int = 0
    n_days: int = 42                 # length of the simulated observation period
    start_date: str = "2026-03-02"   # a Monday, so weeks align with calendar weeks
    n_users: int = 180               # human accounts
    n_service_accounts: int = 12     # machine accounts that run around the clock
    admin_fraction: float = 0.06     # fraction of human users holding admin rights
    n_office_ips: int = 6            # corporate NAT egress addresses (shared)
    n_vpn_ips: int = 20              # VPN concentrator pool (shared, rotating)

    # --- benign "confounders": rare but legitimate behaviour that looks odd ---
    p_travel_day: float = 0.006      # P(user logs in from a new IP on a given day)
    p_late_night_day: float = 0.004  # P(user does a legit late-night session)
    p_typo: float = 0.04             # P(a login starts with 1-3 mistyped passwords)
    p_forgot_password: float = 0.002 # P(a login is a forgotten-password streak)

    # --- injected attack campaigns (counts over the whole period) ---
    attack_counts: Dict[str, int] = field(default_factory=lambda: {
        "brute_force": 14,          # MITRE T1110.001 password guessing
        "password_spray": 12,       # MITRE T1110.003 password spraying
        "odd_hour_login": 24,       # valid account used at 01:00-04:59
        "rare_ip_takeover": 24,     # valid account used from never-seen IP
        "privilege_escalation": 18, # non-admin probing/using /admin endpoints
    })


@dataclass
class PipelineConfig:
    """Parameters of the detection pipeline and evaluation."""

    seed: int = 0
    # All windows are whole weeks so that every split has the same weekday /
    # weekend mix (a weekend-heavy calibration window skews score quantiles).
    burnin_days: int = 7          # week 1     : only used to warm up 7-day profiles
    fit_days: int = 14            # weeks 2-3  : fit detectors (unlabelled, contaminated)
    calib_days: int = 7           # week 4     : calibrate thresholds / ensemble ranks
                                  # weeks 5-6  : held-out test period
    alert_budget: float = 0.02    # fraction of *calibration* events allowed above threshold
    lof_neighbors: int = 35
    lof_max_train: int = 25000    # LOF is O(n^2) -> subsample the fit set
    iforest_estimators: int = 300
    ae_hidden: tuple = (24, 8, 24)
    ae_max_train: int = 30000
    seeds_for_variance: tuple = (0, 1, 2, 3, 4)
    budget_grid: tuple = (0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)
    lof_k_grid: tuple = (10, 35, 100, 250)
