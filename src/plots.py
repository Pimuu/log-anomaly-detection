"""All figures of the study (matplotlib only, PNG at 150 dpi).

Visual conventions: one fixed colour per detector and per attack type across
every figure, a neutral grey for normal traffic, recessive grid lines, and no
dual y-axes.  Colours follow a colour-vision-deficiency-checked categorical
order (blue, orange, aqua, yellow, magenta, green).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NORMAL = "#a3a29c"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
METHOD_COLORS = dict(zip(["IsolationForest", "LOF", "Autoencoder", "Ensemble-mean",
                          "Ensemble-max", "Rules"], CATEGORICAL))
ATTACKS = ["brute_force", "password_spray", "odd_hour_login", "rare_ip_takeover",
           "privilege_escalation"]
ATTACK_COLORS = dict(zip(ATTACKS, CATEGORICAL))
ATTACK_LABEL = {"brute_force": "Brute force", "password_spray": "Password spray",
                "odd_hour_login": "Odd-hour login", "rare_ip_takeover": "Rare-IP takeover",
                "privilege_escalation": "Privilege escalation", "normal": "Normal"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 8.5, "font.size": 9.5, "lines.linewidth": 2,
})


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def fig_data_overview(res, pcfg, out_dir):
    """Hour-of-day profile per population and daily volume with the splits."""
    logs, labels = res["logs"], res["labels"]
    ts = pd.to_datetime(logs["timestamp"])
    svc = logs["user"].str.startswith("svc_")
    pop = np.where(labels.is_anomaly == 1, "Attack events",
                   np.where(svc, "Service accounts", "Human users (normal)"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    colors = {"Human users (normal)": NORMAL, "Service accounts": CATEGORICAL[0],
              "Attack events": CATEGORICAL[1]}
    for p, c in colors.items():
        h = ts[pop == p].dt.hour.value_counts(normalize=True).reindex(range(24), fill_value=0)
        ax.plot(h.index, 100 * h.values, color=c, label=p, marker="o", ms=3)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Share of the population's events (%)")
    ax.set_title("(a) When does activity happen?", loc="left")
    ax.legend()

    ax = axes[1]
    day = (ts - ts.iloc[0].normalize()).dt.days
    daily = pd.crosstab(day, labels.is_anomaly)
    ax.bar(daily.index, daily[0], color=NORMAL, width=0.8, label="Normal events")
    ax.bar(daily.index, daily.get(1, 0), bottom=daily[0], color=CATEGORICAL[1], width=0.8,
           label="Attack events")
    b, f, c = pcfg.burnin_days, pcfg.fit_days, pcfg.calib_days
    top = daily.sum(axis=1).max()
    for x0, x1, name in [(-0.5, b - 0.5, "burn-in"), (b - 0.5, b + f - 0.5, "fit"),
                         (b + f - 0.5, b + f + c - 0.5, "calib."),
                         (b + f + c - 0.5, day.max() + 0.5, "test")]:
        ax.axvline(x0, color=INK2, lw=0.8, ls=":")
        ax.text((x0 + x1) / 2, top * 1.06, name, ha="center", color=INK2, fontsize=8.5)
    ax.set_ylim(0, top * 1.15)
    ax.set_xlabel("Day of simulation")
    ax.set_ylabel("Events per day")
    ax.set_title("(b) Daily volume and chronological split", loc="left", pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    return _save(fig, out_dir, "fig1_data_overview.png")


def fig_features(res, out_dir):
    """Distributions of six key engineered features per traffic class."""
    feats, labels = res["feats"], res["labels"]
    show = [("ip_fail_5m", "log(1+failed logins from IP, 5 min)"),
            ("ip_users_30m", "log(1+distinct accounts from IP, 30 min)"),
            ("user_hour_rarity", "Hour-of-day surprisal for the account"),
            ("user_ip_age", "log(1+age of user-IP pair, s)"),
            ("user_forbidden_1h", "log(1+HTTP 403 for account, 1 h)"),
            ("user_ep_share_7d", "Share of account's 7-day events on endpoint")]
    classes = ["normal"] + ATTACKS
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    rng = np.random.default_rng(0)
    for ax, (col, title) in zip(axes.ravel(), show):
        data = []
        for k in classes:
            v = feats.loc[labels.attack_type == k, col].to_numpy()
            data.append(v if len(v) < 20000 else rng.choice(v, 20000, replace=False))
        bp = ax.boxplot(data, vert=False, showfliers=False, widths=0.6, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.5),
                        whiskerprops=dict(color=INK2, lw=1), capprops=dict(color=INK2, lw=1))
        for patch, k in zip(bp["boxes"], classes):
            patch.set_facecolor(NORMAL if k == "normal" else ATTACK_COLORS[k])
            patch.set_edgecolor(SURFACE)
            patch.set_alpha(0.9)
        ax.set_yticks(range(1, len(classes) + 1))
        ax.set_yticklabels([ATTACK_LABEL[k] for k in classes])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.grid(axis="y", visible=False)
    for ax in axes[:, 1:].ravel():
        ax.set_yticklabels([])
    fig.suptitle("Engineered features by traffic class (boxes: IQR, whiskers: 1.5 IQR)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, out_dir, "fig2_feature_distributions.png")


def fig_scores(res, methods, out_dir):
    """Anomaly-score distribution (test period) normal vs. anomalous."""
    y = res["labels"]["is_anomaly"].to_numpy()[res["test"]]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for ax, m in zip(axes.ravel(), methods):
        s = res["scores"][m][res["test"]]
        c = METHOD_COLORS[m]
        if m == "Rules":
            k = np.arange(0, 4)
            for off, mask, col, lbl in [(-0.2, y == 0, NORMAL, "Normal"),
                                        (0.2, y == 1, c, "Anomalous")]:
                frac = [(s[mask] == v).mean() for v in k]
                ax.bar(k + off, frac, width=0.38, color=col, label=lbl)
            ax.axvline(0.5, color=INK, ls="--", lw=1.2)
            ax.set_xticks(k)
            ax.set_xlabel("Number of rules fired")
            ax.set_yscale("log")
        else:
            log_x = m in ("LOF", "Autoencoder")      # strongly right-skewed scores
            lo, hi = np.quantile(s, 0.0005), s.max()
            bins = np.geomspace(max(lo, 1e-4), hi, 60) if log_x else np.linspace(lo, hi, 60)
            ax.hist(s[y == 0], bins=bins, density=True, color=NORMAL, alpha=0.7, label="Normal")
            ax.hist(s[y == 1], bins=bins, density=True, color=c, alpha=0.25)
            ax.hist(s[y == 1], bins=bins, density=True, histtype="step", color=c, lw=1.6,
                    label="Anomalous")
            ax.axvline(res["thresholds"][m], color=INK, ls="--", lw=1.2)
            ax.set_yscale("log")
            if log_x:
                ax.set_xscale("log")
            ax.set_xlabel("Anomaly score" + (" (log scale)" if log_x else ""))
        ax.set_title(m, loc="left")
        lo_y, hi_y = ax.get_ylim()
        ax.set_ylim(lo_y, hi_y * 60)          # headroom (log axis) for the legend
        ax.legend(loc="upper left", ncol=2, fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Density / share (log scale)")
    fig.suptitle("Test-period anomaly scores; dashed line = 2 % alert-budget threshold",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, out_dir, "fig3_score_distributions.png")


def fig_timeline(res, method, out_dir):
    """Anomaly score of every test event over time + hourly alert counts."""
    test = res["test"]
    ts = pd.to_datetime(res["logs"]["timestamp"])[test].to_numpy()
    lab = res["labels"][test]
    s = res["scores"][method][test]
    thr = res["thresholds"][method]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    ax = axes[0]
    nm = (lab.is_anomaly == 0).to_numpy()
    ax.scatter(ts[nm], s[nm], s=2, color=NORMAL, alpha=0.35, lw=0, label="Normal", rasterized=True)
    for k in ATTACKS:
        mk = (lab.attack_type == k).to_numpy()
        ax.scatter(ts[mk], s[mk], s=10, color=ATTACK_COLORS[k], lw=0, label=ATTACK_LABEL[k])
    ax.axhline(thr, color=INK, ls="--", lw=1.2, label="Alert threshold")
    ax.set_ylabel(f"{method} score")
    ax.set_title(f"(a) Detected anomalies over the test period - {method}", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), markerscale=2, fontsize=8)
    ax = axes[1]
    pred = s >= thr
    df = pd.DataFrame({"t": ts, "tp": pred & ~nm, "fp": pred & nm}).set_index("t")
    hourly = df.resample("1h").sum()
    w = 1 / 24 * 0.9
    ax.bar(hourly.index, hourly.tp, width=w, color=INK2, label="True alerts")
    ax.bar(hourly.index, hourly.fp, width=w, bottom=hourly.tp, color="#e34948",
           label="False alerts")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("Alerts per hour")
    ax.set_title("(b) Hourly alert volume at the operating threshold", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, out_dir, "fig4_timeline.png")


def fig_confusion(res, methods, out_dir):
    y = res["labels"]["is_anomaly"].to_numpy()[res["test"]]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    for ax, m in zip(axes.ravel(), methods):
        pred = res["scores"][m][res["test"]] >= res["thresholds"][m]
        cm = confusion_matrix(y, pred, labels=[0, 1])
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({norm[i, j]:.1%} of row)", ha="center",
                        va="center", fontsize=9, color="white" if norm[i, j] > 0.6 else INK)
        ax.set_xticks([0, 1], ["Pred. normal", "Pred. anomaly"])
        ax.set_yticks([0, 1], ["True normal", "True anomaly"])
        ax.grid(False)
        ax.set_title(m, loc="left")
    fig.suptitle("Confusion matrices on the test period at the 2 % alert-budget threshold",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, out_dir, "fig5_confusion_matrices.png")


def fig_roc_pr(res, methods, out_dir):
    y = res["labels"]["is_anomaly"].to_numpy()[res["test"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for m in methods:
        s = res["scores"][m][res["test"]]
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        axes[0].plot(fpr, tpr, color=METHOD_COLORS[m], label=m)
        axes[1].plot(rec, prec, color=METHOD_COLORS[m], label=m)
    axes[0].plot([0, 1], [0, 1], color=INK2, ls=":", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_xlim(1e-4, 1)
    axes[0].set_xlabel("False-positive rate (log scale)")
    axes[0].set_ylabel("True-positive rate")
    axes[0].set_title("(a) ROC", loc="left")
    axes[1].axhline(y.mean(), color=INK2, ls=":", lw=1, label="Chance (prevalence)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("(b) Precision-recall", loc="left")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return _save(fig, out_dir, "fig6_roc_pr.png")


def fig_by_type(per_type, methods, out_dir):
    """Heatmaps: per-type ROC-AUC and campaign detection (mean over seeds)."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
    for ax, (col, title, cmap) in zip(axes, [
            ("type_auc", "(a) ROC-AUC per attack type", "Blues"),
            ("campaign_detection", "(b) Campaigns with >= 1 alert (pooled)", "Blues")]):
        if col == "campaign_detection":   # pool campaigns over seeds
            d = per_type.assign(det=per_type.campaign_detection * per_type.n_campaigns)
            g = d.groupby(["method", "attack_type"])[["det", "n_campaigns"]].sum()
            piv = (g.det / g.n_campaigns).unstack()
        else:
            piv = per_type.pivot_table(index="method", columns="attack_type", values=col,
                                       aggfunc="mean")
        piv = piv.reindex(index=methods, columns=ATTACKS)
        ax.imshow(piv.to_numpy(), cmap=cmap, vmin=0.0 if col != "type_auc" else 0.5, vmax=1)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.iat[i, j]
                dark = (v - (0.5 if col == "type_auc" else 0)) / (0.5 if col == "type_auc" else 1) > 0.6
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if dark else INK)
        ax.set_xticks(range(len(ATTACKS)), [ATTACK_LABEL[a].replace(" ", "\n") for a in ATTACKS],
                      fontsize=8)
        ax.set_yticks(range(len(methods)), methods)
        ax.grid(False)
        ax.set_title(title, loc="left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig7_per_attack_type.png")


def fig_sensitivity(budget, abl, lofk, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax = axes[0]
    g = budget.groupby(["method", "budget"])["f1"].agg(["mean", "std"]).reset_index()
    for m, d in g.groupby("method", sort=False):
        ax.errorbar(100 * d.budget, d["mean"], yerr=d["std"], color=METHOD_COLORS[m], label=m,
                    marker="o", ms=4, capsize=2, lw=1.6)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 5, 10], ["0.25", "0.5", "1", "2", "5", "10"])
    ax.set_xlabel("Alert budget (% of calibration events)")
    ax.set_ylabel("Test F1 (mean +/- sd, 5 seeds)")
    ax.set_title("(a) Operating-point sensitivity", loc="left")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    base = abl[abl.removed_group == "none"].set_index("seed")["avg_precision"]
    d = abl[abl.removed_group != "none"].copy()
    d["delta"] = d["avg_precision"] - d["seed"].map(base)
    g = d.groupby("removed_group")["delta"].agg(["mean", "std"]).sort_values("mean")
    ax.barh(g.index, g["mean"], xerr=g["std"], color=METHOD_COLORS["IsolationForest"],
            height=0.6, capsize=3, error_kw=dict(ecolor=INK2, lw=1))
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Change in average precision vs. all features")
    ax.set_title("(b) Isolation Forest: drop one feature group", loc="left")
    ax.grid(axis="y", visible=False)

    ax = axes[2]
    g = lofk.groupby("k").mean(numeric_only=True)
    for a in ATTACKS:
        ax.plot(g.index, g[f"auc_{a}"], marker="o", ms=4, color=ATTACK_COLORS[a],
                label=f"AUC: {ATTACK_LABEL[a]}")
    ax.plot(g.index, g["avg_precision"], marker="s", ms=4, color=INK, ls="--",
            label="Average precision (all)")
    ax.set_xscale("log")
    ax.set_xticks(list(g.index), [str(k) for k in g.index])
    ax.set_xlabel("LOF neighbourhood size k")
    ax.set_ylabel("Score (mean of 5 seeds)")
    ax.set_title("(c) LOF: the best k differs by attack type", loc="left")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    return _save(fig, out_dir, "fig8_sensitivity_ablation.png")
