"""Every figure is drawn from results/metrics.json, never from in-memory state."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap  # noqa: E402

from duralloc.config import FIGURES_DIR  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"dp": "#2a78d6", "fqi": "#eb6834", "myopic": "#1baf7a", "immunize": "#eda100"}
LABEL = {"dp": "DP optimum", "fqi": "Fitted Q (learned)", "myopic": "Myopic", "immunize": "Immunize",
         "static": "Static"}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281", "#0d366b"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 2,
    })


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _map_axes(ax, m):
    lv, sl = np.array(m["levels"]) * 100, np.array(m["slopes"]) * 100
    ax.set_xlabel("level (long-end yield, %)")
    ax.set_ylabel("slope (long − short, %)")
    return lv, sl


def policy_map_figure(metrics: dict):
    """Headline: long-bucket weight over the (level, slope) plane, DP against learned."""
    _style()
    pm = metrics["policymap"]
    maps = pm["maps"]
    t = pm["headline_t"]
    dp, fq = maps["dp"][str(t)], maps["fqi"][str(t)]
    steps = np.round(np.arange(0, 0.71, 0.1), 2)
    cmap = ListedColormap(SEQ_BLUE[:len(steps)])
    norm = BoundaryNorm(np.append(steps - 0.05, steps[-1] + 0.05), cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, m, title in ((axes[0], dp, "DP optimum"), (axes[1], fq, "Fitted Q (learned)")):
        lv, sl = _map_axes(ax, m)
        im = ax.pcolormesh(lv, sl, np.array(m["long_weight"]).T, cmap=cmap, norm=norm, shading="nearest")
        hold = np.array(m["hold"]).T.astype(float)
        if hold.any() and not hold.all():
            ax.contourf(lv, sl, hold, levels=[0.5, 1.5], colors="none", hatches=["////"])
            ax.contour(lv, sl, hold, levels=[0.5], colors=INK, linewidths=1.0)
        ax.set_title(f"{title}: long-bucket weight")
    cb = fig.colorbar(im, ax=axes[:2], ticks=steps, shrink=0.9, pad=0.01)
    cb.set_label("weight in 20y bucket", color=INK2)
    diff = np.array(fq["long_weight"]) - np.array(dp["long_weight"])
    lv, sl = _map_axes(axes[2], dp)
    im2 = axes[2].pcolormesh(lv, sl, diff.T, cmap=DIVERGING, vmin=-0.5, vmax=0.5, shading="nearest")
    agree = pm["agreement"][str(t)]["long_within_step"]
    axes[2].set_title(f"Learned − DP  (within one step: {agree:.0%})")
    fig.colorbar(im2, ax=axes[2], shrink=0.9, pad=0.01).set_label("weight difference", color=INK2)
    w = np.round(np.array(dp["w_prev"]), 2)
    fig.suptitle(f"Allocation to long duration at t={t}, from previous allocation {w.tolist()}, "
                 f"cost {pm['cost_bp']:g} bp.  Hatched: no-trade region (holds current allocation).",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    return _save(fig, "policy_map.png")


def hold_region_figure(metrics: dict):
    """No-trade region with and without costs, DP and learned."""
    _style()
    hr = metrics["policymap"]["hold_regions"]
    costs = sorted(hr["dp"].keys(), key=float)
    fig, axes = plt.subplots(2, len(costs), figsize=(4.6 * len(costs), 8), constrained_layout=True,
                             squeeze=False)
    for j, c in enumerate(costs):
        for i, name in enumerate(("dp", "fqi")):
            m = hr[name][c]
            ax = axes[i, j]
            lv, sl = _map_axes(ax, m)
            ax.pcolormesh(lv, sl, np.array(m["hold"]).T.astype(float),
                          cmap=ListedColormap(["#f0efec", SERIES[name]]), vmin=0, vmax=1,
                          shading="nearest")
            ax.set_title(f"{LABEL[name]}, cost {float(c):g} bp: hold {np.mean(m['hold']):.0%}")
    fig.suptitle("Where the policy holds its current allocation (colored) — previous allocation "
                 f"{np.round(hr['dp'][costs[0]]['w_prev'], 2).tolist()}, t=0",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    return _save(fig, "no_trade_region.png")


def cost_sweep_figure(metrics: dict):
    _style()
    cs = metrics["cost_sweep"]
    rows = cs["rows"]
    x = np.array([r["cost_bp"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.axhline(0, color=AXIS, lw=1)
    ax.grid(True, axis="y")
    for name in ("dp", "fqi", "myopic"):
        est = [r["policies"][name]["advantage_over_static"] for r in rows]
        m = np.array([e["mean"] for e in est])
        lo, hi = np.array([e["ci_low"] for e in est]), np.array([e["ci_high"] for e in est])
        ax.fill_between(x, lo, hi, color=SERIES[name], alpha=0.18, lw=0)
        ax.plot(x, m, color=SERIES[name], marker="o", ms=4, label=LABEL[name])
        be = cs["breakeven_bp"].get(name)
        if be is not None:
            ax.axvline(be, color=SERIES[name], lw=1, ls=":")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, x.max() * 1.15)
    ax.set_xlabel("proportional transaction cost (bp of notional traded)")
    ax.set_ylabel("advantage over static (bp / year)")
    ax.set_title("Where the dynamic advantage dies (95% CI bands; dotted: break-even where reached)")
    ax.legend(loc="upper right")
    return _save(fig, "cost_sweep.png")


def offmodel_figure(metrics: dict):
    _style()
    om = metrics["offmodel"]["worlds"]
    worlds = list(om.keys())
    names = ["dp", "fqi", "myopic"]
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.axhline(0, color=AXIS, lw=1)
    ax.grid(True, axis="y")
    width = 0.22
    for k, name in enumerate(names):
        for i, w in enumerate(worlds):
            e = om[w][name]["advantage_over_static"]
            xi = i + (k - 1) * width
            ax.errorbar(xi, e["mean"], yerr=[[e["mean"] - e["ci_low"]], [e["ci_high"] - e["mean"]]],
                        fmt="o", ms=8, color=SERIES[name], ecolor=SERIES[name], capsize=3,
                        label=LABEL[name] if i == 0 else None)
    ax.set_xticks(range(len(worlds)), [w if w != "train" else "training process" for w in worlds])
    ax.set_ylabel("advantage over static (bp / year)")
    ax.set_title("Trained on two factors, evaluated off-model (95% CI)")
    ax.legend(loc="best")
    return _save(fig, "offmodel.png")


def historical_figure(metrics: dict):
    _style()
    h = metrics["historical"]
    months = np.array(h["series"]["month"][1:], dtype="datetime64[D]")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    ax1.axhline(0, color=AXIS, lw=1)
    ax1.grid(True, axis="y")
    for name in ("dp", "fqi", "myopic"):
        cum = np.array(h["walk"][name]["cum_advantage_bp"])
        ax1.plot(months, cum, color=SERIES[name], lw=1.6, label=LABEL[name])
    ax1.set_ylabel("cumulative advantage over static (bp)")
    ax1.set_title(f"Real Treasury curve {h['period'][0][:4]}–{h['period'][1][:4]}, consecutive 24-month episodes")
    ax1.legend(loc="upper left")
    dist = h["windows_advantage_distribution_bp_per_year"]
    names = ["dp", "fqi", "myopic"]
    for i, name in enumerate(names):
        q = dist[name]["quantiles"]
        ax2.plot([q["p05"], q["p95"]], [i, i], color=SERIES[name], lw=2)
        ax2.plot([q["p25"], q["p75"]], [i, i], color=SERIES[name], lw=8, solid_capstyle="round")
        ax2.plot(q["p50"], i, "o", color=SURFACE, mec=INK, ms=7)
        e = h["walk"][name]["advantage_over_static"]
        ax2.errorbar(e["mean"], i - 0.28, xerr=[[e["mean"] - e["ci_low"]], [e["ci_high"] - e["mean"]]],
                     fmt="D", ms=5, color=INK2, capsize=3)
    ax2.axvline(0, color=AXIS, lw=1)
    ax2.set_yticks(range(len(names)), [LABEL[n] for n in names])
    ax2.set_xlabel("advantage over static (bp / year)")
    ax2.set_title("24-month windows: p5–p95, p25–p75, median;\n◆ walk mean with calendar-year cluster-bootstrap CI",
                  fontsize=10)
    return _save(fig, "historical.png")


def all_figures(metrics: dict) -> list:
    out = []
    for key, fn in (("policymap", policy_map_figure), ("policymap", hold_region_figure),
                    ("cost_sweep", cost_sweep_figure), ("offmodel", offmodel_figure),
                    ("historical", historical_figure)):
        if key in metrics:
            out.append(fn(metrics))
    return out
