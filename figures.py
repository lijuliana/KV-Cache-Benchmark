"""Generate publication-quality figures for the paper from results.tsv.

Outputs:
  figures/pareto.png            — clean Pareto front, log-y, deployment region
                                   highlighted, only Pareto-optimal points labeled
  figures/substrate_sweep.png   — K4/V2 crossover at large substrate; deployment
                                   leaders bold, others as context
  figures/kv_asymmetry.png      — matched-pair K-bits vs V-bits at fixed byte
                                   budget; the 3.2-3.8x asymmetry visualized
  figures/headdim_sweep.png     — bar chart at hd128-large showing INT4
                                   strictly dominates every group-wise variant
  figures/family_pareto.png     — family-coloured Pareto (kept for legacy)
  figures/score_trajectory.png  — research-progress plot (kept for appendix)
"""
import os
import csv
import math
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Consistent style across all figures
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "legend.frameon": False,
    "legend.fontsize": 10,
})

KEEP_GATE = 0.10  # Δbpb threshold for "production-acceptable"

SUBSTRATES = ["small", "medium", "large", "hd64", "hd128", "hd128_large"]
SUBSTRATE_COLORS = {
    "small":       "#9c27b0",
    "medium":      "#1976d2",
    "large":       "#d32f2f",
    "hd64":        "#388e3c",
    "hd128":       "#f57c00",
    "hd128_large": "#c2185b",
    "small_legacy": "#999999",
}

FAMILY_LABELS = {
    "quant":      "INT-N (per-tok)",
    "quant_grp":  "INT-N grouped",
    "quant_asym": "INT-N asymmetric",
    "quant_mix":  "Mixed K/V",
    "evict":      "Sliding window",
    "evict_sink": "StreamingLLM sink+W",
    "evict_topk": "Top-k by ‖K‖",
    "evict_h2o":  "H₂O heavy-hitter",
    "lowrank":    "Low-rank (SVD/RP)",
    "headprune":  "Head pruning",
    "hybrid":     "Hybrid recency-tier",
    "stack":      "Eviction × INT4 stack",
    "identity":   "Identity (BF16)",
}
FAMILY_COLORS = {
    "quant":      "#0d47a1",   # deep blue — main story
    "quant_mix":  "#1976d2",   # also blue (related — main story extended)
    "quant_grp":  "#90caf9",   # light blue (related but worse)
    "quant_asym": "#64b5f6",   # light blue
    "evict":      "#c62828",   # red
    "evict_sink": "#ef5350",
    "evict_topk": "#ff7043",   # orange
    "evict_h2o":  "#ffb300",   # amber — best of eviction
    "lowrank":    "#7b1fa2",   # purple
    "headprune":  "#5d4037",   # brown
    "hybrid":     "#388e3c",   # green
    "stack":      "#00838f",   # teal
    "identity":   "#000000",   # black, marker
}
FAMILY_MARKER = {
    "quant":      "o",
    "quant_mix":  "D",
    "quant_grp":  "s",
    "quant_asym": "P",
    "evict":      "v",
    "evict_sink": "<",
    "evict_topk": ">",
    "evict_h2o":  "X",
    "lowrank":    "^",
    "headprune":  "h",
    "hybrid":     "p",
    "stack":      "*",
    "identity":   "o",
}


def family_of(name):
    n = name.lower()
    if n.startswith("identity"):
        return "identity"
    if n.startswith("stack_"):
        return "stack"
    if n.startswith("hybrid_"):
        return "hybrid"
    if "_asym" in n:
        return "quant_asym"
    if n.startswith("mixed_"):
        return "quant_mix"
    if "_group" in n or re.search(r"int\d+_g\d+", n):
        return "quant_grp"
    if re.match(r"int\d+", n):
        return "quant"
    if n.startswith("sliding_window"):
        return "evict"
    if n.startswith("sink"):
        return "evict_sink"
    if n.startswith("topk_"):
        return "evict_topk"
    if n.startswith("h2o_"):
        return "evict_h2o"
    if n.startswith("svd_") or n.startswith("randproj"):
        return "lowrank"
    if n.startswith("headprune"):
        return "headprune"
    return "identity"


def parse_row_substrate(desc):
    for tag in ("hd128_large", "hd64_large", "hd64", "hd128",
                "small", "medium", "large"):
        if re.search(r"\[" + re.escape(tag) + r"(?:\s|\b|\])", desc):
            return tag
    return "small_legacy"


def parse_compressor_name(desc):
    m = re.match(r"([a-zA-Z0-9_+:]+)", desc)
    return m.group(1) if m else (desc.split()[0] if desc else "")


def short_name(name):
    """Pretty short display label for a compressor."""
    n = name.lower()
    rules = [
        (r"^int(\d+)_sym_per_tok_per_head$", lambda m: f"INT{m.group(1)}"),
        (r"^int(\d+)_group(\d+)$",           lambda m: f"INT{m.group(1)}-g{m.group(2)}"),
        (r"^int(\d+)_asym_per_tok_per_head$", lambda m: f"INT{m.group(1)} asym"),
        (r"^mixed_K(\d+)_V(\d+)$",           lambda m: f"K{m.group(1)}V{m.group(2)}"),
        (r"^hybrid_recent(\d+)_int(\d+)_old$", lambda m: f"R{m.group(1)}+INT{m.group(2)}"),
        (r"^sliding_window_(\d+)$",          lambda m: f"slide W={m.group(1)}"),
        (r"^sink(\d+)_W(\d+)$",              lambda m: f"sink+W={m.group(2)}"),
        (r"^topk_knorm_(\d+)pct$",           lambda m: f"top-k {m.group(1)}%"),
        (r"^h2o_R(\d+)_K(\d+)pct$",          lambda m: f"H₂O R={m.group(1)} K={m.group(2)}%"),
        (r"^svd_r(\d+)$",                    lambda m: f"SVD r={m.group(1)}"),
        (r"^headprune_(\d+)of(\d+)$",        lambda m: f"prune {m.group(1)}/{m.group(2)}"),
        (r"^stack_(.+)\+int4.*$",            lambda m: f"{m.group(1)} × INT4"),
        (r"^identity",                       lambda m: "Identity"),
    ]
    for pat, fn in rules:
        m = re.match(pat, n)
        if m:
            return fn(m)
    return name


def load_results(path="results.tsv"):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                if row["status"] in ("marker", "crash"):
                    continue
                ratio = float(row["compression_ratio"])
                if ratio <= 0:
                    continue
                desc = row["description"]
                name = parse_compressor_name(desc)
                rows.append({
                    "commit": row["commit"],
                    "compression_score": float(row["compression_score"]),
                    "compression_ratio": ratio,
                    "val_bpb_delta": max(0.0, float(row["val_bpb_delta"])),
                    "compressed_bpb": float(row["compressed_bpb"]),
                    "status": row["status"],
                    "description": desc,
                    "substrate": parse_row_substrate(desc),
                    "name": name,
                    "short": short_name(name),
                    "family": family_of(name),
                })
            except (KeyError, ValueError):
                continue
    return rows


def is_pareto(rows):
    """Mark each row 'pareto' if no other row dominates it (lower Δ AND higher ratio)."""
    for r in rows:
        r["pareto"] = True
    for r in rows:
        for o in rows:
            if o is r:
                continue
            # o strictly dominates r if o.Δ <= r.Δ AND o.ratio >= r.ratio
            # (with at least one strict)
            if (o["val_bpb_delta"] <= r["val_bpb_delta"]
                and o["compression_ratio"] >= r["compression_ratio"]
                and (o["val_bpb_delta"] < r["val_bpb_delta"]
                     or o["compression_ratio"] > r["compression_ratio"])):
                r["pareto"] = False
                break
    return rows


# --------------------------------------------------------------------------
# FIG 1 — Clean Pareto front (medium substrate)
# --------------------------------------------------------------------------

def plot_pareto(rows, out_path):
    """Clean Pareto plot. Design choices:
       - Collapse 12 family colours down to 5 broad groups (quantization,
         hybrid-quant, eviction, low-rank, head-pruning) so the legend
         doesn't compete with the data.
       - Draw the empirical Pareto frontier as a stepped black line so
         the reader sees the frontier, not a cloud.
       - Annotate only the *deployment-band* Pareto points (the ones a
         reviewer needs to act on); off-frontier and cliff points are
         shown but unlabelled.
       - Legend goes outside the plot area on the right so it never
         competes with data points.
       - Generous right margin (gridspec) so labels never clip the edge."""
    sub = [r for r in rows if r["substrate"] == "medium"]
    if not sub:
        return
    sub = is_pareto(sub)

    # 5 broad family groups
    GROUP = {
        "quant":      ("Quantization (per-tok)",  "#0d47a1", "o"),
        "quant_mix":  ("Quantization (mixed K/V)","#1976d2", "D"),
        "quant_grp":  ("Quantization (grouped)",  "#90caf9", "s"),
        "quant_asym": ("Quantization (asym)",     "#64b5f6", "P"),
        "evict":      ("Eviction",                "#c62828", "v"),
        "evict_sink": ("Eviction",                "#c62828", "v"),
        "evict_topk": ("Eviction",                "#c62828", "v"),
        "evict_h2o":  ("Eviction (H₂O)",          "#ffb300", "X"),
        "lowrank":    ("Low-rank",                "#7b1fa2", "^"),
        "headprune":  ("Head pruning",            "#5d4037", "h"),
        "hybrid":     ("Hybrid recency-tier",     "#388e3c", "p"),
        "stack":      ("Eviction × INT4",         "#00838f", "*"),
        "identity":   ("Identity",                "#000000", "o"),
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.6),
                           gridspec_kw={"right": 0.74})

    # Background bands
    ax.axhspan(1e-5, KEEP_GATE, facecolor="#c8e6c9", alpha=0.32, zorder=0)
    ax.axhspan(KEEP_GATE, 5.0, facecolor="#ffcdd2", alpha=0.20, zorder=0)
    ax.axhline(KEEP_GATE, color="#d32f2f", ls="--", lw=1.0, alpha=0.7, zorder=1)

    # Pareto frontier as stepped line connecting Pareto-optimal points
    pareto_pts = sorted(
        [r for r in sub if r["pareto"]],
        key=lambda r: r["compression_ratio"],
    )
    if len(pareto_pts) >= 2:
        xs = [r["compression_ratio"] for r in pareto_pts]
        ys = [max(r["val_bpb_delta"], 1.5e-5) for r in pareto_pts]
        ax.step(xs, ys, where="post", color="black", lw=1.2,
                alpha=0.55, zorder=2.5,
                label=None)

    # Plot non-Pareto first (small + faded), then Pareto on top
    seen_groups = set()
    for r in sub:
        grp_label, c, m = GROUP.get(r["family"], ("Other", "#888", "o"))
        is_p = r["pareto"]
        size = 130 if is_p else 38
        alpha = 1.0 if is_p else 0.32
        ec = "black" if is_p else "none"
        lw = 0.6 if is_p else 0
        y = max(r["val_bpb_delta"], 1.5e-5)
        label = grp_label if grp_label not in seen_groups else None
        ax.scatter(r["compression_ratio"], y, s=size, c=c, marker=m,
                   alpha=alpha, edgecolors=ec, linewidths=lw,
                   label=label, zorder=4 if is_p else 2)
        seen_groups.add(grp_label)

    # Annotate ONLY deployment-band Pareto points (Δ < 0.10).
    # That's the set a reviewer needs to act on.
    deployment_labels = {
        "int8_sym_per_tok_per_head":  (8, -2, "left"),
        "int4_sym_per_tok_per_head":  (8, -2, "left"),
        "mixed_K8_V4":                (-8, -14, "right"),
        "mixed_K4_V2":                (8, 4, "left"),
        "int4_group16":               (8, 12, "left"),
        "int4_asym_per_tok_per_head": (-8, 12, "right"),
        "h2o_R64_K90pct":             (-8, 8, "right"),
    }
    for r in sub:
        if not r["pareto"]:
            continue
        if r["val_bpb_delta"] >= KEEP_GATE:
            continue  # cliff Pareto points stay unlabeled
        offset = deployment_labels.get(r["name"], (8, 8, "left"))
        dx, dy, ha = offset
        y = max(r["val_bpb_delta"], 1.5e-5)
        c = GROUP.get(r["family"], ("", "#000", ""))[1]
        ax.annotate(r["short"], (r["compression_ratio"], y),
                    fontsize=10.5, fontweight="bold",
                    xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", color=c,
                    zorder=10)

    # Region labels — top-left corner of each band, away from data
    ax.text(0.018, 0.04, "DEPLOYMENT BAND",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, fontweight="bold", color="#1b5e20", alpha=0.85)
    ax.text(0.018, 0.96, "QUALITY CLIFF",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold", color="#b71c1c", alpha=0.85)
    ax.text(0.018, 0.91, r"$\Delta_{\mathrm{bpb}} > 0.10$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color="#b71c1c", alpha=0.75)

    ax.set_yscale("log")
    ax.set_ylim(1e-5, 3.5)
    ax.set_xlim(0.9, 36)
    ax.set_xlabel("Compression ratio  (uncompressed / compressed bytes)")
    ax.set_ylabel(r"Quality loss  $\Delta_{\mathrm{bpb}}$  (log scale)")
    ax.set_title("KV-cache Pareto front on the medium substrate "
                 "(only deployment-band Pareto points are annotated)")

    # Legend OUTSIDE the plot area on the right
    handles, labels = ax.get_legend_handles_labels()
    # Dedup while preserving order
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l in seen or l is None:
            continue
        h2.append(h); l2.append(l); seen.add(l)
    ax.legend(h2, l2, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, frameon=False, borderaxespad=0)

    ax.grid(True, which="major", alpha=0.20)
    ax.grid(True, which="minor", alpha=0.08)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# FIG 2 — Substrate sweep showing the K4/V2 crossover
# --------------------------------------------------------------------------

def plot_substrate_sweep(rows, out_path):
    """Focused on the K4/V2 crossover at large. Three lines only:
    INT4 (the previous deployment leader), K4V2 (the new leader at D=10),
    INT2 (the F2 'aggressive quant tolerates scale' reference). All
    three have data on small/medium/large; using only these avoids the
    'lots of single-data-point lines' problem."""
    by = {}
    for r in rows:
        if r["substrate"] not in ("small", "medium", "large"):
            continue
        by.setdefault(r["name"], {})[r["substrate"]] = r

    focal = [
        ("int4_sym_per_tok_per_head", "INT4",  "#0d47a1", "-",  "o"),
        ("mixed_K4_V2",               "K4V2",  "#d32f2f", "-",  "D"),
        ("int2_sym_per_tok_per_head", "INT2",  "#7b1fa2", "--", "v"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x_pos = {"small": 0, "medium": 1, "large": 2}
    x_lbl = ["small\n(D=3, ~7M)", "medium\n(D=6, ~26M)", "large\n(D=10, ~139M)"]

    # Background shading: green deployment band, red cliff
    ax.axhspan(0, KEEP_GATE, facecolor="#c8e6c9", alpha=0.32, zorder=0)
    ax.axhspan(KEEP_GATE, 1.0, facecolor="#ffcdd2", alpha=0.18, zorder=0)
    ax.axhline(KEEP_GATE, color="#d32f2f", ls=":", lw=1.0, alpha=0.85,
               zorder=1)

    # Plot the three focal compressors
    for name, label, color, ls, marker in focal:
        if name not in by:
            continue
        d = by[name]
        xs = [x_pos[s] for s in ("small", "medium", "large") if s in d]
        ys = [d[s]["val_bpb_delta"] for s in ("small", "medium", "large") if s in d]
        ax.plot(xs, ys, ls=ls, marker=marker, color=color, lw=2.6,
                ms=11, label=label, zorder=3,
                markeredgecolor="white", markeredgewidth=1.0)

    # End-of-line value labels (right of each marker at large)
    for name, label, color, _, _ in focal:
        if name not in by or "large" not in by[name]:
            continue
        d = by[name]["large"]
        ax.annotate(f"{label}: Δ={d['val_bpb_delta']:.3f}",
                    xy=(2, d['val_bpb_delta']), xytext=(8, 0),
                    textcoords="offset points",
                    fontsize=10, fontweight="bold", color=color,
                    va="center")

    # Highlight the crossover with a single annotation arrow
    if "mixed_K4_V2" in by and "large" in by["mixed_K4_V2"]:
        d = by["mixed_K4_V2"]["large"]
        ax.annotate(
            "Crossover:\nK4V2 displaces INT4\nas the α=20 leader",
            xy=(2, d["val_bpb_delta"]),
            xytext=(1.05, 0.32),
            fontsize=10, fontweight="bold", color="#d32f2f",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#d32f2f", lw=1.3,
                            connectionstyle="arc3,rad=-0.2"),
        )

    # Region labels in the corners
    ax.text(0.018, 0.94, "QUALITY CLIFF  (Δ > 0.10)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold", color="#b71c1c", alpha=0.85)
    ax.text(0.018, 0.04, "DEPLOYMENT BAND  (Δ < 0.10)",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, fontweight="bold", color="#1b5e20", alpha=0.85)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(x_lbl)
    ax.set_xlim(-0.25, 2.85)
    ax.set_ylim(-0.02, 0.72)
    ax.set_ylabel(r"Quality loss  $\Delta_{\mathrm{bpb}}$")
    ax.set_title("Substrate-scale sweep: only K4V2 crosses into the "
                 "deployment band at D=10\n"
                 "(the K4V2 vs INT4 crossover is the headline scaling result)")
    ax.legend(loc="upper right", fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# FIG 3 — K-vs-V asymmetry visualization
# --------------------------------------------------------------------------

def plot_kv_asymmetry(out_path):
    """Matched-pair K-bits vs V-bits at fixed byte budget. The 3.2-3.8x
    consistency is the visual punchline.

    Data is from §4.4 (Table tab:kv-asym in paper); we hardcode it here
    because the pairs span runs and substrates that share a byte budget
    but not a (substrate, byte budget) pair-key in results.tsv.
    """
    # (substrate, ratio_label, K-side label, K-side Δ, V-side label, V-side Δ)
    pairs = [
        ("large\n(D=10, HD=96)",       "5.05×", "K4V2", 0.049, "K2V4", 0.158),
        ("hd128-large\n(D=10, HD=128)", "5.12×", "K4V2", 0.047, "K2V4", 0.172),
        ("large\n(D=10, HD=96)",       "3.10×", "K8V2", 0.047, "K2V8", 0.157),
        ("hd128-large\n(D=10, HD=128)", "3.12×", "K8V2", 0.045, "K2V8", 0.171),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    n = len(pairs)
    x = list(range(n))
    bar_w = 0.36
    k_color = "#0d47a1"   # blue — high-K wins
    v_color = "#c62828"   # red — high-V loses

    k_vals = [p[3] for p in pairs]
    v_vals = [p[5] for p in pairs]
    bars_k = ax.bar([xi - bar_w/2 for xi in x], k_vals, bar_w,
                    color=k_color, label="more bits on K (high-K)")
    bars_v = ax.bar([xi + bar_w/2 for xi in x], v_vals, bar_w,
                    color=v_color, label="more bits on V (high-V)")

    # Numerical annotations on bars
    for bar, val, lab in zip(bars_k, k_vals, [p[2] for p in pairs]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f"{lab}\nΔ={val:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=k_color, fontweight="bold")
    for bar, val, lab in zip(bars_v, v_vals, [p[4] for p in pairs]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f"{lab}\nΔ={val:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=v_color, fontweight="bold")

    # Annotate the "K-bit penalty" multiplier above each pair
    for xi, p in zip(x, pairs):
        mult = p[5] / p[3]
        ax.annotate(f"{mult:.1f}× penalty", xy=(xi, max(p[3], p[5]) + 0.025),
                    ha="center", fontsize=9.5, fontweight="bold",
                    color="#37474f")

    # X-axis labels: substrate + ratio
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p[0]}\n@ {p[1]} budget" for p in pairs],
                       fontsize=9.5)

    ax.set_ylim(0, 0.23)
    ax.set_ylabel(r"Quality loss  $\Delta\mathrm{val\_bpb}$")
    ax.set_title("K-vs-V precision asymmetry is structural — "
                 "spending bits on V costs 3.2–3.8× more Δbpb\n"
                 "than spending bits on K, at every fixed byte budget we tested",
                 fontsize=12)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# FIG 4 — Head_dim group-wise sweep (replace flat-line plot with bars)
# --------------------------------------------------------------------------

def plot_headdim_sweep(rows, out_path):
    """At hd128-large (D=10, HD=128), compare INT4 vs each group-wise
    variant. Vanilla INT4 strictly dominates, shown as a clear bar gap."""
    # Pull from hd128 sweep (HEAD_DIM at D=6) AND hd128_large
    targets = ["int4_sym_per_tok_per_head", "int4_group32",
               "int4_group16", "int4_group8"]
    target_labels = ["INT4 (vanilla)", "INT4-g32", "INT4-g16", "INT4-g8"]

    sweeps = {}  # substrate -> {name: row}
    for r in rows:
        if r["substrate"] in ("hd64", "hd128"):
            sweeps.setdefault(r["substrate"], {})[r["name"]] = r

    # Use both HD=64 and HD=128 sweeps at fixed D=6
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={"width_ratios": [1, 1]})

    for idx, (sub_key, sub_label) in enumerate([("hd64", "HD=64 (D=6)"),
                                                  ("hd128", "HD=128 (D=6)")]):
        ax = axes[idx]
        if sub_key not in sweeps:
            ax.text(0.5, 0.5, f"no data for {sub_key}",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        d = sweeps[sub_key]

        # Get bars data
        names_present = [n for n in targets if n in d]
        labels_present = [target_labels[targets.index(n)] for n in names_present]
        ratios = [d[n]["compression_ratio"] for n in names_present]
        deltas = [d[n]["val_bpb_delta"] for n in names_present]
        scores20 = [d[n]["compression_ratio"] - 20 * max(d[n]["val_bpb_delta"], 0)
                    for n in names_present]

        x = list(range(len(names_present)))
        # Color INT4 (vanilla) distinctively
        colors = ["#0d47a1" if "vanilla" in lbl else "#90caf9"
                  for lbl in labels_present]
        bars = ax.bar(x, scores20, color=colors, edgecolor="black", lw=0.5)

        # Annotate each bar with ratio and Δ
        for bar, r_v, d_v in zip(bars, ratios, deltas):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.05,
                    f"r={r_v:.2f}×\nΔ={d_v:.4f}",
                    ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels_present, rotation=15, ha="right")
        ax.set_ylabel(r"Composite score $S(\alpha\!=\!20)$")
        ax.set_title(sub_label)
        ax.set_ylim(0, max(scores20) * 1.20 if scores20 else 1)
        ax.axhline(max(scores20) if scores20 else 0,
                   color="#0d47a1", ls=":", lw=0.8, alpha=0.5)

    fig.suptitle("Group-wise INT4 quantization is strictly dominated by "
                 "vanilla INT4 at every head_dim we tested",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# FIG 5 (legacy) — Family-coloured Pareto
# --------------------------------------------------------------------------

def plot_family_pareto_legacy(rows, out_path):
    """Kept for completeness; now uses the cleaner styling."""
    plot_pareto(rows, out_path)


# --------------------------------------------------------------------------
# FIG 6 — Score trajectory (research progress, mostly for appendix)
# --------------------------------------------------------------------------

def plot_score_trajectory(rows, out_path):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 4.0))
    xs = list(range(1, len(rows) + 1))
    ys = [r["compression_score"] for r in rows]
    statuses = [r["status"] for r in rows]
    colors = {"keep": "#2e7d32", "discard": "#ef9a9a"}
    bar_colors = [colors.get(s, "#888") for s in statuses]
    ax.bar(xs, ys, color=bar_colors, edgecolor="black", linewidth=0.3)
    # Running max
    running, cur = [], -math.inf
    for s, st in zip(ys, statuses):
        if st == "keep":
            cur = max(cur, s)
        running.append(cur if cur != -math.inf else 0)
    ax.plot(xs, running, c="#0d47a1", lw=2, label="best so far")
    ax.axhline(1.0, c="gray", ls=":", lw=1, label="identity baseline")
    ax.set_xlabel("Experiment index")
    ax.set_ylabel(r"$S(\alpha=10)$")
    ax.set_title("Composite-score trajectory across all experiments "
                 "(green = kept, red = discarded)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    rows = load_results()
    if not rows:
        print("No rows in results.tsv yet; nothing to plot.")
        raise SystemExit(0)

    plot_pareto(rows, os.path.join(FIG_DIR, "pareto.png"))
    plot_pareto(rows, os.path.join(FIG_DIR, "family_pareto.png"))  # legacy alias
    plot_substrate_sweep(rows, os.path.join(FIG_DIR, "substrate_sweep.png"))
    plot_kv_asymmetry(os.path.join(FIG_DIR, "kv_asymmetry.png"))
    plot_headdim_sweep(rows, os.path.join(FIG_DIR, "headdim_sweep.png"))
    plot_score_trajectory(rows, os.path.join(FIG_DIR, "score_trajectory.png"))

    print(f"Wrote {len(rows)} rows -> figures/{{pareto,substrate_sweep,"
          f"kv_asymmetry,headdim_sweep,score_trajectory}}.png")
