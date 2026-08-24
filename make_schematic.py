from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent

h = 1.0
d = 0.38
xs = -0.95
xi = 0.55
fig, axs = plt.subplots(1, 3, figsize=(8.7, 3.0), gridspec_kw={"width_ratios": [1.2, 1.0, 0.9]})

# Physical billiard strip.
ax = axs[0]
ax.axhline(0, lw=2)
ax.axhline(h, lw=2)
ax.plot([xi], [d], "o", ms=7)
ax.plot([xs], [0], "s", ms=6)
ax.text(xi + 0.05, d + 0.03, "ion", fontsize=9)
ax.text(xs - 0.03, -0.08, "noisy point", ha="center", va="top", fontsize=8)
ax.text(-1.28, -0.02, r"$y=0$", va="top", fontsize=9)
ax.text(-1.28, h + 0.02, r"$y=h$", va="bottom", fontsize=9)
ax.annotate("", xy=(1.06, d), xytext=(1.06, 0), arrowprops=dict(arrowstyle="<->", lw=0.9))
ax.text(1.11, d / 2, r"$d$", va="center", fontsize=9)
ax.annotate("", xy=(1.32, h), xytext=(1.32, 0), arrowprops=dict(arrowstyle="<->", lw=0.9))
ax.text(1.37, h / 2, r"$h$", va="center", fontsize=9)
ax.plot([xs, xi], [0, d], lw=1.7, label="direct")

Y1 = 2 * h - d
xb1 = xs + (xi - xs) * (h / Y1)
ax.plot([xs, xb1, xi], [0, h, d], lw=1.7, label="1 bounce")

Y2 = 2 * h + d
xb2a = xs + (xi - xs) * (h / Y2)
xb2b = xs + (xi - xs) * (2 * h / Y2)
ax.plot([xs, xb2a, xb2b, xi], [0, h, 0, d], lw=1.7, label="2 bounces")
ax.scatter([xb1, xb2a], [h, h], s=15)
ax.scatter([xb2b], [0], s=15)
ax.text(0.02, 1.04, "billiard paths", transform=ax.transAxes, fontsize=9, weight="bold")
ax.set_xlim(-1.34, 1.53)
ax.set_ylim(-0.14, 1.14)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(frameon=False, fontsize=7.2, loc="upper center", bbox_to_anchor=(0.60, 0.98))

# Unfolded image copies.
ax = axs[1]
for y in [0, h, 2 * h, 3 * h]:
    ax.axhline(y, lw=0.75, ls="--", alpha=0.7)
ax.plot([0], [0], "s", ms=6)
ax.text(0, -0.17, "source", ha="center", fontsize=8)
alphas = [d, 2 * h - d, 2 * h + d]
labels = [r"$d$", r"$2h-d$", r"$2h+d$"]
xt = [0.72, 0.94, 1.16]
for a, label, x in zip(alphas, labels, xt):
    ax.plot([x], [a], "o", ms=6)
    ax.plot([0, x], [0, a], lw=1.5)
    ax.annotate("", xy=(x + 0.12, a), xytext=(x + 0.12, 0), arrowprops=dict(arrowstyle="<->", lw=0.75))
    ax.text(x + 0.16, a / 2, label, rotation=90, va="center", fontsize=7.5)
ax.text(0.02, 1.04, "unfolding", transform=ax.transAxes, fontsize=9, weight="bold")
ax.text(0.03, 0.93, r"$\alpha_n=|d-2nh|$", transform=ax.transAxes, fontsize=9)
ax.set_xlim(-0.15, 1.55)
ax.set_ylim(-0.14, 2.65)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# Return-depth spectrum.
ax = axs[2]
N = 3
ns = np.arange(-N, N + 1)
a = np.abs(d - 2 * ns * h)
order = np.argsort(a)
a = a[order]
for aa in a:
    height = np.exp(-0.42 * aa)
    ax.vlines(aa, 0, height, lw=1.7)
    ax.plot([aa], [height], "o", ms=4)
for aa, label in zip([d, 2 * h - d, 2 * h + d], [r"$d$", r"$2h-d$", r"$2h+d$"]):
    ax.text(aa, 0.04, label, rotation=90, ha="center", va="bottom", fontsize=7.3)
ax.set_xlabel(r"unfolded depth $a$", fontsize=8.5)
ax.set_ylabel(r"weight $e^{-ka}$", fontsize=8.5)
ax.set_yticks([])
ax.tick_params(axis="x", labelsize=7)
ax.text(0.02, 1.04, "return-depth spectrum", transform=ax.transAxes, fontsize=9, weight="bold")
ax.text(0.05, 0.91, r"$\mu_h=\sum_n\delta_{\alpha_n}$", transform=ax.transAxes, fontsize=8.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.55, w_pad=0.7)
fig.savefig(root / "fig_unfolding_schematic.pdf", bbox_inches="tight")
fig.savefig(root / "fig_unfolding_schematic.png", dpi=240, bbox_inches="tight")
