"""Relate BEM singular-mode index to source oscillation scale.

The parallel slab orders spatial modes by Fourier wavenumber. A non-flat
geometry does not, so this script checks the right singular vectors directly.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import surface_noise_tools as rp

ROOT = Path(__file__).resolve().parent


def trapezoid_weights(x):
    x = np.asarray(x, float)
    w = np.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def response_svd(h_over_d, panel_h=0.035):
    d = rp.ION_Y
    edges = rp.geom_slotted(4.0, h_over_d * d, 1.0, 1.0)
    xscan = np.linspace(-2 * d, 2 * d, 41)
    wx = trapezoid_weights(xscan)
    solver = rp.Solver2D(edges, panel_h, ion_hint=np.array([0.0, d]))

    active = {"plate_left", "plate_right"}
    mask = np.array([name in active for name in solver.P.name])
    lengths = np.asarray(solver.P.length[mask], float)
    mids = np.asarray(solver.P.mid[mask], float)
    names = np.asarray(solver.P.name, dtype=object)[mask]

    rows = []
    for x0 in xscan:
        _, ky = solver.kernels(np.array([x0, d]), rp.FD)
        rows.append(np.abs(ky[mask]) ** 2)
    H = np.asarray(rows, float)
    B = np.sqrt(wx)[:, None] * H * np.sqrt(lengths)[None, :]
    _, singular, vt = np.linalg.svd(B, full_matrices=False)
    return singular, vt, lengths, mids, names


def sign_changes(vcoeff, lengths, mids, names, threshold=1e-3):
    profile = vcoeff / np.sqrt(lengths)
    total = 0
    for component in ("plate_left", "plate_right"):
        idx = np.where(names == component)[0]
        idx = idx[np.argsort(mids[idx, 0])]
        values = profile[idx]
        keep = np.abs(values) > threshold * np.max(np.abs(values))
        values = values[keep]
        if len(values) > 1:
            total += int(np.sum(np.sign(values[1:]) * np.sign(values[:-1]) < 0))
    return total


def analyze(h_over_d):
    singular, vt, lengths, mids, names = response_svd(h_over_d)
    nzc = np.array([
        sign_changes(vt[j], lengths, mids, names) for j in range(len(singular))
    ])
    rho, pvalue = spearmanr(np.arange(1, 21), nzc[:20])
    return {
        "h_over_d": h_over_d,
        "singular": singular,
        "vt": vt,
        "lengths": lengths,
        "mids": mids,
        "names": names,
        "nzc": nzc,
        "rho20": float(rho),
        "pvalue20": float(pvalue),
    }


far = analyze(13.1428571429)
close = analyze(2.0)
print(f"close-cover Spearman rho, first 20 modes: {close['rho20']:.6f}")
print(f"far-cover Spearman rho, first 20 modes:   {far['rho20']:.6f}")

fig, axes = plt.subplots(2, 1, figsize=(5.8, 5.2), gridspec_kw={"height_ratios": [1.45, 1.0]})
ax = axes[0]
x = close["mids"][:, 0] / rp.ION_Y
for j in (0, 4, 8, 12):
    mode = close["vt"][j] / np.sqrt(close["lengths"])
    mode = mode / max(np.max(np.abs(mode)), 1e-30)
    for component in ("plate_left", "plate_right"):
        idx = np.where(close["names"] == component)[0]
        idx = idx[np.argsort(x[idx])]
        ax.plot(x[idx], mode[idx], label=(fr"$j={j + 1}$" if component == "plate_left" else None))
ax.set_xlabel(r"noisy-surface coordinate $x/d$")
ax.set_ylabel("right singular mode (normalized)")
ax.legend(frameon=False, ncol=2, fontsize=8)
ax.text(0.02, 0.94, r"close cover $h/d=2$", transform=ax.transAxes, ha="left", va="top", fontsize=9)

ax = axes[1]
j = np.arange(1, 21)
ax.plot(j, far["nzc"][:20], "o-", label=r"far cover $h/d=13.14$")
ax.plot(j, close["nzc"][:20], "s-", label=r"close cover $h/d=2$")
ax.set_xlabel("singular-mode index $j$")
ax.set_ylabel(r"sign changes $N_{\mathrm{zc}}$")
ax.set_xticks([1, 5, 10, 15, 20])
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(pad=0.6)
fig.savefig(ROOT / "fig_bem_modes.pdf", bbox_inches="tight")
fig.savefig(ROOT / "fig_bem_modes.png", dpi=220, bbox_inches="tight")
