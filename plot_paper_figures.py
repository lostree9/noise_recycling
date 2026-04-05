#!/usr/bin/env python3
"""
plot_paper_figures.py — Generate all figures for the PRA paper.

Produces exactly 5 figures:
  fig1_geometry_gallery.pdf   — Cross-section sketches (Fig 1)
  fig2_slot_mechanism.pdf     — Slot kernel enhancement (Fig 2)
  fig3_twoplate.pdf           — Two-plate results (Fig 3)
  fig4_blade.pdf              — Blade results (Fig 4)
  fig5_stylus.pdf             — Stylus results (Fig 5)
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc, Circle, Wedge
from matplotlib.gridspec import GridSpec
from collections import defaultdict

# ── PRA-compatible style ────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
    'text.usetex': False,
})

# PRA single column = 3.375 in, double column = 7.0 in
COL_W = 3.375
DCOL_W = 7.0

C_SLOT = '#2563eb'
C_TWOPLATE = '#dc2626'
C_BLADE = '#059669'
C_STYLUS = '#7c3aed'
C_GRAY = '#94a3b8'


def group_by_type(results):
    groups = defaultdict(list)
    for r in results:
        gtype = r.get('params', {}).get('type', 'unknown')
        if not r.get('params', {}).get('_convergence', False):
            groups[gtype].append(r)
    return dict(groups)


def savefig(fig, out_dir, name):
    for ext in ['.pdf', '.png']:
        fig.savefig(os.path.join(out_dir, name + ext), bbox_inches='tight')
    print(f"  {name}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Fig 1: Geometry gallery
# ═══════════════════════════════════════════════════════════════════
def fig1_geometry_gallery(results, out_dir):
    # Use gridspec with equal width ratios; enough height for blade's square aspect
    fig, axes = plt.subplots(1, 4, figsize=(DCOL_W, 2.4),
                             gridspec_kw={'width_ratios': [1, 1, 1, 1]})

    label_kw = dict(fontsize=8, fontweight='bold', va='top', ha='left',
                    bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))

    # (a) Surface slot
    ax = axes[0]
    W, g, t = 50, 6, 6
    ax.add_patch(Rectangle((-W - g/2, 0), W, t, fc='#bfdbfe', ec='#1e40af', lw=0.8))
    ax.add_patch(Rectangle((g/2, 0), W, t, fc='#bfdbfe', ec='#1e40af', lw=0.8))
    ax.plot([-g/2, g/2], [0, 0], '-', color=C_GRAY, lw=1)
    ax.plot(0, 40, 'r*', ms=7, zorder=10)
    ax.annotate('', xy=(g/2, -4), xytext=(-g/2, -4),
                arrowprops=dict(arrowstyle='<->', color='#64748b', lw=0.5))
    ax.text(0, -7, '$g$', ha='center', fontsize=6, color='#64748b')
    # Dimension annotations
    ax.set_xlim(-62, 62); ax.set_ylim(-12, 52)
    ax.set_aspect('equal')
    ax.text(-58, 49, '(a) Surface slot', **label_kw)

    # (b) Two-plate
    ax = axes[1]
    h_draw = 40
    ax.add_patch(Rectangle((-50, 0), 44, 2, fc='#fecaca', ec='#991b1b', lw=0.8))
    ax.add_patch(Rectangle((6, 0), 44, 2, fc='#fecaca', ec='#991b1b', lw=0.8))
    ax.add_patch(Rectangle((-50, h_draw-2), 44, 2, fc='#fecaca', ec='#991b1b', lw=0.8))
    ax.add_patch(Rectangle((6, h_draw-2), 44, 2, fc='#fecaca', ec='#991b1b', lw=0.8))
    ax.plot(0, h_draw/2, 'r*', ms=7, zorder=10)
    # h annotation
    ax.annotate('', xy=(52, 0), xytext=(52, h_draw),
                arrowprops=dict(arrowstyle='<->', color='#64748b', lw=0.5))
    ax.text(54, h_draw/2, '$h$', fontsize=6, color='#64748b', va='center')
    ax.set_xlim(-55, 62); ax.set_ylim(-12, 52)
    ax.set_aspect('equal')
    ax.text(-51, 49, '(b) Two-plate', **label_kw)

    # (c) Blade — scale down so it fits the same vertical range
    ax = axes[2]
    R, bl = 14, 22
    theta = 45
    for angle in [theta, 180-theta, 180+theta, 360-theta]:
        rad = np.radians(angle)
        cx, cy = R * np.cos(rad), R * np.sin(rad)
        dx, dy = bl * np.cos(rad), bl * np.sin(rad)
        ax.plot([cx, cx+dx], [cy, cy+dy], '-', color=C_BLADE, lw=3,
                solid_capstyle='butt')
    ax.plot(0, 0, 'r*', ms=7, zorder=10)
    # Angle arc
    arc = Arc((0, 0), 16, 16, angle=0, theta1=0, theta2=theta,
              color='#64748b', lw=0.5)
    ax.add_patch(arc)
    ax.text(10, 3, r'$\theta$', fontsize=6, color='#64748b')
    ax.set_xlim(-42, 42); ax.set_ylim(-42, 42)
    ax.set_aspect('equal')
    ax.text(-38, 39, '(c) Blade', **label_kw)

    # (d) Stylus
    ax = axes[3]
    tip_r, base_r = 4, 14
    verts_r = [0, tip_r, base_r, base_r, 0]
    verts_z = [tip_r, 0, -18, -18, -18]
    ax.fill(verts_r, verts_z, color='#ddd6fe', ec='#5b21b6', lw=0.8)
    ax.fill([-x for x in verts_r], verts_z, color='#ddd6fe', ec='#5b21b6', lw=0.8)
    ax.plot([-42, -14], [32, 32], '-', color='#5b21b6', lw=2)
    ax.plot([14, 42], [32, 32], '-', color='#5b21b6', lw=2)
    ax.plot(0, 18, 'r*', ms=7, zorder=10)
    ax.axvline(0, color='gray', ls=':', lw=0.3, ymin=0.1, ymax=0.9)
    ax.set_xlim(-48, 48); ax.set_ylim(-24, 42)
    ax.set_aspect('equal')
    ax.text(-44, 39, '(d) Stylus', **label_kw)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.15)
    savefig(fig, out_dir, 'fig1_geometry_gallery')


# ═══════════════════════════════════════════════════════════════════
# Fig 2: Slot mechanism (uses existing kernel data)
# ═══════════════════════════════════════════════════════════════════
def fig2_slot_mechanism(results, out_dir):
    groups = group_by_type(results)
    slot = groups.get('surface_slot', [])
    if not slot:
        print("  No slot data for fig2")
        return

    # Find flush, mid, deep
    by_name = {r['params']['name']: r for r in slot}
    flush = by_name.get('slot_aspect_0.00')
    mid = by_name.get('slot_aspect_1.00')
    deep = by_name.get('slot_aspect_4.00')

    if not all([flush, mid, deep]):
        print("  Missing slot results for fig2")
        return

    g = flush['params']['g_gap']

    fig, axes = plt.subplots(1, 3, figsize=(DCOL_W, 2.2))

    # (a) |K_y|^2 on top surface
    ax = axes[0]
    for data, label, color in [
        (flush, '$t/g = 0$', 'black'),
        (mid, '$t/g = 1$', C_SLOT),
        (deep, '$t/g = 4$', C_TWOPLATE),
    ]:
        pts = np.array(data['midpoints'])
        Ky = np.array(data['K_y'])
        t_elec = data['params'].get('t_electrode', 0)
        if t_elec < 0.5:
            mask = np.ones(len(pts), dtype=bool)
        else:
            tol = max(g * 0.1, 0.5)
            mask = (np.abs(pts[:, 1] - t_elec) < tol) & (np.abs(pts[:, 0]) > g/2 - tol)
        x_top = pts[mask, 0]
        Ky_top = Ky[mask]
        idx = np.argsort(x_top)
        ax.plot(x_top[idx], Ky_top[idx]**2, '-', label=label, color=color, lw=1)

    ax.axvline(-g/2, color='gray', ls=':', alpha=0.4, lw=0.5)
    ax.axvline(g/2, color='gray', ls=':', alpha=0.4, lw=0.5)
    ax.set_xlabel(r'$x$ [$\mu$m]')
    ax.set_ylabel(r'$|K_y|^2$')
    ax.set_title('(a) Kernel on top surface')
    ax.set_xlim(-80, 80)
    ax.legend(fontsize=6, frameon=False)

    # (b) Enhancement ratio
    ax = axes[1]
    pts_f = np.array(flush['midpoints'])
    Ky_f = np.array(flush['K_y'])
    pts_d = np.array(deep['midpoints'])
    Ky_d = np.array(deep['K_y'])
    t_d = deep['params'].get('t_electrode', 0)
    tol = max(g * 0.1, 0.5)

    idx_f = np.argsort(pts_f[:, 0])
    x_f = pts_f[idx_f, 0]
    Ky2_f = Ky_f[idx_f]**2

    mask_d = (np.abs(pts_d[:, 1] - t_d) < tol) & (np.abs(pts_d[:, 0]) > g/2 - tol)
    x_d = pts_d[mask_d, 0]
    Ky2_d = Ky_d[mask_d]**2
    idx_d = np.argsort(x_d)
    x_d = x_d[idx_d]
    Ky2_d = Ky2_d[idx_d]

    Ky2_f_interp = np.interp(x_d, x_f, Ky2_f)
    ratio = np.where(Ky2_f_interp > 0, Ky2_d / Ky2_f_interp, 1.0)

    ax.plot(x_d, ratio, color=C_TWOPLATE, lw=1)
    ax.axhline(1.0, color='black', ls='--', alpha=0.3, lw=0.5)
    ax.axvline(-g/2, color='gray', ls=':', alpha=0.4, lw=0.5)
    ax.axvline(g/2, color='gray', ls=':', alpha=0.4, lw=0.5)
    ax.set_xlabel(r'$x$ [$\mu$m]')
    ax.set_ylabel(r'$|K_y|^2_{t/g=4} / |K_y|^2_{t/g=0}$')
    ax.set_title('(b) Enhancement ratio')
    ax.set_xlim(-80, 80)

    # (c) Cumulative G_y from gap edge
    ax = axes[2]
    for data, label, color in [
        (flush, '$t/g = 0$', 'black'),
        (mid, '$t/g = 1$', C_SLOT),
        (deep, '$t/g = 4$', C_TWOPLATE),
    ]:
        pts = np.array(data['midpoints'])
        Ky = np.array(data['K_y'])
        dl = np.array(data['dl'])
        t_elec = data['params'].get('t_electrode', 0)
        if t_elec < 0.5:
            mask = np.ones(len(pts), dtype=bool)
        else:
            mask = (np.abs(pts[:, 1] - t_elec) < tol) & (np.abs(pts[:, 0]) > g/2 - tol)
        x_top = np.abs(pts[mask, 0])
        Ky_top = Ky[mask]
        dl_top = dl[mask]
        dist = x_top - g/2
        idx = np.argsort(dist)
        cumul = np.cumsum(Ky_top[idx]**2 * dl_top[idx])
        if cumul[-1] > 0:
            ax.plot(dist[idx], cumul / cumul[-1], '-', label=label, color=color, lw=1)

    ax.set_xlabel(r'Distance from gap edge [$\mu$m]')
    ax.set_ylabel(r'Cumulative $\mathcal{G}_y^{(\mathrm{top})}$')
    ax.set_title('(c) Spatial extent')
    ax.set_xlim(0, 120)
    ax.legend(fontsize=6, frameon=False)

    fig.tight_layout(pad=0.5)
    savefig(fig, out_dir, 'fig2_slot_mechanism')


# ═══════════════════════════════════════════════════════════════════
# Fig 3: Two-plate
# ═══════════════════════════════════════════════════════════════════
def fig3_twoplate(results, out_dir):
    groups = group_by_type(results)
    tp = groups.get('two_plate', [])
    if not tp:
        print("  No two-plate data for fig3")
        return

    h_sweep = sorted([r for r in tp
                      if 'twoplate_h_' in r['params'].get('name', '')],
                     key=lambda r: r['params']['h_separation'])
    g_sweep = sorted([r for r in tp
                      if 'twoplate_g_' in r['params'].get('name', '')],
                     key=lambda r: r['params']['g_gap'])

    fig, axes = plt.subplots(1, 3, figsize=(DCOL_W, 2.2))

    # (a) G_y vs h
    ax = axes[0]
    if h_sweep:
        h_vals = [r['params']['h_separation'] for r in h_sweep]
        gy = [r['G_y'] for r in h_sweep]
        gx = [r['G_x'] for r in h_sweep]
        ax.semilogy(h_vals, gy, 'o-', color=C_TWOPLATE, ms=4, label=r'$\mathcal{G}_y$')
        ax.semilogy(h_vals, gx, 's--', color=C_GRAY, ms=3, label=r'$\mathcal{G}_x$')
        ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel(r'$h$ [$\mu$m]'); ax.set_ylabel(r'$\mathcal{G}_j$')
    ax.set_title(r'(a) $\mathcal{G}_j$ vs separation')

    # (b) G_y vs g
    ax = axes[1]
    if g_sweep:
        g_vals = [r['params']['g_gap'] for r in g_sweep]
        gy = [r['G_y'] for r in g_sweep]
        ax.semilogy(g_vals, gy, 'o-', color=C_TWOPLATE, ms=4)
    ax.set_xlabel(r'$g$ [$\mu$m]'); ax.set_ylabel(r'$\mathcal{G}_y$')
    ax.set_title(r'(b) $\mathcal{G}_y$ vs gap width')

    # (c) Anisotropy
    ax = axes[2]
    if h_sweep:
        h_vals = [r['params']['h_separation'] for r in h_sweep]
        ratio = [r['G_y'] / max(r['G_x'], 1e-30) for r in h_sweep]
        ax.plot(h_vals, ratio, 'o-', color=C_TWOPLATE, ms=4)
        ax.axhline(1, ls='--', color='gray', lw=0.5)
    ax.set_xlabel(r'$h$ [$\mu$m]')
    ax.set_ylabel(r'$\mathcal{G}_y / \mathcal{G}_x$')
    ax.set_title('(c) Anisotropy')

    fig.tight_layout(pad=0.5)
    savefig(fig, out_dir, 'fig3_twoplate')


# ═══════════════════════════════════════════════════════════════════
# Fig 4: Blade
# ═══════════════════════════════════════════════════════════════════
def fig4_blade(results, out_dir):
    groups = group_by_type(results)
    blade = groups.get('blade', [])
    if not blade:
        print("  No blade data for fig4")
        return

    theta_sweep = sorted([r for r in blade
                          if 'blade_theta_' in r['params'].get('name', '')],
                         key=lambda r: r['params']['blade_angle'])
    R_sweep = sorted([r for r in blade
                      if 'blade_R_' in r['params'].get('name', '')],
                     key=lambda r: r['params']['blade_separation'])
    bt_sweep = sorted([r for r in blade
                       if 'blade_bt_' in r['params'].get('name', '')],
                      key=lambda r: r['params']['blade_thickness'])

    fig, axes = plt.subplots(1, 3, figsize=(DCOL_W, 2.2))

    # (a) Angle sweep
    ax = axes[0]
    if theta_sweep:
        thetas = [r['params']['blade_angle'] for r in theta_sweep]
        gy = [r['G_y'] for r in theta_sweep]
        gx = [r['G_x'] for r in theta_sweep]
        ax.plot(thetas, gy, 'o-', color=C_BLADE, ms=4, label=r'$\mathcal{G}_y$')
        ax.plot(thetas, gx, 's--', color=C_GRAY, ms=3, label=r'$\mathcal{G}_x$')
        ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel(r'$\theta$ [deg]'); ax.set_ylabel(r'$\mathcal{G}_j$')
    ax.set_title(r'(a) Blade angle')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))

    # (b) R sweep (log-log)
    ax = axes[1]
    if R_sweep:
        Rs = np.array([r['params']['blade_separation'] for r in R_sweep])
        gy = np.array([r['G_y'] for r in R_sweep])
        ax.loglog(Rs, gy, 'o-', color=C_BLADE, ms=4, label=r'$\mathcal{G}_y$')
        mid = len(gy) // 2
        scale = gy[mid] * Rs[mid]**3
        R_ref = np.linspace(Rs.min(), Rs.max(), 50)
        ax.loglog(R_ref, scale * R_ref**(-3), ':', color='gray', lw=0.8,
                  label=r'$\propto R^{-3}$')
        ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel(r'$R$ [$\mu$m]'); ax.set_ylabel(r'$\mathcal{G}_y$')
    ax.set_title(r'(b) Distance scaling')

    # (c) Thickness sweep
    ax = axes[2]
    if bt_sweep:
        bts = [r['params']['blade_thickness'] for r in bt_sweep]
        gy = [r['G_y'] for r in bt_sweep]
        ax.plot(bts, gy, 'o-', color=C_BLADE, ms=4)
    ax.set_xlabel(r'$b_t$ [$\mu$m]'); ax.set_ylabel(r'$\mathcal{G}_y$')
    ax.set_title(r'(c) Blade thickness')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))

    fig.tight_layout(pad=0.5)
    savefig(fig, out_dir, 'fig4_blade')


# ═══════════════════════════════════════════════════════════════════
# Fig 5: Stylus
# ═══════════════════════════════════════════════════════════════════
def fig5_stylus(results, out_dir):
    groups = group_by_type(results)
    stylus = groups.get('stylus', [])
    if not stylus:
        print("  No stylus data for fig5")
        return

    rtip_sweep = sorted([r for r in stylus
                         if 'stylus_rtip_' in r['params'].get('name', '')],
                        key=lambda r: r['params']['r_tip'])
    zion_sweep = sorted([r for r in stylus
                         if 'stylus_zion_' in r['params'].get('name', '')],
                        key=lambda r: r['params']['z_ion'])

    fig, axes = plt.subplots(1, 3, figsize=(DCOL_W, 2.2))

    # (a) G_z vs r_tip
    ax = axes[0]
    if rtip_sweep:
        rtips = [r['params']['r_tip'] for r in rtip_sweep]
        gz = [r.get('G_y', r.get('G_z', 0)) for r in rtip_sweep]
        ax.semilogy(rtips, gz, 'o-', color=C_STYLUS, ms=4)
    ax.set_xlabel(r'$r_{\mathrm{tip}}$ [$\mu$m]')
    ax.set_ylabel(r'$\mathcal{G}_z$')
    ax.set_title(r'(a) Tip radius')

    # (b) G_z vs z_ion (log-log)
    ax = axes[1]
    if zion_sweep:
        z_arr = np.array([r['params']['z_ion'] for r in zion_sweep])
        gz_arr = np.array([r.get('G_y', r.get('G_z', 0)) for r in zion_sweep])
        mask = gz_arr > 0
        ax.loglog(z_arr[mask], gz_arr[mask], 'o-', color=C_STYLUS, ms=4,
                  label=r'$\mathcal{G}_z$')
        if np.sum(mask) >= 2:
            coeffs = np.polyfit(np.log(z_arr[mask]), np.log(gz_arr[mask]), 1)
            mid = np.sum(mask) // 2
            scale = gz_arr[mask][mid] * z_arr[mask][mid]**4
            z_ref = np.linspace(z_arr[mask].min(), z_arr[mask].max(), 50)
            ax.loglog(z_ref, scale * z_ref**(-4), ':', color='gray', lw=0.8,
                      label=r'$\propto z^{-4}$')
            ax.loglog(z_ref, np.exp(coeffs[1]) * z_ref**coeffs[0], '--',
                      color=C_STYLUS, lw=0.8, alpha=0.5,
                      label=rf'fit: $z^{{{coeffs[0]:.1f}}}$')
        ax.legend(fontsize=6, frameon=False)
    ax.set_xlabel(r'$z_{\mathrm{ion}}$ [$\mu$m]')
    ax.set_ylabel(r'$\mathcal{G}_z$')
    ax.set_title(r'(b) Distance scaling')

    # (c) G_z * z^4 collapse
    ax = axes[2]
    if zion_sweep:
        z_arr = np.array([r['params']['z_ion'] for r in zion_sweep])
        gz_arr = np.array([r.get('G_y', r.get('G_z', 0)) for r in zion_sweep])
        mask = gz_arr > 0
        ax.plot(z_arr[mask], gz_arr[mask] * z_arr[mask]**4, 'o-',
                color=C_STYLUS, ms=4)
    ax.set_xlabel(r'$z_{\mathrm{ion}}$ [$\mu$m]')
    ax.set_ylabel(r'$\mathcal{G}_z \cdot z_{\mathrm{ion}}^4$')
    ax.set_title(r'(c) $d^{-4}$ deviation')

    fig.tight_layout(pad=0.5)
    savefig(fig, out_dir, 'fig5_stylus')


# ═══════════════════════════════════════════════════════════════════
# Master entry point
# ═══════════════════════════════════════════════════════════════════
def generate_all_figures(results, out_dir):
    """Generate all paper figures."""
    os.makedirs(out_dir, exist_ok=True)
    fig1_geometry_gallery(results, out_dir)
    fig2_slot_mechanism(results, out_dir)
    fig3_twoplate(results, out_dir)
    fig4_blade(results, out_dir)
    fig5_stylus(results, out_dir)
    print(f"All figures saved to {out_dir}/")


if __name__ == "__main__":
    import glob
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "paper_outputs"

    results = []
    seen = set()
    for pattern in ["**/*_result.json", "*_result.json"]:
        for fpath in sorted(glob.glob(os.path.join(results_dir, pattern),
                                      recursive=True)):
            try:
                with open(fpath) as f:
                    data = json.load(f)
                name = data.get('params', {}).get('name', '')
                if name not in seen:
                    seen.add(name)
                    results.append(data)
            except Exception:
                pass
    print(f"Loaded {len(results)} results")
    generate_all_figures(results, out_dir)