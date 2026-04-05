#!/usr/bin/env python3
"""
decompose_gj_extended.py — Decompose G_j by spatial region for all geometry types.

For each geometry type, classifies boundary facets into physically meaningful
regions and computes the partial G_j from each.

Usage:
    python decompose_gj_extended.py <results_dir>
"""
import json
import glob
import os
import sys
import numpy as np
from collections import defaultdict


def classify_surface_slot(pts, params):
    """Classify boundary points for surface slot geometry."""
    g = params.get('g_gap', 10)
    t = params.get('t_electrode', 0)
    tol = max(g * 0.1, 0.5)

    regions = {}
    if t < 0.5:
        regions['top_surface'] = np.ones(len(pts), dtype=bool)
        regions['gap_sidewalls'] = np.zeros(len(pts), dtype=bool)
        regions['gap_bottom'] = np.zeros(len(pts), dtype=bool)
        regions['outer_walls'] = np.zeros(len(pts), dtype=bool)
    else:
        is_top = (np.abs(pts[:, 1] - t) < tol) & (np.abs(pts[:, 0]) > g/2 - tol)
        is_gap_wall = ((np.abs(np.abs(pts[:, 0]) - g/2) < tol) &
                       (pts[:, 1] > tol) & (pts[:, 1] < t - tol))
        is_gap_floor = (np.abs(pts[:, 1]) < tol) & (np.abs(pts[:, 0]) < g/2 + tol)
        is_other = ~(is_top | is_gap_wall | is_gap_floor)
        regions['top_surface'] = is_top
        regions['gap_sidewalls'] = is_gap_wall
        regions['gap_bottom'] = is_gap_floor
        regions['outer_walls'] = is_other
    return regions


def classify_two_plate(pts, params):
    """Classify boundary points for two-plate geometry."""
    g = params.get('g_gap', 10)
    h = params.get('h_separation', 200)
    tol = max(g * 0.15, 1.0)

    regions = {}
    # Bottom plate electrodes: y ≈ 0, |x| > g/2
    regions['bottom_plate'] = (np.abs(pts[:, 1]) < tol) & (np.abs(pts[:, 0]) > g/2 - tol)
    # Top plate electrodes: y ≈ h, |x| > g/2
    regions['top_plate'] = (np.abs(pts[:, 1] - h) < tol) & (np.abs(pts[:, 0]) > g/2 - tol)
    # Near-gap region on bottom plate: within 2g of gap edge
    regions['bottom_near_gap'] = (regions['bottom_plate'] &
                                   (np.abs(pts[:, 0]) < g/2 + 2*g))
    # Near-gap region on top plate
    regions['top_near_gap'] = (regions['top_plate'] &
                                (np.abs(pts[:, 0]) < g/2 + 2*g))
    # Far from gap
    regions['bottom_far'] = regions['bottom_plate'] & ~regions['bottom_near_gap']
    regions['top_far'] = regions['top_plate'] & ~regions['top_near_gap']
    # Other
    regions['other'] = ~(regions['bottom_plate'] | regions['top_plate'])
    return regions


def classify_blade(pts, params):
    """Classify boundary points for blade geometry by quadrant."""
    theta_deg = params.get('blade_angle', 45)
    theta = np.radians(theta_deg)
    R = params.get('blade_separation', 150)

    angles = np.arctan2(pts[:, 1], pts[:, 0])  # angle of each point from origin

    regions = {}
    # Four blades at angles theta, pi-theta, pi+theta, 2pi-theta
    blade_angles = [theta, np.pi - theta, np.pi + theta, 2*np.pi - theta]
    blade_names = ['upper_right', 'upper_left', 'lower_left', 'lower_right']

    assigned = np.zeros(len(pts), dtype=bool)
    for name, ba in zip(blade_names, blade_angles):
        # Points near this blade: within angular tolerance and at distance > R
        ba_norm = np.arctan2(np.sin(ba), np.cos(ba))  # normalize to [-pi, pi]
        angle_diff = np.abs(np.arctan2(np.sin(angles - ba_norm),
                                        np.cos(angles - ba_norm)))
        dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        mask = (angle_diff < np.radians(30)) & (dist > R * 0.7)
        regions[f'blade_{name}'] = mask
        assigned |= mask

    regions['other'] = ~assigned

    # Aggregate: "horizontal pair" vs "vertical pair"
    # Upper blades contribute to G_y, lower to G_y (both pairs),
    # but the directional coupling differs
    regions['upper_blades'] = regions['blade_upper_right'] | regions['blade_upper_left']
    regions['lower_blades'] = regions['blade_lower_right'] | regions['blade_lower_left']
    regions['left_blades'] = regions['blade_upper_left'] | regions['blade_lower_left']
    regions['right_blades'] = regions['blade_upper_right'] | regions['blade_lower_right']

    return regions


def classify_stylus(pts, params):
    """Classify boundary points for stylus geometry."""
    r_tip = params.get('r_tip', 10)
    h_tip = params.get('h_tip', 0)
    h_ground = params.get('h_ground', 200)
    r_ground_inner = params.get('r_ground_inner', 50)
    tol = max(r_tip * 0.3, 1.0)

    regions = {}
    # Stylus cap: near tip, z ≈ h_tip, r small
    dist_from_tip = np.sqrt(pts[:, 0]**2 + (pts[:, 1] - h_tip)**2)
    regions['stylus_cap'] = dist_from_tip < r_tip * 2.5
    # Stylus sidewall: below tip
    regions['stylus_wall'] = (~regions['stylus_cap'] &
                               (pts[:, 1] < h_tip) &
                               (pts[:, 0] < r_tip * 5))
    # Ground plane
    regions['ground_plane'] = (np.abs(pts[:, 1] - h_ground) < tol)
    # Other
    regions['other'] = ~(regions['stylus_cap'] | regions['stylus_wall'] |
                          regions['ground_plane'])
    return regions


CLASSIFIERS = {
    'surface_slot': classify_surface_slot,
    'surface_d_scaling': classify_surface_slot,
    'surface_gap_width': classify_surface_slot,
    'surface_backplane': classify_surface_slot,
    'two_plate': classify_two_plate,
    'blade': classify_blade,
    'stylus': classify_stylus,
}


def decompose_result(result_file):
    """Decompose a single result file."""
    with open(result_file) as f:
        data = json.load(f)

    pts = np.array(data.get('midpoints', []))
    Ky = np.array(data.get('K_y', []))
    Kx = np.array(data.get('K_x', []))
    dl = np.array(data.get('dl', []))
    params = data.get('params', {})
    gtype = params.get('type', data.get('geometry_type', 'unknown'))
    name = params.get('name', os.path.basename(result_file))

    if len(pts) == 0 or len(Ky) == 0:
        print(f"  {name}: no boundary data")
        return None

    classifier = CLASSIFIERS.get(gtype)
    if classifier is None:
        print(f"  {name}: no classifier for type '{gtype}'")
        return None

    regions = classifier(pts, params)

    G_y_total = np.sum(Ky**2 * dl)
    G_x_total = np.sum(Kx**2 * dl)

    # For stylus (axisymmetric), include 2π r' weight
    is_axisym = data.get('axisymmetric', False)
    if is_axisym:
        r_weight = np.maximum(pts[:, 0], 1e-10)
        G_y_total = 2 * np.pi * np.sum(Ky**2 * r_weight * dl)
        G_x_total = 2 * np.pi * np.sum(Kx**2 * r_weight * dl)

    print(f"\n  {name} ({gtype})")
    print(f"  {'Region':25s} {'N':>6s} {'G_y_part':>12s} {'% G_y':>8s} "
          f"{'G_x_part':>12s}")
    print(f"  {'-'*65}")

    decomp = {'name': name, 'type': gtype, 'params': params}

    # Only print the primary regions (skip aggregated ones for blade)
    skip_aggregate = {'upper_blades', 'lower_blades', 'left_blades', 'right_blades'}

    for rname, mask in regions.items():
        if rname in skip_aggregate:
            continue
        n = int(np.sum(mask))
        if is_axisym:
            gy_part = 2 * np.pi * np.sum(Ky[mask]**2 * r_weight[mask] * dl[mask]) if n > 0 else 0
            gx_part = 2 * np.pi * np.sum(Kx[mask]**2 * r_weight[mask] * dl[mask]) if n > 0 else 0
        else:
            gy_part = np.sum(Ky[mask]**2 * dl[mask]) if n > 0 else 0
            gx_part = np.sum(Kx[mask]**2 * dl[mask]) if n > 0 else 0
        pct = 100 * gy_part / G_y_total if G_y_total > 0 else 0

        print(f"  {rname:25s} {n:6d} {gy_part:12.4e} {pct:7.1f}% {gx_part:12.4e}")
        decomp[rname] = {
            'G_y': float(gy_part),
            'G_x': float(gx_part),
            'pct_G_y': float(pct),
            'n_facets': n,
        }

    print(f"  {'-'*65}")
    print(f"  {'TOTAL':25s} {len(pts):6d} {G_y_total:12.4e} {'100.0':>7s}% {G_x_total:12.4e}")

    decomp['total'] = {'G_y': float(G_y_total), 'G_x': float(G_x_total)}
    return decomp


def main():
    results_dir = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else "results"
    files = sorted(glob.glob(os.path.join(results_dir, "**", "*_result.json"),
                             recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(results_dir, "*_result.json")))

    print(f"Found {len(files)} result files in {results_dir}")

    all_decomp = defaultdict(list)
    for f in files:
        decomp = decompose_result(f)
        if decomp:
            all_decomp[decomp['type']].append(decomp)

    # ── Per-type summaries ──
    for gtype, decomps in sorted(all_decomp.items()):
        print(f"\n\n{'='*70}")
        print(f"SUMMARY: {gtype} ({len(decomps)} geometries)")
        print(f"{'='*70}")

        if gtype in ('two_plate',):
            # Show top vs bottom plate contribution
            print(f"{'Name':25s} {'G_y(bot)':>12s} {'G_y(top)':>12s} "
                  f"{'top/bot':>8s} {'G_y(total)':>12s}")
            print('-' * 72)
            for d in decomps:
                gt = d.get('total', {}).get('G_y', 0)
                gb = d.get('bottom_plate', {}).get('G_y', 0)
                gtp = d.get('top_plate', {}).get('G_y', 0)
                ratio = gtp / gb if gb > 0 else float('inf')
                print(f"{d['name']:25s} {gb:12.4e} {gtp:12.4e} "
                      f"{ratio:8.2f} {gt:12.4e}")

        elif gtype == 'blade':
            # Show per-blade contributions
            print(f"{'Name':25s} {'G_y(UL)':>10s} {'G_y(UR)':>10s} "
                  f"{'G_y(LL)':>10s} {'G_y(LR)':>10s} {'G_y(tot)':>12s}")
            print('-' * 80)
            for d in decomps:
                gt = d.get('total', {}).get('G_y', 0)
                gul = d.get('blade_upper_left', {}).get('G_y', 0)
                gur = d.get('blade_upper_right', {}).get('G_y', 0)
                gll = d.get('blade_lower_left', {}).get('G_y', 0)
                glr = d.get('blade_lower_right', {}).get('G_y', 0)
                print(f"{d['name']:25s} {gul:10.3e} {gur:10.3e} "
                      f"{gll:10.3e} {glr:10.3e} {gt:12.4e}")

        elif gtype == 'stylus':
            # Show cap vs wall vs ground
            print(f"{'Name':25s} {'G_z(cap)':>12s} {'%(cap)':>8s} "
                  f"{'G_z(wall)':>12s} {'G_z(gnd)':>12s} {'G_z(tot)':>12s}")
            print('-' * 85)
            for d in decomps:
                gt = d.get('total', {}).get('G_y', 0)
                gc = d.get('stylus_cap', {}).get('G_y', 0)
                gw = d.get('stylus_wall', {}).get('G_y', 0)
                gg = d.get('ground_plane', {}).get('G_y', 0)
                pct = 100 * gc / gt if gt > 0 else 0
                print(f"{d['name']:25s} {gc:12.4e} {pct:7.1f}% "
                      f"{gw:12.4e} {gg:12.4e} {gt:12.4e}")

    # Save all decompositions
    out_file = os.path.join(results_dir, "decomposition_extended.json")
    # Convert defaultdict to regular dict for JSON
    save_data = {k: v for k, v in all_decomp.items()}
    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved decompositions to {out_file}")


if __name__ == "__main__":
    main()
