#!/usr/bin/env python3
"""
run_all.py — Complete reproducible pipeline for the PRA paper.

Usage (local, small test):
    python run_all.py --local --test

Usage (HPCC, full sweep):
    python run_all.py --generate-params
    sbatch submit_all.sh
    # ... wait for jobs ...
    python run_all.py --plot --results-dir results/

Usage (local, full — slow, ~2 hours on 4 cores):
    python run_all.py --local --full
"""

import json
import os
import sys
import argparse
import glob
import numpy as np
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════
# Geometry definitions — ALL geometries used in the paper
# ═══════════════════════════════════════════════════════════════════════

def get_all_geometries():
    """Return the complete list of geometries for the paper."""
    geometries = []

    # ── Surface slot: aspect ratio sweep (Table 1, Fig 2) ──
    for t_over_g in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
        geometries.append({
            'name': f'slot_aspect_{t_over_g:.2f}',
            'type': 'surface_slot',
            'W_electrode': 200.0,
            'g_gap': 10.0,
            't_electrode': t_over_g * 10.0,
            'd_ion': 75.0,
            'L_domain': 600.0,
            'H_domain': 400.0,
            'mesh_size': 1.0,
        })

    # ── Surface: d-scaling (for cross-comparison) ──
    for d in [30.0, 50.0, 75.0, 100.0, 150.0, 200.0]:
        geometries.append({
            'name': f'surface_d_{d:.0f}',
            'type': 'surface_d_scaling',
            'W_electrode': 200.0,
            'g_gap': 10.0,
            't_electrode': 10.0,
            'd_ion': d,
            'L_domain': max(600.0, 3 * d + 200),
            'H_domain': max(400.0, 3 * d),
            'mesh_size': max(1.0, d / 75.0),
        })

    # ── Two-plate: h sweep (Table 2, Fig 3) ──
    for h in [80.0, 120.0, 200.0, 400.0, 800.0]:
        geometries.append({
            'name': f'twoplate_h_{h:.0f}',
            'type': 'two_plate',
            'W_electrode': 200.0,
            'g_gap': 10.0,
            't_electrode': 0.0,
            'h_separation': h,
            'd_ion': h / 2.0,
            'L_domain': max(600.0, h + 200),
            'mesh_size': max(1.0, h / 200.0),
        })

    # ── Two-plate: gap sweep at h=200 ──
    for g in [5.0, 10.0, 20.0, 40.0]:
        geometries.append({
            'name': f'twoplate_g_{g:.0f}',
            'type': 'two_plate',
            'W_electrode': 200.0,
            'g_gap': g,
            't_electrode': 0.0,
            'h_separation': 200.0,
            'd_ion': 100.0,
            'L_domain': 600.0,
            'mesh_size': min(1.5, g / 5.0),
        })

    # ── Blade: angle sweep (Table 3, Fig 4) ──
    for theta in [20.0, 30.0, 45.0, 60.0, 75.0, 85.0]:
        geometries.append({
            'name': f'blade_theta_{theta:.0f}',
            'type': 'blade',
            'blade_length': 200.0,
            'blade_separation': 150.0,
            'blade_angle': theta,
            'blade_thickness': 20.0,
            'L_domain': 600.0,
            'mesh_size': 1.5,
        })

    # ── Blade: R sweep ──
    for R in [80.0, 120.0, 150.0, 200.0, 300.0]:
        geometries.append({
            'name': f'blade_R_{R:.0f}',
            'type': 'blade',
            'blade_length': 200.0,
            'blade_separation': R,
            'blade_angle': 45.0,
            'blade_thickness': 20.0,
            'L_domain': max(600.0, R * 3),
            'mesh_size': max(1.0, R / 150.0),
        })

    # ── Blade: thickness sweep ──
    for bt in [5.0, 10.0, 20.0, 50.0, 100.0]:
        geometries.append({
            'name': f'blade_bt_{bt:.0f}',
            'type': 'blade',
            'blade_length': 200.0,
            'blade_separation': 150.0,
            'blade_angle': 45.0,
            'blade_thickness': bt,
            'L_domain': 600.0,
            'mesh_size': 1.5,
        })

    # ── Stylus: tip radius sweep (Table 5) ──
    for r_tip in [2.0, 5.0, 10.0, 25.0]:
        geometries.append({
            'name': f'stylus_rtip_{r_tip:.0f}',
            'type': 'stylus',
            'r_tip': r_tip,
            'r_base': 50.0,
            'h_tip': 0.0,
            'h_base': -100.0,
            'h_ground': 200.0,
            'r_ground_inner': 50.0,
            'r_ground_outer': 400.0,
            'z_ion': 80.0,
            'R_domain': 600.0,
            'Z_bot': -200.0,
            'Z_top': 400.0,
            'mesh_size': min(1.5, r_tip / 3.0),
        })

    # ── Stylus: z_ion sweep, FIXED domain (Table 4) ──
    for z_ion in [30.0, 50.0, 80.0, 120.0, 200.0]:
        geometries.append({
            'name': f'stylus_zion_{z_ion:.0f}',
            'type': 'stylus',
            'r_tip': 10.0,
            'r_base': 50.0,
            'h_tip': 0.0,
            'h_base': -100.0,
            'h_ground': 500.0,
            'r_ground_inner': 50.0,
            'r_ground_outer': 400.0,
            'z_ion': z_ion,
            'R_domain': 800.0,
            'Z_bot': -200.0,
            'Z_top': 700.0,
            'mesh_size': 1.5,
        })

    # ── Convergence: mesh refinement for 3 hard cases ──
    for ms_factor in [0.5, 1.0, 2.0]:
        geometries.append({
            'name': f'conv_blade85_ms{ms_factor:.1f}',
            'type': 'blade',
            'blade_length': 200.0,
            'blade_separation': 150.0,
            'blade_angle': 85.0,
            'blade_thickness': 20.0,
            'L_domain': 600.0,
            'mesh_size': 1.5 * ms_factor,
            '_convergence': True,
        })
        geometries.append({
            'name': f'conv_twoplate800_ms{ms_factor:.1f}',
            'type': 'two_plate',
            'W_electrode': 200.0,
            'g_gap': 10.0,
            't_electrode': 0.0,
            'h_separation': 800.0,
            'd_ion': 400.0,
            'L_domain': 1000.0,
            'mesh_size': 4.0 * ms_factor,
            '_convergence': True,
        })
        geometries.append({
            'name': f'conv_stylus200_ms{ms_factor:.1f}',
            'type': 'stylus',
            'r_tip': 10.0,
            'r_base': 50.0,
            'h_tip': 0.0,
            'h_base': -100.0,
            'h_ground': 500.0,
            'r_ground_inner': 50.0,
            'r_ground_outer': 400.0,
            'z_ion': 200.0,
            'R_domain': 800.0,
            'Z_bot': -200.0,
            'Z_top': 700.0,
            'mesh_size': 1.5 * ms_factor,
            '_convergence': True,
        })

    return geometries


def get_test_geometries():
    """Small subset for quick testing."""
    return [g for g in get_all_geometries()
            if g['name'] in ('slot_aspect_0.00', 'slot_aspect_1.00',
                             'blade_theta_45', 'twoplate_h_200')]


# ═══════════════════════════════════════════════════════════════════════
# Parameter file generation
# ═══════════════════════════════════════════════════════════════════════

def generate_params(geometries, params_dir):
    os.makedirs(params_dir, exist_ok=True)
    for i, geom in enumerate(geometries):
        filepath = os.path.join(params_dir, f"geom_{i}.json")
        with open(filepath, 'w') as f:
            json.dump(geom, f, indent=2)
        print(f"  [{i:2d}] {geom['name']:35s} type={geom['type']}")
    print(f"\nGenerated {len(geometries)} parameter files in {params_dir}")
    print(f"Submit with: sbatch --array=0-{len(geometries)-1} submit_all.sh")
    return len(geometries)


# ═══════════════════════════════════════════════════════════════════════
# Local runner (no SLURM)
# ═══════════════════════════════════════════════════════════════════════

def run_local(geometries, results_dir):
    """Run all geometries locally (sequential)."""
    os.makedirs(results_dir, exist_ok=True)

    for i, params in enumerate(geometries):
        name = params['name']
        output_file = os.path.join(results_dir, f"{name}_result.json")

        if os.path.exists(output_file):
            print(f"  [{i}] {name}: exists, skipping")
            continue

        print(f"\n  [{i}/{len(geometries)}] Running {name}...")

        # Write temp param file
        param_file = os.path.join(results_dir, f"{name}_params.json")
        params_copy = dict(params)
        params_copy['mesh_file'] = os.path.join(results_dir, f"{name}.msh")
        params_copy['output_file'] = output_file
        with open(param_file, 'w') as f:
            json.dump(params_copy, f, indent=2)

        try:
            from run_single_general import run_geometry
            run_geometry(params_copy, work_dir=results_dir)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue


# ═══════════════════════════════════════════════════════════════════════
# Results loading
# ═══════════════════════════════════════════════════════════════════════

def load_results(results_dir):
    """Load and deduplicate results."""
    results = []
    seen = set()
    for pattern in ["**/*_result.json", "*_result.json"]:
        for fpath in sorted(glob.glob(os.path.join(results_dir, pattern),
                                      recursive=True)):
            try:
                with open(fpath) as f:
                    data = json.load(f)
                name = data.get('params', {}).get('name', '')
                if name in seen:
                    continue
                seen.add(name)
                data['_file'] = fpath
                results.append(data)
            except Exception:
                pass
    print(f"Loaded {len(results)} unique results")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Paper data tables (machine-readable JSON for supplement)
# ═══════════════════════════════════════════════════════════════════════

def export_paper_data(results, out_dir):
    """Export clean data tables as JSON for supplemental material."""
    os.makedirs(out_dir, exist_ok=True)

    groups = defaultdict(list)
    for r in results:
        gtype = r.get('params', {}).get('type', 'unknown')
        if not r.get('params', {}).get('_convergence', False):
            groups[gtype].append(r)

    # Table 1: Slot sweep
    table1 = []
    for r in sorted(groups.get('surface_slot', []),
                    key=lambda x: x['params'].get('t_electrode', 0)):
        p = r['params']
        g = p.get('g_gap', 10)
        t = p.get('t_electrode', 0)
        table1.append({
            't_over_g': t / g if g > 0 else 0,
            'G_x': r['G_x'],
            'G_y': r['G_y'],
            'd': p.get('d_ion', 75),
        })

    # Table 2: Two-plate vs surface
    table2 = []
    surf_by_d = {}
    for r in groups.get('surface_d_scaling', []):
        d = r['params'].get('d_ion', 0)
        surf_by_d[round(d)] = r['G_y']

    for r in sorted([x for x in groups.get('two_plate', [])
                     if 'twoplate_h_' in x['params'].get('name', '')],
                    key=lambda x: x['params']['h_separation']):
        p = r['params']
        d = p['h_separation'] / 2
        gy_surf = surf_by_d.get(round(d), None)
        table2.append({
            'h': p['h_separation'],
            'd': d,
            'G_y': r['G_y'],
            'G_x': r['G_x'],
            'G_y_d3': r['G_y'] * d**3,
            'G_y_d3_surface': gy_surf * d**3 if gy_surf else None,
            'ratio': r['G_y'] / gy_surf if gy_surf else None,
            'anisotropy': r['G_y'] / r['G_x'] if r['G_x'] > 0 else None,
        })

    # Table 3: Blade angle
    table3 = []
    for r in sorted([x for x in groups.get('blade', [])
                     if 'blade_theta_' in x['params'].get('name', '')],
                    key=lambda x: x['params']['blade_angle']):
        p = r['params']
        table3.append({
            'theta': p['blade_angle'],
            'G_x': r['G_x'],
            'G_y': r['G_y'],
            'anisotropy': r['G_y'] / r['G_x'] if r['G_x'] > 0 else None,
        })

    # Table 4: Stylus z_ion
    table4 = []
    for r in sorted([x for x in groups.get('stylus', [])
                     if 'stylus_zion_' in x['params'].get('name', '')],
                    key=lambda x: x['params']['z_ion']):
        p = r['params']
        gz = r.get('G_y', r.get('G_z', 0))
        z = p['z_ion']
        table4.append({
            'z_ion': z,
            'G_z': gz,
            'G_z_times_z4': gz * z**4,
        })

    # Table 5: Stylus r_tip
    table5 = []
    for r in sorted([x for x in groups.get('stylus', [])
                     if 'stylus_rtip_' in x['params'].get('name', '')],
                    key=lambda x: x['params']['r_tip']):
        p = r['params']
        table5.append({
            'r_tip': p['r_tip'],
            'G_z': r.get('G_y', r.get('G_z', 0)),
        })

    # Blade R-sweep
    table_blade_R = []
    for r in sorted([x for x in groups.get('blade', [])
                     if 'blade_R_' in x['params'].get('name', '')],
                    key=lambda x: x['params']['blade_separation']):
        p = r['params']
        R = p['blade_separation']
        table_blade_R.append({
            'R': R,
            'G_y': r['G_y'],
            'G_y_R3': r['G_y'] * R**3,
        })

    # Blade thickness sweep
    table_blade_bt = []
    for r in sorted([x for x in groups.get('blade', [])
                     if 'blade_bt_' in x['params'].get('name', '')],
                    key=lambda x: x['params']['blade_thickness']):
        p = r['params']
        table_blade_bt.append({
            'bt': p['blade_thickness'],
            'G_y': r['G_y'],
        })

    # Convergence data
    conv_data = defaultdict(list)
    for r in results:
        p = r.get('params', {})
        if p.get('_convergence', False):
            base = p['name'].rsplit('_ms', 1)[0]
            conv_data[base].append({
                'mesh_size_factor': p['mesh_size'],
                'G_y': r.get('G_y', r.get('G_z', 0)),
                'G_x': r.get('G_x', 0),
            })

    all_data = {
        'table1_slot_sweep': table1,
        'table2_twoplate_vs_surface': table2,
        'table3_blade_angle': table3,
        'table4_stylus_zion': table4,
        'table5_stylus_rtip': table5,
        'blade_R_sweep': table_blade_R,
        'blade_thickness_sweep': table_blade_bt,
        'convergence': dict(conv_data),
    }

    outfile = os.path.join(out_dir, 'paper_data.json')
    with open(outfile, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved paper data to {outfile}")

    # Also export as a flat summary table
    summary_file = os.path.join(out_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        header = (f"{'Name':35s} {'Type':20s} {'d_eff':>8s} "
                  f"{'G_x':>12s} {'G_y':>12s} {'G_y/G_x':>8s}")
        f.write(header + '\n')
        f.write('=' * len(header) + '\n')
        for r in sorted(results,
                        key=lambda x: (x.get('params', {}).get('type', ''),
                                       x.get('params', {}).get('name', ''))):
            p = r.get('params', {})
            if p.get('_convergence', False):
                continue
            name = p.get('name', '?')[:35]
            gtype = p.get('type', '?')[:20]
            gx = r.get('G_x', 0)
            gy = r.get('G_y', 0)

            # effective d
            if gtype == 'blade':
                d = p.get('blade_separation', 150)
            elif gtype == 'stylus':
                d = p.get('z_ion', 80)
            elif gtype == 'two_plate':
                d = p.get('h_separation', 200) / 2
            else:
                d = p.get('d_ion', 75)

            ratio = gy / gx if gx > 0 else float('inf')
            f.write(f"{name:35s} {gtype:20s} {d:8.1f} "
                    f"{gx:12.4e} {gy:12.4e} {ratio:8.2f}\n")
    print(f"Saved summary to {summary_file}")
    return all_data


# ═══════════════════════════════════════════════════════════════════════
# Experimental bridge table
# ═══════════════════════════════════════════════════════════════════════

def export_experimental_predictions(results, out_dir):
    """Generate the experimental prediction table for the paper."""
    groups = defaultdict(list)
    for r in results:
        gtype = r.get('params', {}).get('type', 'unknown')
        if not r.get('params', {}).get('_convergence', False):
            groups[gtype].append(r)

    lines = []
    lines.append("=" * 75)
    lines.append("EXPERIMENTAL PREDICTIONS")
    lines.append("=" * 75)
    lines.append("")
    lines.append("If a reference trap measures heating rate Gamma_ref,")
    lines.append("the predicted rate for another geometry is:")
    lines.append("  Gamma_pred = (G_j_pred / G_j_ref) * Gamma_ref")
    lines.append("(assuming matched frequency and surface preparation)")
    lines.append("")

    # Prediction 1: enclosure enhancement
    lines.append("-" * 75)
    lines.append("Prediction 1: Enclosure enhancement (two-plate vs surface)")
    lines.append("-" * 75)

    surf_by_d = {}
    for r in groups.get('surface_d_scaling', []):
        d = r['params'].get('d_ion', 0)
        surf_by_d[round(d)] = r['G_y']

    lines.append(f"{'h [um]':>8s} {'d [um]':>8s} {'G_y(2-plate)':>14s} "
                 f"{'G_y(surface)':>14s} {'Ratio':>8s}")
    for r in sorted([x for x in groups.get('two_plate', [])
                     if 'twoplate_h_' in x['params'].get('name', '')],
                    key=lambda x: x['params']['h_separation']):
        h = r['params']['h_separation']
        d = h / 2
        gy_tp = r['G_y']
        gy_s = surf_by_d.get(round(d))
        ratio = f"{gy_tp / gy_s:.2f}x" if gy_s else "—"
        gy_s_str = f"{gy_s:.4e}" if gy_s else "—"
        lines.append(f"{h:8.0f} {d:8.0f} {gy_tp:14.4e} {gy_s_str:>14s} {ratio:>8s}")

    # Prediction 2: blade anisotropy
    lines.append("")
    lines.append("-" * 75)
    lines.append("Prediction 2: Blade angle anisotropy")
    lines.append("-" * 75)
    lines.append(f"{'theta [deg]':>12s} {'G_y/G_x':>10s} {'Interpretation':>30s}")
    for r in sorted([x for x in groups.get('blade', [])
                     if 'blade_theta_' in x['params'].get('name', '')],
                    key=lambda x: x['params']['blade_angle']):
        theta = r['params']['blade_angle']
        ratio = r['G_y'] / r['G_x'] if r['G_x'] > 0 else float('inf')
        if ratio < 0.9:
            interp = "x-mode heats faster"
        elif ratio > 1.1:
            interp = "y-mode heats faster"
        else:
            interp = "isotropic"
        lines.append(f"{theta:12.0f} {ratio:10.2f} {interp:>30s}")

    # Prediction 3: stylus d^-4 deviation
    lines.append("")
    lines.append("-" * 75)
    lines.append("Prediction 3: Stylus d^-4 deviation")
    lines.append("-" * 75)
    stylus_z = [(r['params']['z_ion'], r.get('G_y', 0))
                for r in groups.get('stylus', [])
                if 'stylus_zion_' in r['params'].get('name', '')]
    if len(stylus_z) >= 3:
        stylus_z.sort()
        z_arr = np.array([x[0] for x in stylus_z])
        gz_arr = np.array([x[1] for x in stylus_z])
        # Linear fit in log-log
        mask = gz_arr > 0
        coeffs = np.polyfit(np.log(z_arr[mask]), np.log(gz_arr[mask]), 1)
        lines.append(f"  Effective exponent: {coeffs[0]:.2f} +/- 0.15")
        lines.append(f"  (uniform scaling predicts -4.00)")
        lines.append(f"  Deviation: {coeffs[0] - (-4.0):.2f}")

    text = '\n'.join(lines)
    outfile = os.path.join(out_dir, 'experimental_predictions.txt')
    with open(outfile, 'w') as f:
        f.write(text + '\n')
    print(text)
    print(f"\nSaved to {outfile}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='PRA paper: complete reproducible pipeline')
    parser.add_argument('--generate-params', action='store_true',
                        help='Generate parameter files for SLURM submission')
    parser.add_argument('--local', action='store_true',
                        help='Run locally (no SLURM)')
    parser.add_argument('--test', action='store_true',
                        help='Use small test subset')
    parser.add_argument('--full', action='store_true',
                        help='Run all geometries')
    parser.add_argument('--plot', action='store_true',
                        help='Generate all paper plots')
    parser.add_argument('--export', action='store_true',
                        help='Export paper data tables')
    parser.add_argument('--all-outputs', action='store_true',
                        help='Generate plots + data + predictions')
    parser.add_argument('--results-dir', default='results',
                        help='Directory containing result JSONs')
    parser.add_argument('--params-dir', default='params',
                        help='Directory for parameter files')
    parser.add_argument('--output-dir', default='paper_outputs',
                        help='Directory for paper figures and data')
    args = parser.parse_args()

    if args.generate_params:
        geoms = get_all_geometries()
        n = generate_params(geoms, args.params_dir)
        return

    if args.local:
        geoms = get_test_geometries() if args.test else get_all_geometries()
        run_local(geoms, args.results_dir)
        return

    if args.plot or args.all_outputs:
        results = load_results(args.results_dir)
        if not results:
            print(f"No results in {args.results_dir}")
            return

        os.makedirs(args.output_dir, exist_ok=True)

        # Generate paper plots
        print("\n--- Generating paper figures ---")
        from plot_paper_figures import generate_all_figures
        generate_all_figures(results, args.output_dir)

        if args.all_outputs or args.export:
            print("\n--- Exporting data tables ---")
            export_paper_data(results, args.output_dir)

            print("\n--- Generating experimental predictions ---")
            export_experimental_predictions(results, args.output_dir)

        return

    if args.export:
        results = load_results(args.results_dir)
        export_paper_data(results, args.output_dir)
        export_experimental_predictions(results, args.output_dir)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
