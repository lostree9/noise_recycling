#!/usr/bin/env python3
"""
run_single_general.py — Run full pipeline for one geometry.
Dispatches to the correct mesh generator and G_j solver based on geometry type.
"""
import json
import sys
import os
import numpy as np

from mpi4py import MPI
import dolfinx

try:
    from dolfinx.io import gmshio
except ImportError:
    from dolfinx.io import gmsh as gmshio


def read_mesh_robust(mesh_file, comm, gdim=2):
    """Read gmsh mesh, handling different DOLFINx API versions."""
    print(f"  DOLFINx version: {dolfinx.__version__}")
    result = gmshio.read_from_msh(mesh_file, comm, gdim=gdim)

    if hasattr(result, 'mesh'):
        return result.mesh, result.cell_tags, result.facet_tags

    if isinstance(result, tuple):
        return result[0], result[1], result[2]

    raise RuntimeError(f"Cannot unpack gmshio result: {type(result)}")


def run_geometry(params, work_dir="."):
    """Run full pipeline for one parameter set."""
    name = params.get('name', 'unnamed')
    gtype = params.get('type', 'surface_slot')

    print(f"\n{'='*60}")
    print(f"Running geometry: {name} (type: {gtype})")
    print(f"DOLFINx version: {dolfinx.__version__}")
    print(f"{'='*60}")

    mesh_file = os.path.join(work_dir, f"{name}.msh")
    output_file = os.path.join(work_dir, f"{name}_result.json")

    if os.path.exists(output_file):
        print(f"  Output exists, skipping.")
        return

    # ── 1. Generate mesh ──
    print(f"\n--- Generating mesh ---")

    is_axisym = (gtype == 'stylus')

    if gtype in ('surface_slot', 'surface_d_scaling', 'surface_gap_width',
                 'surface_backplane'):
        from generate_mesh import create_surface_trap_mesh
        ion_pos = create_surface_trap_mesh(
            params, output_file=mesh_file,
            mesh_size=params.get('mesh_size', 1.5)
        )
    else:
        from generate_mesh_general import create_mesh
        ion_pos = create_mesh(
            params, output_file=mesh_file,
            mesh_size=params.get('mesh_size', 1.5)
        )

    params['ion_pos'] = ion_pos.tolist()
    params['mesh_file'] = mesh_file
    params['output_file'] = output_file

    # ── 2. Load mesh ──
    print(f"\n--- Loading mesh ---")
    msh, cell_tags, facet_tags = read_mesh_robust(mesh_file, MPI.COMM_WORLD, gdim=2)

    tdim = msh.topology.dim
    num_cells = msh.topology.index_map(tdim).size_local
    num_verts = msh.topology.index_map(0).size_local
    print(f"  Mesh: {num_cells} cells, {num_verts} vertices")

    # Check facet tags
    for tag_val in [1, 2, 3, 4]:
        facets = facet_tags.find(tag_val)
        if len(facets) > 0:
            print(f"  Facet tag {tag_val}: {len(facets)} facets")

    # ── 3. Compute G_j ──
    print(f"\n--- Computing G_j ---")

    if is_axisym:
        from compute_gj_axi import compute_geometry_functional_axisym
        result = compute_geometry_functional_axisym(
            msh, facet_tags, ion_pos,
            electrode_tag=1, outer_tag=2, gap_tag=3, axis_tag=4,
        )
    else:
        from compute_gj import compute_geometry_functional
        # For blade trap, there's no gap tag (tag 3 may not exist)
        gap_tag = 3
        if gtype == 'blade':
            # Check if gap tag exists
            if len(facet_tags.find(3)) == 0:
                gap_tag = 2  # lump with outer

        result = compute_geometry_functional(
            msh, facet_tags, ion_pos,
            electrode_tag=1, outer_tag=2, gap_tag=gap_tag,
        )

    result['params'] = params
    result['dolfinx_version'] = dolfinx.__version__
    result['geometry_type'] = gtype

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nDone: {name}")
    print(f"  G_x = {result.get('G_x', 'N/A')}")
    print(f"  G_y = {result.get('G_y', 'N/A')}")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_single_general.py <param_file.json>")
        sys.exit(1)

    param_file = sys.argv[1]
    with open(param_file, 'r') as f:
        params = json.load(f)

    work_dir = os.path.dirname(os.path.abspath(param_file))
    run_geometry(params, work_dir=work_dir)
