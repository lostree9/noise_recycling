#!/usr/bin/env python3
"""
compute_gj.py — Compute geometry functional G_j for a 2D trap cross-section.

Method:
    1. Decompose Green's function: G(r, r_src) = G_0(r, r_src) + H(r, r_src)
       where G_0 = -(1/2π) ln|r - r_src| is the free-space 2D Green's function
       and H is the regular part satisfying:
           -ΔH = 0          in Ω
           H = -G_0          on ∂Ω  (so that G = 0 on ∂Ω)

    2. Solve for H using FEniCSx. The boundary condition is smooth since
       r_src is in the interior.

    3. Compute the full normal derivative on the boundary:
       ∂G/∂n = ∂G_0/∂n + ∂H/∂n

    4. Do this for 3 source positions: r_0, r_0 + δ·ê_x, r_0 + δ·ê_y
       and finite-difference to get K_j(r') = ∂_{r₀,j} [∂G/∂n]

    5. Integrate |K_j|² over electrode surfaces → G_j
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import fem, mesh, io
try:
    from dolfinx.io import gmshio
except ImportError:
    from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import LinearProblem
import ufl
import json
import sys
import os


def free_space_green_2d(x, x_src):
    """Free-space 2D Green's function: G_0 = -(1/2π) ln|x - x_src|"""
    dx = x[0] - x_src[0]
    dy = x[1] - x_src[1]
    r2 = dx**2 + dy**2
    r2 = np.maximum(r2, 1e-30)  # avoid log(0)
    return -1.0 / (2.0 * np.pi) * 0.5 * np.log(r2)


def grad_free_space_green_2d(x, x_src):
    """Gradient of free-space 2D Green's function w.r.t. x."""
    dx = x[0] - x_src[0]
    dy = x[1] - x_src[1]
    r2 = dx**2 + dy**2
    r2 = np.maximum(r2, 1e-30)
    gx = -1.0 / (2.0 * np.pi) * dx / r2
    gy = -1.0 / (2.0 * np.pi) * dy / r2
    return gx, gy


def solve_regular_part(msh, facet_tags, x_src, electrode_tag=1,
                       outer_tag=2, gap_tag=3):
    """
    Solve for the regular part H of the Green's function.

    -ΔH = 0 in Ω
    H = (1/2π) ln|x - x_src| on ∂Ω  (i.e., H = -G_0 on boundary)

    Returns the FEM solution H and the function space.
    """
    V = fem.functionspace(msh, ("Lagrange", 2))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Bilinear form: a(u,v) = ∫ ∇u · ∇v dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    # RHS: zero (Laplace equation)
    L = fem.Constant(msh, PETSc.ScalarType(0.0)) * v * ufl.dx

    # Dirichlet BC: H = -G_0 = (1/2π) ln|x - x_src| on ALL boundaries
    # (electrode, outer, and gap boundaries)

    def bc_value(x):
        """H on boundary = -G_0 = (1/2π) ln|x - x_src|"""
        dx = x[0] - x_src[0]
        dy = x[1] - x_src[1]
        r2 = dx**2 + dy**2
        r2 = np.maximum(r2, 1e-30)
        return 1.0 / (2.0 * np.pi) * 0.5 * np.log(r2)

    # Find all boundary facets
    boundary_facets = np.concatenate([
        facet_tags.find(electrode_tag),
        facet_tags.find(outer_tag),
        facet_tags.find(gap_tag),
    ])
    boundary_dofs = fem.locate_dofs_topological(
        V, msh.topology.dim - 1, boundary_facets
    )

    # Interpolate BC function
    H_bc = fem.Function(V)
    H_bc.interpolate(bc_value)

    bc = fem.dirichletbc(H_bc, boundary_dofs)

    # Assemble and solve manually (compatible with DOLFINx 0.8+/0.9+)
    a_compiled = fem.form(a)
    L_compiled = fem.form(L)

    A = fem.petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    b = fem.petsc.assemble_vector(L_compiled)
    fem.petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, [bc])

    H = fem.Function(V)
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.solve(b, H.x.petsc_vec)
    H.x.scatter_forward()

    return H, V


def compute_boundary_normal_derivative(msh, facet_tags, H, x_src,
                                        target_tags=[1]):
    """
    Compute ∂G/∂n = ∂G_0/∂n + ∂H/∂n on specified boundary facets.

    Returns arrays of (midpoint coordinates, normal derivatives, facet lengths)
    for all facets with tags in target_tags.
    """
    tdim = msh.topology.dim
    fdim = tdim - 1

    # Get facets of interest
    target_facets = np.concatenate([facet_tags.find(tag) for tag in target_tags])

    if len(target_facets) == 0:
        return np.array([]), np.array([]), np.array([])

    # We need connectivity
    msh.topology.create_connectivity(fdim, tdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)

    midpoints = []
    normals = []
    dG_dn_values = []
    lengths = []

    # Get mesh geometry
    x_mesh = msh.geometry.x

    for facet_idx in target_facets:
        # Get the cell containing this facet
        cells = f_to_c.links(facet_idx)
        if len(cells) == 0:
            continue
        cell = cells[0]

        # Get facet vertices
        facet_entities = dolfinx.mesh.entities_to_geometry(
            msh, fdim, np.array([facet_idx]), False
        )
        verts = x_mesh[facet_entities[0]]

        # Midpoint
        mid = 0.5 * (verts[0] + verts[1])

        # Tangent and outward normal (2D: rotate tangent by -90°)
        tangent = verts[1] - verts[0]
        length = np.linalg.norm(tangent[:2])
        t_hat = tangent[:2] / length
        n_hat = np.array([t_hat[1], -t_hat[0]])  # outward normal (may need sign check)

        # Facet length
        lengths.append(length)
        midpoints.append(mid[:2])

        # ∂G_0/∂n at the midpoint
        gx, gy = grad_free_space_green_2d(mid[:2], x_src)
        dG0_dn = gx * n_hat[0] + gy * n_hat[1]

        # ∂H/∂n at the midpoint — evaluate gradient of FEM solution
        # Use cell-based evaluation
        point = mid.reshape(1, 3)

        # Create bounding box tree for point evaluation
        bb_tree = dolfinx.geometry.bb_tree(msh, tdim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, point)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(
            msh, cell_candidates, point
        )

        if len(colliding_cells.links(0)) > 0:
            eval_cell = colliding_cells.links(0)[0]

            # Evaluate gradient of H at midpoint
            # We need to compute ∇H, so we project gradient first
            # Simpler: use finite differences of H evaluation
            eps = length * 0.01
            p_plus = mid.copy()
            p_minus = mid.copy()
            # Shift slightly into the domain (along normal)
            p_eval = mid[:2] + n_hat * eps  # just inside domain

            p_x_plus = np.array([p_eval[0] + eps, p_eval[1], 0.0]).reshape(1, 3)
            p_x_minus = np.array([p_eval[0] - eps, p_eval[1], 0.0]).reshape(1, 3)
            p_y_plus = np.array([p_eval[0], p_eval[1] + eps, 0.0]).reshape(1, 3)
            p_y_minus = np.array([p_eval[0], p_eval[1] - eps, 0.0]).reshape(1, 3)

            try:
                H_xp = H.eval(p_x_plus, [eval_cell])[0]
                H_xm = H.eval(p_x_minus, [eval_cell])[0]
                H_yp = H.eval(p_y_plus, [eval_cell])[0]
                H_ym = H.eval(p_y_minus, [eval_cell])[0]

                dH_dx = (H_xp - H_xm) / (2 * eps)
                dH_dy = (H_yp - H_ym) / (2 * eps)
                dH_dn = dH_dx * n_hat[0] + dH_dy * n_hat[1]
            except Exception:
                dH_dn = 0.0
        else:
            dH_dn = 0.0

        dG_dn = dG0_dn + dH_dn
        dG_dn_values.append(dG_dn)
        normals.append(n_hat)

    return (np.array(midpoints), np.array(dG_dn_values),
            np.array(lengths), np.array(normals))


def compute_geometry_functional(msh, facet_tags, ion_pos,
                                 electrode_tag=1, outer_tag=2, gap_tag=3,
                                 fd_delta=None):
    """
    Compute G_x and G_y for the given mesh and ion position.

    Uses finite differences in the ion position to get K_j.
    """
    if fd_delta is None:
        # Default: 1% of ion height (for surface traps, d ≈ ion_pos[1])
        fd_delta = max(ion_pos[1] * 0.005, 0.1)

    print(f"  Ion position: ({ion_pos[0]:.4f}, {ion_pos[1]:.4f})")
    print(f"  FD delta: {fd_delta:.4f}")

    results = {}
    source_positions = {
        'center': ion_pos.copy(),
        'x_plus': ion_pos + np.array([fd_delta, 0.0]),
        'x_minus': ion_pos - np.array([fd_delta, 0.0]),
        'y_plus': ion_pos + np.array([0.0, fd_delta]),
        'y_minus': ion_pos - np.array([0.0, fd_delta]),
    }

    for name, x_src in source_positions.items():
        print(f"  Solving for source at {name}: ({x_src[0]:.4f}, {x_src[1]:.4f})")
        H, V = solve_regular_part(msh, facet_tags, x_src,
                                   electrode_tag, outer_tag, gap_tag)

        # Compute ∂G/∂n on electrode surfaces only
        midpts, dGdn, dl, normals = compute_boundary_normal_derivative(
            msh, facet_tags, H, x_src, target_tags=[electrode_tag]
        )
        results[name] = {
            'midpoints': midpts,
            'dGdn': dGdn,
            'dl': dl,
            'normals': normals,
        }

    # --- Finite difference for K_j ---
    dGdn_center = results['center']['dGdn']
    dl = results['center']['dl']

    # K_x(r') = ∂/∂x₀ [∂G/∂n'](r')
    K_x = (results['x_plus']['dGdn'] - results['x_minus']['dGdn']) / (2 * fd_delta)

    # K_y(r') = ∂/∂y₀ [∂G/∂n'](r')
    K_y = (results['y_plus']['dGdn'] - results['y_minus']['dGdn']) / (2 * fd_delta)

    # --- Integrate |K_j|² over electrode surfaces ---
    G_x = np.sum(K_x**2 * dl)
    G_y = np.sum(K_y**2 * dl)
    G_total = G_x + G_y

    print(f"\n  Results:")
    print(f"    G_x = {G_x:.6e}")
    print(f"    G_y = {G_y:.6e}")
    print(f"    G_total = {G_total:.6e}")
    print(f"    d (ion height) = {ion_pos[1]:.4f}")
    print(f"    G_y * d^3 = {G_y * ion_pos[1]**3:.6e} (should be ~ const for scaling)")

    return {
        'G_x': float(G_x),
        'G_y': float(G_y),
        'G_total': float(G_total),
        'ion_pos': ion_pos.tolist(),
        'midpoints': results['center']['midpoints'].tolist(),
        'K_x': K_x.tolist(),
        'K_y': K_y.tolist(),
        'dGdn_center': dGdn_center.tolist(),
        'dl': dl.tolist(),
    }


if __name__ == "__main__":
    param_file = sys.argv[1] if len(sys.argv) > 1 else "params.json"

    with open(param_file, 'r') as f:
        params = json.load(f)

    mesh_file = params.get('mesh_file', 'trap.msh')
    output_file = params.get('output_file', 'gj_result.json')

    # Load mesh
    print(f"Loading mesh from {mesh_file}...")
    msh, cell_tags, facet_tags = gmshio.read_from_msh(
        mesh_file, MPI.COMM_WORLD, gdim=2
    )
    print(f"  Mesh: {msh.topology.index_map(2).size_local} cells, "
          f"{msh.topology.index_map(0).size_local} vertices")

    # Ion position
    ion_pos = np.array(params.get('ion_pos', [0.0, 85.0]))

    # Compute G_j
    result = compute_geometry_functional(
        msh, facet_tags, ion_pos,
        electrode_tag=1, outer_tag=2, gap_tag=3,
    )

    # Add parameters to result
    result['params'] = params

    # Save
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {output_file}")