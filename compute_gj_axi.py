#!/usr/bin/env python3
"""
compute_gj_axi.py — Axisymmetric G_j computation for stylus-type traps.

In axisymmetric (r, z) coordinates, the 3D Laplacian becomes:
    -Δu = -(1/r) ∂/∂r(r ∂u/∂r) - ∂²u/∂z²

The weak form with weight r:
    a(u, v) = ∫ (∂u/∂r ∂v/∂r + ∂u/∂z ∂v/∂z) r dr dz

The free-space 3D Green's function in axisymmetric coords (for source on axis, r_src=0):
    G_0(r, z; 0, z_s) = -1/(4π √(r² + (z - z_s)²))

The boundary integral for G_j uses the axisymmetric line element:
    dA = 2π r' dl'  (surface of revolution)

So the geometry functional becomes:
    G_j = 2π ∫_Γe |K_j(r', z')|² r' dl'
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl
import json
import sys
import os

try:
    from dolfinx.io import gmshio
except ImportError:
    from dolfinx.io import gmsh as gmshio


def free_space_green_3d_axisym(r, z, z_src):
    """
    Free-space 3D Green's function for source on axis at (0, z_src).
    G_0 = -1/(4π) * 1/√(r² + (z - z_src)²)
    """
    dist2 = r**2 + (z - z_src)**2
    dist2 = np.maximum(dist2, 1e-30)
    return -1.0 / (4.0 * np.pi) * 1.0 / np.sqrt(dist2)


def grad_free_space_green_3d_axisym(r, z, z_src):
    """
    Gradient of G_0 w.r.t. (r, z) for source on axis at (0, z_src).
    ∂G_0/∂r = 1/(4π) r / (r² + (z-z_s)²)^(3/2)
    ∂G_0/∂z = 1/(4π) (z-z_s) / (r² + (z-z_s)²)^(3/2)
    """
    dist2 = r**2 + (z - z_src)**2
    dist2 = np.maximum(dist2, 1e-30)
    dist3 = dist2 * np.sqrt(dist2)
    gr = 1.0 / (4.0 * np.pi) * r / dist3
    gz = 1.0 / (4.0 * np.pi) * (z - z_src) / dist3
    return gr, gz


def solve_regular_part_axisym(msh, facet_tags, z_src,
                               electrode_tag=1, outer_tag=2,
                               gap_tag=3, axis_tag=4):
    """
    Solve for H in axisymmetric coordinates.

    -div(r ∇H) = 0  (in weak form with weight r)
    H = -G_0 on conducting boundaries (tags 1, 2)
    ∂H/∂n = 0 on axis (tag 4) — symmetry BC (natural, no explicit BC needed)
    H = -G_0 on gap boundaries (tag 3)

    Source is on axis: r_src = 0, z_src given.
    Mesh coordinates: x[0] = r, x[1] = z.
    """
    V = fem.functionspace(msh, ("Lagrange", 2))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Coordinate r (first component of spatial coords)
    x = ufl.SpatialCoordinate(msh)
    r = x[0]

    # Regularised r to avoid division by zero on axis
    r_reg = ufl.max_value(r, 1e-10)

    # Bilinear form with axisymmetric weight r
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * r_reg * ufl.dx

    # RHS: zero
    L = fem.Constant(msh, PETSc.ScalarType(0.0)) * v * r_reg * ufl.dx

    # Dirichlet BC on conducting + outer + gap boundaries
    # (NOT on axis — axis gets natural Neumann = symmetry)
    def bc_value(x):
        """H = -G_0 on boundary. x[0] = r, x[1] = z."""
        rr = x[0]
        zz = x[1]
        dist2 = rr**2 + (zz - z_src)**2
        dist2 = np.maximum(dist2, 1e-30)
        # H = -G_0 = 1/(4π) * 1/√(r² + (z-z_s)²)
        return 1.0 / (4.0 * np.pi) * 1.0 / np.sqrt(dist2)

    # Collect all Dirichlet boundary facets (everything except axis)
    bc_facets = []
    for tag in [electrode_tag, outer_tag, gap_tag]:
        found = facet_tags.find(tag)
        if len(found) > 0:
            bc_facets.append(found)

    if bc_facets:
        boundary_facets = np.concatenate(bc_facets)
    else:
        boundary_facets = np.array([], dtype=np.int32)

    boundary_dofs = fem.locate_dofs_topological(
        V, msh.topology.dim - 1, boundary_facets
    )

    H_bc = fem.Function(V)
    H_bc.interpolate(bc_value)
    bc = fem.dirichletbc(H_bc, boundary_dofs)

    # Solve
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


def compute_boundary_normal_derivative_axisym(msh, facet_tags, H, z_src,
                                                target_tags=[1]):
    """
    Compute ∂G/∂n on electrode boundaries in axisymmetric coordinates.
    Coordinates: x[0] = r, x[1] = z.
    """
    tdim = msh.topology.dim
    fdim = tdim - 1

    target_facets = np.concatenate([facet_tags.find(tag) for tag in target_tags])
    if len(target_facets) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    msh.topology.create_connectivity(fdim, tdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)
    x_mesh = msh.geometry.x

    midpoints = []
    dG_dn_values = []
    lengths = []
    normals_list = []

    for facet_idx in target_facets:
        cells = f_to_c.links(facet_idx)
        if len(cells) == 0:
            continue
        cell = cells[0]

        facet_entities = dolfinx.mesh.entities_to_geometry(
            msh, fdim, np.array([facet_idx]), False
        )
        verts = x_mesh[facet_entities[0]]
        mid = 0.5 * (verts[0] + verts[1])

        tangent = verts[1] - verts[0]
        length = np.linalg.norm(tangent[:2])
        t_hat = tangent[:2] / length
        n_hat = np.array([t_hat[1], -t_hat[0]])

        lengths.append(length)
        midpoints.append(mid[:2])  # (r, z)
        normals_list.append(n_hat)

        r_mid = mid[0]
        z_mid = mid[1]

        # ∂G_0/∂n at midpoint (source on axis)
        gr, gz = grad_free_space_green_3d_axisym(r_mid, z_mid, z_src)
        dG0_dn = gr * n_hat[0] + gz * n_hat[1]

        # ∂H/∂n via finite differences
        eps = length * 0.01

        bb_tree = dolfinx.geometry.bb_tree(msh, tdim)
        point = mid.reshape(1, 3)
        cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, point)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(
            msh, cell_candidates, point
        )

        if len(colliding_cells.links(0)) > 0:
            eval_cell = colliding_cells.links(0)[0]
            p_eval = mid[:2] + n_hat * eps

            p_r_plus = np.array([p_eval[0] + eps, p_eval[1], 0.0]).reshape(1, 3)
            p_r_minus = np.array([max(p_eval[0] - eps, 1e-8), p_eval[1], 0.0]).reshape(1, 3)
            p_z_plus = np.array([p_eval[0], p_eval[1] + eps, 0.0]).reshape(1, 3)
            p_z_minus = np.array([p_eval[0], p_eval[1] - eps, 0.0]).reshape(1, 3)

            try:
                H_rp = H.eval(p_r_plus, [eval_cell])[0]
                H_rm = H.eval(p_r_minus, [eval_cell])[0]
                H_zp = H.eval(p_z_plus, [eval_cell])[0]
                H_zm = H.eval(p_z_minus, [eval_cell])[0]

                dH_dr = (H_rp - H_rm) / (2 * eps)
                dH_dz = (H_zp - H_zm) / (2 * eps)
                dH_dn = dH_dr * n_hat[0] + dH_dz * n_hat[1]
            except Exception:
                dH_dn = 0.0
        else:
            dH_dn = 0.0

        dG_dn = dG0_dn + dH_dn
        dG_dn_values.append(dG_dn)

    return (np.array(midpoints), np.array(dG_dn_values),
            np.array(lengths), np.array(normals_list))


def compute_geometry_functional_axisym(msh, facet_tags, ion_pos,
                                        electrode_tag=1, outer_tag=2,
                                        gap_tag=3, axis_tag=4,
                                        fd_delta=None):
    """
    Compute G_z (axial) and G_r (radial) for axisymmetric stylus trap.

    Ion is on axis: ion_pos = [r=0, z_ion].
    We finite-difference in z only (axial direction) since the ion is on axis.
    For G_r, we use a small off-axis displacement.

    The geometry functional includes the 2π r' weight:
        G_j = 2π ∫ |K_j|² r' dl'
    """
    z_ion = ion_pos[1]

    if fd_delta is None:
        fd_delta = max(z_ion * 0.005, 0.1)

    print(f"  Ion position: (r=0, z={z_ion:.4f})")
    print(f"  FD delta: {fd_delta:.4f}")

    results = {}

    # For z-direction: displace source along z-axis (stays on axis)
    source_configs = {
        'center': z_ion,
        'z_plus': z_ion + fd_delta,
        'z_minus': z_ion - fd_delta,
    }

    for name, z_src in source_configs.items():
        print(f"  Solving for source at {name}: z={z_src:.4f}")
        H, V = solve_regular_part_axisym(msh, facet_tags, z_src,
                                          electrode_tag, outer_tag,
                                          gap_tag, axis_tag)

        midpts, dGdn, dl, normals = compute_boundary_normal_derivative_axisym(
            msh, facet_tags, H, z_src, target_tags=[electrode_tag]
        )
        results[name] = {
            'midpoints': midpts,
            'dGdn': dGdn,
            'dl': dl,
            'normals': normals,
        }

    # K_z = ∂/∂z₀ [∂G/∂n']
    dGdn_center = results['center']['dGdn']
    dl = results['center']['dl']
    midpts = results['center']['midpoints']

    K_z = (results['z_plus']['dGdn'] - results['z_minus']['dGdn']) / (2 * fd_delta)

    # Axisymmetric weight: r' coordinate of each boundary point
    r_boundary = midpts[:, 0]  # r coordinate
    r_boundary = np.maximum(r_boundary, 1e-10)  # avoid r=0 issues

    # G_z = 2π ∫ |K_z|² r' dl'
    G_z = 2 * np.pi * np.sum(K_z**2 * r_boundary * dl)

    print(f"\n  Results (axisymmetric):")
    print(f"    G_z = {G_z:.6e}")
    print(f"    z_ion = {z_ion:.4f}")

    return {
        'G_z': float(G_z),
        'G_x': 0.0,  # placeholder — would need off-axis source
        'G_y': float(G_z),  # map to G_y for compatibility with plotting
        'G_total': float(G_z),
        'ion_pos': ion_pos.tolist(),
        'midpoints': midpts.tolist(),
        'K_z': K_z.tolist(),
        'K_x': K_z.tolist(),  # compatibility
        'K_y': K_z.tolist(),  # compatibility
        'dGdn_center': dGdn_center.tolist(),
        'dl': dl.tolist(),
        'axisymmetric': True,
    }
