"""Focused numerical utilities for the surface-noise transfer manuscript.
Extracted from the larger thesis-era production driver to keep the public repository minimal.
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy import stats as sps
from scipy import integrate, optimize, special

EPS = 1e-12
ION_Y = 0.35
FD = 0.02

def _check_geometry(d: float, h: float) -> None:
    if not (d > 0.0 and h > d):
        raise ValueError(f'require 0 < d < h, got d={d}, h={h}')

def _image_pair_sum_closed(d: float, h: float, power: int) -> float:
    _check_geometry(d, h)
    if power <= 2:
        raise ValueError('power must exceed 2 for convergence')
    q = d / h
    z = special.zeta
    bracket = q ** (-power) + 2.0 * z(power - 1, 1.0) + 2.0 * z(power, 1.0 + q) + z(power - 1, 2.0 - q) + (q - 1.0) * z(power, 2.0 - q) + z(power - 1, 2.0 + q) - (q + 1.0) * z(power, 2.0 + q)
    return float((2.0 * h) ** (-power) * bracket)

def exact_plate3d_Gz(d: float, h: float) -> float:
    return float(3.0 / math.pi * _image_pair_sum_closed(d, h, 4))

def exact_strip_kernel_closed(x, d: float, h: float) -> np.ndarray:
    _check_geometry(d, h)
    xx = np.atleast_1d(np.asarray(x, float))
    theta = math.pi * d / h
    u = math.pi * xx / h
    cu = np.cosh(u)
    c = math.cos(theta)
    return math.pi / (2.0 * h * h) * (1.0 - c * cu) / (cu - c) ** 2

@dataclass
class Edge2D:
    p0: np.ndarray
    p1: np.ndarray
    name: str
    ray: str = 'reflect'

@dataclass
class Panels2D:
    mid: np.ndarray
    tan: np.ndarray
    length: np.ndarray
    name: List[str] = field(default_factory=list)

def _pseg(p, a, b) -> float:
    ab = b - a
    den = float(ab @ ab)
    u = 0.0 if den < EPS else float(np.clip((p - a) @ ab / den, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + u * ab)))

def panelize2d(edges: List[Edge2D], h: float, ion: Optional[np.ndarray]=None, r_ref: float=1.5) -> Panels2D:
    h_fine = h / 3.0
    mids, tans, lens, names = ([], [], [], [])
    for e in edges:
        L = float(np.linalg.norm(e.p1 - e.p0))
        if L < EPS:
            continue
        t = (e.p1 - e.p0) / L
        size = h_fine if ion is not None and _pseg(ion, e.p0, e.p1) < r_ref else h
        n = max(1, int(math.ceil(L / size)))
        s = np.linspace(0.0, L, n + 1)
        for a, b in zip(s[:-1], s[1:]):
            mids.append(e.p0 + t * 0.5 * (a + b))
            tans.append(t)
            lens.append(b - a)
            names.append(e.name)
    return Panels2D(np.array(mids), np.array(tans), np.array(lens), names)

def bem2d_matrix(P: Panels2D, n_sub: int=8, near_fac: float=6.0) -> np.ndarray:
    mid, tan, ln = (P.mid, P.tan, P.length)
    N = len(ln)
    dx = mid[:, None, :] - mid[None, :, :]
    r = np.sqrt(np.sum(dx * dx, axis=2)) + np.eye(N)
    A = -np.log(r) / (2.0 * math.pi) * ln[None, :]
    near = (r < near_fac * np.maximum(ln[None, :], ln[:, None])) & ~np.eye(N, dtype=bool)
    ii, jj = np.where(near)
    if len(ii):
        u = (np.arange(n_sub) + 0.5) / n_sub - 0.5
        for i, j in zip(ii, jj):
            pts = mid[j] + np.outer(u * ln[j], tan[j])
            rr = np.linalg.norm(mid[i] - pts, axis=1)
            A[i, j] = float(np.sum(-np.log(np.maximum(rr, EPS))) / (2.0 * math.pi) * ln[j] / n_sub)
    np.fill_diagonal(A, ln * (1.0 - np.log(ln / 2.0)) / (2.0 * math.pi))
    return A

class Solver2D:
    def __init__(self, edges: List[Edge2D], panel_h: float=0.05, ion_hint: Optional[np.ndarray]=None):
        self.P = panelize2d(edges, panel_h, ion=ion_hint)
        self.lu = lu_factor(bem2d_matrix(self.P))
    def density(self, src) -> np.ndarray:
        r = np.linalg.norm(self.P.mid - np.asarray(src, float)[None, :], axis=1)
        return lu_solve(self.lu, np.log(np.maximum(r, EPS)) / (2.0 * math.pi))
    def kernels(self, ion, fd: float=FD) -> Tuple[np.ndarray, np.ndarray]:
        ion = np.asarray(ion, float)
        sx = (self.density(ion + [fd, 0]) - self.density(ion + [-fd, 0])) / (2 * fd)
        sy = (self.density(ion + [0, fd]) - self.density(ion + [0, -fd])) / (2 * fd)
        return (sx, sy)
    def active_mask(self, active: Sequence[str]) -> np.ndarray:
        return np.array([n in active for n in self.P.name])
    def G(self, ion, active: Sequence[str], fd: float=FD) -> Dict:
        sx, sy = self.kernels(ion, fd)
        act, ln = (self.active_mask(active), self.P.length)
        Gx = float(np.sum(ln[act] * sx[act] ** 2)); Gy = float(np.sum(ln[act] * sy[act] ** 2))
        return dict(Gx=Gx, Gy=Gy, Gtrace_xy=Gx + Gy, Axy=math.log((Gx + 1e-300) / (Gy + 1e-300)), n_panels=int(len(ln)))
    def C(self, ion1, ion2, active: Sequence[str], fd: float=FD) -> Dict:
        sx1, sy1 = self.kernels(ion1, fd); sx2, sy2 = self.kernels(ion2, fd)
        act, ln = (self.active_mask(active), self.P.length)
        return dict(Cxx=float(np.sum(ln[act] * sx1[act] * sx2[act])), Cyy=float(np.sum(ln[act] * sy1[act] * sy2[act])))

def bem2d_G(edges: List[Edge2D], active: Sequence[str], ion, panel_h: float=0.05, fd: float=FD) -> Dict:
    return Solver2D(edges, panel_h, ion_hint=np.asarray(ion, float)).G(ion, active, fd)

def _poly(pts, names, rays=None) -> List[Edge2D]:
    P = [np.array(p, float) for p in pts]
    rays = rays or ['reflect'] * len(pts)
    return [Edge2D(P[i], P[(i + 1) % len(P)], names[i], rays[i]) for i in range(len(P))]

def geom_strip(W: float, T: float, open_sides: bool=True) -> List[Edge2D]:
    esc = 'escape' if open_sides else 'reflect'
    return _poly([(-W, T), (W, T), (W, 0), (-W, 0)], ['cover', 'remote_right', 'plate', 'remote_left'], ['reflect', esc, 'reflect', esc])

def geom_slotted(W: float, T: float, s: float, D: float, blade_angle_deg: float=0.0, blade_depth: Optional[float]=None) -> List[Edge2D]:
    off = 0.0
    if blade_angle_deg > 0.0:
        D = blade_depth if blade_depth is not None else D
        off = D * math.tan(math.radians(blade_angle_deg))
    return _poly([(-W, T), (W, T), (W, 0), (s / 2, 0), (s / 2 + off, -D), (-s / 2 - off, -D), (-s / 2, 0), (-W, 0)], ['cover', 'remote_right', 'plate_right', 'slot_wall_right', 'slot_bottom', 'slot_wall_left', 'plate_left', 'remote_left'], ['reflect', 'escape', 'reflect', 'reflect', 'reflect', 'reflect', 'reflect', 'escape'])

ACTIVE = {'plates': {'plate_left', 'plate_right', 'plate'}, 'trench': {'plate_left', 'plate_right', 'slot_wall_left', 'slot_wall_right', 'slot_bottom'}, 'covered': {'plate_left', 'plate_right', 'slot_wall_left', 'slot_wall_right', 'slot_bottom', 'cover'}, 'blade': {'slot_wall_left', 'slot_wall_right', 'plate_left', 'plate_right'}, 'strip': {'plate'}}

@dataclass
class RaySegs:
    a: np.ndarray
    b: np.ndarray
    escape: np.ndarray
    name: List[str] = field(default_factory=list)

def ray_segs(edges: List[Edge2D]) -> RaySegs:
    return RaySegs(np.array([e.p0 for e in edges]), np.array([e.p1 for e in edges]), np.array([e.ray == 'escape' for e in edges]), [e.name for e in edges])

def _chord_point_dist(p: np.ndarray, q: np.ndarray, r0: np.ndarray) -> np.ndarray:
    v = q - p; den = np.einsum('ij,ij->i', v, v)
    u = np.where(den > EPS, np.einsum('ij,ij->i', r0[None, :] - p, v) / np.maximum(den, EPS), 0.0)
    proj = p + np.clip(u, 0.0, 1.0)[:, None] * v
    return np.linalg.norm(r0[None, :] - proj, axis=1)

def trace_gate_counts(edges: List[Edge2D], ion, rho: float, launch_x: np.ndarray, n_per_bin: int, u_bin: np.ndarray, u_ang: np.ndarray, launch_y: float=0.0, max_bounces: int=400) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    S = ray_segs(edges); nb = len(launch_x)
    x0 = np.repeat(np.asarray(launch_x, float), n_per_bin)
    th = np.arcsin(np.clip(2.0 * u_ang.ravel() - 1.0, -1.0, 1.0))
    pos = np.stack([x0, np.full_like(x0, launch_y + 1e-09)], axis=1)
    vel = np.stack([np.sin(th), np.cos(th)], axis=1)
    bin_id = np.repeat(np.arange(nb), n_per_bin)
    up, dn = (np.zeros(nb), np.zeros(nb)); alive = np.ones(len(pos), dtype=bool); bounces = np.zeros(len(pos), dtype=int)
    ion = np.asarray(ion, float); vseg = S.b - S.a
    for _ in range(max_bounces):
        if not alive.any(): break
        idx = np.where(alive)[0]; p, v = (pos[idx], vel[idx])
        den = v[:, 0:1] * vseg[None, :, 1] - v[:, 1:2] * vseg[None, :, 0]
        safe = np.where(np.abs(den) > 1e-14, den, 1.0)
        w = S.a[None, :, :] - p[:, None, :]
        t = (w[:, :, 0] * vseg[None, :, 1] - w[:, :, 1] * vseg[None, :, 0]) / safe
        uu = (w[:, :, 0] * v[:, 1:2] - w[:, :, 1] * v[:, 0:1]) / safe
        ok = (np.abs(den) > 1e-14) & (t > 1e-09) & (uu >= -1e-12) & (uu <= 1 + 1e-12)
        tt = np.where(ok, t, np.inf); jhit = np.argmin(tt, axis=1); thit = tt[np.arange(len(idx)), jhit]
        gone = ~np.isfinite(thit); q = p + np.where(gone, 1000000.0, thit)[:, None] * v
        hitg = _chord_point_dist(p, q, ion) < rho
        if hitg.any():
            gi, vy = (bin_id[idx[hitg]], v[hitg, 1]); np.add.at(up, gi[vy > 0], 1.0); np.add.at(dn, gi[vy <= 0], 1.0)
        die = gone | S.escape[jhit]; keep = ~die
        if keep.any():
            k, jk = (idx[keep], jhit[keep]); tk = vseg[jk]; tk = tk / np.linalg.norm(tk, axis=1)[:, None]
            nk = np.stack([-tk[:, 1], tk[:, 0]], axis=1); vk = v[keep]
            vr = vk - 2.0 * np.einsum('ij,ij->i', vk, nk)[:, None] * nk
            pos[k] = q[keep] + 1e-09 * vr; vel[k] = vr; bounces[k] += 1
        alive[idx[die]] = False
    mb = float(np.mean(bounces))
    return (up / n_per_bin, dn / n_per_bin, dict(mean_bounces=mb, escape_frac=float(np.mean(~alive)), gamma_eff=1.0 / max(mb + 1.0, EPS), N_eff=mb + 1.0))

def reconstruct_kernel(edges: List[Edge2D], d: float, rho: float, launch_x: np.ndarray, n_per_bin: int, delta: float, seed: int=0, max_bounces: int=400, launch_y: float=0.0, x_ion: float=0.0, sampling: str='random') -> Tuple[np.ndarray, Dict[str, float]]:
    rng = np.random.default_rng(seed); u_bin = np.zeros((len(launch_x), n_per_bin), dtype=float)
    if sampling == 'random': u_ang = rng.random((len(launch_x), n_per_bin))
    elif sampling == 'stratified':
        shift = rng.random((len(launch_x), 1)); u_ang = (np.arange(n_per_bin, dtype=float)[None, :] + shift) / n_per_bin; u_ang %= 1.0
    else: raise ValueError(f'unknown sampling mode: {sampling}')
    up_p, dn_p, gp = trace_gate_counts(edges, [x_ion, d + delta], rho, launch_x, n_per_bin, u_bin, u_ang, launch_y, max_bounces)
    up_m, dn_m, gm = trace_gate_counts(edges, [x_ion, d - delta], rho, launch_x, n_per_bin, u_bin, u_ang, launch_y, max_bounces)
    est = (up_p - up_m - (dn_p - dn_m)) / (2.0 * delta) / rho
    diag = {k: 0.5 * (gp[k] + gm[k]) for k in gp}
    return (est, diag)
