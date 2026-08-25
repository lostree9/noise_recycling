"""Billiard/path representation of anomalous trapped-ion heating.

This script reproduces the exact slab results used in the research note:
  * vertical and tangential field-response multipliers,
  * local-white anomalous-heating geometry factors,
  * exact h/d=2 zeta/eta identities,
  * finite-correlation-length (Gaussian) source filtering,
  * two-ion vertical-field cross correlations.

Conventions follow a two-dimensional noisy boundary plane (physical 3D slab).
Overall source PSD and e^2/(4 m hbar omega) factors are omitted from geometry-only
ratios.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import j0, zeta

ROOT = Path(__file__).resolve().parent


def kperp(k: float, d: float, h: float | None) -> float:
    """Vertical-field Fourier multiplier K_y(k)."""
    if h is None:
        return k * math.exp(-d * k)
    if k < 1e-10 / max(d, h):
        return 1.0 / h
    return k * math.exp(-d * k) * (1.0 + math.exp(-2.0 * (h-d) * k)) / (-math.expm1(-2.0*h*k))


def potential_ratio(k: float, d: float, h: float | None) -> float:
    """Potential multiplier P(k) at ion height; K_x = i k_x P."""
    if h is None:
        return math.exp(-d * k)
    if k < 1e-10 / max(d, h):
        return (h-d) / h
    return math.exp(-d*k) * (1.0 - math.exp(-2.0*(h-d)*k)) / (-math.expm1(-2.0*h*k))


def G_perp(d: float, h: float | None) -> float:
    # D=2 radial measure: d^2k/(2pi)^2 -> k dk/(2pi)
    return quad(lambda k: k * kperp(k,d,h)**2 / (2.0*math.pi), 0, np.inf,
                epsabs=1e-11, epsrel=2e-10, limit=400)[0]


def G_tangent(d: float, h: float | None) -> float:
    # angular average k_x^2 -> k^2/2
    return quad(lambda k: k**3 * potential_ratio(k,d,h)**2 / (4.0*math.pi), 0, np.inf,
                epsabs=1e-11, epsrel=2e-10, limit=400)[0]


def G_gaussian_perp(d: float, h: float | None, xi: float) -> float:
    # source spatial spectrum proportional to exp[-(xi k)^2/2]; amplitude cancels in ratios
    return quad(lambda k: k * kperp(k,d,h)**2 * math.exp(-0.5*(xi*k)**2)/(2.0*math.pi),
                0, np.inf, epsabs=1e-11, epsrel=3e-10, limit=400)[0]


def beta_eff_fixed_h(d: float, h: float, relstep: float=2e-4) -> float:
    # beta_eff = - d ln G / d ln d at fixed physical h
    dp=d*math.exp(relstep); dm=d*math.exp(-relstep)
    if dp >= h:
        dp = d*(1+0.5*(h/d-1))
    gp=G_perp(dp,h); gm=G_perp(dm,h)
    return -(math.log(gp)-math.log(gm))/(math.log(dp)-math.log(dm))


def cross_perp_white(R: float, d: float, h: float | None) -> float:
    # S_12 = int k dk/(2pi) K_y(k)^2 J0(kR)
    return quad(lambda k: k * kperp(k,d,h)**2 * j0(k*R)/(2.0*math.pi),
                0, np.inf, epsabs=2e-10, epsrel=2e-8, limit=800)[0]


def write_mode_table() -> None:
    gpo=G_perp(1.0,None); gto=G_tangent(1.0,None)
    rows=[]
    for hd in [1.10,1.20,1.50,2.00,3.00,6.00,10.00]:
        gp=G_perp(1.0,hd); gt=G_tangent(1.0,hd)
        rows.append([hd,gp/gpo,gt/gto,gp/gt,beta_eff_fixed_h(1.0,hd)])
    with open(ROOT/'billiard_heating_mode_table.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['h_over_d','Gy_over_open','Gx_over_open','Gy_over_Gx','beta_eff_fixed_h']); w.writerows(rows)


def write_corr_length_table() -> None:
    rows=[]; d=1.0; h=2.0
    for xid in [0.0,0.05,0.10,0.20,0.50,1.0,2.0,5.0,10.0]:
        xi=max(xid*d,1e-9)
        ratio=G_gaussian_perp(d,h,xi)/G_gaussian_perp(d,None,xi)
        rows.append([xid,ratio])
    with open(ROOT/'billiard_heating_correlation_length.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['xi_over_d','enclosed_over_open_Gy']); w.writerows(rows)


def write_cross_table() -> None:
    rows=[]; d=1.0
    s0_open=cross_perp_white(0,d,None); s0_h2=cross_perp_white(0,d,2*d)
    for Rd in [0.0,0.5,1.0,1.5,2.0,3.0,4.0]:
        co=cross_perp_white(Rd*d,d,None)/s0_open
        ce=cross_perp_white(Rd*d,d,2*d)/s0_h2
        rows.append([Rd,co,ce,1+co,1-co,1+ce,1-ce])
    with open(ROOT/'billiard_heating_two_ion.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['R_over_d','corr_open','corr_h_over_d_2','COM_factor_open','stretch_factor_open','COM_factor_h2','stretch_factor_h2']); w.writerows(rows)


def exact_checks() -> None:
    z3=float(zeta(3.0,1.0)); eta3=0.75*z3
    gp_ratio=G_perp(1,2)/G_perp(1,None)
    gt_ratio=G_tangent(1,2)/G_tangent(1,None)
    assert abs(gp_ratio-z3) < 2e-10, (gp_ratio,z3)
    assert abs(gt_ratio-eta3) < 2e-10, (gt_ratio,eta3)
    print(f'h/d=2: Gy/Gy_open = {gp_ratio:.12f} = zeta(3)')
    print(f'h/d=2: Gx/Gx_open = {gt_ratio:.12f} = eta(3)=3 zeta(3)/4')
    print(f'h/d=2: Gy/Gx = {G_perp(1,2)/G_tangent(1,2):.12f} = 8/3')


if __name__ == '__main__':
    exact_checks(); write_mode_table(); write_corr_length_table(); write_cross_table()
    print('wrote billiard-heating CSV tables')
