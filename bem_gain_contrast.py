"""Quadrature-consistent BEM stress test for the surface-noise response operator.

The continuum operator is

    (T s)(x) = integral_{Gamma_e} H(x,p) s(p) d ell_p,
    H(x,p) = |K_y(p;x)|^2,

viewed as T: L^2(Gamma_e,d ell) -> L^2(X,dx), with X the fixed scan
interval. In orthonormal piecewise-constant input coordinates and trapezoidal
output quadrature,

    B_ip = sqrt(w_i) H(x_i,p) sqrt(Delta ell_p).
"""
from pathlib import Path
import csv
import numpy as np
import surface_noise_tools as rp

root = Path(__file__).resolve().parent

def trapezoid_weights(x):
    x = np.asarray(x, float)
    if len(x) < 2: return np.ones_like(x)
    w = np.empty_like(x); w[0] = 0.5*(x[1]-x[0]); w[-1] = 0.5*(x[-1]-x[-2])
    if len(x) > 2: w[1:-1] = 0.5*(x[2:]-x[:-2])
    return w

def metrics(T, active, panel_h=0.05):
    d = rp.ION_Y; edges = rp.geom_slotted(4.0, T, 1.0, 1.0)
    xscan = np.linspace(-2*d, 2*d, 41); wx = trapezoid_weights(xscan)
    sol = rp.Solver2D(edges, panel_h, ion_hint=np.array([0.0, d]))
    act = np.array([n in active for n in sol.P.name]); lens = np.asarray(sol.P.length[act], float)
    H=[]
    for x0 in xscan:
        _,ky = sol.kernels(np.array([x0,d]), rp.FD); H.append(np.abs(ky[act])**2)
    H=np.asarray(H,float)
    B=np.sqrt(wx)[:,None]*H*np.sqrt(lens)[None,:]
    s=np.linalg.svd(B,compute_uv=False); psv=s*s; psv/=psv.sum()
    er=float(np.exp(-np.sum(psv[psv>0]*np.log(psv[psv>0])))); tail10=float(s[9]/s[0]) if len(s)>9 else float('nan')
    G=rp.bem2d_G(edges,active,np.array([0.0,d]),panel_h=panel_h)['Gy']
    return G,er,tail10,len(s),int(act.sum()),float(lens.min()),float(lens.max())

rows=[]; d=rp.ION_Y
panel_sizes=[0.07,0.05,0.035]
cover_ratios=[13.1428571429,8.5714285714,5.7142857143,4.2857142857,3.2857142857,2.5714285714,2.0]
for panel in panel_sizes:
    for hd in cover_ratios:
        T=hd*d
        for mode,active in [('fixed_noisy_surface',{'plate_left','plate_right'}),('all_enclosure_surfaces_noisy',rp.ACTIVE['covered'])]:
            G,er,t10,nsv,npan,lmin,lmax=metrics(T,active,panel)
            rows.append([panel,hd,mode,G,er,t10,nsv,npan,lmin,lmax])
            print(f"panel={panel:.3f} h/d={hd:7.3f} {mode:30s} G={G:.8g} r_eff={er:.6f} s10/s1={t10:.8g} panels={npan} dl=[{lmin:.4g},{lmax:.4g}]")
with open(root/'bem_gain_contrast.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['panel_h','h_over_d','mode','Gy','entropy_effective_rank_L2','sigma10_over_sigma1_L2','n_singular','n_active_panels','panel_length_min','panel_length_max']); w.writerows(rows)
