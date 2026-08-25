from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.special import j0, jv

ROOT = Path(__file__).resolve().parent


def potential_normal(k, hd=np.inf):
    k = np.asarray(k, float)
    if np.isinf(hd):
        return np.exp(-k)
    h = float(hd)
    out = np.empty_like(k)
    small = np.abs(k) < 1e-12
    out[small] = 1.0 / h
    z = k[~small]
    out[~small] = np.exp(-z) * (1.0 + np.exp(-2.0*(h-1.0)*z)) / (-np.expm1(-2.0*h*z))
    return out


def potential_tangent(k, hd=np.inf):
    k = np.asarray(k, float)
    if np.isinf(hd):
        return np.exp(-k)
    h = float(hd)
    out = np.empty_like(k)
    small = np.abs(k) < 1e-12
    out[small] = (h-1.0)/h
    z = k[~small]
    out[~small] = np.exp(-z) * (1.0 - np.exp(-2.0*(h-1.0)*z)) / (-np.expm1(-2.0*h*z))
    return out


def spatial_psd(k, xi=0.0):
    return np.exp(-0.5*(xi*k)**2)


_KGRID = np.linspace(0.0, 45.0, 5000)
_BESSEL_CACHE = {}
_COV_CACHE = {}

def _bessel_arrays(N, spacing):
    key=(int(N),float(spacing))
    if key not in _BESSEL_CACHE:
        rs=np.arange(N,dtype=float)*spacing
        z=rs[:,None]*_KGRID[None,:]
        _BESSEL_CACHE[key]=(j0(z), jv(2,z))
    return _BESSEL_CACHE[key]


def covariance_matrix(N=10, spacing=0.6, hd=np.inf, xi=0.0, component='normal'):
    key=(int(N),float(spacing),float(hd) if np.isfinite(hd) else 'inf',float(xi),component)
    if key in _COV_CACHE:
        return _COV_CACHE[key].copy()
    k=_KGRID
    J0,J2=_bessel_arrays(N,spacing)
    P=spatial_psd(k,xi)
    if component=='normal':
        K=k*potential_normal(k,hd)
        base=k*P*K*K/(2*np.pi)
        vals=np.trapezoid(J0*base[None,:],k,axis=1)
    elif component=='x':
        pot=potential_tangent(k,hd)
        base=k**3*P*pot*pot/(4*np.pi)
        vals=np.trapezoid((J0-J2)*base[None,:],k,axis=1)
    else:
        raise ValueError(component)
    idx=np.abs(np.arange(N)[:,None]-np.arange(N)[None,:])
    C=vals[idx]
    _COV_CACHE[key]=C
    return C.copy()

def participation_rank(C):
    return float(np.trace(C)**2 / np.trace(C@C))


def entropy_rank(C):
    lam = np.linalg.eigvalsh((C+C.T)/2)
    lam = np.clip(lam, 0, None)
    p = lam/lam.sum()
    p = p[p>0]
    return float(np.exp(-np.sum(p*np.log(p))))


def eigsorted(C):
    lam, V = np.linalg.eigh((C+C.T)/2)
    order=np.argsort(lam)[::-1]
    lam=lam[order]; V=V[:,order]
    for j in range(V.shape[1]):
        if V[:,j].sum() < 0: V[:,j] *= -1
    return lam,V


def mode_exposure(C, b):
    b=np.asarray(b,float); b=b/np.linalg.norm(b)
    return float(b@C@b)


def make_figure():
    N=10; a=0.6
    Copen=covariance_matrix(N,a,np.inf,0,'normal')
    C2=covariance_matrix(N,a,2.0,0,'normal')
    l0,V0=eigsorted(Copen); l2,V2=eigsorted(C2)

    fig,axs=plt.subplots(2,2,figsize=(7.2,5.5))
    ax=axs[0,0]
    im=ax.imshow(Copen/np.sqrt(np.outer(np.diag(Copen),np.diag(Copen))),vmin=-0.15,vmax=1,aspect='auto')
    ax.set_title('(a) Open normalized covariance',loc='left',fontsize=9,fontweight='bold')
    ax.set_xlabel('ion $j$'); ax.set_ylabel('ion $i$')
    fig.colorbar(im,ax=ax,fraction=.047,pad=.03,label=r'$C_{ij}/\sqrt{C_{ii}C_{jj}}$')

    ax=axs[0,1]
    im=ax.imshow(C2/np.sqrt(np.outer(np.diag(C2),np.diag(C2))),vmin=-0.15,vmax=1,aspect='auto')
    ax.set_title(r'(b) Passive cover $h/d=2$',loc='left',fontsize=9,fontweight='bold')
    ax.set_xlabel('ion $j$'); ax.set_ylabel('ion $i$')
    fig.colorbar(im,ax=ax,fraction=.047,pad=.03,label=r'$C_{ij}/\sqrt{C_{ii}C_{jj}}$')

    ax=axs[1,0]
    j=np.arange(1,N+1)
    ax.semilogy(j,l0/l0.sum(),'o-',label='open')
    ax.semilogy(j,l2/l2.sum(),'s-',label=r'$h/d=2$')
    ax.set_xlabel(r'collective channel index $\mu$')
    ax.set_ylabel(r'$\lambda_\mu/\mathrm{Tr}\,C$')
    ax.set_title('(c) Collective noise-channel spectrum',loc='left',fontsize=9,fontweight='bold')
    ax.legend(frameon=False,fontsize=8)
    ins=ax.inset_axes([0.54,0.48,0.42,0.42])
    ii=np.arange(1,N+1)
    ins.plot(ii,V0[:,0],'o-',ms=2.5,lw=1,label='open')
    ins.plot(ii,V2[:,0],'s-',ms=2.5,lw=1,label=r'$h/d=2$')
    ins.set_xticks([1,5,10]); ins.set_yticks([]); # inset identity is given in the manuscript caption
    ins.tick_params(labelsize=6)

    ax=axs[1,1]
    hs=np.array([1.25,1.4,1.6,2,2.5,3,4,6,10,30])
    for xi,marker in [(0.0,'o'),(0.5,'s'),(1.0,'^')]:
        rr=[]
        for h in hs:
            C=covariance_matrix(N,a,h,xi,'normal')
            rr.append(participation_rank(C))
        ax.plot(hs,rr,marker=marker,label=fr'$\xi/d={xi:g}$')
    ax.axhline(participation_rank(Copen),linestyle='--',linewidth=1,label='open, local')
    ax.set_xscale('log')
    ax.set_xlabel(r'cover height $h/d$')
    ax.set_ylabel(r'effective noise rank $r_{\rm eff}$')
    ax.set_title('(d) Geometry and correlation complexity',loc='left',fontsize=9,fontweight='bold')
    ax.legend(frameon=False,fontsize=7)

    fig.tight_layout()
    fig.savefig(ROOT/'fig_many_ion_quantum_noise.pdf')
    fig.savefig(ROOT/'fig_many_ion_quantum_noise.png',dpi=240)
    plt.close(fig)

    # Leading covariance eigenchannel profiles, as a small companion figure/data.
    df=[]
    for name,C in [('open',Copen),('h2',C2)]:
        lam,V=eigsorted(C)
        for i in range(N):
            df.append((name,i,lam[0],V[i,0]))
    pd.DataFrame(df,columns=['geometry','ion','lambda1','V_i1']).to_csv(ROOT/'many_ion_leading_jump.csv',index=False)


def make_gate_exposure():
    Rvals=np.linspace(0.2,3.0,50)
    rows=[]
    for R in Rvals:
        for comp in ['normal','x']:
            for hdname,hd in [('open',np.inf),('h2',2.0)]:
                C=covariance_matrix(2,R,hd,0,comp)
                plus=mode_exposure(C,[1,1])
                minus=mode_exposure(C,[1,-1])
                rows.append((R,comp,hdname,plus,minus))
    df=pd.DataFrame(rows,columns=['R_over_d','component','geometry','common','differential'])
    df.to_csv(ROOT/'gate_heating_exposure.csv',index=False)

    fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
    for ax,comp,title in zip(axs,['normal','x'],['normal-field bus','tangential-field bus']):
        op=df[(df.component==comp)&(df.geometry=='open')].sort_values('R_over_d')
        h2=df[(df.component==comp)&(df.geometry=='h2')].sort_values('R_over_d')
        ax.plot(op.R_over_d,h2.common.values/op.common.values,label='common bus')
        ax.plot(op.R_over_d,h2.differential.values/op.differential.values,label='differential bus')
        ax.axhline(1,linewidth=1)
        ax.set_xlabel(r'ion separation $R/d$')
        ax.set_ylabel(r'enclosed/open heating exposure')
        ax.set_title(title,fontsize=9)
        ax.legend(frameon=False,fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT/'fig_gate_heating_exposure.pdf')
    fig.savefig(ROOT/'fig_gate_heating_exposure.png',dpi=240)
    plt.close(fig)


def verification_table():
    rows=[]
    N=10; a=.6
    for xi in [0,0.5,1.0]:
        for hd in [np.inf,3,2,1.5]:
            Cy=covariance_matrix(N,a,hd,xi,'normal')
            Cx=covariance_matrix(N,a,hd,xi,'x')
            rows.append((xi,hd,participation_rank(Cy),entropy_rank(Cy),participation_rank(Cx),entropy_rank(Cx)))
    pd.DataFrame(rows,columns=['xi_over_d','h_over_d','rank_y','entropy_rank_y','rank_x','entropy_rank_x']).to_csv(ROOT/'many_ion_rank_table.csv',index=False)

    # Numerical Loewner checks for several finite arrays/source spectra.
    checks=[]
    for N in [3,5,10,16]:
        for a in [0.25,0.6,1.0]:
            for xi in [0,0.5,1.0]:
                for h in [1.3,1.5,2,3,6]:
                    yo=covariance_matrix(N,a,np.inf,xi,'normal')
                    yh=covariance_matrix(N,a,h,xi,'normal')
                    xo=covariance_matrix(N,a,np.inf,xi,'x')
                    xh=covariance_matrix(N,a,h,xi,'x')
                    my=float(np.linalg.eigvalsh(yh-yo).min())
                    mx=float(np.linalg.eigvalsh(xo-xh).min())
                    checks.append((N,a,xi,h,my,mx))
    pd.DataFrame(checks,columns=['N','spacing_over_d','xi_over_d','h_over_d','min_eig_normal_difference','min_eig_tangent_difference']).to_csv(ROOT/'loewner_numerical_checks.csv',index=False)


if __name__=='__main__':
    make_figure()
    make_gate_exposure()
    verification_table()
    print('many-ion quantum-noise outputs written')
