import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import k0,k1
from numpy.polynomial.legendre import leggauss
import surface_noise_tools as rp
root=Path(__file__).resolve().parent

# 2D strip transfer
x=np.linspace(-25,25,100001); u=np.linspace(0,8,161)
fig,ax=plt.subplots(figsize=(5.5,3.0)); ax.plot(u,np.exp(-u)*(1+u+u**3/3),label='open')
for hd in [1.5,2,3,6]:
    K=rp.exact_strip_kernel_closed(x,1.0,hd); H=K*K; norm=np.trapezoid(H,x)
    vals=[np.trapezoid(H*np.cos(q*x),x)/norm for q in u]; ax.plot(u,vals,label=fr'$h/d={hd:g}$')
ax.set_xlabel(r'spatial wavenumber $qd$'); ax.set_ylabel(r'normalized heating transfer $M_h(q)$'); ax.set_ylim(-0.03,1.03); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(root/'fig_transfer_2d.pdf'); plt.close(fig)

# 3D slab transfer by radial convolution
zg,wg=leggauss(96); th=np.pi*(zg+1); wt=np.pi*wg; cth=np.cos(th); pgrid=np.linspace(0,30,1800)
def kh_open(r): return np.asarray(r)*np.exp(-np.asarray(r))
def kh_slab(r,hd):
    r=np.asarray(r,float); out=np.empty_like(r); small=r<1e-9; out[small]=1/hd
    z=r[~small]; out[~small]=z*np.exp(-z)*(1+np.exp(-2*(hd-1)*z))/(-np.expm1(-2*hd*z)); return out
def H2(q,hd=None):
    kp=kh_open(pgrid) if hd is None else kh_slab(pgrid,hd)
    rr=np.sqrt(np.maximum(q*q+pgrid[:,None]**2-2*q*pgrid[:,None]*cth[None,:],0)); kr=kh_open(rr) if hd is None else kh_slab(rr,hd)
    return np.trapezoid(kp*np.sum(kr*wt[None,:],axis=1)*pgrid,pgrid)/(2*np.pi)**2
def M3_open(v):
    v=np.asarray(v,float); out=np.ones_like(v); m=v>0; z=v[m]; out[m]=(z*z/2+z**4/16)*k0(z)+(z+z**3/6)*k1(z); return out
u3=np.linspace(0,8,81); fig,ax=plt.subplots(figsize=(5.5,3.0)); ax.plot(u3,M3_open(u3),label='open')
for hd in [1.5,2,3,6]:
    G=rp.exact_plate3d_Gz(1.0,hd); vals=[1.0]+[H2(q,hd)/G for q in u3[1:]]; ax.plot(u3,vals,label=fr'$h/d={hd:g}$')
ax.set_xlabel(r'spatial wavenumber $qd$'); ax.set_ylabel(r'normalized 3D heating transfer $M_h(q)$'); ax.set_ylim(-0.03,1.03); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(root/'fig_transfer_3d.pdf'); plt.close(fig)

# Ray transfer
rdata=np.genfromtxt(root/'ray_transfer_spectrum.csv',delimiter=',',names=True); K=rp.exact_strip_kernel_closed(x,1.0,2.0); H=K*K; norm=np.trapezoid(H,x); ue=np.linspace(0,6.3,250); Me=[np.trapezoid(H*np.cos(q*x),x)/norm for q in ue]
fig,ax=plt.subplots(figsize=(5.5,3.0)); ax.plot(ue,Me,label='exact strip'); ax.plot(rdata['qd'],rdata['M_ray'],'o',label='ray reconstruction'); ax.axhline(0.5,linewidth=1); ax.set_xlabel(r'spatial wavenumber $qd$'); ax.set_ylabel(r'$M_h(q)$ at $h/d=2$'); ax.set_ylim(0,1.03); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(root/'fig_ray_transfer.pdf'); plt.close(fig)

# BEM figures
b=np.genfromtxt(root/'bem_gain_contrast.csv',delimiter=',',names=True,dtype=None,encoding='utf-8'); panels=sorted(set(b['panel_h'])); series=[]
for ph in panels:
    ss=b[np.isclose(b['panel_h'],ph)]; fx=np.sort(ss[ss['mode']=='fixed_noisy_surface'],order='h_over_d'); series.append((ph,fx,fx['Gy']/fx['Gy'][-1],fx['sigma10_over_sigma1_L2']/fx['sigma10_over_sigma1_L2'][-1]))
hd=series[0][1]['h_over_d']; Gstack=np.vstack([z[2] for z in series]); Tstack=np.vstack([z[3] for z in series]); fig,ax=plt.subplots(figsize=(5.5,3.0)); ax.plot(hd,Gstack.mean(axis=0),'o-',label=r'total coupling $G/G_{\rm far}$'); ax.fill_between(hd,Gstack.min(axis=0),Gstack.max(axis=0),alpha=0.2); ax.plot(hd,Tstack.mean(axis=0),'s-',label=r'normalized $\sigma_{10}/\sigma_1$'); ax.fill_between(hd,Tstack.min(axis=0),Tstack.max(axis=0),alpha=0.2); ax.invert_xaxis(); ax.set_xlabel(r'cover height $h/d$ (closing $\rightarrow$)'); ax.set_ylabel('ratio to far-cover value'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(root/'fig_bem_fixed_surface.pdf'); plt.close(fig)
fine=min(panels); ss=b[np.isclose(b['panel_h'],fine)]; fixed=np.sort(ss[ss['mode']=='fixed_noisy_surface'],order='h_over_d'); alln=np.sort(ss[ss['mode']=='all_enclosure_surfaces_noisy'],order='h_over_d'); fig,ax=plt.subplots(figsize=(5.5,3.0)); ax.plot(fixed['h_over_d'],fixed['sigma10_over_sigma1_L2']/fixed['sigma10_over_sigma1_L2'][-1],'o-',label='fixed noisy plate'); ax.plot(alln['h_over_d'],alln['sigma10_over_sigma1_L2']/alln['sigma10_over_sigma1_L2'][-1],'s-',label='added enclosure surfaces also noisy'); ax.invert_xaxis(); ax.set_xlabel(r'cover height $h/d$ (closing $\rightarrow$)'); ax.set_ylabel(r'normalized $\sigma_{10}/\sigma_1$'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(root/'fig_bem_control.pdf'); plt.close(fig)
print('figures written')
