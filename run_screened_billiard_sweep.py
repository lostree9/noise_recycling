from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import quad

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import screened_bem_billiards as sb

W=5.0; d=1.0; target=0.11; fd=0.01
qvals=np.array([2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0])
GEOMS=[
 ('flat_h1.70',dict(h=1.70,eps=0.0,kappa=1.0,phase=0.0,bump=None)),
 ('flat_h1.85',dict(h=1.85,eps=0.0,kappa=1.0,phase=0.0,bump=None)),
 ('flat_h2.00',dict(h=2.00,eps=0.0,kappa=1.0,phase=0.0,bump=None)),
 ('flat_h2.20',dict(h=2.20,eps=0.0,kappa=1.0,phase=0.0,bump=None)),
 ('flat_h2.50',dict(h=2.50,eps=0.0,kappa=1.0,phase=0.0,bump=None)),
 ('cos_m0.15',dict(h=2.00,eps=-0.15,kappa=1.0,phase=0.0,bump=None)),
 ('cos_m0.075',dict(h=2.00,eps=-0.075,kappa=1.0,phase=0.0,bump=None)),
 ('cos_p0.075',dict(h=2.00,eps=0.075,kappa=1.0,phase=0.0,bump=None)),
 ('cos_p0.15',dict(h=2.00,eps=0.15,kappa=1.0,phase=0.0,bump=None)),
 ('wave_a',dict(h=2.15,eps=0.18,kappa=0.8,phase=0.4,bump=None)),
 ('wave_b',dict(h=2.15,eps=0.18,kappa=1.3,phase=0.7,bump=None)),
 ('wave_c',dict(h=2.20,eps=0.20,kappa=1.8,phase=0.9,bump=None)),
 ('wave_d',dict(h=2.10,eps=0.15,kappa=2.3,phase=0.5,bump=None)),
 ('bump_left',dict(h=2.50,eps=0.0,kappa=1.0,phase=0.0,bump=(-0.55,0.8,0.45))),
 ('bump_offaxis',dict(h=2.70,eps=0.0,kappa=1.0,phase=0.0,bump=(-0.90,1.2,0.35))),
 ('bump_up',dict(h=2.40,eps=0.0,kappa=1.0,phase=0.0,bump=(0.25,-0.7,0.50))),
]

def exact_partial_flat(q,h):
    def base(kx,hh):
        k=math.hypot(kx,q)
        if hh is None:
            return k*math.exp(-d*k)
        return k*math.exp(-d*k)*(1+math.exp(-2*(hh-d)*k))/(-math.expm1(-2*hh*k))
    num=quad(lambda x:base(x,h),0,np.inf,epsabs=1e-12,epsrel=1e-11,limit=300)[0]
    den=quad(lambda x:base(x,None),0,np.inf,epsabs=1e-12,epsrel=1e-11,limit=300)[0]
    return num/den-1

far,_=sb.curved_box(W=W,h=7.0,target=target,ncurve=100)
farK={q:sb.Ky_at_source(far,q,ion=(0,d),fd=fd,order=8)[0] for q in qvals}
rows=[]; curves=[]
for name,kw in GEOMS:
    mesh,yfun=sb.curved_box(W=W,target=target,ncurve=100,**kw)
    L,xstar=sb.one_bounce_length(yfun,d=d,W=W)
    delta=L-d
    corr=[]
    for q in qvals:
        kh,_=sb.Ky_at_source(mesh,q,ion=(0,d),fd=fd,order=8)
        c=abs(kh/farK[q]-1.0); corr.append(c)
        curves.append(dict(name=name,qd=q,correction=c))
    slope,_=sb.fit_exponent(qvals,corr)
    rows.append(dict(name=name,deltaL=delta,xstar=xstar,slope=slope,
                     rel_error=(slope-delta)/delta,mincorr=min(corr),maxcorr=max(corr)))
    print(name,delta,xstar,slope,(slope-delta)/delta,flush=True)

out=pd.DataFrame(rows)
out.to_csv(ROOT/'screened_bem_sweep_v2.csv',index=False)
pd.DataFrame(curves).to_csv(ROOT/'screened_bem_curves_v2.csv',index=False)

mesh,_=sb.curved_box(W=W,h=2.0,target=target,ncurve=100)
valid=[]
for q in qvals:
    kh,_=sb.Ky_at_source(mesh,q,ion=(0,d),fd=fd,order=8)
    bem=abs(kh/farK[q]-1.0)
    ex=exact_partial_flat(q,2.0)
    valid.append(dict(qd=q,bem_correction=bem,exact_correction=ex,relative_error=(bem-ex)/ex))
pd.DataFrame(valid).to_csv(ROOT/'screened_bem_flat_validation.csv',index=False)

print('pearson',out.deltaL.corr(out.slope))
print('spearman',out.deltaL.corr(out.slope,method='spearman'))
print('mae',np.mean(abs(out.slope-out.deltaL)))
print('medrel',np.median(abs(out.rel_error)))
print('flat amp max rel',max(abs(pd.DataFrame(valid).relative_error)))
