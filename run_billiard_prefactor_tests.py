from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import screened_bem_billiards as sb
W=5.; d=1.

def lpp(yfun,x,step=2e-4):
    def L(z):
        y=yfun(z); return math.hypot(z,y)+math.hypot(z,y-d)
    return (L(x+step)-2*L(x)+L(x-step))/step**2

# Focusing family: centered saddle, same L=3 => delta L=2, vary curvature.
q=5.0; far,_=sb.curved_box(W=W,h=7,target=.10,ncurve=100); kref=sb.Ky_at_source(far,q,ion=(0,d),fd=.008)[0]
rows=[]
for eps in np.linspace(-.2,.2,9):
    h=2.0-eps
    mesh,y=sb.curved_box(W=W,h=h,eps=float(eps),kappa=1.0,phase=0,target=.10,ncurve=100)
    L,x=sb.one_bounce_length(y,d=d,W=W); cur=lpp(y,x)
    kval=sb.Ky_at_source(mesh,q,ion=(0,d),fd=.008)[0]
    corr=abs(kval/kref-1)
    rows.append((eps,h,L-d,x,cur,1/math.sqrt(cur),corr,corr*math.exp(q*(L-d))))
focus=pd.DataFrame(rows,columns=['eps','h','deltaL','xstar','Lpp','focus','corr','scaled_amp'])
focus.to_csv(ROOT/'billiard_focusing_family_v2.csv',index=False)

# Shape derivative.
qvals=np.array([2.5,3.5,4.5,5.5,6.5]); far,_=sb.curved_box(W=W,h=7,target=.09,ncurve=110)
refs={qq:sb.Ky_at_source(far,qq,ion=(0,d),fd=.008)[0] for qq in qvals}
rows=[]
for qq in qvals:
    cc=[]
    for eps in [-.015,.015]:
        mesh,_=sb.curved_box(W=W,h=2,eps=eps,kappa=1,target=.09,ncurve=110)
        kval=sb.Ky_at_source(mesh,qq,ion=(0,d),fd=.008)[0]
        cc.append(abs(kval/refs[qq]-1))
    deriv=(math.log(cc[1])-math.log(cc[0]))/.03
    rows.append((qq,deriv,-2*qq))
shape=pd.DataFrame(rows,columns=['qd','measured','predicted']); shape.to_csv(ROOT/'billiard_shape_derivative_v2.csv',index=False)

# Multiple local minima utility.
def local_minima(yfun):
    def L(x):
        y=yfun(x); return math.hypot(x,y)+math.hypot(x,y-d)
    xs=np.linspace(-W,W,4001); vals=np.array([L(x) for x in xs]); out=[]
    for i in range(1,len(xs)-1):
        if vals[i]<=vals[i-1] and vals[i]<=vals[i+1]:
            r=minimize_scalar(L,bounds=(xs[i-1],xs[i+1]),method='bounded',options={'xatol':1e-12})
            x=r.x; hh=2e-4; cur=(L(x+hh)-2*L(x)+L(x-hh))/hh**2
            out.append((r.fun,x,cur))
    ded=[]
    for item in sorted(out):
        if not any(abs(item[1]-j[1])<1e-4 for j in ded): ded.append(item)
    return ded

q=4.5; far,_=sb.curved_box(W=W,h=7,target=.10,ncurve=100); kref=sb.Ky_at_source(far,q,ion=(0,d),fd=.008)[0]
rows=[]
for phase in np.linspace(0,.9,10):
    mesh,y=sb.curved_box(W=W,h=2,eps=.4,kappa=2.5,phase=float(phase),target=.10,ncurve=120)
    mins=local_minima(y); Lmin=mins[0][0]; near=[m for m in mins if m[0]-Lmin<1.2]
    kval=sb.Ky_at_source(mesh,q,ion=(0,d),fd=.008)[0]
    corr=abs(kval/kref-1); amp=corr*math.exp(q*(Lmin-d))
    saddle=sum(math.exp(-q*(L-Lmin))/math.sqrt(cur) for L,x,cur in near)
    rows.append((phase,Lmin-d,amp,saddle,len(near),near[1][0]-Lmin if len(near)>1 else np.nan))
multi=pd.DataFrame(rows,columns=['phase','deltaL_min','scaled_amp','saddle_sum','n_saddles','second_gap']); multi.to_csv(ROOT/'billiard_multiple_saddle_v2.csv',index=False)

print('focus corr',focus.focus.corr(focus.scaled_amp))
print('focus R2',focus.focus.corr(focus.scaled_amp)**2)
print('shape max rel',max(abs((shape.measured-shape.predicted)/shape.predicted)))
print('multi corr',multi.saddle_sum.corr(multi.scaled_amp))
