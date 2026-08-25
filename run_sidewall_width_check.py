from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import screened_bem_billiards as sb

qvals=np.array([2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0])
d=1.0; target=.11; fd=.01
geoms={
    'bump_offaxis':dict(h=2.70,eps=0.0,kappa=1.0,phase=0.0,bump=(-0.90,1.2,0.35)),
    'wave_b':dict(h=2.15,eps=0.18,kappa=1.3,phase=0.7,bump=None),
}
rows=[]
for W in [4.0,5.0,7.0]:
    far,_=sb.curved_box(W=W,h=7.0,target=target,ncurve=100)
    farK={q:sb.Ky_at_source(far,q,ion=(0,d),fd=fd,order=8)[0] for q in qvals}
    for name,kw in geoms.items():
        mesh,yfun=sb.curved_box(W=W,target=target,ncurve=100,**kw)
        L,xstar=sb.one_bounce_length(yfun,d=d,W=W)
        corr=[]
        for q in qvals:
            kh,_=sb.Ky_at_source(mesh,q,ion=(0,d),fd=fd,order=8)
            corr.append(abs(kh/farK[q]-1.0))
        slope,_=sb.fit_exponent(qvals,corr)
        rows.append(dict(W_over_d=W,name=name,deltaL=L-d,xstar=xstar,slope=slope,
                         rel_error=(slope-(L-d))/(L-d),mincorr=min(corr),maxcorr=max(corr)))
        print(W,name,L-d,xstar,slope,flush=True)
pd.DataFrame(rows).to_csv(ROOT/'sidewall_width_check.csv',index=False)
print('wrote sidewall_width_check.csv')
