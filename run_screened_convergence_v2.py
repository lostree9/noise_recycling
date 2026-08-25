import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import screened_bem_billiards as sb
q=np.array([2.5,3,3.5,4,4.5,5,5.5,6.0]); W=5.; d=1.
GEOMS={
 'flat':dict(h=2.0,eps=0,kappa=1,phase=0,bump=None),
 'offaxis':dict(h=2.7,eps=0,kappa=1,phase=0,bump=(-.9,1.2,.35)),
}
rows=[]
for target,fd in [(0.14,.01),(0.11,.01),(0.085,.01),(0.11,.015),(0.11,.007)]:
 far,_=sb.curved_box(W=W,h=7,target=target,ncurve=100)
 fK={qq:sb.Ky_at_source(far,qq,ion=(0,d),fd=fd)[0] for qq in q}
 for name,kw in GEOMS.items():
  mesh,y=sb.curved_box(W=W,target=target,ncurve=100,**kw)
  L,x=sb.one_bounce_length(y,d=d,W=W)
  corr=[]
  for qq in q:
   kh,_=sb.Ky_at_source(mesh,qq,ion=(0,d),fd=fd)
   corr.append(abs(kh/fK[qq]-1))
  slope,_=sb.fit_exponent(q,corr)
  rows.append((target,fd,name,L-d,slope,min(corr),max(corr)))
  print(rows[-1],flush=True)
pd.DataFrame(rows,columns=['target','fd','name','pred','slope','mincorr','maxcorr']).to_csv(ROOT/'screened_bem_convergence_v2.csv',index=False)
