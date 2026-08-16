import numpy as np, time, csv
from pathlib import Path
import surface_noise_tools as rp
root=Path(__file__).resolve().parent
d=rp.ION_Y; h=2*d; rho=0.15; delta=0.10*d
xs=np.linspace(-1.4,1.4,31)
edges=rp.geom_strip(6.0,h)
exact=-np.pi*rp.exact_strip_kernel_closed(xs,d,h)
ests=[]; t0=time.time()
for seed in range(5):
    est,diag=rp.reconstruct_kernel(edges,d,rho,xs,8000,delta,seed=seed,max_bounces=400,sampling='stratified')
    ests.append(est); print(seed,diag,'elapsed',time.time()-t0,flush=True)
mean=np.mean(ests,axis=0); sem=np.std(ests,axis=0,ddof=1)/np.sqrt(len(ests))
scale=float(np.dot(mean,exact)/np.dot(exact,exact)); r=float(np.corrcoef(mean,exact)[0,1]); rmse=float(np.linalg.norm(mean-exact)/np.linalg.norm(exact))
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
fine=np.linspace(xs.min(),xs.max(),4001)
km=CubicSpline(xs,mean)(fine)/(-np.pi); ke=rp.exact_strip_kernel_closed(fine,d,h)
Hm=km*km; He=ke*ke
def M(q,H): return np.trapezoid(H*np.cos(q*fine),fine)/np.trapezoid(H,fine)
def qhalf(H): return brentq(lambda q:M(q,H)-0.5,0.01,30)
qm=qhalf(Hm); qe=qhalf(He)
print('r',r,'scale',scale,'rmse',rmse,'qhalf exact',qe*d,'ray',qm*d,'relerr',(qm-qe)/qe)
rows=[]
for u in [0,1,2,3,4,5,6]:
    q=u/d; rows.append((u,M(q,He),M(q,Hm))); print('u',u,'exact',rows[-1][1],'ray',rows[-1][2])
with open(root/'ray_transfer.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['x','exact_sum_phi','ray_sum_phi','ray_sem']); w.writerows(zip(xs,exact,mean,sem))
with open(root/'ray_transfer_spectrum.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['qd','M_exact','M_ray']); w.writerows(rows)
with open(root/'ray_transfer_summary.txt','w') as f:
    f.write(f'r={r:.8f}\nscale={scale:.8f}\nrel_rmse={rmse:.8f}\nqhalf_exact_d={qe*d:.8f}\nqhalf_ray_d={qm*d:.8f}\nqhalf_relerr={(qm-qe)/qe:.8f}\n')
