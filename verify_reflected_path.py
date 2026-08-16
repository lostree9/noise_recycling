"""Independent numerical checks for Spatial transfer of surface noise in enclosed ion traps."""
import math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import k0,k1
from numpy.polynomial.legendre import leggauss
import surface_noise_tools as rp
root=Path(__file__).resolve().parent

def kh_closed(k,d,h):
    k=np.asarray(k,float); out=np.empty_like(k); small=k<1e-10; out[small]=1/h
    z=k[~small]; out[~small]=z*np.cosh((h-d)*z)/np.sinh(h*z); return out

def kh_images(k,d,h,M=200):
    k=np.asarray(k,float); n=np.arange(-M,M+1); a=np.abs(d-2*n*h)
    return k*np.exp(-k[...,None]*a).sum(axis=-1)

def A(k,d,h): return (1+np.exp(-2*(h-d)*k))/(1-np.exp(-2*h*k))
def W0(D,a,b):
    if D==1: return 2/(math.pi*(a+b)**3)
    if D==2: return 3/(math.pi*(a+b)**4)
    raise ValueError

def Mopen3(u):
    if u==0: return 1.0
    return (u*u/2+u**4/16)*k0(u)+(u+u**3/6)*k1(u)

ks=np.array([0.05,0.2,0.7,1.5,4.0]); d=1.; h=2.3
err=np.max(np.abs(kh_closed(ks,d,h)-kh_images(ks,d,h,400)))
print(f'image-depth identity max abs error = {err:.3e}'); assert err<1e-11
kg=np.linspace(0.02,10,5000); av=A(kg,d,h); maxdiff=np.max(np.diff(av))
print(f'max forward difference of A_h = {maxdiff:.3e}'); assert maxdiff<0
for D in [1,2]:
    a,b=0.8,2.1; s=a+b
    if D==1: val=(1/(2*math.pi))*2*quad(lambda p:p*p*math.exp(-s*p),0,np.inf)[0]
    else: val=(1/(2*math.pi)**2)*2*math.pi*quad(lambda p:p**3*math.exp(-s*p),0,np.inf)[0]
    er=abs(val-W0(D,a,b)); print(f'D={D} W_ab(0) error = {er:.3e}'); assert er<1e-12
z,w=leggauss(96); th=np.pi*(z+1); wt=np.pi*w; c=np.cos(th); pg=np.linspace(0,30,1800)
def kopen(r): return r*np.exp(-r)
def kslab(r,hd):
    r=np.asarray(r,float); out=np.empty_like(r); sm=r<1e-10; out[sm]=1/hd
    x=r[~sm]; out[~sm]=x*np.exp(-x)*(1+np.exp(-2*(hd-1)*x))/(-np.expm1(-2*hd*x)); return out
def H2(q,hd=None):
    kp=kopen(pg) if hd is None else kslab(pg,hd)
    rr=np.sqrt(np.maximum(q*q+pg[:,None]**2-2*q*pg[:,None]*c[None,:],0))
    kr=kopen(rr) if hd is None else kslab(rr,hd)
    ang=np.sum(kr*wt[None,:],axis=1)
    return np.trapezoid(kp*ang*pg,pg)/(2*np.pi)**2
G0=3/(16*np.pi)
for q in [1.,2.,4.,6.]:
    num=H2(q,None)/G0; ex=Mopen3(q); er=abs(num-ex)
    print(f'open D=2 q={q:g}: numerical={num:.10f}, Bessel={ex:.10f}, error={er:.2e}'); assert er<2e-6
for hd in [1.5,2.0,3.0]:
    vals=[H2(q,hd)/H2(q,None) for q in [6.,8.,10.]]
    print(f'h/d={hd:g} finite/open ratios q=6,8,10: '+', '.join(f'{v:.6f}' for v in vals)); assert abs(vals[-1]-1)<abs(vals[0]-1)
summary={}
for line in (root/'ray_transfer_summary.txt').read_text().splitlines():
    k,v=line.split('='); summary[k]=float(v)
print('ray certificate:',summary); assert summary['r']>0.999; assert abs(summary['scale']-1)<0.01; assert abs(summary['qhalf_relerr'])<0.01
import csv
rows=list(csv.DictReader(open(root/'bem_gain_contrast.csv')))
def get(panel,mode,hd): return next(r for r in rows if abs(float(r['panel_h'])-panel)<1e-9 and r['mode']==mode and abs(float(r['h_over_d'])-hd)<1e-6)
reductions=[]
for panel in [0.07,0.05,0.035]:
    far=get(panel,'fixed_noisy_surface',13.1428571429); close=get(panel,'fixed_noisy_surface',2.0)
    gain=float(close['Gy'])/float(far['Gy']); tail=float(close['sigma10_over_sigma1_L2'])/float(far['sigma10_over_sigma1_L2']); er=float(close['entropy_effective_rank_L2'])/float(far['entropy_effective_rank_L2'])
    reductions.append(1-tail); print(f'BEM panel={panel:.3f}: gain={gain:.6f}, sigma10 ratio={tail:.6f}, r_eff ratio={er:.6f}'); assert gain>3.3 and tail<0.7 and er<0.93
assert max(reductions)-min(reductions)<0.01
far=get(0.035,'all_enclosure_surfaces_noisy',13.1428571429); close=get(0.035,'all_enclosure_surfaces_noisy',2.0)
control=float(close['sigma10_over_sigma1_L2'])/float(far['sigma10_over_sigma1_L2']); print(f'BEM all-surfaces-noisy control sigma10 ratio={control:.6f}'); assert control>3.0
print('ALL CHECKS PASSED')
