from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.special import k0
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

ROOT=Path(__file__).resolve().parent

@dataclass
class PanelMesh:
    a: np.ndarray
    b: np.ndarray
    mid: np.ndarray
    length: np.ndarray
    name: np.ndarray


def polyline_panels(points, names, target=0.10):
    aa=[]; bb=[]; nn=[]
    for p0,p1,name in zip(points[:-1], points[1:], names):
        p0=np.asarray(p0,float); p1=np.asarray(p1,float)
        L=float(np.linalg.norm(p1-p0)); n=max(1,int(math.ceil(L/target)))
        # The response is sampled at x=0 on the noisy lower plate.  Force a
        # collocation midpoint there so resolution sweeps do not also move the
        # sampled boundary point.
        if name == 'plate' and p0[0] < 0.0 < p1[0] and n % 2 == 0:
            n += 1
        for i in range(n):
            a=p0+(p1-p0)*i/n; b=p0+(p1-p0)*(i+1)/n
            aa.append(a); bb.append(b); nn.append(name)
    aa=np.asarray(aa); bb=np.asarray(bb); mid=(aa+bb)/2; length=np.linalg.norm(bb-aa,axis=1)
    return PanelMesh(aa,bb,mid,length,np.asarray(nn))


def curved_box(W=5.0,h=2.0,eps=0.0,kappa=1.0,phase=0.0,target=0.10,ncurve=120, bump=None):
    # counter-clockwise: bottom L->R, right side up, cover R->L, left side down
    pts=[]; names=[]
    pts.append((-W,0.0)); pts.append((W,0.0)); names.append('plate')
    # right wall
    def ycover(x):
        y=h+eps*math.cos(kappa*x+phase)
        if bump is not None:
            amp,xc,sig=bump
            y += amp*math.exp(-0.5*((x-xc)/sig)**2)
        return y
    pts.append((W,ycover(W))); names.append('right')
    xs=np.linspace(W,-W,ncurve+1)[1:]
    for x in xs:
        pts.append((x,ycover(x))); names.append('cover')
    pts.append((-W,0.0)); names.append('left')
    return polyline_panels(pts,names,target=target), ycover


def gauss_panel_nodes(mesh:PanelMesh, order=8):
    z,w=np.polynomial.legendre.leggauss(order)
    # nodes shape (N,order,2); weights include ds
    ctr=(mesh.a+mesh.b)/2; half=(mesh.b-mesh.a)/2
    nodes=ctr[:,None,:]+z[None,:,None]*half[:,None,:]
    weights=(mesh.length[:,None]/2)*w[None,:]
    return nodes,weights


def matrix_screened(mesh:PanelMesh,q:float,order=8):
    nodes,weights=gauss_panel_nodes(mesh,order)
    N=len(mesh.length)
    # evaluate each target midpoint against each panel quadrature nodes
    # (N,N,order)
    diff=mesh.mid[:,None,None,:]-nodes[None,:,:,:]
    r=np.linalg.norm(diff,axis=3)
    # collocation midpoint is never one of even-order GL nodes, so r>0 incl self panels
    G=k0(q*np.maximum(r,1e-14))/(2*math.pi)
    A=np.sum(G*weights[None,:,:],axis=2)
    # Replace the logarithmically singular self-panel quadrature by an
    # adaptive one-dimensional integral.  With s=(L/2)t^2 the endpoint
    # logarithm is integrable and the transformed integrand is regular.
    for j,L in enumerate(mesh.length):
        def self_integrand(t):
            if t == 0.0:
                return 0.0
            ss=0.5*L*t*t
            return (L*t)*k0(q*ss)/math.pi
        A[j,j]=quad(self_integrand,0.0,1.0,epsabs=1e-12,epsrel=1e-11,limit=100)[0]
    return A


def boundary_density(mesh:PanelMesh,q:float,src,A_lu=None,order=8):
    if A_lu is None:
        A_lu=lu_factor(matrix_screened(mesh,q,order=order))
    r=np.linalg.norm(mesh.mid-np.asarray(src,float)[None,:],axis=1)
    rhs=-k0(q*np.maximum(r,1e-14))/(2*math.pi)
    return lu_solve(A_lu,rhs)


def Ky_at_source(mesh:PanelMesh,q:float,ion=(0.0,1.0),fd=0.01,xsource=0.0,order=8):
    A_lu=lu_factor(matrix_screened(mesh,q,order=order))
    ion=np.asarray(ion,float)
    sp=boundary_density(mesh,q,ion+[0,fd],A_lu,order)
    sm=boundary_density(mesh,q,ion-[0,fd],A_lu,order)
    ky=(sp-sm)/(2*fd)
    ids=np.where(mesh.name=='plate')[0]
    i=ids[np.argmin(np.abs(mesh.mid[ids,0]-xsource))]
    return float(ky[i]), float(mesh.mid[i,0])


def one_bounce_length(ycover,d=1.0,W=5.0,xsource=0.0,xion=0.0):
    def L(x):
        y=ycover(x)
        return math.hypot(x-xsource,y)+math.hypot(x-xion,y-d)
    # coarse seed then bounded local minimizations around candidate minima
    xs=np.linspace(-W,W,2001); vals=np.array([L(x) for x in xs])
    candidates=[]
    for i in range(1,len(xs)-1):
        if vals[i]<=vals[i-1] and vals[i]<=vals[i+1]:
            lo=xs[i-1]; hi=xs[i+1]
            res=minimize_scalar(L,bounds=(lo,hi),method='bounded',options={'xatol':1e-12})
            candidates.append((res.fun,res.x))
    if not candidates:
        i=int(np.argmin(vals)); return vals[i],xs[i]
    return min(candidates)


def fit_exponent(qs,corr):
    qs=np.asarray(qs); corr=np.asarray(corr)
    mask=(corr>0)&np.isfinite(corr)
    x=qs[mask]; y=np.log(corr[mask])
    # high-q last ~half
    cut=max(0,len(x)//2)
    coef=np.polyfit(x[cut:],y[cut:],1)
    return -coef[0],coef[1]


def demo_family():
    d=1.0; W=5.0; qvals=np.array([2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0])
    geometries=[
        ('flat',dict(h=2.0,eps=0.0,kappa=1.0,phase=0.0,bump=None)),
        ('cos_down',dict(h=2.0,eps=-0.10,kappa=1.0,phase=0.0,bump=None)),
        ('cos_up05',dict(h=2.0,eps=0.05,kappa=1.0,phase=0.0,bump=None)),
        ('cos_up15',dict(h=2.0,eps=0.15,kappa=1.0,phase=0.0,bump=None)),
        ('phase',dict(h=2.0,eps=0.15,kappa=1.2,phase=0.8,bump=None)),
        ('bump_offaxis',dict(h=2.3,eps=0.0,kappa=1.0,phase=0.0,bump=(-0.55,1.15,0.45))),
    ]
    # far-cover reference with identical finite side width
    far,_=curved_box(W=W,h=7.0,target=0.085,ncurve=100)
    rows=[]
    for name,kw in geometries:
        mesh,yfun=curved_box(W=W,target=0.085,ncurve=100,**kw)
        L,xstar=one_bounce_length(yfun,d=d,W=W)
        delta=L-d
        cs=[]
        for q in qvals:
            kc,_=Ky_at_source(mesh,q,ion=(0,d),fd=0.008,order=8)
            ko,_=Ky_at_source(far,q,ion=(0,d),fd=0.008,order=8)
            cs.append(abs(kc/ko-1.0))
        slope,_=fit_exponent(qvals,cs)
        rows.append((name,delta,xstar,slope,*cs))
    return rows

if __name__=='__main__':
    # lightweight flat sanity test only
    d=1.; W=5.
    flat,yf=curved_box(W=W,h=2,target=.11,ncurve=80)
    far,_=curved_box(W=W,h=7,target=.11,ncurve=80)
    for q in [2.5,3.5,4.5,5.5]:
        kf,_=Ky_at_source(flat,q,ion=(0,d),fd=.01,order=8)
        ko,_=Ky_at_source(far,q,ion=(0,d),fd=.01,order=8)
        print(q,kf,ko,abs(kf/ko-1))
