from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor
import screened_bem_billiards as sb

ROOT=Path(__file__).resolve().parent


def plate_kernels(mesh,q,ions,fd=.008,order=8):
    A_lu=lu_factor(sb.matrix_screened(mesh,q,order=order))
    ids=np.where(mesh.name=='plate')[0]
    K=[]
    for ion in ions:
        ion=np.asarray(ion,float)
        sp=sb.boundary_density(mesh,q,ion+[0,fd],A_lu,order)
        sm=sb.boundary_density(mesh,q,ion-[0,fd],A_lu,order)
        K.append(((sp-sm)/(2*fd))[ids])
    return np.asarray(K),mesh.length[ids],mesh.mid[ids,0]


def covariance(mesh,q,ions):
    K,w,x=plate_kernels(mesh,q,ions)
    C=(K*w[None,:])@K.T
    return 0.5*(C+C.T)


def eig(C):
    lam,V=np.linalg.eigh(C)
    o=np.argsort(lam)[::-1]; lam=lam[o]; V=V[:,o]
    for j in range(V.shape[1]):
        if V[:,j].sum()<0: V[:,j]*=-1
    return lam,V


def main():
    xs=np.linspace(-1.2,1.2,5)
    ions=np.c_[xs,np.ones_like(xs)]
    q=.8
    flat,_=sb.curved_box(W=5,h=2.2,target=.10,ncurve=120)
    curved,yfun=sb.curved_box(W=5,h=2.2,target=.10,ncurve=120,bump=(-.75,1.1,.45))
    Cf=covariance(flat,q,ions); Cc=covariance(curved,q,ions)
    lf,Vf=eig(Cf); lc,Vc=eig(Cc)

    rows=[]
    for geom,C,l,V in [('flat',Cf,lf,Vf),('depressed',Cc,lc,Vc)]:
        for i in range(len(xs)):
            for j in range(len(xs)):
                rows.append((geom,i,j,xs[i],xs[j],C[i,j]))
    pd.DataFrame(rows,columns=['geometry','i','j','x_i_over_d','x_j_over_d','Cij']).to_csv(ROOT/'curved_multi_ion_covariance.csv',index=False)
    pd.DataFrame({'ion':np.arange(5),'x_over_d':xs,'lead_flat':Vf[:,0],'lead_depressed':Vc[:,0]}).to_csv(ROOT/'curved_multi_ion_leading_channel.csv',index=False)

    fig,axs=plt.subplots(1,3,figsize=(7.2,2.55),gridspec_kw={'width_ratios':[1,1,1.08]})
    for ax,C,title in [(axs[0],Cf,'(a) Flat cover'),(axs[1],Cc,'(b) Off-axis depression')]:
        R=C/np.sqrt(np.outer(np.diag(C),np.diag(C)))
        im=ax.imshow(R,vmin=0,vmax=1,origin='lower',extent=[xs[0],xs[-1],xs[0],xs[-1]],aspect='auto')
        ax.set_xlabel(r'$x_j/d$'); ax.set_ylabel(r'$x_i/d$')
        ax.set_title(title,loc='left',fontsize=8.8,fontweight='bold')
    ax=axs[2]
    ax.plot(xs,Vf[:,0],'o-',label='flat')
    ax.plot(xs,Vc[:,0],'s-',label='depressed')
    ax.set_xlabel(r'probe position $x_i/d$')
    ax.set_ylabel(r'leading channel $V_{i1}$')
    ax.set_title(r'(c) Leading channel at $q_zd=0.8$',loc='left',fontsize=8.3,fontweight='bold')
    ax.legend(frameon=False,fontsize=7)
    ax.text(.03,.05,fr'$r_{{\rm eff}}$: {np.trace(Cf)**2/np.trace(Cf@Cf):.3f} $\to$ {np.trace(Cc)**2/np.trace(Cc@Cc):.3f}',transform=ax.transAxes,fontsize=7)
    fig.subplots_adjust(left=.075,right=.92,bottom=.20,top=.88,wspace=.40)
    cax=fig.add_axes([.935,.24,.012,.56])
    fig.colorbar(im,cax=cax,label='normalized covariance')
    fig.savefig(ROOT/'fig_curved_multi_ion_noise.pdf')
    fig.savefig(ROOT/'fig_curved_multi_ion_noise.png',dpi=240)
    plt.close(fig)
    print('written')

if __name__=='__main__': main()
