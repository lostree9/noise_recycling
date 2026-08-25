from pathlib import Path
import numpy as np
import pandas as pd
import many_ion_quantum_noise as mq
ROOT=Path(__file__).resolve().parent

def transverse_modes_uniform(N=10,beta=0.05):
    D=np.eye(N)
    for i in range(N):
        for j in range(N):
            if i==j: continue
            c=beta/abs(i-j)**3
            D[i,i]-=c
            D[i,j]+=c
    w2,V=np.linalg.eigh(D)
    o=np.argsort(w2)[::-1]
    return w2[o],V[:,o]

N=10; spacing=.6
rows=[]
for hdname,hd in [('open',np.inf),('h2',2.0)]:
    C=mq.covariance_matrix(N,spacing,hd,0,'normal')
    lam,Venv=mq.eigsorted(C)
    w2,Vmech=transverse_modes_uniform(N,.05)
    O=np.abs(Vmech.T@Venv)**2
    heat=np.diag(Vmech.T@C@Vmech)
    for mu in range(N):
        rows.append((hdname,mu+1,w2[mu],heat[mu]/np.trace(C),int(np.argmax(O[:,mu])+1),float(np.max(O[:,mu]))))
pd.DataFrame(rows,columns=['geometry','mechanical_mode','omega2_over_omega_t2','noise_weight_over_trace','best_environment_channel','squared_overlap']).to_csv(ROOT/'normal_mode_projection.csv',index=False)
print('written')
