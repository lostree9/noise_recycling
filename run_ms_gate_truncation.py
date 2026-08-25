from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from ms_gate_heating_channel import reduced_qubit_channel

ROOT=Path(__file__).resolve().parent
# Same three weak-diffusion points used for the production coefficient.
nus=np.array([1e-4,2e-4,5e-4])
rows=[]
for nph in range(5,11):
    F0=reduced_qubit_channel(nph=nph,heating_dimless=0.0)
    vals=[]
    for nu in nus:
        F=reduced_qubit_channel(nph=nph,heating_dimless=nu/(2*np.pi))
        vals.append((1-F)-(1-F0))
    slope,intercept=np.polyfit(nus,np.asarray(vals),1)
    rows.append((nph,F0,slope,intercept))
    print(nph,F0,slope,intercept,flush=True)
pd.DataFrame(rows,columns=['nph','F0','slope','intercept']).to_csv(ROOT/'ms_gate_truncation_convergence.csv',index=False)
