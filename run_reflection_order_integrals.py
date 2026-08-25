from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
h, eps, kappa, phase, W, d = 2.0, 0.15, 1.2, 0.8, 4.0, 1.0

def ytop(x):
    return h + eps*np.cos(kappa*x + phase)

def phi1(x):
    y = ytop(x)
    return np.sqrt(x*x+y*y) + np.sqrt(x*x+(y-d)**2)

# Source at (0,0), one cover reflection at x1, one lower-plane reflection at x2,
# then the ion at (0,d).  The integrals are deliberately left unweighted here:
# they test only the exponential actions selected by Laplace's method.
x = np.linspace(-3.0, 3.0, 601)
X1, X2 = np.meshgrid(x, x, indexing='ij')
Y1 = ytop(X1)
Phi1 = phi1(x)
Phi2 = (np.sqrt(X1**2 + Y1**2)
        + np.sqrt((X2-X1)**2 + Y1**2)
        + np.sqrt(X2**2 + d**2))
L1, L2 = float(Phi1.min()), float(Phi2.min())
rows = []
for q in np.arange(3.0, 8.5, 0.5):
    I1 = np.trapezoid(np.exp(-q*(Phi1-L1)), x)*np.exp(-q*L1)
    inner = np.trapezoid(np.exp(-q*(Phi2-L2)), x, axis=1)
    I2 = np.trapezoid(inner, x)*np.exp(-q*L2)
    rows.append((q, I1, I2))

pd.DataFrame(rows, columns=['q','I1','I2']).to_csv(ROOT/'billiard_reflection_integrals.csv', index=False)
print(f'L1/d={L1:.12f}, L2/d={L2:.12f}')
