from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent

def kron(*ops):
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Iq = kron(I2, I2)
XX = kron(X, X)
Sx = kron(X, I2) + kron(I2, X)

def annih(n):
    a = np.zeros((n, n), complex)
    for j in range(1, n):
        a[j - 1, j] = np.sqrt(j)
    return a

def reduced_qubit_channel(nph=9, heating_dimless=0.0, rtol=2e-8, atol=2e-10):
    """Primitive single-loop MS gate under classical motional diffusion.

    Units: delta=1, gate time T=2*pi, g/delta=1/4.  ``heating_dimless``
    is dot(nbar)/delta.  Classical symmetric electric-field noise gives the
    motional master equation dot(nbar) [D[a] + D[a^dagger]], for which the
    mean occupation increases at exactly dot(nbar) in an infinite oscillator.
    """
    a = annih(nph)
    ad = a.conj().T
    Aph = kron(Iq, a)
    Ad = kron(Iq, ad)
    S = kron(Sx, np.eye(nph))
    T = 2 * np.pi
    g = 0.25
    Ddim = 4 * nph
    n_op = Ad @ Aph
    aa = Aph @ Ad

    def rhs(t, y):
        rho = y.reshape((Ddim, Ddim))
        H = g * S @ (Aph * np.exp(-1j * t) + Ad * np.exp(1j * t))
        dr = -1j * (H @ rho - rho @ H)
        if heating_dimless:
            # Equal upward/downward rates: classical force-noise diffusion.
            dr += heating_dimless * (
                Aph @ rho @ Ad - 0.5 * (n_op @ rho + rho @ n_op)
                + Ad @ rho @ Aph - 0.5 * (aa @ rho + rho @ aa)
            )
        return dr.ravel()

    p0 = np.zeros((nph, nph), complex)
    p0[0, 0] = 1
    E = np.empty((4, 4), dtype=object)
    for i in range(4):
        for j in range(4):
            op = np.zeros((4, 4), complex)
            op[i, j] = 1
            rho0 = np.kron(op, p0)
            sol = solve_ivp(rhs, (0, T), rho0.ravel(), rtol=rtol, atol=atol, method="DOP853")
            rho = sol.y[:, -1].reshape((Ddim, Ddim)).reshape(4, nph, 4, nph)
            E[i, j] = np.einsum("iaja->ij", rho)

    # The closed single-loop trajectory implements exp(+i*pi/4 X1 X2), up to
    # a global phase, at g/delta=1/4.
    U = expm(1j * np.pi / 4 * XX)
    Fe = 0j
    for i in range(4):
        for j in range(4):
            M = U.conj().T @ E[i, j] @ U
            Fe += M[i, j]
    Fe = (Fe / 16).real
    Favg = (4 * Fe + 1) / 5
    return float(Favg)

def main():
    # Mean quanta added during one gate: nu = dot(nbar) * T.
    nus = np.array([0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
    rows = []
    for nu in nus:
        heating = nu / (2 * np.pi)
        F = reduced_qubit_channel(nph=7, heating_dimless=heating)
        rows.append((nu, F, 1 - F))
        print(nu, F, 1 - F, flush=True)
    df = pd.DataFrame(rows, columns=["quanta_per_gate", "F_avg", "infidelity"])
    df.to_csv(ROOT / "ms_gate_heating_channel.csv", index=False)

    # Differential weak-heating coefficient after removing the finite-Fock
    # no-noise truncation floor.  Fit the three smallest nonzero points.
    floor = float(df.infidelity.iloc[0])
    x = df.quanta_per_gate.values[1:4]
    y = df.infidelity.values[1:4] - floor
    m, b = np.polyfit(x, y, 1)
    with open(ROOT / "ms_gate_heating_summary.txt", "w") as f:
        f.write(f"weak_heating_coefficient_infidelity_per_quanta={m:.10g}\n")
        f.write(f"fit_intercept_after_floor_subtraction={b:.10g}\n")
        f.write(f"F_no_noise_nph7={df.F_avg.iloc[0]:.12g}\n")
        f.write("nph8_small_error_coefficient_approx=0.39997\n")
        f.write("nph10_small_error_coefficient_approx=0.39997\n")

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(df.quanta_per_gate, df.infidelity - floor, "o-", label="Lindblad simulation")
    xx = np.linspace(0, 0.01, 120)
    ax.plot(xx, m * xx + b, "--", label=fr"weak-heating fit: ${m:.3f}\,\dot{{\bar n}}t_g$")
    ax.set_xlabel(r"mean added quanta $\dot{\bar n}t_g$")
    ax.set_ylabel("heating-induced average infidelity")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "fig_ms_gate_heating_channel.pdf")
    fig.savefig(ROOT / "fig_ms_gate_heating_channel.png", dpi=220)
    plt.close(fig)

if __name__ == "__main__":
    main()
