# Geometry-controlled correlated quantum noise in enclosed ion traps

Reproducibility code for Ayush Nadiger's undergraduate honors-thesis manuscript, dated April 2026. The manuscript and bibliography are intentionally kept as an April 2026 record; later journal versions of cited preprints are not substituted into this version.

The current manuscript asks how passive conducting geometry restructures the spatial quantum-noise environment seen by a trapped-ion system. The calculation follows

\[
\text{surface covariance}\to\text{Green operator}\to\text{billiard returns}\to C_{ij}(\omega)\to\text{collective motional noise}.
\]

In the parallel slab, electrostatic images and billiard unfolding give the unfolded normal depths

\[
\alpha_n=|d-2nh|.
\]

Their return-pair spectrum determines both single-ion anomalous heating and the full many-ion electric-field covariance. At \(h=2d\), the local-noise normal and tangential heating ratios are exactly \(\zeta(3)\) and \(\eta(3)=3\zeta(3)/4\). For any finite equal-height ion array with unchanged stationary source statistics, the passive cover obeys the Loewner inequalities

\[
C_h^{(y)}-C_\infty^{(y)}\succeq0,
\qquad
C_\infty^{(x)}-C_h^{(x)}\succeq0.
\]

Beyond parallel walls, a screened boundary-element calculation tests the non-flat billiard picture: large invariant-direction wavenumber selects specular stationary paths, path length fixes the exponential action, and curvature enters through the focusing prefactor.

## Main analysis scripts

- `billiard_anomalous_heating.py` — exact slab heating, polarization, correlation-length, and two-ion calculations.
- `many_ion_quantum_noise.py` — N-ion covariance matrices, Loewner checks, collective-channel spectra, participation ranks, and bus-mode exposures.
- `normal_mode_projection.py` — illustrative projection onto a finite-chain mechanical-mode basis.
- `screened_bem_billiards.py` — singularity-corrected screened BEM utilities for curved covers.
- `curved_multi_ion_noise.py` — five-probe curved-boundary covariance slice.
- `ms_gate_heating_channel.py` — primitive Mølmer-Sørensen gate channel under symmetric classical motional diffusion.
- `verify_qis_results_final.py` — integrated numerical certificate for the current manuscript.

## Production / convergence scripts

- `run_screened_billiard_sweep.py` — 16-cover curved-geometry production sweep and flat exact benchmark.
- `run_screened_convergence_v2.py` — panel-size and finite-difference convergence tests.
- `run_billiard_prefactor_tests.py` — focusing, shape-derivative, and multiple-saddle tests.
- `run_reflection_order_integrals.py` — one- and two-reflection Laplace-action integrals.
- `run_ms_gate_truncation.py` — Fock-space truncation convergence of the gate calculation.

The CSV files in the repository are the recorded production outputs used for the quoted numerical values. `screened_bem_curves_v2.csv` stores the individual large-wavenumber correction curves behind the 16-cover slope fits.

The older `verify_reflected_path.py`, `run_ray_transfer.py`, `bem_gain_contrast.py`, and `bem_mode_diagnostic.py` files record earlier stages of the same thesis project. They are retained for provenance but are not the principal evidence in the integrated manuscript.

## Python environment

Tested with Python 3.13 and the pinned versions in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Verify the frozen numerical snapshot

After cloning the repository, the quickest end-to-end check is

```bash
python verify_qis_results_final.py
```

It should end with

```text
ALL FINAL QIS CHECKS PASSED
```

The verifier is standalone: if manuscript TeX is not present in the clone, manuscript-source checks are explicitly skipped rather than treated as a numerical failure.

## Regenerate the main recorded outputs

The more expensive production suite is

```bash
python billiard_anomalous_heating.py
python many_ion_quantum_noise.py
python normal_mode_projection.py
python ms_gate_heating_channel.py
python run_ms_gate_truncation.py
python curved_multi_ion_noise.py
python run_screened_billiard_sweep.py
python run_screened_convergence_v2.py
python run_billiard_prefactor_tests.py
python run_reflection_order_integrals.py
python verify_qis_results_final.py
```

The screened-BEM sweeps are the slowest part of this sequence.

## Numerical provenance

Commit `81c8d5aee5150e1bb124f645ad419d792ec6ca84` is the original frozen numerical snapshot for the early slab, ray, and slotted-BEM calculations. Later commits extend the same undergraduate thesis project with the return-pair heating formulation, many-ion covariance theorem, collective-noise analysis, corrected screened-BEM saddle tests, and the primitive gate-level calculation.