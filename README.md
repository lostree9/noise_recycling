# Reproducibility bundle: geometry-controlled correlated electric-field noise

This repository accompanies the manuscript **Geometry-controlled correlated electric-field noise in enclosed ion traps from billiard return spectra** (April 2026).

The calculation follows

\[
\text{surface covariance}\to\text{Green operator}\to\text{billiard returns}\to C_{ij}(\omega)\to\text{collective motional noise}.
\]

In the parallel slab, electrostatic images and billiard unfolding give the unfolded normal depths

\[
\alpha_n=|d-2nh|.
\]

Their return-pair spectrum determines both single-ion anomalous heating and the full many-ion electric-field covariance. At \(h=2d\), the local-noise normal and tangential heating ratios are exactly \(\zeta(3)\) and \(\eta(3)=3\zeta(3)/4\). For any finite equal-height ion array with unchanged stationary source statistics, the passive cover obeys

\[
C_h^{(y)}-C_\infty^{(y)}\succeq0,
\qquad
C_\infty^{(x)}-C_h^{(x)}\succeq0.
\]

Beyond parallel walls, a screened boundary-element calculation tests the non-flat billiard picture: large invariant-direction wavenumber selects specular stationary paths, path length fixes the exponential action, and curvature enters through the focusing prefactor.

## Environment

Tested with Python 3.13 and the pinned packages in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Fast verification

Run

```bash
python verify_pra_results_final.py
```

The final line should be

```text
ALL FINAL PRA CHECKS PASSED
```

The verifier is standalone. Manuscript-source consistency checks are skipped when the separately distributed TeX source is absent.

## Main analysis scripts

- `billiard_anomalous_heating.py`: exact slab heating, polarization, finite-correlation-length, and two-ion checks.
- `many_ion_quantum_noise.py`: N-ion covariance matrices, Loewner checks, collective covariance-channel spectra, effective noise rank, and bus-mode exposure.
- `normal_mode_projection.py`: illustrative mechanical-mode projection.
- `screened_bem_billiards.py`: singularity-corrected screened BEM utilities for curved covers.
- `curved_multi_ion_noise.py`: five-probe curved-boundary covariance slice.
- `ms_gate_heating_channel.py`: primitive Mølmer-Sørensen gate channel under symmetric classical motional diffusion.

## Production and convergence scripts

- `run_screened_billiard_sweep.py`: 16-cover curved-geometry production sweep and flat exact benchmark.
- `run_screened_convergence_v2.py`: panel-size and finite-difference convergence tests.
- `run_sidewall_width_check.py`: lateral-closure check at `W/d = 4, 5, 7` for two representative non-flat covers.
- `run_billiard_prefactor_tests.py`: focusing, shape-derivative, and multiple-saddle tests.
- `run_reflection_order_integrals.py`: one- and two-reflection Laplace-action tests.
- `run_ms_gate_truncation.py`: Fock-space truncation convergence for the gate calculation.

The CSV and TXT files are the frozen production outputs used for the quoted numerical values. In addition to correlation coefficients, the final verifier checks one-scale residuals for the focusing and split-saddle laws and the side-wall-width stability. The manuscript itself is packaged separately in the submission-source archive.

The older `verify_qis_results_final.py`, `verify_reflected_path.py`, `run_ray_transfer.py`, `bem_gain_contrast.py`, and `bem_mode_diagnostic.py` are retained for provenance; `verify_pra_results_final.py` is the current submission certificate.

## Numerical provenance

Commit `81c8d5aee5150e1bb124f645ad419d792ec6ca84` is the original frozen numerical snapshot for the early slab, ray, and slotted-BEM calculations. Later commits extend the same thesis project with the return-pair heating formulation, many-ion covariance theorem, collective-noise analysis, corrected screened-BEM saddle tests, and the primitive gate-level calculation.
