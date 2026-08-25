# Geometry-controlled correlated quantum noise in enclosed ion traps

Reproducibility code for Ayush Nadiger's undergraduate honors-thesis manuscript, dated April 2026. The manuscript and bibliography are intentionally kept as an April 2026 record; later journal versions of cited preprints are not substituted into this version.

The current manuscript connects anomalous trapped-ion heating to a broader spatial quantum-noise operator. For a fixed noisy electrode surface and passive grounded geometry, the calculation follows

\[
\text{surface covariance}\to\text{Green operator}\to\text{billiard returns}\to C_{ij}(\omega)\to\text{collective motional noise}.
\]

In the parallel slab, electrostatic images and billiard unfolding give the same unfolded normal depths

\[
\alpha_n=|d-2nh|.
\]

Their return-pair spectrum determines single-ion heating and the full many-ion electric-field covariance. At \(h=2d\), the local-noise normal and tangential heating ratios are exactly \(\zeta(3)\) and \(\eta(3)=3\zeta(3)/4\). For any finite equal-height ion array with unchanged stationary source statistics, the passive cover obeys the Loewner inequalities

\[
C_h^{(y)}-C_\infty^{(y)}\succeq0,
\qquad
C_\infty^{(x)}-C_h^{(x)}\succeq0.
\]

Beyond parallel walls, a screened boundary-element calculation tests the non-flat billiard picture: large longitudinal wavenumber selects specular stationary paths, path length fixes the exponential action, and curvature enters through the focusing prefactor.

## Main scripts

- `verify_reflected_path.py` checks the original strip identities and numerical certificates.
- `run_ray_transfer.py` reproduces the constructive direction-resolved ray reconstruction.
- `billiard_anomalous_heating.py` reproduces the exact slab heating, polarization, correlation-length, and two-ion results used in the thesis manuscript.
- `many_ion_quantum_noise.py` builds N-ion covariance matrices, collective channel spectra, effective noise ranks, and bus-mode exposure ratios.
- `normal_mode_projection.py` compares the environmental eigenvectors with an illustrative finite-chain motional-mode basis.
- `screened_bem_billiards.py` contains the singularity-corrected screened BEM used for curved covers.
- `curved_multi_ion_noise.py` builds the five-probe curved-boundary covariance slice.
- `ms_gate_heating_channel.py` reconstructs a primitive Mølmer-Sørensen gate channel under symmetric classical motional diffusion.
- `verify_qis_results_final.py` is the integrated numerical certificate for the current manuscript.
- `make_schematic.py` generates the strip/unfolding/return-depth schematic.

The older `bem_gain_contrast.py` and `bem_mode_diagnostic.py` files record the earlier slotted-enclosure stress tests retained as project history; they are no longer the main non-flat evidence in the integrated manuscript.

## Python environment

Tested with Python 3.13 and the package versions listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce the principal calculations

```bash
python billiard_anomalous_heating.py
python many_ion_quantum_noise.py
python normal_mode_projection.py
python ms_gate_heating_channel.py
python curved_multi_ion_noise.py
python verify_qis_results_final.py
```

The final verification script ends with

```text
ALL FINAL QIS CHECKS PASSED
```

when the recorded numerical outputs agree with the manuscript claims.

## Numerical provenance

Commit `81c8d5aee5150e1bb124f645ad419d792ec6ca84` remains the original frozen numerical snapshot for the early slab, ray, and slotted-BEM calculations. Subsequent commits extend the same undergraduate thesis project with the return-pair heating formulation, many-ion covariance theorem, collective-noise analysis, corrected screened-BEM saddle tests, and primitive gate-level calculation.