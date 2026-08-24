# Spatial transfer of surface noise in enclosed ion traps

Reproducibility code for Ayush Nadiger's undergraduate honors-thesis manuscript, dated April 2026.

The paper studies a fixed noisy electrode surface under a passive grounded enclosure. In the parallel strip, electrostatic images and billiard unfolding give the same path depths

\[
\alpha_n = |d - 2nh|.
\]

Those depths set the spatial transfer kernel through a Laplace transform. The enclosure also creates extra reflected/image contributions from the same fixed noisy surface, so the ion couples to that source through more than the direct path. The manuscript calls the resulting zero-wavenumber increase in total coupling **noise recycling**. No new fluctuating source is implied by the term.

The paper then proves the fine-scale direct-path limit and tests the geometric picture with ray tracing and BEM.

## Files

- `verify_reflected_path.py` checks the main analytic and numerical identities.
- `run_ray_transfer.py` reproduces the direction-resolved billiard reconstruction.
- `bem_gain_contrast.py` reproduces the quadrature-consistent non-flat BEM sweep.
- `bem_mode_diagnostic.py` checks that higher BEM singular modes correspond to finer source patterns in the slotted test geometry.
- `make_figures.py` regenerates the transfer, ray, and BEM summary figures.
- `make_schematic.py` generates the strip/unfolding/return-depth schematic used near Sec. III.
- `surface_noise_tools.py` contains the shared exact-kernel, BEM, and ray-tracing routines.
- `ray_transfer_spectrum.csv`, `ray_transfer_summary.txt`, and `bem_gain_contrast.csv` store the recorded numerical outputs used by the paper.

## Python environment

Tested with Python 3.13 and the package versions listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce the checks

```bash
python verify_reflected_path.py
python run_ray_transfer.py
python bem_gain_contrast.py
python bem_mode_diagnostic.py
python make_figures.py
python make_schematic.py
```

`verify_reflected_path.py` ends with `ALL CHECKS PASSED` when the numerical certificates agree with the manuscript formulas.

The BEM spectrum uses quadrature-normalized coordinates,

```text
B_ip = sqrt(w_i) H(x_i,p) sqrt(Delta ell_p)
```

so its singular values approximate the continuum `L^2(Gamma_e) -> L^2(X)` response operator rather than a mesh-coordinate-dependent matrix.

## Numerical snapshot

Commit `81c8d5aee5150e1bb124f645ad419d792ec6ca84` is the original frozen numerical snapshot for the slab, ray, and BEM results. Later repository commits add the unfolding schematic and the singular-mode spatial-scale diagnostic without changing those recorded baseline results.
