# Spatial transfer of surface noise in enclosed ion traps

Reproducibility repository for the August 2026 preprint by Ayush Nadiger.

## Manuscript

The submission source is `noise_recycling_PRA.tex`; the compiled manuscript is `noise_recycling_PRA.pdf`.

Build with REVTeX 4.2:

```bash
pdflatex noise_recycling_PRA.tex
pdflatex noise_recycling_PRA.tex
```

## Python environment

Tested with Python 3.13 and the versions in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Verification

```bash
python verify_reflected_path.py
```

A successful run ends with:

```text
ALL CHECKS PASSED
```

The verification script checks numerical consequences of the analytic formulas; the proofs are in the manuscript.

## Computational components

- `run_ray_transfer.py` reproduces the direction-resolved, height-differenced specular-ray reconstruction used in the manuscript.
- `bem_gain_contrast.py` reproduces the quadrature-consistent `L^2 -> L^2` BEM stress test at three panel resolutions.
- `make_figures.py` regenerates the manuscript figures from the exact formulas and recorded CSV data.
- `surface_noise_tools.py` contains the shared exact-kernel, BEM, and ray-tracing routines used by the focused scripts.
- `verify_reflected_path.py` independently checks the principal analytic/numerical identities and recorded results.

The BEM singular-value calculation uses orthonormal quadrature coordinates,

```text
B_ip = sqrt(w_i) H(x_i,p) sqrt(Delta ell_p)
```

so that the computed spectrum approximates the continuum `L^2(Gamma_e) -> L^2(X)` response operator rather than a mesh-coordinate-dependent matrix.
