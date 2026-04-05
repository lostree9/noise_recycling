# Geometry-Dependent Electric Field Noise in Trapped-Ion Systems

Code and data for: *"Geometry-dependent electric field noise in trapped-ion systems: noise recycling in open electrode geometries"*

## Quick start

```bash
# 1. Generate parameter files
python run_all.py --generate-params

# 2a. Run on SLURM cluster
sbatch submit_all.sh

# 2b. OR run locally (slow, ~2 hours)
python run_all.py --local --full

# 3. Generate all paper figures and data tables
python run_all.py --all-outputs --results-dir results/
```

## Requirements

- Python 3.10+
- [DOLFINx](https://github.com/FEniCS/dolfinx) 0.8+ (for FEM solves)
- [Gmsh](https://gmsh.info/) Python API (for meshing)
- NumPy, Matplotlib (for analysis/plotting)

On a cluster with Apptainer/Singularity, use the DOLFINx container:
```bash
export FENICS_SIF=$HOME/dolfinx-stable.sif
```

## Repository structure

```
├── run_all.py                  # Master pipeline script
├── plot_paper_figures.py       # Generate all paper figures (Figs 1-5)
├── generate_mesh.py            # Surface slot mesh generator
├── generate_mesh_general.py    # Blade, two-plate, stylus mesh generators
├── compute_gj.py               # 2D geometry functional computation
├── compute_gj_axi.py           # Axisymmetric (stylus) computation
├── run_single_general.py       # Single-geometry pipeline (mesh → solve → G_j)
├── decompose_gj_extended.py    # Spatial decomposition of G_j by region
├── submit_all.sh               # SLURM array job submission
├── name_light_edit.tex         # Paper LaTeX source
├── references.bib              # Bibliography
└── paper_outputs/              # Generated figures and data
    ├── fig1_geometry_gallery.pdf
    ├── fig2_slot_mechanism.pdf
    ├── fig3_twoplate.pdf
    ├── fig4_blade.pdf
    ├── fig5_stylus.pdf
    ├── paper_data.json
    ├── experimental_predictions.txt
    └── summary.txt
```

## Geometry families

| Family | Parameters | Key result |
|--------|-----------|------------|
| Surface slot | $t/g \in [0, 4]$ | 30% enhancement from gap recycling |
| Two-plate | $h \in [80, 800]$ µm | 2.5× enclosure enhancement |
| Blade | $\theta \in [20°, 85°]$ | Anisotropy $\mathcal{G}_y/\mathcal{G}_x$ from 0.5 to 3.9 |
| Stylus (axisym.) | $z_\mathrm{ion} \in [30, 200]$ µm | Effective exponent −4.7 (vs. −4.0) |

## Method

The geometry functional $\mathcal{G}_j = \int_{\Gamma_e} |K_j|^2 \, dl'$ is computed by:
1. Decomposing the Dirichlet Green's function as $G = G_0 + H$
2. Solving $-\Delta H = 0$ with FEniCSx (P2 elements, direct solver)
3. Finite-differencing the ion position to obtain $K_j$
4. Integrating $|K_j|^2$ over electrode boundaries

## Citation

```bibtex
@article{Nadiger2026,
  author  = {Nadiger, Ayush},
  title   = {Geometry-dependent electric field noise in trapped-ion systems: 
             noise recycling in open electrode geometries},
  journal = {Phys. Rev. A},
  year    = {2026},
}
```

## License

MIT
