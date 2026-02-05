# Neural Basis Method

Neural Basis Method (NBM) for PDE-governed Dynamical Systems.

Commercial licensing and support: commercial licensing, enterprise support, and custom development are available.

Name usage: “NBM”, “NBM-OL”, “NBM Operator Learning”, and “Neural Basis Method” are project names, including common variants in capitalization and hyphenation. See `TRADEMARKS.md` for permitted and prohibited uses.

---

## Repository layout

- `src/nbm/`
  - `core/`
    - `neural_basis_engine.py` (public class: `NeuralBasisEngine`)
  - `solvers/`
    - `sc_darcy_be_picard.py`
  - `utils/`
    - `collocation.py`, `fields.py`, `metrics.py`, `spectrum.py`
- `examples/`
  - `sc_darcy_be_picard_demo.py`
  - `sc_darcian_solver_fvm_picard_massflux.py`
  - `data/`
    - `kappa50x50_2.mat`

This repo currently uses a lightweight workflow: example scripts insert `repo_root/src` into `sys.path`, so you can run demos without installing the package.

---

## Quick start

### 1) Clone

```bash
git clone https://github.com/neural-basis-method/nbm.git
cd nbm
```

### 2) Run the demo

```bash
python examples/sc_darcy_be_picard_demo.py
```

Notes:
- The demo script inserts `repo_root/src` into `sys.path`, so you can import `nbm.*` without installing the package.
- Outputs (configs/metrics) may be written under `runs/` depending on your demo settings.

---

## What the demo runs

`examples/sc_darcy_be_picard_demo.py` does the following:

1. Runs the NBM solver for slightly-compressible Darcy flow (space-basis + Backward Euler + Picard):
   - `nbm.solvers.sc_darcy_be_picard.run_sc_space_be_picard`
2. Optionally runs an FVM reference solver:
   - `examples/sc_darcian_solver_fvm_picard_massflux.py`
3. Optionally compares spectra on the same grid:
   - `nbm.utils.spectrum.compare_spectra_samegrid_from_faces`

The demo is configured inside `main()` (no argparse). You can edit:
- grid resolution (`nx, ny`)
- physical parameters (`rho0, cf, p0, epsilon, mu`)
- time stepping (`dt, n_steps, picard_iters`)
- boundary conditions (`bc_left, g_right, g_bottom, g_top`)
- permeability settings (file vs constant)
- neural basis settings (`layer_num, layer_width, basis_num, shape, concat_layers, ridge, orthogonal`)

---

## Data files

The SC-Darcy demo can load permeability from:

- `examples/data/kappa50x50_2.mat`

If you change the file location, update the demo config accordingly.

---

## Development workflow (Spyder-friendly)

This repo is designed so you can develop locally in Spyder without packaging:

- Open the repo folder in Spyder (project).
- Run `examples/sc_darcy_be_picard_demo.py` directly.
- The script handles `sys.path` injection (`repo_root/src`) so imports work.

If you prefer, you can also add `repo_root/src` to Spyder’s Python path:
- Spyder Preferences → Python interpreter → PYTHONPATH manager → add `<repo_root>/src`

---

## Git hygiene (recommended)

This project should not commit local macOS metadata or run outputs.

Add these to `.gitignore` (repo root):

```gitignore
# macOS
.DS_Store

# local runs / outputs
runs/
```

If you already accidentally tracked them once, remove from git tracking:

```bash
git rm -r --cached runs
git rm -r --cached **/.DS_Store
git commit -m "Ignore local runs and macOS metadata"
```

---

## License

TBD (private repo for now). If/when you publish, choose a license and add `LICENSE` at the repo root.

---

## Citation

TBD (add a `CITATION.cff` when you are ready to publish).


Commercial licensing and support
Commercial licensing, enterprise support, and custom development are available.
<!--Contact: yuhe.wang@me.com-->

Name usage
“NBM”, “NBM-OL”,  “NBM Operatolr Learning”, and “Neural Basis Method” are project names, including common variants in capitalization and hyphenation. See TRADEMARKS.md for permitted and prohibited uses.
