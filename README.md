# Neural Basis Method for modeling and learning PDE-governed systems

NBM (Neural Basis Method) is a projection-based, solver-first framework that unifies simulation and learning for PDE-governed systems. Rather than treating a PDE as a loss to be minimized by non-convex training, NBM freezes a multilayer residual network to generate a finite dimensional approximation space of neural basis functions, then computes the solution by solving a deterministic projection problem, typically a weighted least squares system that enforces the PDE operators and boundary conditions directly. This makes the computational structure explicit, preserves physical scaling and numerical stabilization, and yields solver-level residuals that are interpretable, comparable across terms, and actionable for diagnosis.

NBM is paired with operator learning (NBM-OL) for parametric many-query problems. By representing solution families across parameter instances in the same fixed neural basis space, NBM-OL learns a map from parameter encodings (and time, when applicable) to the neural basis coefficients. Training is self-supervised using the same residual metric as the NBM solver, so the learned operator is constrained by an operator-consistent objective rather than heuristic, dimensionless loss balancing. Beyong forward prediction, this residual-structured formualtion also provides a natural route to physics-constrained inverse problems, where measurements can be appended as additional residual blocks to uncover system parameters guided by model-data identifiability and compatiblility.  

Current implementations in this repository focus on fluid flow problems, including porous media flow and transport, incompressible Navier–Stokes systems, and network-flow models such as pipeline and blood-flow networks.

---

## What is in this repo

- `src/nbm/`  

- `examples/`  



---

## Quick start

### 1) Clone

```bash
git clone https://github.com/neural-basis-method/nbm.git
cd nbm
```

### 2) Create an environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

If your repo has a `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3) Run the demo

```bash
python examples/sc_darcy_be_picard_demo.py
```

Notes:

- The demo script inserts `repo_root/src` into `sys.path`, so you can `import nbm.*` without installing the package.
- Outputs (configs/metrics) may be written under `runs/` depending on demo settings.


---

## Citation

If you use NBM / NBM-OL in academic work, please cite the accompanying paper:

- *Modeling and learning multiscale advective Darcian dynamics with the Neural Basis Method* (add DOI/arXiv when available)

---

## License and trademarks

- See `LICENSE` for the software license.
- Name usage: “NBM”, “NBM-OL”, “NBM Operator Learning”, and “Neural Basis Method” are project names, including common variants in capitalization and hyphenation. See `TRADEMARKS.md` for permitted and prohibited uses.
- Commercial licensing and support: commercial licensing, enterprise support, and custom development are available.
