# Neural Basis Method for Solving and Learning PDE-Governed Systems

NBM (Neural Basis Method) is a projection-based, solver-first framework that unifies simulation and learning for PDE-governed systems. Rather than treating a PDE as a loss to be minimized by non-convex training, NBM freezes a multilayer residual network to generate a finite dimensional approximation space of neural basis functions, then computes the solution by solving a deterministic projection problem, typically a weighted least squares system that enforces the PDE operators and boundary conditions directly. This makes the computational structure explicit, preserves physical scaling and numerical stabilization, and yields solver-level residuals that are interpretable, comparable across terms, and actionable for diagnosis.

NBM is paired with operator learning (NBM-OL) for parametric many-query problems. By representing solution families across parameter instances in the same fixed neural basis space, NBM-OL learns a map from parameter encodings (and time, when applicable) to the neural basis coefficients. Training is self-supervised using the same residual metric as the NBM solver, so the learned operator is constrained by an operator-consistent objective rather than heuristic, dimensionless loss balancing. Beyong forward prediction, this residual-structured formualtion also provides a natural route to physics-constrained inverse problems, where measurements can be appended as additional residual blocks to uncover system parameters guided by model-data identifiability and compatiblility.  

Current implementations in this repository focus on fluid flow problems, including porous media flow and transport, incompressible Navier–Stokes systems, and network-flow models such as pipeline and blood-flow networks.

<table>
  <tr>
    <td align="center" width="50%">
      <img src="docs/figs/darcy_transport.svg" width="100%">
      <br/><em>Constant-tracer injection in a CO₂ storage reservoir: the prediction is produced by NBM-OL, achieving ~5400× speedup for the full 200-day dynamics.</em>
    </td>
    <td align="center" width="50%">
      <img src="docs/figs/kolmogorov_xi1_t=20-25.gif" width="100%">
      <br/><em>Kolmogorov flow (bounded, t=20-25s): columns show x-velocity, y-velocity, and speed; rows show prediction, reference, and error. Relative L2: 0.2% (spectral), 1.2% (speed); speedup: ~25000× (for 25s horizon).</em>
    </td>
  </tr>
</table>



---

## What is in this repo

- `src/nbm/`  

- `examples/`  



---

## Quick start (no install, run examples directly)

This repo uses the `src/` layout. The examples are runnable **without installing** anything:
each demo prepends `repo_root/src` into `sys.path`, so `import nbm.*` works out of the box.


### 1) Clone

```bash
git clone https://github.com/neural-basis-method/nbm.git
cd nbm
```

### 2) Run the demo

```bash
python examples/sc_darcy_be_picard_demo.py
```

---

## Citation

If you use NBM / NBM-OL in academic work, please cite the accompanying paper:

- *Modeling and learning multiscale advective Darcian dynamics with the Neural Basis Method* (add DOI/arXiv when available)

---

## License and trademarks

- See `LICENSE` for the software license.
- Name usage: “NBM”, “NBM-OL”, “NBM Operator Learning”, and “Neural Basis Method” are project names, including common variants in capitalization and hyphenation. See `TRADEMARKS.md` for permitted and prohibited uses.
- Commercial licensing and support: commercial licensing, enterprise support, and custom development are available.
