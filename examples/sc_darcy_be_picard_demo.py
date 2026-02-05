#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
examples/sc_darcy_be_picard_demo.py

1) Run NBM SC-Darcy (space basis + Backward Euler + Picard) via:
      nbm.solvers.sc_darcy_be_picard.run_sc_space_be_picard
2) (Optional) Run FVM reference and print error metrics.
3) (Optional) Compare spectra.

Assumed repo layout (your screenshot):
  src/nbm/core/neural_basis_engine.py
  src/nbm/utils/metrics.py
  examples/sc_darcy_be_picard_demo.py
  examples/sc_darcian_solver_fvm_picard_massflux.py
"""

from __future__ import annotations

import sys
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    # ============================================================
    # 0) Make sure we can import from src/ and examples/
    # ============================================================
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "examples"))

    # ============================================================
    # 1) ALL CONFIGS HERE
    # ============================================================
    # ---- output ----
    run_id = now_tag()
    out_dir = repo_root / "runs" / "sc_darcy_be_picard_demo" / run_id
    ensure_dir(out_dir)

    # ============================================================
    # 1) ALL CONFIGS
    # ============================================================

    # ---- domain/collocation ----
    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    nx, ny = 50, 50
    Lx, Ly = x_max - x_min, y_max - y_min
    hx, hy = Lx / nx, Ly / ny

    # ---- physical ----
    rho0 = 24.0
    cf = 2e-4
    p0 = 1500.0
    epsilon = 0.25
    mu = 0.04

    # ---- time ----
    T_days = 90.0
    dt = 10.0
    n_steps = 1  # set to int(round(T_days/dt)) for full horizon
    picard_iters = 25

    # ---- BC ----
    bc_left = 1500.0
    u_right = 0.0
    u_top = 5.0
    u_bottom = 0.0
    # mass-flux g = rho * u
    g_right = rho0 * u_right
    g_bottom = rho0 * u_bottom
    g_top = -rho0 * u_top

    # ---- permeability ----
    use_kappa_file = True
    kappa_file = repo_root / "examples" / "data" / "kappa50x50_2.mat"
    constant_kappa = False

    # ---- neural basis ----
    layer_num = 2
    layer_width = 1000
    basis_num = layer_width
    shape = 3.5
    include_const = False
    concat_layers = None  # set like [0, 2, "final"] if needed
    orthogonal = False
    ridge = -1.0

    # ---- behavior ----
    check_error = True   # run FVM compare
    make_plot = False

    # ---- seeds ----
    seeds = tuple(random.randint(0, 10**6) for _ in range(3))
    seeds = (380837, 921489, 59285)  # comment to unfix
    print("[demo] seeds:", seeds)

    cfg = dict(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        nx=nx, ny=ny,
        rho0=rho0, cf=cf, p0=p0, epsilon=epsilon, mu=mu,
        dt=dt, n_steps=n_steps, picard_iters=picard_iters,
        bc_left=bc_left, g_right=g_right, g_bottom=g_bottom, g_top=g_top,
        use_kappa_file=use_kappa_file, kappa_file=str(kappa_file),
        constant_kappa=constant_kappa,
        layer_num=layer_num, layer_width=layer_width, basis_num=basis_num,
        shape=shape, include_const=include_const, concat_layers=concat_layers,
        orthogonal=orthogonal, ridge=ridge,
        check_error=check_error, make_plot=make_plot,
    )
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("[demo] config saved.")

    # ============================================================
    # 2) RUN NBM SOLVER
    # ============================================================
      
    from nbm.solvers.sc_darcy_be_picard import run_sc_space_be_picard

    sol = run_sc_space_be_picard(
        nx=nx, ny=ny, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        layer_num=layer_num, layer_width=layer_width, basis_num=basis_num,
        shape=shape, include_const=include_const, concat_layers=concat_layers,

        rho0=rho0, cf=cf, p0=p0, epsilon=epsilon, mu=mu,
        use_kappa_file=use_kappa_file, kappa_file=str(kappa_file),
        constant_kappa=constant_kappa,

        q_fun=None,
        bc_left=bc_left, g_right=g_right, g_bottom=g_bottom, g_top=g_top,

        n_steps=n_steps, dt=dt, picard_iters=picard_iters,
        ridge=ridge, orthogonal=orthogonal, seed=seeds,
    )

    print("\n[demo] solver returned keys:", list(sol.keys()))
    if "cond_A" in sol:
        print("[demo] cond_A:", sol["cond_A"])

    # ============================================================
    # 3) OPTIONAL: RUN FVM REFERENCE + ERROR METRICS
    # ============================================================
    if check_error:
        print("\n[demo] FVM reference solve")

        from sc_darcian_solver_fvm_picard_massflux import run_sc_darcian_solver_fvm_picard
        from nbm.utils.metrics import rel_metrics, print_err
        from nbm.utils.spectrum import compare_spectra_samegrid_from_faces

        out = run_sc_darcian_solver_fvm_picard(
            Nx=nx, Ny=ny, Lx=Lx, Ly=Ly,
            n_steps=n_steps, dt=dt,
            cf=cf, p0=p0, rho0=rho0,
            mu=mu, epsilon=epsilon,
            use_kappa_file=use_kappa_file,
            constant_kappa=constant_kappa,
            kappa_val=0.3 * 6.328 * 0.0008,
            kappa_file=str(kappa_file),
            p_left=bc_left,
            m_right=g_right,
            m_top=g_bottom,
            m_bottom=g_top,
            picard_tol=1e-6,
            picard_maxit=25,
            verbose=True,
            do_plot=False
        )

        # --- NBM outputs to compare ---
        hat = sol['fields']
        p = np.asarray(hat["p"])
        vx = np.asarray(hat["vx"])
        vy = np.asarray(hat["vy"])
        vx_f = np.asarray(hat["vx_f"])
        vy_f = np.asarray(hat["vy_f"])

        # FVM outputs
        p_ref = np.asarray(out["p"])
        vx_ref = np.asarray(out["vx"])
        vy_ref = np.asarray(out["vy"])
        vx_f_ref = np.asarray(out["vx_f"])
        vy_f_ref = np.asarray(out["vy_f"])


        # --- metrics ---
        err_p    = rel_metrics(p_ref, p)
        err_vx = rel_metrics(vx_ref, vx)
        err_vy = rel_metrics(vy_ref, vy)

        print("\n[demo] errors vs FVM:")
        print_err("p", err_p)
        print_err("vx", err_vx)
        print_err("vy", err_vy)


        m = compare_spectra_samegrid_from_faces(
            vx_f, vy_f, vx_f_ref, vy_f_ref,
            hx, hy, detrend=True, window=True, nbins=None
        )
        print("\n[demo] spectrum metrics:")
        for kk, vv in m.items():
            print(f"  {kk}: {vv}")


    print("\n[demo] done.")


if __name__ == "__main__":
    main()
