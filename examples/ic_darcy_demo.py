#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 02:09:33 2025

With divergence free basis

@author: y.wang
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
    # 0) Make sure we can import from src/
    # ============================================================
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    # ============================================================
    # 1) ALL CONFIGS HERE
    # ============================================================
    # ---- output ----
    run_id = now_tag()
    out_dir = repo_root / "runs" / "ic_darcy_demo" / run_id
    ensure_dir(out_dir)

    # ---- domain/collocation ----
    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    nx, ny = 50, 50
    Lx, Ly = x_max - x_min, y_max - y_min
    hx, hy = Lx / nx, Ly / ny

    # ---- physical ----
    mu = 0.04

    # ---- BC ----
    p_left = 1500.0
    g_right = 1.0
    g_top = 5.0
    g_bottom = -5.0

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

    # ---- behavior ----
    check_error = True   # run FVM compare 
    make_plot = False   

    seeds = tuple(random.randint(0, 10**6) for _ in range(2))
    print("[demo] seeds:", seeds)
    
    cfg = dict(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        nx=nx, ny=ny,
        mu=mu,
        p_left=p_left, g_right=g_right, g_bottom=g_bottom, g_top=g_top,
        use_kappa_file=use_kappa_file, kappa_file=str(kappa_file),
        constant_kappa=constant_kappa, 
        layer_num=layer_num, layer_width=layer_width, basis_num=basis_num,
        shape=shape, include_const=include_const, concat_layers=concat_layers,
        orthogonal=orthogonal, 
        check_error=check_error, make_plot=make_plot,
    )
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("[demo] config saved.")    
    
    # ============================================================
    # 2) RUN NBM SOLVER
    # ============================================================
    from nbm.solvers.ic_darcy import run_ic_darcy
    
    sol = run_ic_darcy(
        nx=nx, ny=ny, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        layer_num=layer_num, layer_width=layer_width, basis_num=basis_num,
        shape=shape, include_const=include_const, concat_layers=concat_layers,

        mu=mu,
        use_kappa_file=use_kappa_file, kappa_file=str(kappa_file),
        constant_kappa=constant_kappa, 

        p_left=p_left, g_right=g_right, g_bottom=g_bottom, g_top=g_top,

        orthogonal=orthogonal, seed=seeds,
    )
    
    print("\n[demo] solver returned keys:", list(sol.keys()))
    if "cond_A" in sol:
        print("[demo] cond_A:", sol["cond_A"])
        
    
    # ============================================================
    # 3) OPTIONAL: RUN FVM REFERENCE + ERROR METRICS
    # ============================================================
    if check_error:
        print("\n[demo] FVM reference solve")

        from ic_darcy_fvm import run_ic_darcy_fvm
        from nbm.utils.metrics import rel_metrics, print_err
        from nbm.utils.spectrum import compare_spectra_samegrid_from_faces

        
        out = run_ic_darcy_fvm(
            Lx=Lx, Ly=Ly,
            Nx=nx, Ny=nx,
            bc_left=p_left, bc_right=g_right,
            bc_bottom=g_bottom, bc_top=g_top,
            kappa_file=str(kappa_file)
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


