#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 02:09:33 2025

@author: y.wang
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np

from scipy.linalg import lstsq
from scipy.io import loadmat

from nbm.core.neural_basis_engine import NeuralBasisEngine
from nbm.utils.fields import kappa_xy_constant
from nbm.utils.collocation import structured_collocation_2d, structured_collocation_2d_cc_bd, remove_corners


# -------------------------
# Weighting strategy
# -------------------------

@dataclass
class Weights:
    bc_left: np.ndarray   # (n_left,1) or scalar 
    bc_right: np.ndarray  # (n_right,1) or scalar
    bc_bottom: np.ndarray # (n_bottom,1) or scalar
    bc_top: np.ndarray    # (n_top,1) or scalar
    darcy: np.ndarray     # (n_pde,1) or scalar


def default_weights(
    *,
    hx: float,
    hy: float,
    mu: float,
    kappa_pde: np.ndarray,   # (n_pde,1)
    kappa_mean: float,
) -> Weights:
    """
    
    """
    hc = float(min(hx, hy))
    
    w_darcy = hc * np.sqrt(mu) * np.sqrt(1/kappa_mean) 
    
    w_left  = np.sqrt(hc) * np.sqrt(1/hc) * np.sqrt(kappa_mean) * np.sqrt(1/mu)
    w_right = np.sqrt(hc) * np.sqrt(1/hc) * np.sqrt(mu) * np.sqrt(1/kappa_mean) 
    w_top = np.sqrt(hc) * np.sqrt(1/hc) * np.sqrt(mu) * np.sqrt(1/kappa_mean) 
    w_bottom = np.sqrt(hc) * np.sqrt(1/hc) * np.sqrt(mu) * np.sqrt(1/kappa_mean) 

    return Weights(
        bc_left=w_left,
        bc_right=w_right,
        bc_bottom=w_top,
        bc_top=w_bottom,
        darcy=w_darcy,
    )


# -------------------------
# Core solver
# -------------------------

def run_ic_darcy(
    *,
    # domain + grid
    nx: int = 50,
    ny: int = 50,
    x_min: float = -1.0,
    x_max: float = 1.0,
    y_min: float = -1.0,
    y_max: float = 1.0,
    # basis configuration
    layer_num: int = 2,
    layer_width: int = 500,
    basis_num: int = 500,
    shape: float = 3.5,
    include_const: bool = False,
    concat_layers: Optional[Tuple[Any, ...]] = None,   # e.g. (0,2,'final') or None
    orthogonal: bool = False,
    seed: Tuple[int, int, int] = (0, 1),
    # physical parameters
    mu: float = 1.0,
    # permeability
    use_kappa_file: bool = False,
    kappa_file: str = "kappa50x50_2.mat",
    constant_kappa: bool = True,
    kappa_val: float = 0.3 * 6.328 * 0.0008,
    # BC: left Dirichlet p=bc_left; others Neumann mass flux n·J=g_*
    p_left: float = 1500.0,
    g_right: float = 0.0,
    g_bottom: float = 0.0,
    g_top: float = 120.0,

    # LS solve controls
    ls_cond: float = 1e-12,
    lapack_driver: str = "gelsd",
    # weighting
    weight_fn: Optional[Callable[..., Weights]] = None,
    # diagnostics
    verbose: bool = True,
) -> Dict[str, Any]:        
        

    # ---------- collocation spacings ----------
    Lx = float(x_max - x_min)
    Ly = float(y_max - y_min)
    hx = Lx / nx
    hy = Ly / ny


    # boundary points: include boundary nodes and cell center internal nodes
    x_all = structured_collocation_2d_cc_bd(nx, ny, x_min, x_max, y_min, y_max)
    x_all_grid = x_all.reshape(ny+2, nx+2, 2)
    x_pde = x_all_grid[1:-1, 1:-1, :].reshape(-1, 2)

    tol = 1e-12
    x_left   = x_all[np.abs(x_all[:, 0] - x_min) < tol]
    x_right  = x_all[np.abs(x_all[:, 0] - x_max) < tol]
    x_bottom = x_all[np.abs(x_all[:, 1] - y_min) < tol]
    x_top    = x_all[np.abs(x_all[:, 1] - y_max) < tol]

    corners = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]], dtype=float)
    x_left_wo   = remove_corners(x_left, corners, tol)
    x_right_wo  = remove_corners(x_right, corners, tol)
    x_bottom_wo = remove_corners(x_bottom, corners, tol)
    x_top_wo    = remove_corners(x_top, corners, tol)

    # boundary stacking order: right, bottom, top, left
    x_bd = np.vstack([x_right_wo, x_bottom_wo, x_top_wo, x_left_wo])
    
    n_right = x_right_wo.shape[0]
    n_bottom = x_bottom_wo.shape[0]
    n_top = x_top_wo.shape[0]
    n_left = x_left_wo.shape[0]

    i1 = 0
    i2 = i1 + n_right
    i3 = i2 + n_bottom
    i4 = i3 + n_top
    i5 = i4 + n_left
    assert i5 == x_bd.shape[0]

    if verbose:
        print(f"Collocation points: x_pde {x_pde.shape}")
        print(f"Boundary points:    x_bd  {x_bd.shape}")
        
    # ---------- permeability field ----------
    if constant_kappa:
        k_pde = kappa_xy_constant(x_pde[:, 0], x_pde[:, 1], val=kappa_val).reshape(-1, 1)
    elif use_kappa_file:
        kappa = loadmat(kappa_file)['kappa']
        kappa = kappa * 0.1 * 6.328 * 0.0008
        k_pde = kappa.reshape(-1, 1)
    else:
        raise ValueError('not constant_kappa nor use_kappa_file')
        
    kappa_mean = float(np.mean(k_pde))
    kappa_min = float(np.min(k_pde))
    kappa_max = float(np.max(k_pde))
    kappa_ratio = kappa_max / max(kappa_min, 1e-300)

    if verbose:
        print(f"\nkappa stats: mean={kappa_mean:.3e}, min={kappa_min:.3e}, max={kappa_max:.3e}, max/min={kappa_ratio:.3e}")


    # ---------- build basis engines ----------
    layer_widths = [layer_width] * (layer_num - 1) + [basis_num]
    shape_factors = [1.0] + [0.1] * (layer_num - 1)
    residual_strengths = [0.2] * (layer_num - 1) + [0.0]    
    
    div_basis = NeuralBasisEngine(
        x_dim=2, 
        layer_widths=layer_widths,
        nlin_type='tanh', 
        include_const=include_const,
        shape_factors=shape_factors,
        residual_strengths=residual_strengths, 
        seed=int(seed[0]),
    )
    div_basis.init_pde_basis(shape=shape, radius=1.5, orthogonal=orthogonal)
    
    p_basis = NeuralBasisEngine(
        x_dim=2, 
        layer_widths=layer_widths,
        nlin_type='tanh', 
        include_const=include_const,
        shape_factors=shape_factors,
        residual_strengths=residual_strengths, 
        seed=int(seed[1]),
    )
    p_basis.init_pde_basis(shape=shape, radius=1.5, orthogonal=orthogonal)
    
    eval_range = np.array([[x_min, x_max], [y_min, y_max]], dtype=float)
    div_basis.set_eval_range(eval_range)
    p_basis.set_eval_range(eval_range)
    
    # concat_layers forwarded to eval_basis
    kwargs = {"concat_layers": list(concat_layers)} if concat_layers else {}
    
    
    # ---------- precompute basis matrices on PDE points ----------
    # p
    P_x  = p_basis.eval_basis(x_pde, eval_list=['u0'],  **kwargs)['u0']
    P_y  = p_basis.eval_basis(x_pde, eval_list=['u1'],  **kwargs)['u1']
    
    # velocity basis
    # U_x = +dpsi/dy -> 'u1'；   U_y = -dpsi/dx -> -'u0'
    U_x =  div_basis.eval_basis(x_pde, eval_list=['u1'], **kwargs)['u1']
    U_y = -div_basis.eval_basis(x_pde, eval_list=['u0'], **kwargs)['u0']
        
    # boundary
    P_L = p_basis.eval_basis(x_left_wo,  eval_list=['u'], **kwargs)['u']
    U_R = div_basis.eval_basis(x_right_wo,  eval_list=['u1'], **kwargs)['u1']
    U_T = -div_basis.eval_basis(x_top_wo,  eval_list=['u0'], **kwargs)['u0']
    U_B = -div_basis.eval_basis(x_bottom_wo,  eval_list=['u0'], **kwargs)['u0']

    if include_const:
        basis_num += 1
    
    # LS system dimensions
    if concat_layers:
        basis_num += layer_width * (len(concat_layers) - 1)
        
    # ---------- allocate global LS system (one step reusable buffers) ----------
    col_num = basis_num * 2   
    row_bd = x_bd.shape[0]
    row_pde = x_pde.shape[0]
    row_num = row_bd + row_pde * 2
    
    A = np.zeros((row_num, col_num), dtype=float)
    b  = np.zeros((row_num, 1), dtype=float)

    # row masks
    ind_bd = np.zeros(row_num, dtype=bool)
    ind_bd[:row_bd] = True    
    
    ind_right  = np.zeros(row_num, dtype=bool)
    ind_bottom = np.zeros(row_num, dtype=bool)
    ind_top    = np.zeros(row_num, dtype=bool)
    ind_left   = np.zeros(row_num, dtype=bool)
    ind_right[i1:i2] = True
    ind_bottom[i2:i3] = True
    ind_top[i3:i4] = True
    ind_left[i4:i5] = True

    ind_pde_u = np.zeros(row_num, dtype=bool)
    ind_pde_v = np.zeros(row_num, dtype=bool)
    ind_pde = np.zeros(row_num, dtype=bool)
    
    ind_pde_u[row_bd:row_bd+row_pde] = True
    ind_pde_v[row_bd+row_pde:row_bd+row_pde*2] = True   
    ind_pde[row_bd:] = True
    
    # ---------- weights ----------

    if weight_fn is None:
        W = default_weights(
            hx=hx, hy=hy, 
            mu=mu,
            kappa_pde=k_pde,
            kappa_mean=kappa_mean,
        )
    else:
        W = weight_fn(
            hx=hx, hy=hy, 
            mu=mu,
            kappa_pde=k_pde,
            kappa_mean=kappa_mean,
        )

    # convert weights to arrays for broadcasting in assembly
    def as_col(w, n: int) -> np.ndarray:
        wv = np.asarray(w, dtype=float)
        if wv.ndim == 0:
            return np.full((n, 1), float(wv))
        if wv.ndim == 1:
            return wv.reshape(-1, 1)
        return wv

    w_right = as_col(W.bc_right, n_right)
    w_bottom = as_col(W.bc_bottom, n_bottom)
    w_top = as_col(W.bc_top, n_top)
    w_left = as_col(W.bc_left, n_left)
    w_darcy = as_col(W.darcy, row_pde)
    
    # ---- boundary rows ----
    A[ind_right,  0:basis_num] = U_R * w_right 
    A[ind_bottom, 0:basis_num] = U_B * w_bottom
    A[ind_top,    0:basis_num] = U_T * w_top
    A[ind_left,   basis_num:] = P_L * w_left
       
    b[i1:i2] = g_right * w_right #1
    b[i2:i3] = g_bottom * w_bottom#-5
    b[i3:i4] = g_top  * w_top  #5
    b[i4:i5] = p_left * w_left
    
    # ---- darcy rows ----
    A[ind_pde_u, 0:basis_num] = U_x * w_darcy
    A[ind_pde_u, basis_num: ] =  k_pde * P_x * w_darcy
    A[ind_pde_v, 0:basis_num] = U_y * w_darcy
    A[ind_pde_v, basis_num: ] =  k_pde * P_y * w_darcy
    
    # -------------------------------------------------------------------------
    # Least square solution
    # -------------------------------------------------------------------------
    
    print('\nSolving Least Square using Scipy lstsq:')   
    coef, residuals, rank, svals = lstsq(A, b, cond=ls_cond, lapack_driver=lapack_driver)

    cond_A = np.nan
    if svals is not None and len(svals) >= 2 and float(svals[-1]) > 0:
        cond_A = float(svals[0] / svals[-1])
        print(f"Condition number = {cond_A:.3e}")     

    fitted = A @ coef
    res = fitted - b
    
    mse_total = float(np.mean(res**2))
    mse_bd = float(np.mean(res[ind_bd]**2))
    mse_pde = float(np.mean(res[ind_pde]**2))
    
    denom = float(np.mean(b**2) + 1e-300)
    rel_mse_total = float(mse_total / denom)
    rel_mse_bd = float(mse_bd / denom)
    rel_mse_pde = float(mse_pde / denom)
    
    print('\tLS rel_MSE:', rel_mse_total)
    print('\tBD rel_MSE:', rel_mse_bd)
    print('\tPDE rel_MSE:', rel_mse_pde)
       
    # -------------------------
    # Reconstruct fields on common collocation
    # -------------------------
    coef_u = coef[: basis_num]
    coef_p = coef[basis_num: ]


    sample_c = structured_collocation_2d(
        nx, ny,
        x_min + hx / 2, x_max - hx / 2,
        y_min + hy / 2, y_max - hy / 2
    )

    sample_x = structured_collocation_2d(
        nx - 1, ny,
        x_min + hx, x_max - hx,
        y_min + hy / 2, y_max - hy / 2
    )

    sample_y = structured_collocation_2d(
        nx, ny - 1,
        x_min + hx / 2, x_max - hx / 2,
        y_min + hy, y_max - hy
    )

    sample_x_f = structured_collocation_2d(
        nx + 1, ny,
        x_min, x_max,
        y_min + hy / 2, y_max - hy / 2
    )

    sample_y_f = structured_collocation_2d(
        nx, ny + 1,
        x_min + hx / 2, x_max - hx / 2,
        y_min, y_max
    )

    
    # pressure
    p_set  = p_basis.eval_basis(sample_c, eval_list=['u','u0','u1'], **kwargs)
    p_c      = (p_set['u']  @ coef_p).reshape(ny, nx)

    
    vx = (div_basis.eval_basis(sample_x, eval_list=["u1"], **kwargs)["u1"] @ coef_u).reshape(ny, nx-1)
    vy = (-div_basis.eval_basis(sample_y, eval_list=["u0"], **kwargs)["u0"] @ coef_u).reshape(ny-1, nx)

    vx_f = (div_basis.eval_basis(sample_x_f, eval_list=["u1"], **kwargs)["u1"] @ coef_u).reshape(ny, nx+1)
    vy_f = (-div_basis.eval_basis(sample_y_f, eval_list=["u0"], **kwargs)["u0"] @ coef_u).reshape(ny+1, nx)

    return dict(
        fields=dict(
            p=p_c,
            vx=vx,
            vy=vy,
            vx_f=vx_f,
            vy_f=vy_f,
        )
    )
 
                
