#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Slightly-compressible Darcy (2D), spatial neural-basis + implicit time stepping 
with Picard iterations, solved via one global least-squares system per Picard step.

Unknowns:
- Mass flux J = (Jx, Jy), decomposed as J = J_div + J_grad
  * J_div  = ( dpsi/dy, -dpsi/∂x )   (divergence-free contribution)
  * J_grad = ( dpsi/∂x,  dpsi/∂y )   (gradient contribution, carries divergence)
- Pressure p expanded in its own neural basis

Equations (collocation at cell centers):
1) Darcy:  J + M* ∇p = 0, where M* = rho^n * k / mu  (lagged mobility from previous p)
2) Mass:   (S/dt) p^{n+1} + ∇·J = (S/dt) p^n + q, with S = epsilon * rho0 * cf

Boundary conditions (in this version):
- Left boundary: Dirichlet pressure p = bc_left
- Right/Bottom/Top: Neumann mass-flux n·J = g_* 


Author: harrywang
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any, List

import numpy as np
from numpy.linalg import norm

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
    mass: np.ndarray      # (n_pde,1) or scalar


def default_weights(
    *,
    hx: float,
    hy: float,
    rho0: float,
    mu: float,
    kappa_pde: np.ndarray,   # (n_pde,1)
    kappa_mean: float,
) -> Weights:
    """
    
    """
    hc = float(min(hx, hy))
    
    w_darcy = hc * (1/rho0) * np.sqrt(mu) * np.sqrt(1/kappa_mean) 
    w_mass  = hc * (1/rho0) * np.sqrt(mu) * np.sqrt(1/kappa_mean) 
    
    w_left  = np.sqrt(hc) * np.sqrt(1/hc) * np.sqrt(kappa_mean) * np.sqrt(1/mu)
    w_right = np.sqrt(hc) * np.sqrt(1/hc) * (1/rho0) * np.sqrt(mu) * np.sqrt(1/kappa_mean) 
    w_top = np.sqrt(hc) * np.sqrt(1/hc) * (1/rho0) * np.sqrt(mu) * np.sqrt(1/kappa_mean) 
    w_bottom = np.sqrt(hc) * np.sqrt(1/hc) * (1/rho0) * np.sqrt(mu) * np.sqrt(1/kappa_mean) 

    return Weights(
        bc_left=w_left,
        bc_right=w_right,
        bc_bottom=w_top,
        bc_top=w_bottom,
        darcy=w_darcy,
        mass=w_mass,
    )


# -------------------------
# Core solver
# -------------------------

def run_sc_space_be_picard(
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
    seed: Tuple[int, int, int] = (0, 1, 2),
    # physical parameters
    rho0: float = 24.0,
    cf: float = 2e-4,
    p0: float = 1500.0,
    epsilon: float = 0.25,
    mu: float = 0.04,
    # permeability
    use_kappa_file: bool = False,
    kappa_file: str = "kappa50x50_2.mat",
    constant_kappa: bool = True,
    kappa_val: float = 0.3 * 6.328 * 0.0008,
    # source term q(x,y)
    q_fun: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    # BC: left Dirichlet p=bc_left; others Neumann mass flux n·J=g_*
    bc_left: float = 1500.0,
    g_right: float = 0.0,
    g_bottom: float = 0.0,
    g_top: float = 120.0,
    # time stepping
    n_steps: int = 9,
    dt: float = 10.0,
    picard_iters: int = 10,
    omega0: float = 1.0,
    tol_update: float = 1e-6,
    # LS solve controls
    ridge: float = -1.0,  # ridge > 0 enables block-diagonal Tikhonov
    ls_cond: float = 1e-12,
    lapack_driver: str = "gelsd",
    # weighting
    weight_fn: Optional[Callable[..., Weights]] = None,
    # diagnostics
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Returns a dict with:
      - coefficients: coef_jdiv, coef_jcurl, coef_p
      - basis objects: jdiv_basis, jcurl_basis, p_basis
      - collocation points: x_pde, x_bd, boundary sets
      - permeability arrays: k_pde, stats
      - assembled masks and last-step diagnostics
      - reconstructed fields on cell centers/faces (final step)
      - full time history (coeffs + Picard iteration traces)
    """

    # ---------- grid spacings ----------
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
        kappa = kappa * 0.01 * 6.328 * 0.0008
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
    residual_strengths = [0.4] * (layer_num - 1) + [0.0]

    jdiv_basis = NeuralBasisEngine(
        x_dim=2,
        layer_widths=layer_widths,
        nlin_type="tanh",
        include_const=False,
        shape_factors=shape_factors,
        residual_strengths=residual_strengths,
        seed=int(seed[0]),
    )
    jdiv_basis.init_pde_basis(shape=shape, radius=1.5, orthogonal=orthogonal)

    jcurl_basis = NeuralBasisEngine(
        x_dim=2,
        layer_widths=layer_widths,
        nlin_type="tanh",
        include_const=False,
        shape_factors=shape_factors,
        residual_strengths=residual_strengths,
        seed=int(seed[1]),
    )
    jcurl_basis.init_pde_basis(shape=shape, radius=1.5, orthogonal=orthogonal)

    p_basis = NeuralBasisEngine(
        x_dim=2,
        layer_widths=layer_widths,
        nlin_type="tanh",
        include_const=include_const,
        shape_factors=shape_factors,
        residual_strengths=residual_strengths,
        seed=int(seed[2]),
    )
    p_basis.init_pde_basis(shape=shape, radius=1.5, orthogonal=orthogonal)

    eval_range = np.array([[x_min, x_max], [y_min, y_max]], dtype=float)
    jdiv_basis.set_eval_range(eval_range)
    jcurl_basis.set_eval_range(eval_range)
    p_basis.set_eval_range(eval_range)

    # concat_layers forwarded to eval_basis
    eval_kwargs = {"concat_layers": list(concat_layers)} if concat_layers else {}

    if concat_layers:
        basis_num += layer_width * (len(concat_layers) - 1)

    # ---------- precompute basis matrices on PDE points ----------
    # p
    P   = p_basis.eval_basis(x_pde, eval_list=["u"],  **eval_kwargs)["u"]
    P_x = p_basis.eval_basis(x_pde, eval_list=["u0"], **eval_kwargs)["u0"]
    P_y = p_basis.eval_basis(x_pde, eval_list=["u1"], **eval_kwargs)["u1"]

    # J_div: Jx_div = +dpsi/dy -> u1 ; Jy_div = -dpsi/dx -> -u0
    Jx_div_pde =  jdiv_basis.eval_basis(x_pde, eval_list=["u1"], **eval_kwargs)["u1"]
    Jy_div_pde = -jdiv_basis.eval_basis(x_pde, eval_list=["u0"], **eval_kwargs)["u0"]

    # J_grad: Jx_grad = dpsi/dx -> u0 ; Jy_grad = dpsi/dy -> u1
    Jx_grad_pde = jcurl_basis.eval_basis(x_pde, eval_list=["u0"], **eval_kwargs)["u0"]
    Jy_grad_pde = jcurl_basis.eval_basis(x_pde, eval_list=["u1"], **eval_kwargs)["u1"]

    # divergence of gradient part: dJx/dx + dJy/dy = u00 + u11
    div_grad_pde = (
        jcurl_basis.eval_basis(x_pde, eval_list=["u00"], **eval_kwargs)["u00"]
      + jcurl_basis.eval_basis(x_pde, eval_list=["u11"], **eval_kwargs)["u11"]
    )

    # ---------- boundary basis matrices ----------
    # left Dirichlet for p
    P_L = p_basis.eval_basis(x_left_wo, eval_list=["u"], **eval_kwargs)["u"]

    # right boundary n=(1,0): n·J = Jx
    Jx_div_R =  jdiv_basis.eval_basis(x_right_wo, eval_list=["u1"], **eval_kwargs)["u1"]
    Jx_grad_R = jcurl_basis.eval_basis(x_right_wo, eval_list=["u0"], **eval_kwargs)["u0"]

    # bottom boundary n=(0,-1): n·J = -Jy
    Jy_div_B = -jdiv_basis.eval_basis(x_bottom_wo, eval_list=["u0"], **eval_kwargs)["u0"]
    Jy_grad_B =  jcurl_basis.eval_basis(x_bottom_wo, eval_list=["u1"], **eval_kwargs)["u1"]

    # top boundary n=(0,1): n·J = +Jy
    Jy_div_T = -jdiv_basis.eval_basis(x_top_wo, eval_list=["u0"], **eval_kwargs)["u0"]
    Jy_grad_T =  jcurl_basis.eval_basis(x_top_wo, eval_list=["u1"], **eval_kwargs)["u1"]

    # sizes for block assembly
    nb_jdiv = Jx_div_pde.shape[1]
    nb_jcurl = Jx_grad_pde.shape[1]
    nb_p = P.shape[1]

    if verbose:
        print(f"\nBasis sizes: nb_jdiv={nb_jdiv}, nb_jcurl={nb_jcurl}, nb_p={nb_p}")
        if concat_layers:
            print(f"(concat_layers enabled) basis_num={basis_num})")

    # ---------- allocate global LS system (one step reusable buffers) ----------
    row_bd = x_bd.shape[0]
    row_pde = x_pde.shape[0]
    row_num = row_bd + 3 * row_pde
    col_num = nb_jdiv + nb_jcurl + nb_p

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
    ind_pde_m = np.zeros(row_num, dtype=bool)

    ind_pde_u[row_bd : row_bd + row_pde] = True
    ind_pde_v[row_bd + row_pde : row_bd + 2 * row_pde] = True
    ind_pde_m[row_bd + 2 * row_pde : row_bd + 3 * row_pde] = True

    ind_pde = np.zeros(row_num, dtype=bool)
    ind_pde[row_bd:] = True

    # ---------- weights ----------

    if weight_fn is None:
        W = default_weights(
            hx=hx, hy=hy, 
            rho0=rho0, mu=mu,
            kappa_pde=k_pde,
            kappa_mean=kappa_mean,
        )
    else:
        W = weight_fn(
            hx=hx, hy=hy, 
            rho0=rho0, mu=mu,
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
    w_mass = as_col(W.mass, row_pde)

    # ---------- initial condition for p coefficients ----------
    S = epsilon * rho0 * cf
    p_init_vec = np.full((row_pde, 1), float(p0), dtype=float)
    coef_p, *_ = lstsq(P, p_init_vec, cond=ls_cond, lapack_driver=lapack_driver)

    coef_jdiv = np.zeros((nb_jdiv, 1), dtype=float)
    coef_jcurl = np.zeros((nb_jcurl, 1), dtype=float)

    # ---------- assembly+solve for one Picard step ----------
    def assemble_and_solve_step(coef_p_prev: np.ndarray) -> Dict[str, Any]:
        """
        Build LS system for one BE step using lagged mobility from coef_p_prev,
        then solve for (coef_jdiv, coef_jcurl, coef_p_new).
        """
        # lagged mobility: M* = rho^n * k / mu
        p_prev = P @ coef_p_prev                               # (n_pde,1)
        rho_n  = rho0 * (1.0 + cf * (p_prev - p0))            # (n_pde,1)
        M_c    = (rho_n * k_pde) / mu                          # (n_pde,1)

        # RHS for mass equation: (S/dt)*p^n + q
        b_mass = (S / dt) * p_prev
        if q_fun is not None:
            qv = q_fun(x_pde[:, 0], x_pde[:, 1]).reshape(-1, 1)
            b_mass = b_mass + qv

        # ---- boundary rows ----
        # right: Jx = g_right
        A[ind_right, 0:nb_jdiv] = Jx_div_R * w_right
        A[ind_right, nb_jdiv:nb_jdiv+nb_jcurl] = Jx_grad_R * w_right
        A[ind_right, nb_jdiv+nb_jcurl:] = 0.0
        b[i1:i2, :] = float(g_right) * w_right

        # bottom: -Jy = g_bottom  (you used Jy directly in your script; keep your convention)
        A[ind_bottom, 0:nb_jdiv] = Jy_div_B * w_bottom
        A[ind_bottom, nb_jdiv:nb_jdiv+nb_jcurl] = Jy_grad_B * w_bottom
        A[ind_bottom, nb_jdiv+nb_jcurl:] = 0.0
        b[i2:i3, :] = float(g_bottom) * w_bottom

        # top: +Jy = -g_top (keep your sign convention from script)
        A[ind_top, 0:nb_jdiv] = Jy_div_T * w_top
        A[ind_top, nb_jdiv:nb_jdiv+nb_jcurl] = Jy_grad_T * w_top
        A[ind_top, nb_jdiv+nb_jcurl:] = 0.0
        b[i3:i4, :] = float(-g_top) * w_top

        # left: p = bc_left
        A[ind_left, 0:nb_jdiv] = 0.0
        A[ind_left, nb_jdiv:nb_jdiv+nb_jcurl] = 0.0
        A[ind_left, nb_jdiv+nb_jcurl:] = P_L * w_left
        b[i4:i5, :] = float(bc_left) * w_left

        # ---- Darcy rows (cell centers) ----
        # Darcy_x: Jx_div + Jx_grad + M* * px = 0
        A[ind_pde_u, 0:nb_jdiv] = Jx_div_pde * w_darcy
        A[ind_pde_u, nb_jdiv:nb_jdiv+nb_jcurl] = Jx_grad_pde * w_darcy
        A[ind_pde_u, nb_jdiv+nb_jcurl:] = (M_c * P_x) * w_darcy
        b[ind_pde_u, :] = 0.0

        # Darcy_y: Jy_div + Jy_grad + M* * py = 0
        A[ind_pde_v, 0:nb_jdiv] = Jy_div_pde * w_darcy
        A[ind_pde_v, nb_jdiv:nb_jdiv+nb_jcurl] = Jy_grad_pde * w_darcy
        A[ind_pde_v, nb_jdiv+nb_jcurl:] = (M_c * P_y) * w_darcy
        b[ind_pde_v, :] = 0.0

        # ---- Mass rows ----
        # (S/dt) p + div(J) = (S/dt) p^n + q
        # div(J_div)=0; div(J_grad)=div_grad_pde * coef_jcurl
        A[ind_pde_m, 0:nb_jdiv] = 0.0
        A[ind_pde_m, nb_jdiv:nb_jdiv+nb_jcurl] = div_grad_pde * w_mass
        A[ind_pde_m, nb_jdiv+nb_jcurl:] = ((S / dt) * P) * w_mass
        b[ind_pde_m, :] = b_mass * w_mass

        # ---- ridge regularization (optional) ----

        # if ridge is not None and ridge > 0:
        #     # block-diagonal Tikhonov for each coefficient group
        #     I_jd = np.sqrt(ridge) * np.eye(nb_jdiv)
        #     I_jc = np.sqrt(ridge) * np.eye(nb_jcurl)
        #     I_p  = np.sqrt(ridge) * np.eye(nb_p)

        #     A = np.vstack([
        #         A,
        #         np.hstack([I_jd, np.zeros((nb_jdiv, nb_jcurl)), np.zeros((nb_jdiv, nb_p))]),
        #         np.hstack([np.zeros((nb_jcurl, nb_jdiv)), I_jc, np.zeros((nb_jcurl, nb_p))]),
        #         np.hstack([np.zeros((nb_p, nb_jdiv)), np.zeros((nb_p, nb_jcurl)), I_p]),
        #     ])
        #     b = np.vstack([b, np.zeros((nb_jdiv + nb_jcurl + nb_p, 1))])

        # ---- solve LS ----
        coef, residuals, rank, svals = lstsq(A, b, cond=ls_cond, lapack_driver=lapack_driver)

        # condition number estimate from singular values (if available)
        cond_A = np.nan
        if svals is not None and len(svals) >= 2 and float(svals[-1]) > 0:
            cond_A = float(svals[0] / svals[-1])
            print(f"Condition number = {cond_A:.3e}")

        fitted = A @ coef
        res = fitted - b

        # group blocks
        cd = coef[0:nb_jdiv]
        cc = coef[nb_jdiv:nb_jdiv+nb_jcurl]
        cp = coef[nb_jdiv+nb_jcurl:]

        # basic residual norms (on un-augmented rows only)
        res0 = res[:row_num, :]
        b0 = b[:row_num, :]

        mse_total = float(np.mean(res0**2))
        mse_bd = float(np.mean(res0[ind_bd]**2))
        mse_pde = float(np.mean(res0[ind_pde]**2))
        mse_u = float(np.mean(res0[ind_pde_u]**2))
        mse_v = float(np.mean(res0[ind_pde_v]**2))
        mse_m = float(np.mean(res0[ind_pde_m]**2))

        denom = float(np.mean(b0**2) + 1e-300)
        rel_mse_total = float(mse_total / denom)
        rel_mse_bd = float(mse_bd / denom)
        rel_mse_pde = float(mse_pde / denom)
        rel_mse_u = float(mse_u / denom)
        rel_mse_v = float(mse_v / denom)
        rel_mse_m = float(mse_m / denom)
        
        print('\tLS rel_MSE:', rel_mse_total)
        print('\tBD rel_MSE:', rel_mse_bd)
        print('\tPDE rel_MSE:', rel_mse_pde)
        print('\tPDE_u rel_MSE:', rel_mse_u)
        print('\tPDE_v rel_MSE:', rel_mse_v)
        print('\tPDE_s rel_MSE:', rel_mse_m)

        out = dict(
            coef_jdiv=cd,
            coef_jcurl=cc,
            coef_p=cp,
            cond_A=cond_A,
            mse_total=mse_total,
            mse_bd=mse_bd,
            mse_pde=mse_pde,
            mse_u=mse_u,
            mse_v=mse_v,
            mse_m=mse_m,
            rel_mse_total=rel_mse_total,
            rel_mse_bd=rel_mse_bd,
            rel_mse_pde=rel_mse_pde,
            rel_mse_u=rel_mse_u,
            rel_mse_v=rel_mse_v,
            rel_mse_m=rel_mse_m,
            res_norm_total=float(norm(res0)),
            res_norm_bd=float(norm(res0[ind_bd])),
            res_norm_u=float(norm(res0[ind_pde_u])),
            res_norm_v=float(norm(res0[ind_pde_v])),
            res_norm_m=float(norm(res0[ind_pde_m])),
        )
        return out

    # ---------- time stepping ----------
    history: List[Dict[str, Any]] = []
    omega = float(omega0)

    for step in range(int(n_steps)):
        cp = coef_p.copy()

        it_hist = dict(rel_dp=[], res_total=[], res_bd=[], res_u=[], res_v=[], res_m=[], cond_A=[])

        converged = False
        for it in range(int(picard_iters)):
            out = assemble_and_solve_step(cp)
            cp_new = out["coef_p"]

            # Picard update size (use coefficient-space)
            dp = cp_new - cp
            rel_dp = float(norm(dp) / (norm(cp) + 1e-12))

            it_hist["rel_dp"].append(rel_dp)
            it_hist["res_total"].append(out["res_norm_total"])
            it_hist["res_bd"].append(out["res_norm_bd"])
            it_hist["res_u"].append(out["res_norm_u"])
            it_hist["res_v"].append(out["res_norm_v"])
            it_hist["res_m"].append(out["res_norm_m"])
            it_hist["cond_A"].append(out["cond_A"])

            if verbose:
                print(
                    f"[step {step+1}/{n_steps} | it {it+1}/{picard_iters}] "
                    f"relΔp={rel_dp:.2e}, "
                    f"r_tot={out['res_norm_total']:.2e}, "
                    f"r_bd={out['res_norm_bd']:.2e}, "
                    f"r_u={out['res_norm_u']:.2e}, "
                    f"r_v={out['res_norm_v']:.2e}, "
                    f"r_m={out['res_norm_m']:.2e}, "
                    f"cond={out['cond_A']:.2e}"
                )

            # relaxed update
            cp = (1.0 - omega) * cp + omega * cp_new

            if rel_dp < tol_update:
                converged = True
                coef_jdiv = out["coef_jdiv"]
                coef_jcurl = out["coef_jcurl"]
                coef_p = cp
                break

            # crude damping if stagnation grows
            if it >= 3 and it_hist["rel_dp"][-1] > max(it_hist["rel_dp"][-3:]):
                omega = max(0.5 * omega, 0.1)

            coef_jdiv = out["coef_jdiv"]
            coef_jcurl = out["coef_jcurl"]
            coef_p = cp

        history.append(
            dict(
                step=step,
                converged=converged,
                omega=omega,
                iters=(it + 1),
                it_hist=it_hist,
                coef_jdiv=coef_jdiv.copy(),
                coef_jcurl=coef_jcurl.copy(),
                coef_p=coef_p.copy(),
            )
        )

        if verbose:
            status = "OK" if converged else "MAX_ITERS"
            print(
                f"[step {step+1}/{n_steps} | {status}] "
                f"||p||={norm(coef_p):.3e}, "
                f"||jdiv||={norm(coef_jdiv):.3e}, "
                f"||jcurl||={norm(coef_jcurl):.3e}"
            )

    # -------------------------
    # Reconstruct fields (final step) on common grids
    # -------------------------

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
    p_set = p_basis.eval_basis(sample_c, eval_list=["u", "u0", "u1"], **eval_kwargs)
    p_c  = (p_set["u"]  @ coef_p).reshape(ny, nx)
    px_c = (p_set["u0"] @ coef_p).reshape(ny, nx)
    py_c = (p_set["u1"] @ coef_p).reshape(ny, nx)

    # density at cell centers
    rho_c = rho0 * (1.0 + cf * (p_c - p0))

    # J on interior faces
    Jx_div = (jdiv_basis.eval_basis(sample_x, eval_list=["u1"], **eval_kwargs)["u1"] @ coef_jdiv).reshape(ny, nx - 1)
    Jy_div = (-jdiv_basis.eval_basis(sample_y, eval_list=["u0"], **eval_kwargs)["u0"] @ coef_jdiv).reshape(ny - 1, nx)
    Jx_cur = (jcurl_basis.eval_basis(sample_x, eval_list=["u0"], **eval_kwargs)["u0"] @ coef_jcurl).reshape(ny, nx - 1)
    Jy_cur = (jcurl_basis.eval_basis(sample_y, eval_list=["u1"], **eval_kwargs)["u1"] @ coef_jcurl).reshape(ny - 1, nx)

    Jx = Jx_div + Jx_cur
    Jy = Jy_div + Jy_cur

    # J on boundary-including faces
    Jx_div_f = (jdiv_basis.eval_basis(sample_x_f, eval_list=["u1"], **eval_kwargs)["u1"] @ coef_jdiv).reshape(ny, nx + 1)
    Jy_div_f = (-jdiv_basis.eval_basis(sample_y_f, eval_list=["u0"], **eval_kwargs)["u0"] @ coef_jdiv).reshape(ny + 1, nx)
    Jx_cur_f = (jcurl_basis.eval_basis(sample_x_f, eval_list=["u0"], **eval_kwargs)["u0"] @ coef_jcurl).reshape(ny, nx + 1)
    Jy_cur_f = (jcurl_basis.eval_basis(sample_y_f, eval_list=["u1"], **eval_kwargs)["u1"] @ coef_jcurl).reshape(ny + 1, nx)

    Jx_f = Jx_div_f + Jx_cur_f
    Jy_f = Jy_div_f + Jy_cur_f

    # J at cell centers
    Jx_div_c = (jdiv_basis.eval_basis(sample_c, eval_list=["u1"], **eval_kwargs)["u1"] @ coef_jdiv).reshape(ny, nx)
    Jy_div_c = (-jdiv_basis.eval_basis(sample_c, eval_list=["u0"], **eval_kwargs)["u0"] @ coef_jdiv).reshape(ny, nx)
    Jx_cur_c = (jcurl_basis.eval_basis(sample_c, eval_list=["u0"], **eval_kwargs)["u0"] @ coef_jcurl).reshape(ny, nx)
    Jy_cur_c = (jcurl_basis.eval_basis(sample_c, eval_list=["u1"], **eval_kwargs)["u1"] @ coef_jcurl).reshape(ny, nx)

    Jx_c = Jx_div_c + Jx_cur_c
    Jy_c = Jy_div_c + Jy_cur_c

    # divergence diagnostics (simple: div of J_grad at cell centers)
    div_grad_c = (
        (jcurl_basis.eval_basis(sample_c, eval_list=["u00"], **eval_kwargs)["u00"] @ coef_jcurl).reshape(ny, nx)
      + (jcurl_basis.eval_basis(sample_c, eval_list=["u11"], **eval_kwargs)["u11"] @ coef_jcurl).reshape(ny, nx)
    )

    # recover velocity v = J / rho on faces and centers
    # density on x-face and y-face grids
    px = (p_basis.eval_basis(sample_x, eval_list=["u"], **eval_kwargs)["u"] @ coef_p).reshape(ny, nx - 1)
    py = (p_basis.eval_basis(sample_y, eval_list=["u"], **eval_kwargs)["u"] @ coef_p).reshape(ny - 1, nx)
    rho_x = rho0 * (1.0 + cf * (px - p0))
    rho_y = rho0 * (1.0 + cf * (py - p0))

    px_f = (p_basis.eval_basis(sample_x_f, eval_list=["u"], **eval_kwargs)["u"] @ coef_p).reshape(ny, nx + 1)
    py_f = (p_basis.eval_basis(sample_y_f, eval_list=["u"], **eval_kwargs)["u"] @ coef_p).reshape(ny + 1, nx)
    rho_x_f = rho0 * (1.0 + cf * (px_f - p0))
    rho_y_f = rho0 * (1.0 + cf * (py_f - p0))

    vx = Jx / rho_x
    vy = Jy / rho_y
    vx_f = Jx_f / rho_x_f
    vy_f = Jy_f / rho_y_f
    vx_c = Jx_c / rho_c
    vy_c = Jy_c / rho_c

    # -------------------------
    # Return everything useful
    # -------------------------

    return dict(
        # coefficients
        coef_jdiv=coef_jdiv,
        coef_jcurl=coef_jcurl,
        coef_p=coef_p,
        # basis
        jdiv_basis=jdiv_basis,
        jcurl_basis=jcurl_basis,
        p_basis=p_basis,
        # points
        x_pde=x_pde,
        x_bd=x_bd,
        x_left=x_left_wo,
        x_right=x_right_wo,
        x_bottom=x_bottom_wo,
        x_top=x_top_wo,
        # permeability
        k_pde=k_pde,
        kappa_stats=dict(mean=kappa_mean, min=kappa_min, max=kappa_max, ratio=kappa_ratio),
        # discretization
        domain=dict(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
        # weights used
        weights=dict(
            bc_left=W.bc_left, bc_right=W.bc_right, bc_bottom=W.bc_bottom, bc_top=W.bc_top,
            darcy=W.darcy, mass=W.mass
        ),
        # solver history
        history=history,
        # reconstructed fields (final step)
        fields=dict(
            p=p_c,
            px=px_c,
            py=py_c,
            rho_c=rho_c,
            Jx=Jx,
            Jy=Jy,
            Jx_f=Jx_f,
            Jy_f=Jy_f,
            Jx_c=Jx_c,
            Jy_c=Jy_c,
            div_grad_c=div_grad_c,
            vx=vx,
            vy=vy,
            vx_f=vx_f,
            vy_f=vy_f,
            vx_c=vx_c,
            vy_c=vy_c,
        ),
    )
