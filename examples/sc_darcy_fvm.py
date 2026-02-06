#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Slightly compressible Darcy flow (backward Euler + Picard)

Mass balance:
    ε ∂ρ/∂t + div( ρ u ) = q
Darcy:
    u = - (κ/μ) ∇p
Fluid:
    ρ(p) = ρ0 [1 + cf (p - p0)]

BCs:
  LEFT   : Dirichlet pressure p = p_left
  RIGHT  : prescribed outward MASS FLUX m_right  (outward normal +x)
  TOP    : prescribed outward MASS FLUX m_top    (outward normal -y)
  BOTTOM : prescribed outward MASS FLUX m_bottom (outward normal +y)

Sign: positive means flowing OUT of the domain in the outward normal direction.

Outputs:
  - p
  - vx, vy              (Darcy velocity, interior faces, old shapes)
  - mx, my              (mass flux = rho*u, same shapes as vx, vy)
  - vx_f, vy_f          (Darcy vel on full face-staggered grid, incl boundaries)
  - mx_f, my_f          (mass flux on full face-staggered grid, incl boundaries)
  - vx_c, vy_c          (Darcy vel at cell centers)
  - mx_c, my_c          (mass flux at cell centers)
  - rho_last            (final density field)
  - kappa, dx, dy
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.io import loadmat
import matplotlib.pyplot as plt


# ============================================================
#               kappa field construction
# ============================================================

def make_constant_kappa_field(Nx, Ny, val=0.3*6.328):
    return np.ones((Ny, Nx)) * val


# ============================================================
#                  helper utilities
# ============================================================

def harmonic_average(a, b, eps=1e-12):
    return 2.0*a*b / (a + b + eps)

def rho_linear(p, rho0, cf, p0):
    return rho0 * (1.0 + cf*(p - p0))

def rel_errors(a, b, eps=1e-30):
    diff = a - b
    rel_L1 = np.sum(np.abs(diff)) / (np.sum(np.abs(b)) + eps)
    rel_L2 = np.linalg.norm(diff) / (np.linalg.norm(b) + eps)
    return rel_L1, rel_L2

def rel_errors_percent(a, b, eps=1e-30):
    diff = a - b
    rel_L1 = 100.0 * np.sum(np.abs(diff)) / (np.sum(np.abs(b)) + eps)
    rel_L2 = 100.0 * np.linalg.norm(diff) / (np.linalg.norm(b) + eps)
    return rel_L1, rel_L2

# ============================================================
#   -div( rho_k * (kappa/mu) grad p_new )  matrix
# ============================================================

def laplacian_matrix(Nx, Ny, dx, dy,
                     kappa, mu,
                     rho,
                     dirichlet='left'):
    Ny_, Nx_ = kappa.shape
    assert Ny_ == Ny and Nx_ == Nx
    assert rho.shape == (Ny, Nx)

    def face_Mx(i, jL, jR):
        k_h = harmonic_average(kappa[i, jL], kappa[i, jR])
        rho_avg = 0.5*(rho[i, jL] + rho[i, jR])
        return (rho_avg * k_h) / mu

    def face_My(iT, iB, j):
        k_h = harmonic_average(kappa[iT, j], kappa[iB, j])
        rho_avg = 0.5*(rho[iT, j] + rho[iB, j])
        return (rho_avg * k_h) / mu

    N = Nx*Ny
    A = lil_matrix((N, N))

    for i in range(Ny):
        for j in range(Nx):
            idx = i*Nx + j

            M_right = face_Mx(i, j, j+1) if j < Nx-1 else 0.0
            M_left  = face_Mx(i, j-1, j) if j > 0     else 0.0
            M_up    = face_My(i-1, i, j) if i > 0     else 0.0
            M_down  = face_My(i, i+1, j) if i < Ny-1  else 0.0

            A[idx, idx] = -(M_right + M_left)/dx**2 - (M_up + M_down)/dy**2

            # Dirichlet p_left on j=0
            if (j == 0) and (dirichlet == 'left'):
                M_leftBC = (rho[i,0] * kappa[i,0] / mu)
                A[idx, idx] -= 2.0 * M_leftBC / dx**2

            # optional hook for right Dirichlet
            if (j == Nx-1) and (dirichlet == 'right'):
                M_rightBC = (rho[i,-1] * kappa[i,-1] / mu)
                A[idx, idx] -= 2.0 * M_rightBC / dx**2

            if j < Nx-1:
                A[idx, idx+1] = M_right / dx**2
            if j > 0:
                A[idx, idx-1] = M_left  / dx**2
            if i > 0:
                A[idx, idx-Nx] = M_up    / dy**2
            if i < Ny-1:
                A[idx, idx+Nx] = M_down  / dy**2

    return A.tocsr()


# ============================================================
#   RHS for Picard linearized BE system
# ============================================================

def rhs_vector(Nx, Ny, dx, dy,
               kappa, mu,
               p_left,
               m_right, m_top, m_bottom,
               q,
               epsilon, rho0, cf, p0,
               p_n,
               rho,
               dt,
               dirichlet='left'):
    """
    (ε ρ0 cf / dt) p_new  - div( ρ_k κ/μ ∇p_new )
    =
    (ε ρ0 cf / dt) p_n + BC_mass_flux + q + Dirichlet-left forcing

    m_right, m_top, m_bottom are mass flux BCs [ρ u . n_out].
    """

    Ny_, Nx_ = rho.shape
    assert Ny_ == Ny and Nx_ == Nx
    assert p_n.shape == (Ny, Nx)
    assert q.shape   == (Ny, Nx)

    S_linear = epsilon * (rho0 * cf) / dt

    b = (S_linear * p_n.reshape(-1)).copy()

    for i in range(Ny):
        for j in range(Nx):
            idx = i*Nx + j

            # RIGHT boundary (+x outward)
            if j == Nx-1:
                b[idx] -= m_right / dx

            # TOP boundary (-y outward)
            if i == 0:
                b[idx] += m_top / dy

            # BOTTOM boundary (+y outward)
            if i == Ny-1:
                b[idx] += m_bottom / dy

            # body source
            b[idx] += q[i, j]

    # Left Dirichlet p_left
    if dirichlet == 'left':
        for i in range(Ny):
            idx = i*Nx + 0
            M_leftBC = (rho[i,0] * kappa[i,0] / mu)
            b[idx] += (2.0 * M_leftBC / dx**2) * p_left

    return b, S_linear


# ============================================================
#   One backward-Euler step via Picard
# ============================================================

def BE_picard(
    Nx, Ny, dx, dy,
    kappa, epsilon,
    p_n,
    cf, p0, rho0,
    dt, mu,
    q,
    p_left,
    m_right, m_top, m_bottom,
    dirichlet='left',
    picard_tol=1e-6, picard_maxit=25,
    verbose=False
):
    p_k = p_n.copy()

    for it in range(1, picard_maxit+1):
        rho_k = rho_linear(p_k, rho0, cf, p0)

        A_diff = -laplacian_matrix(
            Nx, Ny, dx, dy,
            kappa, mu,
            rho_k,
            dirichlet=dirichlet
        )

        b_vec, S_linear = rhs_vector(
            Nx, Ny, dx, dy,
            kappa, mu,
            p_left,
            m_right, m_top, m_bottom,
            q,
            epsilon, rho0, cf, p0,
            p_n,
            rho_k,
            dt,
            dirichlet=dirichlet
        )

        N = Nx*Ny
        A_mass = csr_matrix(
            (np.full(N, S_linear), (np.arange(N), np.arange(N))),
            shape=(N, N)
        )

        A_sys = A_mass +  A_diff
        
        # p_new_flat = spsolve(A_sys, b_vec)
        
        from scipy.sparse.linalg import bicgstab, spilu, LinearOperator

        # ---------- ILU ----------
        ilu = spilu(A_sys.tocsc())
        
        iter_count = [0]   
        
        def cb(rk):
            iter_count[0] += 1
        
        def Mv(v):
            return ilu.solve(v)
        
        M = LinearOperator(A_sys.shape, Mv)
        

        # ---------- BICGSTAB ----------
        p_new_flat, info = bicgstab(
            A_sys, b_vec,
            M=M,           
            rtol=1e-6,      
            maxiter=1000,   
            callback=cb,
        )
        print("BICGSTAB iterations =", iter_count[0])
        if info != 0:
            print(f"[BICGSTAB] did not converge, info = {info}")


        p_new = p_new_flat.reshape(Ny, Nx)

        num = np.linalg.norm((p_new - p_k).ravel(), 2)
        den = np.linalg.norm(p_new.ravel(), 2) + 1e-30
        rel_change = num / den

        if verbose:
            print(f"[Picard it={it}] rel_change = {rel_change:.3e}")

        if rel_change < picard_tol:
            rho_k = rho_linear(p_new, rho0, cf, p0)
            return p_new, rho_k, it, rel_change
        
        p_k = p_new

    return p_k, rho_k, picard_maxit, rel_change


# ============================================================
#   Reconstruct cell-center Darcy vel for plotting
# ============================================================

def reconstruct_cell_center_velocity(p, kappa, dx, dy,
                                     p_left,
                                     vx_f, vy_f,
                                     mu):
    """
    vx_c, vy_c at cell centers using ghost gradients.
    vx_f, vy_f are Darcy velocities on faces.
    """
    Ny, Nx = p.shape
    eps = 1e-30

    M_R = kappa[:, -1] / mu
    M_T = kappa[0,  :] / mu
    M_B = kappa[-1, :] / mu

    # grad p from boundary Darcy vel: u = -(kappa/mu) grad p
    gR = - vx_f[:, -1] / (M_R + eps)   # dp/dx at right
    gT = - vy_f[0,   :] / (M_T + eps)  # dp/dy at top
    gB = - vy_f[-1,  :] / (M_B + eps)  # dp/dy at bottom

    pLghost = 2.0*p_left - p[:, 0]
    pRghost = p[:, -1] + dx * gR
    pTghost = p[0,  :] - dy * gT
    pBghost = p[-1, :] + dy * gB

    dpdx = np.zeros_like(p)
    dpdy = np.zeros_like(p)

    if Nx == 1:
        dpdx[:, 0] = (pRghost - pLghost)/(2.0*dx)
    else:
        dpdx[:, 0]    = (p[:, 1] - pLghost)/(2.0*dx)
        if Nx >= 3:
            dpdx[:, 1:-1] = (p[:, 2:] - p[:, :-2])/(2.0*dx)
        dpdx[:, -1]   = (pRghost - p[:, -2])/(2.0*dx)

    if Ny == 1:
        dpdy[0, :] = (pBghost - pTghost)/(2.0*dy)
    else:
        dpdy[0, :]    = (p[1, :] - pTghost)/(2.0*dy)
        if Ny >= 3:
            dpdy[1:-1, :] = (p[2:, :] - p[:-2, :])/(2.0*dy)
        dpdy[-1, :]   = (pBghost - p[-2, :])/(2.0*dy)

    vx_c = -(kappa / mu) * dpdx
    vy_c = -(kappa / mu) * dpdy
    return vx_c, vy_c


# ============================================================
#   Build velocities + mass fluxes (INTERIOR + faces + centers)
# ============================================================

def build_vel_and_flux_fields(p, kappa, dx, dy,
                              rho_last,
                              m_right, m_top, m_bottom,
                              mu,
                              p_left):
    """
    Return:
      vx, vy        (Darcy vel interior faces)
      mx, my        (mass flux interior faces)
      vx_f, vy_f    (Darcy vel full faces, incl BC)
      mx_f, my_f    (mass flux full faces, incl BC)
      vx_c, vy_c    (Darcy vel cell centers)
      mx_c, my_c    (mass flux cell centers)
    """

    Ny, Nx = p.shape

    # --------------------
    # interior Darcy velocity (old vx, vy shapes)
    # --------------------
    vx = np.zeros((Ny, Nx-1))
    vy = np.zeros((Ny-1, Nx))
    for i in range(Ny):
        for j in range(Nx-1):
            kf = harmonic_average(kappa[i, j], kappa[i, j+1])
            vx[i, j] = -(kf / mu) * (p[i, j+1] - p[i, j]) / dx
    for i in range(Ny-1):
        for j in range(Nx):
            kf = harmonic_average(kappa[i, j], kappa[i+1, j])
            vy[i, j] = -(kf / mu) * (p[i+1, j] - p[i, j]) / dy

    # --------------------
    # interior mass flux mx,my with averaged density on each interior face
    # --------------------
    mx = np.zeros_like(vx)
    my = np.zeros_like(vy)
    for i in range(Ny):
        for j in range(Nx-1):
            rho_face = 0.5*(rho_last[i, j] + rho_last[i, j+1])
            mx[i, j] = rho_face * vx[i, j]
    for i in range(Ny-1):
        for j in range(Nx):
            rho_face = 0.5*(rho_last[i, j] + rho_last[i+1, j])
            my[i, j] = rho_face * vy[i, j]

    # --------------------
    # full-face Darcy velocity vx_f, vy_f (size Ny x (Nx+1), (Ny+1) x Nx)
    # --------------------
    vx_f = np.zeros((Ny, Nx+1))
    vy_f = np.zeros((Ny+1, Nx))

    # put interior into middle
    vx_f[:, 1:Nx] = vx
    vy_f[1:Ny, :] = vy

    # boundary Darcy velocity from mass flux BCs / density
    # right boundary (+x outward normal):
    rho_right = rho_last[:, -1]
    vx_f[:, Nx] = -m_right / (rho_right + 1e-30)

    # top boundary (-y outward):
    rho_top = rho_last[0, :]
    vy_f[0, :] = -m_top / (rho_top + 1e-30)

    # bottom boundary (+y outward):
    rho_bottom = rho_last[-1, :]
    vy_f[Ny, :] = -m_bottom / (rho_bottom + 1e-30)

    # left boundary x-face: 
    vx_f[:, 0] = - (kappa[:, 0] / mu) * (2.0 * (p[:, 0] - p_left) / dx)


    # --------------------
    # full-face mass flux mx_f, my_f
    # --------------------
    mx_f = np.zeros_like(vx_f)
    my_f = np.zeros_like(vy_f)

    # interior x-faces (between j and j+1) go to slot j+1 in vx_f layout
    for i in range(Ny):
        for j in range(Nx-1):
            rho_face = 0.5*(rho_last[i, j] + rho_last[i, j+1])
            mx_f[i, j+1] = rho_face * vx_f[i, j+1]

    # left boundary face j=0 ~ cell j=0
    mx_f[:, 0] = rho_last[:, 0] * vx_f[:, 0]
    # right boundary face j=Nx ~ cell j=Nx-1
    mx_f[:, Nx] = rho_last[:, -1] * vx_f[:, Nx]

    # interior y-faces
    for i in range(Ny-1):
        for j in range(Nx):
            rho_face = 0.5*(rho_last[i, j] + rho_last[i+1, j])
            my_f[i+1, j] = rho_face * vy_f[i+1, j]

    # top boundary face i=0 ~ cell i=0
    my_f[0, :] = rho_last[0, :] * vy_f[0, :]
    # bottom boundary face i=Ny ~ cell i=Ny-1
    my_f[Ny, :] = rho_last[-1, :] * vy_f[Ny, :]

    # --------------------
    # cell-center Darcy velocity + mass flux
    # --------------------
    vx_c, vy_c = reconstruct_cell_center_velocity(
        p, kappa, dx, dy,
        p_left,
        vx_f, vy_f,
        mu
    )
    mx_c = rho_last * vx_c
    my_c = rho_last * vy_c

    return vx, vy, mx, my, vx_f, vy_f, mx_f, my_f, vx_c, vy_c, mx_c, my_c


# ============================================================
#     Visualization helper (optional sanity)
# ============================================================

def plot_outputs(Lx, Ly,
                 p,
                 vx_c, vy_c, mx_c, my_c):
    extent = (0.0, Lx, 0.0, Ly)
    Ny, Nx = p.shape
    hx = Lx / Nx
    hy = Ly / Ny

    xs_cc = np.linspace(0.5*hx, Lx - 0.5*hx, Nx)
    ys_cc = np.linspace(0.5*hy, Ly - 0.5*hy, Ny)
    Xcc, Ycc = np.meshgrid(xs_cc, ys_cc, indexing="xy")

    step = max(1, Nx // 28)
    uq = vx_c[::step, ::step]
    vq = vy_c[::step, ::step]
    sp = np.sqrt(uq**2 + vq**2)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axs = axs.ravel()

    im0 = axs[0].imshow(p, origin='lower',
                        extent=extent, cmap='RdYlBu_r')
    axs[0].set_title("p")
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(vx_c, origin='lower',
                        extent=extent, cmap='RdYlBu_r')
    axs[1].set_title("vx_c (Darcy vel)")
    fig.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(vy_c, origin='lower',
                        extent=extent, cmap='RdYlBu_r')
    axs[2].set_title("vy_c (Darcy vel)")
    fig.colorbar(im2, ax=axs[2])

    qplt = axs[3].quiver(Xcc[::step, ::step],
                         Ycc[::step, ::step],
                         uq, vq, sp,
                         cmap='RdYlBu_r',
                         scale=10,
                         width=0.006)
    axs[3].set_title("Darcy vel @ centers")
    fig.colorbar(qplt, ax=axs[3])

    im4 = axs[4].imshow(mx_c, origin='lower',
                        extent=extent, cmap='RdYlBu_r')
    axs[4].set_title("mx_c (mass flux)")
    fig.colorbar(im4, ax=axs[4])

    im5 = axs[5].imshow(my_c, origin='lower',
                        extent=extent, cmap='RdYlBu_r')
    axs[5].set_title("my_c (mass flux)")
    fig.colorbar(im5, ax=axs[5])

    plt.show()
    


# ============================================================
#     Main driver with time marching + Picard
# ============================================================

def run_sc_darcy_fvm(
        Nx=50, Ny=50, Lx=2.0, Ly=2.0,
        n_steps=10, dt=10.0,
        cf=2e-4, p0=1500.0, rho0=1.0,
        mu=0.04, epsilon=0.25,
        use_kappa_file=False, kappa_file='kappa50x50_2.mat',
        constant_kappa=True, kappa_val=0.3*6.328,
        beta1=1.2, beta2=1.2, beta3=1.2,
        p_left=1500.0,
        m_right=0.0,   # outward +x
        m_top=0.0,     # outward -y
        m_bottom=0.0,  # outward +y
        q=None,
        picard_tol=1e-8, picard_maxit=25,
        verbose=True,
        do_plot=True
    ):
    dx = Lx / Nx
    dy = Ly / Ny

    # permeability
    if use_kappa_file:
        kappa = loadmat(kappa_file)['kappa']
        kappa = kappa * 0.01 * 6.328 * 0.0008
        assert kappa.shape == (Ny, Nx)
    elif constant_kappa:
        kappa = make_constant_kappa_field(Nx, Ny, val=kappa_val)
    else:
        raise ValueError('not constant_kappa nor use_kappa_file')

    if q is None:
        q = np.zeros((Ny, Nx))

    p = np.full((Ny, Nx), p0, dtype=float)
    
    m_right = -m_right
    

    for n in range(1, n_steps+1):
        
        p_new, rho_last, it_used, relchg = BE_picard(
            Nx, Ny, dx, dy,
            kappa, epsilon,
            p_n=p,
            cf=cf, p0=p0, rho0=rho0,
            dt=dt, mu=mu,
            q=q,
            p_left=p_left,
            m_right=m_right, m_top=m_top, m_bottom=m_bottom,
            dirichlet='left',
            picard_tol=picard_tol, picard_maxit=picard_maxit,
            verbose=verbose
        )
        if verbose:
            print(f"[time step {n}/{n_steps}] Picard iters={it_used}, final rel_change={relchg:.3e}")
        
        p = p_new
        

    # build all velocity / flux fields
    (vx, vy, mx, my,
     vx_f, vy_f, mx_f, my_f,
     vx_c, vy_c, mx_c, my_c) = build_vel_and_flux_fields(
        p, kappa, dx, dy,
        rho_last,
        m_right=m_right,
        m_top=m_top,
        m_bottom=m_bottom,
        mu=mu,
        p_left=p_left
    )

    if do_plot:
        plot_outputs(Lx, Ly,
                     p,
                     vx_c, vy_c, mx_c, my_c)

    # return EVERYTHING
    return {
        "p": p,
        "vx": vx, "vy": vy,             # Darcy vel interior faces
        "mx": mx, "my": my,             # mass flux interior faces
        "vx_f": vx_f, "vy_f": vy_f,     # Darcy vel faces (incl BC)
        "mx_f": mx_f, "my_f": my_f,     # mass flux faces (incl BC)
        "vx_c": vx_c, "vy_c": vy_c,     # Darcy vel centers
        "mx_c": mx_c, "my_c": my_c,     # mass flux centers
        "rho_last": rho_last,
        "kappa": kappa,
        "dx": dx, "dy": dy
    }


# ============================================================
#                         demo
# ============================================================

if __name__ == "__main__":
    Nx, Ny = 50, 50
    Lx, Ly = 2.0, 2.0
    
    Lf = 2500 / Lx
    
    T_days = 90.0 #/ Lf / Lf
    dt = T_days / 9
    n_steps = int(T_days / dt)
    n_steps = 9
    
    # Example BC:
    rho = 24.0            # lbm/ft^3 (example)
    u_top = 5.0 #* Lf            # ft/day Darcy vel outward (-y)
    m_top = rho * u_top  # lbm/(ft^2 day)

    out = run_sc_darcy_fvm(
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
        n_steps=n_steps, dt=dt,
        cf=2e-4, p0=1500.0, rho0=rho,
        mu=0.04, epsilon=0.25,
        use_kappa_file=False, kappa_file='kappa50x50_2.mat', constant_kappa=True, kappa_val = 0.3*6.328* 0.0008,
        p_left=1500.0,
        m_right=0.0,
        m_top=0.0,
        m_bottom=-m_top,
        q=np.zeros((Ny, Nx)),
        picard_tol=1e-6, picard_maxit=25,
        verbose=True,
        do_plot=False
    )

    
    from down_sampler_utils import (   
        downsample_cell_center_scalar,
        downsample_cell_center_vector_comp,
        downsample_internal_vx,
        downsample_internal_vy,
        downsample_all_vx,
        downsample_all_vy
    )
    fine_solution = np.load("fine_solution_sc_homogeneous_massfluxBC_t090.npz")

    p_fine      = fine_solution["p"]
    vx_fine     = fine_solution["vx"]
    vy_fine     = fine_solution["vy"]
    vx_f_fine   = fine_solution["vx_f"]
    vy_f_fine   = fine_solution["vy_f"]
    
    mx_fine     = fine_solution["mx"]
    my_fine     = fine_solution["my"]
    mx_f_fine   = fine_solution["mx_f"]
    my_f_fine   = fine_solution["my_f"]
    

    mx_c_fine   = fine_solution["mx_c"]
    my_c_fine   = fine_solution["my_c"]
    
    fine_n = 1000
    
    p_ref   = downsample_cell_center_scalar(p_fine, Lx, Ly, fine_n, fine_n, 50, 50)
    vx_ref  = downsample_internal_vx(vx_fine, Lx, Ly, fine_n, fine_n, 50, 50)
    vy_ref  = downsample_internal_vy(vy_fine, Lx, Ly, fine_n, fine_n, 50, 50)
    mx_ref  = downsample_internal_vx(mx_fine, Lx, Ly, fine_n, fine_n, 50, 50)
    my_ref  = downsample_internal_vy(my_fine, Lx, Ly, fine_n, fine_n, 50, 50)
    vx_f_ref  = downsample_all_vx(vx_f_fine, Lx, Ly, fine_n, fine_n, 50, 50)
    vy_f_ref  = downsample_all_vy(vy_f_fine, Lx, Ly, fine_n, fine_n, 50, 50)
    mx_f_ref  = downsample_all_vx(mx_f_fine, Lx, Ly, fine_n, fine_n, 50, 50)
    my_f_ref  = downsample_all_vy(my_f_fine, Lx, Ly, fine_n, fine_n, 50, 50)
    mx_c_ref  = downsample_cell_center_vector_comp(mx_c_fine, Lx, Ly, fine_n, fine_n, 50, 50)
    my_c_ref  = downsample_cell_center_vector_comp(my_c_fine, Lx, Ly, fine_n, fine_n, 50, 50)
    
    ref = {
        "p": p_ref,
        "vx": vx_ref, "vy": vy_ref,
        "mx": mx_ref, "my": my_ref,
        "vx_f": vx_f_ref, "vy_f": vy_f_ref,
        "mx_f": mx_f_ref, "my_f": my_f_ref,
        "mx_c": mx_c_ref, "my_c": my_c_ref
    }

    fields = ["p", "vx", "vy", "mx", "my",
          "vx_f", "vy_f", "mx_f", "my_f",
          "mx_c", "my_c"]

    errors = {}
    
    for key in fields:
        relL1, relL2 = rel_errors(out[key], ref[key])
        errors[key] = (relL1, relL2)
        print(f"{key:6s}:  relL1 = {relL1:.3e},  relL2 = {relL2:.3e}")
        
    for key in fields:
        relL1, relL2 = rel_errors_percent(out[key], ref[key])
        errors[key] = (relL1, relL2)
        print(f"{key:6s}:  relL1 = {relL1:8.3f}%,  relL2 = {relL2:8.3f}%")
        

        
    print('Done!')


