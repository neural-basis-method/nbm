#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 02:09:33 2025

With divergence free basis

@author: y.wang
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.io import loadmat

def laplacian_matrix(Nx, Ny, dx, dy, k, dirichlet='left'):
    N = Nx * Ny
    A = lil_matrix((N, N))
    kx = np.zeros((Ny, Nx+1))
    ky = np.zeros((Ny+1, Nx))

    for i in range(Ny):
        for j in range(Nx-1):
            kx[i, j+1] = 2 * k[i, j] * k[i, j+1] / (k[i, j] + k[i, j+1] + 1e-12)
    for i in range(Ny-1):
        for j in range(Nx):
            ky[i+1, j] = 2 * k[i, j] * k[i+1, j] / (k[i, j] + k[i+1, j] + 1e-12)
    if dirichlet == 'left':
        kx[:, 0] = 2 * k[:, 0] / (1 + 1)
    elif dirichlet == 'right':
        kx[:, Nx] = 2 * k[:, -1] / (1 + 1)

    for i in range(Ny):
        for j in range(Nx):
            idx = i * Nx + j
            k_right = kx[i, j+1] if j < Nx-1 else 0
            k_left  = kx[i, j]   if j > 0     else 0
            k_up    = ky[i, j]   if i > 0     else 0
            k_down  = ky[i+1, j] if i < Ny-1  else 0
            A[idx, idx] = - (k_right + k_left) / dx**2 - (k_up + k_down) / dy**2
            if j == 0 and dirichlet == 'left':
                A[idx, idx] -= 2 * kx[i, j] / dx**2
            if j == Nx-1 and dirichlet == 'right':
                A[idx, idx] -= 2 * kx[i, j+1] / dx**2
            if j < Nx-1:
                A[idx, idx + 1] = k_right / dx**2
            if j > 0:
                A[idx, idx - 1] = k_left / dx**2
            if i > 0:
                A[idx, idx - Nx] = k_up / dy**2
            if i < Ny - 1:
                A[idx, idx + Nx] = k_down / dy**2
    return A.tocsr()

def rhs_vector(Nx, Ny, dx, dy, k, g_left, g_right, g_top, g_bottom, s, dirichlet='left'):
    b = s.copy().reshape(-1)
    for i in range(Ny):
        for j in range(Nx):
            idx = i * Nx + j
            if (dirichlet == 'left' and j == 0) or (dirichlet == 'right' and j == Nx-1):
                continue
            if i == 0:
                b[idx] += g_top / dy
            if i == Ny-1:
                b[idx] -= g_bottom / dy
            if j == 0:
                b[idx] += g_left / dx
            if j == Nx-1:
                b[idx] -= g_right / dx
    if dirichlet == 'left':
        for i in range(Ny):
            idx = i * Nx
            b[idx] -= 2 * k[i, 0] * g_left / dx**2
    return b

def direct_solve(Nx, Ny, dx, dy, k, s, bc_left, bc_right, bc_top, bc_bottom):
    A = laplacian_matrix(Nx, Ny, dx, dy, k, 'left')
    b = rhs_vector(Nx, Ny, dx, dy, k, bc_left, bc_right, bc_top, bc_bottom, s, 'left')
    # x = spsolve(A, b)
    
    from scipy.sparse.linalg import gmres, bicgstab, spilu, LinearOperator

    # ---------- ILU 预条件器 ----------
    ilu = spilu(A.tocsc())
    
    iter_count = [0]   # 用 list 是因为闭包能修改
    
    def cb(rk):
        iter_count[0] += 1
    
    def Mv(v):
        # 解 M^{-1} v，这里 M ≈ A（ILU 分解）
        return ilu.solve(v)
    
    M = LinearOperator(A.shape, Mv)
    
    # # ---------- GMRES 求解 ----------
    # x, info = gmres(
    #     A, b,
    #     M=M,          # 预条件器
    #     rtol=1e-6,     # 误差容忍
    #     restart=100,  # Krylov 维度
    #     maxiter=1000,  # 最多迭代次数
    #     callback=cb,
    #     callback_type="pr_norm"
    # )
    # print("GMRES iterations =", iter_count[0])
    # if info != 0:
    #     print(f"[GMRES] did not converge, info = {info}")
    
    # ---------- BICGSTAB 求解 ----------
    x, info = bicgstab(
        A, b,
        M=M,          # 预条件器
        rtol=1e-6,     # 误差容忍
        maxiter=1000,  # 最多迭代次数
        callback=cb,
    )
    # print("BICGSTAB iterations =", iter_count[0])
    if info != 0:
        print(f"[BICGSTAB] did not converge, info = {info}")
    
    
    
    rel_mse = np.mean((b - A @ x)**2) / np.mean(b**2)
    print('\FVM rel_MSE:', rel_mse)
    
    P = x.reshape(Ny, Nx)
    vx = np.zeros((Ny, Nx-1))
    vy = np.zeros((Ny-1, Nx))
    
    for i in range(Ny):
        for j in range(Nx-1):
            k_face = harmonic_average(k[i,j], k[i,j+1])
            vx[i,j] = -k_face * (P[i,j+1] - P[i,j]) / dx
    for i in range(Ny-1):
        for j in range(Nx):
            k_face = harmonic_average(k[i,j], k[i+1,j])
            vy[i,j] = -k_face * (P[i+1,j] - P[i,j]) / dy
            
    vx_f = np.zeros((Ny, Nx+1)) # internal interfaces + boundary
    vy_f = np.zeros((Ny+1, Nx)) # internal interfaces + boundary
    vx_f[:, 1:Nx] = vx
    vy_f[1:Ny, :] = vy
            
    vx_f[:, Nx] = -bc_right
    vy_f[0,  :] = -bc_top
    vy_f[Ny, :] = -bc_bottom
            
    vx_f[:,0] = vx_f[:, 1] + dx * (vy_f[1:, 0] - vy_f[:-1, 0]) / dy
    
    # check
    # div = ((vx_f[:,1:] - vx_f[:,:-1])/dx + (vy_f[1:,:] - vy_f[:-1,:])/dy)
    # print(np.max(np.abs(div)))  
    
    return P, vx, vy, vx_f, vy_f

def harmonic_average(a, b):
    return 2 * a * b / (a + b + 1e-12)

def compute_divergence(vx, vy, dx, dy):
    Ny, Nx_minus1 = vx.shape
    Ny_minus1, Nx = vy.shape
    div = np.zeros((Ny, Nx))
    for i in range(1, Ny-1):
        for j in range(1, Nx-1):
            dVx = (vx[i,j] - vx[i,j-1]) / dx
            dVy = (vy[i,j] - vy[i-1,j]) / dy
            div[i,j] = dVx + dVy
    return div

def center_velocity_from_gradient_2nd_ghost(P, kappa, dx, dy, g_left, vx_f, vy_f):
    """
    用 ghost + 二阶中心差分在 cell center 计算 Darcy 速度：
      右/上/下是 Neumann(给速度u_n) -> 用 Darcy 得 dp/dn，再生成 ghost 压力
      左是 Dirichlet(p=g_left) -> 二阶 ghost: pLghost = 2*g_left - p0
    返回 vx_c, vy_c (Ny, Nx)
    """
    Ny, Nx = P.shape
    eps = 1e-30

    # 边界处的压力法向导数（用 Darcy: dp/dn = -u_n/kappa）
    gR = - vx_f[:, Nx] / (kappa[:, -1] + eps)   # ∂p/∂x at right boundary
    gT = - vy_f[0,  :] / (kappa[0,  :] + eps)   # ∂p/∂y at top boundary
    gB = - vy_f[Ny, :] / (kappa[-1, :] + eps)   # ∂p/∂y at bottom boundary

    # ghost 压力（注意号）：
    pLghost = 2.0*g_left - P[:, 0]           # 左：Dirichlet
    pRghost = P[:, -1] + dx * gR             # 右：+x 外侧 → 加
    pTghost = P[0,  :] - dy * gT             # 上：−y 外侧 → 减   *** 修正点 ***
    pBghost = P[-1, :] + dy * gB             # 下：+y 外侧 → 加

    dpdx = np.zeros_like(P, dtype=float)
    dpdy = np.zeros_like(P, dtype=float)

    # x 向梯度
    if Nx == 1:
        dpdx[:, 0] = (pRghost - pLghost) / (2.0*dx)
    else:
        dpdx[:, 0]    = (P[:, 1]   - pLghost) / (2.0*dx)
        if Nx >= 3:
            dpdx[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2.0*dx)
        dpdx[:, -1]   = (pRghost   - P[:, -2]) / (2.0*dx)

    # y 向梯度
    if Ny == 1:
        dpdy[0, :] = (pBghost - pTghost) / (2.0*dy)
    else:
        dpdy[0, :]    = (P[1, :]   - pTghost) / (2.0*dy)
        if Ny >= 3:
            dpdy[1:-1, :] = (P[2:, :] - P[:-2, :]) / (2.0*dy)
        dpdy[-1, :]   = (pBghost   - P[-2, :]) / (2.0*dy)
        
    kappa = np.asarray(kappa, dtype=float)
    dpdx  = np.asarray(dpdx, dtype=float)

    vx_c = - kappa * dpdx
    vy_c = - kappa * dpdy
    return vx_c, vy_c


def run_ic_darcian_solver_fvm(Lx=2., Ly=2., Nx=50, Ny=50, bc_left=100., bc_right=1., 
                           bc_bottom=-5, bc_top=5., *, 
                           kappa=None, kappa_file=None):
    
    if (kappa is None) == (kappa_file is None) or (kappa is not None and kappa_file is not None):
        raise ValueError("Provide exactly ONE of `kappa` or `kappa_file`.")
    
    dx = Lx / Nx
    dy = Ly / Ny
    
    if kappa_file is not None:
        kappa_data = loadmat(kappa_file)
        kappa = kappa_data['kappa'] 
        
    # kappa = 10.0 * np.ones((Ny, Nx))
    
    kappa *= 0.1 * 6.328 * 0.0008
    # kappa *= 0.01 * 6.328 * 0.0008 / 0.04 / 2
    
    # Ks2 = np.load("Ks_sample_2.npz", allow_pickle=True)["Ks_sample"]
    # Ks2 = [Ks2[i] for i in range(Ks2.shape[0])]
    # kappa_samples = Ks2
    # kk = kappa_samples[0]
    # kk = kk * 6.328 * 0.0008
    
    # kappa = kk
    
    q = np.zeros((Ny, Nx))
    
    # caution: bc_right/bc_bottom/bc_top are normal velocities in the function argument.
    bc_right = - bc_right
    bc_bottom = bc_bottom  # the positive y direction is opposite to the transnet code
    bc_top = bc_top  # the positive y direction is opposite to the transnet code

    P, vx, vy, vx_f, vy_f = direct_solve(Nx, Ny, dx, dy, kappa, q, bc_left, bc_right, bc_top, bc_bottom)
    vx_c, vy_c = center_velocity_from_gradient_2nd_ghost(P, kappa, dx, dy, bc_left, vx_f, vy_f)
    

    return {
        "p": P,
        "vx": vx, "vy": vy,             # Darcy vel interior faces
        "vx_f": vx_f, "vy_f": vy_f,     # Darcy vel faces (incl BC)
        "vx_c": vx_c, "vy_c": vy_c,     # Darcy vel centers
    }

# ============================================================
#                         简单示例
# ============================================================
 
if __name__ == "__main__":
    
    import time
    
    t1 = time.time()
    out = run_ic_darcian_solver_fvm(Lx=2., Ly=2., 
                               Nx=100, Ny=100, 
                               bc_left=500., bc_right=1., 
                               bc_bottom=-5., bc_top=5.,
                               kappa_file='kappa100x100_1.mat')

    
    t2 = time.time()
      
    print('\ttime:', t2 - t1)

