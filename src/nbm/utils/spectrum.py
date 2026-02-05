#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 14:02:54 2025

@author: harrywang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False
})


def cell_center_velocity(vx_faces, vy_faces):
    # vx_faces: Ny x (Nx+1) at vertical faces
    # vy_faces: (Ny+1) x Nx at horizontal faces
    vx_c = 0.5 * (vx_faces[:, :-1] + vx_faces[:, 1:])
    vy_c = 0.5 * (vy_faces[:-1, :] + vy_faces[1:, :])
    return vx_c, vy_c

def _spectrum_from_cells_with_edges(ux, uy, dx, dy, k_edges, detrend=True, window=True):
    Ny, Nx = ux.shape
    if detrend:
        ux = ux - ux.mean(); uy = uy - uy.mean()
    if window:
        wx = np.hanning(Nx); wy = np.hanning(Ny)
        w2 = np.outer(wy, wx); w2 = w2/np.sqrt((w2**2).mean())
        ux = ux*w2; uy = uy*w2

    Ux = np.fft.fftn(ux); Uy = np.fft.fftn(uy)
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    Kmag = np.sqrt(KX**2 + KY**2)

    dKx = (2*np.pi)/(Nx*dx); dKy = (2*np.pi)/(Ny*dy)
    E2D = 0.5*(np.abs(Ux)**2 + np.abs(Uy)**2)/(Nx*Ny)**2
    E2D = E2D/(dKx*dKy)

    idx = np.digitize(Kmag.ravel(), k_edges) - 1
    nbins = len(k_edges) - 1
    Ek_shell = np.bincount(idx.clip(0, nbins-1), weights=E2D.ravel(), minlength=nbins)[:nbins]
    k = 0.5*(k_edges[:-1] + k_edges[1:])
    dk = np.diff(k_edges); dk[dk==0] = np.inf
    Ek = Ek_shell / dk

    KE = 0.5*np.mean(ux**2 + uy**2)
    integ = np.trapz(Ek, k)
    if integ > 0:
        Ek *= (KE/integ)
    return k, Ek

def compare_spectra_samegrid_from_faces(ux_f_approx, uy_f_approx, ux_f_ref, uy_f_ref,
                                        dx, dy, 
                                        detrend=True, window=True, nbins=None):
    # faces -> cells
    ux_a, uy_a = cell_center_velocity(ux_f_approx, uy_f_approx)
    ux_r, uy_r = cell_center_velocity(ux_f_ref,    uy_f_ref)

    Ny, Nx = ux_a.shape

    if nbins is None:
        nbins = int(np.sqrt(Nx*Ny))
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    Kmax = np.sqrt(KX**2 + KY**2).max()
    k_edges = np.linspace(0.0, Kmax, nbins+1)  # 两者共用

    k, Er = _spectrum_from_cells_with_edges(ux_r, uy_r, dx, dy, k_edges, detrend, window)
    _, Ea = _spectrum_from_cells_with_edges(ux_a, uy_a, dx, dy, k_edges, detrend, window)

    # 
    rel_L1 = np.trapz(np.abs(Ea-Er), k) / max(np.trapz(Er, k), 1e-30)
    rel_L2 = np.sqrt(np.trapz((Ea-Er)**2, k) / max(np.trapz(Er**2, k), 1e-30))
    rel_L1_ = np.sum(np.abs(Ea - Er)) / (np.sum(np.abs(Er)) + 1e-30)
    rel_L2_ = np.linalg.norm(Ea - Er) / (np.linalg.norm(Er) + 1e-30)
    mask = (Ea>0) & (Er>0)
    log_RMSE = float(np.sqrt(np.mean((np.log10(Ea[mask]) - np.log10(Er[mask]))**2))) if np.any(mask) else np.nan
    dk = np.diff(k_edges)
    Ea_shell = Ea * dk
    Er_shell = Er * dk
    cum_a = np.cumsum(Ea_shell) / np.sum(Ea_shell)
    cum_r = np.cumsum(Er_shell) / np.sum(Er_shell)
    KS = np.mean(np.abs(cum_a - cum_r)) / (np.mean(np.abs(cum_r)) + 1e-30)
    Sa = np.sum(Ea_shell)
    Sr = np.sum(Er_shell)
    
    if Sa <= 0 or Sr <= 0:
        KS = np.nan
    else:
        cum_a = np.cumsum(Ea_shell) / Sa
        cum_r = np.cumsum(Er_shell) / Sr
        KS = float(np.max(np.abs(cum_a - cum_r))) 
    
    plt.figure(figsize=(8,4), dpi=240)
    plt.loglog(k[1:], Er[1:], label="Reference")
    plt.loglog(k[1:], Ea[1:], '--', label="Approximation")
    
    # explicit font control
    plt.xlabel("Wavenumber $k$", fontsize=12)
    plt.ylabel("Energy spectra", fontsize=12)
    
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    ax.legend(fontsize=11)
    
    # no grid
    ax.grid(False)
    
    for s in ax.spines.values():
        s.set_color('black')
        s.set_linewidth(0.8)
    
    plt.show()

    return {"rel_L1": rel_L1, "rel_L2": rel_L2, "rel_L1_": rel_L1_, "rel_L2_": rel_L2_, 
            "log_RMSE": log_RMSE, "KS_cum_energy": KS}

