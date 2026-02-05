#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: harrywang
"""

import numpy as np

# Structured collocation (internal points implicitly at cell center)

def structured_collocation_2d_cc_bd(nx, ny, x_min=-1, x_max=1, y_min=-1, y_max=1):
    """
    Cell-centered style collocations including boundary collocation:
    internal points are shifted by half-cell.
    """
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    x_internal = x_min + dx/2 + np.arange(nx) * dx
    y_internal = y_min + dy/2 + np.arange(ny) * dy
    x_coords = np.concatenate(([x_min], x_internal, [x_max]))
    y_coords = np.concatenate(([y_min], y_internal, [y_max]))
    x_2d, y_2d = np.meshgrid(x_coords, y_coords, indexing="xy")
    return np.stack([x_2d.flatten(), y_2d.flatten()], axis=1)

def structured_collocation_2d(nx, ny, x_min=-1, x_max=1, y_min=-1, y_max=1):
    x_1d = np.linspace(x_min, x_max, nx)
    y_1d = np.linspace(y_min, y_max, ny)
    x_2d, y_2d = np.meshgrid(x_1d, y_1d)
    return np.stack([x_2d.flatten(), y_2d.flatten()], axis=1)

def remove_corners(edge_pts, corners, tol=1e-12):
    return np.array([pt for pt in edge_pts if not np.any(
        np.all(np.isclose(pt, corners, atol=tol), axis=1)
    )])