#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def rel_metrics(y, yhat, eps=1e-12):
    y = np.asarray(y).ravel()
    yhat = np.asarray(yhat).ravel()
    m = np.isfinite(y) & np.isfinite(yhat)
    y = y[m]
    yhat = yhat[m]
    mse = float(np.mean((yhat - y) ** 2))
    en = float(np.mean(y ** 2))
    if en < eps:
        return {"mse": mse, "rel_l2": np.nan, "rel_l1": np.nan, "nmse": np.nan}
    l2 = float(np.linalg.norm(yhat - y) / (np.linalg.norm(y) + eps) * 100.0)
    l1 = float(np.sum(np.abs(yhat - y)) / (np.sum(np.abs(y)) + eps) * 100.0)
    nmse = float(np.sum((yhat - y) ** 2) / (np.sum(y ** 2) + eps) * 100.0)
    return {"mse": mse, "rel_l2": l2, "rel_l1": l1, "nmse": nmse}


def print_err(name: str, d: dict):
    def pct(x): return f"{x:10.4f}%" if np.isfinite(x) else f"{'NaN':>11}"
    def num(x): return f"{x:11.4e}" if np.isfinite(x) else f"{'NaN':>11}"
    print(f"{name}:")
    print(f"  MSE   = {num(d.get('mse', np.nan))}")
    print(f"  RelL2 = {pct(d.get('rel_l2', np.nan))}")
    print(f"  RelL1 = {pct(d.get('rel_l1', np.nan))}")
    print(f"  NMSE  = {pct(d.get('nmse', np.nan))}")