#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Basis Engine: Generator and Evaluator 
============================================

This module implements a neural basis generator and evaluator based on a
multi-layer residual network architecture. Basis functions are defined on the
unit d-dimensional hypercube [-1, 1]^d, with optional affine mappings to support
general parallelotope domains. The implementation provides analytic input
derivatives up to third order.

Created on Sun Aug 24 18:47:08 2025
@author: y.wang
"""

import numpy as np

# -----------------------------
# Helpers 
# -----------------------------
def _normalize_shape_factors(shape_factors, layer_widths):
    """
    Return a list of per-layer vectors alpha_l with shape (p_out,).
    Accepts:
      - None        -> all ones (no extra slope)
      - scalar      -> same scalar for all layers/neurons
      - list length L:
          * scalar per layer -> broadcast to that layer's width
          * array per layer  -> must match that layer's width
    """
    L = len(layer_widths)
    if shape_factors is None:
        return [np.ones((w,), float) for w in layer_widths]

    if np.isscalar(shape_factors):
        a = float(shape_factors)
        return [np.full((w,), a, float) for w in layer_widths]

    if len(shape_factors) != L:
        raise ValueError(f"shape_factors length {len(shape_factors)} != #layers {L}")

    out = []
    for i, (sf, w) in enumerate(zip(shape_factors, layer_widths)):
        if np.isscalar(sf):
            out.append(np.full((w,), float(sf), float))
        else:
            arr = np.asarray(sf, float).reshape(-1,)
            if arr.shape[0] != w:
                raise ValueError(f"shape_factors[{i}] length {arr.shape[0]} != layer width {w}")
            out.append(arr.copy())
    return out


def _normalize_residual_strengths(residual_strengths, L):
    """
    Return a list of scalars beta_l (length = #layers).
    Accepts:
      - None     -> all ones (skip enabled with strength 1.0)
      - scalar   -> same scalar for every layer
      - list len L -> per-layer scalars
    """
    if residual_strengths is None:
        return [1.0] * L
    if np.isscalar(residual_strengths):
        return [float(residual_strengths)] * L
    if len(residual_strengths) != L:
        raise ValueError(f"residual_strengths length {len(residual_strengths)} != #layers {L}")
    return [float(x) for x in residual_strengths]


# -----------------------------------------------------------------------
# Orthonormalization utilities (NOT WORKING!! CAUSE ERROR!! DO NOT USE!!)
# -----------------------------------------------------------------------
def _orthonormalize_columns(M: np.ndarray) -> np.ndarray:
    """Orthonormalize columns of M via QR: returns Q with same shape as M."""
    Q, _ = np.linalg.qr(M)
    return Q


def _normalize_rows(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each row of M to unit 2-norm. Keeps shape."""
    s = np.linalg.norm(M, axis=1, keepdims=True)
    s = np.maximum(s, eps)
    return M / s


# ==========================================================
#       Neural Basis Engine: Generator and Evaluator
# ==========================================================
class NeuralBasisEngine:
    """
    Fixed-parameter multilayer residual network with:
      - per-layer shape factors (multiply preactivation before activation),
      - per-layer residual strengths (scale the linear skip),
      - first-layer eval_range scaling,
      - up to 3rd-order input derivatives,
      - optional concatenation of multiple layers' outputs in eval_basis.

    Layer l forward (post-activation residual):
        t_l = h_{l-1} @ W_l^T + b_l
        a_l = alpha_l ⊙ t_l
        z_l = sigma(a_l)
        s_l = β_l * (h_{l-1} @ P_l^T)
        h_l = s_l + z_l
    """

    def __init__(self,
                 x_dim,
                 layer_widths,
                 include_const=False,
                 nlin_type='tanh',
                 shape_factors=None,
                 residual_strengths=None,
                 seed=None):
        assert x_dim >= 1
        assert len(layer_widths) >= 1
        self.x_dim = int(x_dim)
        self.layer_widths = list(layer_widths)
        self.include_const = bool(include_const)
        self.nlin_type = str(nlin_type)

        # First-layer input scaling range (applied to W1, P1, b1 only)
        self.eval_range = np.array([[-1.0, 1.0]] * self.x_dim, dtype=float)

        # Unscaled originals (as set/initialized)
        self.weight_0_list = None
        self.bias_0_list   = None
        self.proj_0_list   = None

        # Scaled versions actually used in forward/eval
        self.weight_list = None
        self.bias_list   = None
        self.proj_list   = None

        # Per-layer settings
        self.shape_list = _normalize_shape_factors(shape_factors, self.layer_widths)  # alpha_l vectors
        self.beta_list  = _normalize_residual_strengths(residual_strengths, len(self.layer_widths))  # beta_l scalars

        # First-layer aliases (debug)
        self.weight_0 = None
        self.bias_0   = None
        self.proj_0   = None
        self.weight   = None
        self.bias     = None
        self.proj     = None

        self._rng = np.random.default_rng(seed)
        self.info = {}

    def __repr__(self):
        es = '*'.join([f'[{lo:.2f},{hi:.2f}]' for (lo, hi) in self.eval_range])
        return f'[NeuralBasis] d={self.x_dim}, layers={self.layer_widths}, act={self.nlin_type}, range={es}'

    # -------------------------
    # Parameter initialization
    # -------------------------
    def set_layers(self, W_list, b_list, P_list):
        """
        Set all (W, b, P) per layer with shape checks, then refresh scaled params.
        W_l, P_l: (p_out, p_in);  b_l: (p_out,)
        """
        L = len(self.layer_widths)
        assert len(W_list) == len(b_list) == len(P_list) == L
        inw = self.x_dim
        for (W, b, P, outw) in zip(W_list, b_list, P_list, self.layer_widths):
            W = np.asarray(W, float)
            b = np.asarray(b, float).reshape(-1,)
            P = np.asarray(P, float)
            assert W.shape == (outw, inw), f"Layer W has {W.shape}, expected {(outw, inw)}"
            assert P.shape == (outw, inw), f"Layer P has {P.shape}, expected {(outw, inw)}"
            assert b.shape == (outw,),     f"Layer b has {b.shape}, expected {(outw,)}"
            inw = outw
        self.weight_0_list = [np.asarray(W, float).copy() for W in W_list]
        self.bias_0_list   = [np.asarray(b, float).copy() for b in b_list]
        self.proj_0_list   = [np.asarray(P, float).copy() for P in P_list]
        self._refresh_scaled_params()
        
    def init_layers_random(self, radius=1.5, scale_W=1.0, scale_P=None, orthogonal=False, seed=None):
        """
        Random initialization for all layers (no geometric filtering).
        Construction order: first determine each layer's `inw` (the previous layer's `outw`;
        for the first layer, `inw = x_dim`), then build the layers following this list.
        """
        rng = np.random.default_rng(self._rng.integers(0, 10**9) if seed is None else seed)
        W_list, b_list, P_list = [], [], []
    
        # 1) Precompute the `inw` for each layer (to avoid any ordering mistakes).
        inw_list = []
        _prev = self.x_dim
        for outw in self.layer_widths:
            inw_list.append(_prev)
            _prev = outw
    
        # 2) Construct each layer.
        for li, (outw, inw) in enumerate(zip(self.layer_widths, inw_list)):
            W = rng.normal(scale=scale_W, size=(outw, inw))
    
            if scale_P is None:
                sp = 1.0 if outw == inw else 1.0 / max(1, np.sqrt(inw))
            else:
                sp = float(scale_P)
            P = np.eye(outw) if outw == inw else rng.normal(scale=sp, size=(outw, inw))
    
            if orthogonal:
                W = _orthonormalize_columns(W)
                P = np.eye(outw) if outw == inw else _orthonormalize_columns(P)
    
            b = rng.uniform(-radius, radius, size=(outw,))
    
            W_list.append(W); P_list.append(P); b_list.append(b)
    
        # 3) Shape sanity check.
        chk_inw = self.x_dim
        for li, (W, P, outw) in enumerate(zip(W_list, P_list, self.layer_widths)):
            assert W.shape == (outw, chk_inw), f"[init_layers_random] W@{li} shape {W.shape} != {(outw, chk_inw)}"
            assert P.shape == (outw, chk_inw), f"[init_layers_random] P@{li} shape {P.shape} != {(outw, chk_inw)}"
            chk_inw = outw
    
        self.set_layers(W_list, b_list, P_list)

    def init_pde_basis(self,
                       shape,
                       radius,
                       *,
                       min_ratio=0.0001,
                       ratio_ref='square_oriented',
                       ref_circle_R=np.sqrt(2.0),
                       allow_tangent=True,
                       symmetric_b=True,
                       max_tries_per_sample=500,
                       orthogonal=False,
                       seed=None):
        """
        PDE-friendly init:
          - If x_dim == 2: build (W1,b1) with 2D line-in-square filtering.
          - If x_dim > 2: fallback random first layer (no geometric filter).
          - P1 is identity when p1 == x_dim, otherwise random; deeper P_l random or identity if square.
          - MUST KEEP orthogonal=False !!!!! orthogonal=True IS NOT WORKING!!!!!

        After building unscaled params, first-layer range scaling is applied in _refresh_scaled_params().
        """
        rng = np.random.default_rng(self._rng.integers(0, 10**9) if seed is None else seed)
        W_list, b_list, P_list = [], [], []

        outw_1 = self.layer_widths[0]

        if self.x_dim == 2:
            W1, b1, tries, rejects = self._make_first_layer_2d(
                outw_1, shape=shape, radius=radius,
                min_ratio=min_ratio, ratio_ref=ratio_ref, ref_circle_R=ref_circle_R,
                allow_tangent=allow_tangent, symmetric_b=symmetric_b,
                max_tries_per_sample=max_tries_per_sample, rng=rng, 
                min_angle_deg=1.0, min_parallel_b_sep=0.05
            )
            self.info["pde_basis_filtered"] = dict(
                asked_basis=int(outw_1), kept=int(len(W1)), tries=int(tries),
                min_ratio=float(min_ratio), ratio_ref=str(ratio_ref), ref_circle_R=float(ref_circle_R),
                allow_tangent=bool(allow_tangent), symmetric_b=bool(symmetric_b),
                reject_rate=(float(rejects) / float(tries)) if tries > 0 else 0.0
            )

        else:
            # fallback: random first layer for d != 2
            W1 = rng.normal(scale=1.0, size=(outw_1, self.x_dim))
            b1 = rng.uniform(-radius, radius, size=(outw_1,))
            self.info["pde_basis_filtered"] = dict(
                asked_basis=int(outw_1), kept=int(len(W1)),
                note="x_dim>2: first layer random (no geometric filter)"
            )

        # P1
        if outw_1 == self.x_dim:
            P1 = np.eye(outw_1)
        else:
            P1 = rng.normal(scale=1.0 / np.sqrt(max(1, self.x_dim)), size=(outw_1, self.x_dim))

        if orthogonal:
            W1 = _orthonormalize_columns(W1)
            if outw_1 == self.x_dim:
                P1 = np.eye(outw_1)
            else:
                P1 = _orthonormalize_columns(P1)

        W_list.append(W1); b_list.append(b1); P_list.append(P1)

        # Deeper layers
        inw = outw_1
        for outw in self.layer_widths[1:]:
            W_l = rng.normal(scale=1.0, size=(outw, inw))
            b_l = rng.uniform(-radius, radius, size=(outw,))
            if outw == inw:
                P_l = np.eye(outw)
            else:
                P_l = rng.normal(scale=1.0 / np.sqrt(max(1, inw)), size=(outw, inw))
            if orthogonal:
                W_l = _orthonormalize_columns(W_l)
                if outw == inw:
                    P_l = np.eye(outw)
                else:
                    P_l = _orthonormalize_columns(P_l)
            W_list.append(W_l); b_list.append(b_l); P_list.append(P_l)
            inw = outw

        self.set_layers(W_list, b_list, P_list)

    
    # -------------------------
    # Range scaling (layer 1)
    # -------------------------
    def set_eval_range(self, eval_range):
        eval_range = np.asarray(eval_range, float)
        assert eval_range.shape == self.eval_range.shape
        self.eval_range = eval_range
        self._refresh_scaled_params()

    def _refresh_scaled_params(self):
        """
        Apply first-layer scaling:
          W1' = W1_0 / s
          P1' = P1_0 / s
          b1' = b1_0 - W1_0 @ (c/s)
        with s=(upper-lower)/2, c=(upper+lower)/2 (per-dim).
        """
        if self.weight_0_list is None:
            raise RuntimeError("Parameters not initialized. Call set_layers/init_* first.")

        lower = self.eval_range[:, 0]
        upper = self.eval_range[:, 1]
        s = (upper - lower) / 2.0
        c = (upper + lower) / 2.0

        W1_0 = self.weight_0_list[0]
        b1_0 = self.bias_0_list[0]
        P1_0 = self.proj_0_list[0]

        W1 = W1_0 / s[None, :]
        P1 = P1_0 / s[None, :]
        b1 = b1_0 - (W1_0 @ (c / s))

        self.weight_list = [W1] + [W.copy() for W in self.weight_0_list[1:]]
        self.bias_list   = [b1] + [b.copy() for b in self.bias_0_list[1:]]
        self.proj_list   = [P1] + [P.copy() for P in self.proj_0_list[1:]]
        
        # Skip the branch offset (constant shift).
        beta1 = float(self.beta_list[0])
        self._skip_const_1 = - beta1 * (P1_0 @ (c / s))   # shape: (outw_1,)

        self.weight_0 = W1_0; self.bias_0 = b1_0; self.proj_0 = P1_0
        self.weight   = W1;   self.bias   = b1;   self.proj   = P1

    # -------------------------
    # Activations and derivatives (up to 3rd order w.r.t. a)
    # -------------------------
    @staticmethod
    def _act_all(a, kind):
        """
        Return elementwise (z, dz, d2, d3) for sigma(a), sigma'(a), sigma''(a), sigma'''(a).
        Currently implemented: tanh
        """
        if kind == 'tanh':
            z  = np.tanh(a)
            dz = 1.0 - z*z        
            d2 = -2.0 * z * dz      
            d3 = -2.0 * (dz*dz + z*d2)
        else:
            raise ValueError(f"Unsupported activation '{kind}' for 3rd-order derivatives")
        return z, dz, d2, d3

    # -------------------------
    # Basis evaluation (+ concat)
    # -------------------------
    def eval_basis(self, x_in, eval_list=('u',), concat_layers=None):
        """
        Evaluate selected layers' outputs (values and derivatives) and concatenate along columns.
        eval_list:
            'u'         : value
            'u0'..'u{d-1}' : 1st derivatives
            'uij'       : 2nd derivatives
            'uijk'      : 3rd derivatives
        concat_layers: None | list of layer indices and/or 'final'
        """
        if (self.weight_list is None) or (self.bias_list is None) or (self.proj_list is None):
            raise RuntimeError("Parameters not initialized. Call set_layers/init_* first.")

        X = np.asarray(x_in, float)
        assert X.ndim == 2 and X.shape[1] == self.x_dim
        M, d = X.shape

        need_first  = any(len(k) >= 2 for k in eval_list)
        need_second = any(len(k) >= 3 for k in eval_list)
        need_third  = any(len(k) >= 4 for k in eval_list)

        # Third order needs second; second needs first.
        if need_third:
            need_second = True
        if need_second:
            need_first = True

        A_list = []
        J_list = []
        H_list = []
        T_list = []   # 3rd-order

        # ---------- layer 1 ----------
        W1, b1, P1 = self.weight_list[0], self.bias_list[0], self.proj_list[0]
        alpha1 = self.shape_list[0]
        beta1  = float(self.beta_list[0])

        t1 = X @ W1.T + b1[None, :]
        a1 = t1 * alpha1[None, :]
        z1, dz1, d21, d31 = self._act_all(a1, self.nlin_type)

        dz1_eff = dz1 * alpha1[None, :]
        d21_eff = d21 * (alpha1[None, :]**2)
        d31_eff = d31 * (alpha1[None, :]**3)

        # skip
        s1 = beta1 * (X @ P1.T) + self._skip_const_1[None, :]
        A  = s1 + z1

        # 1st-order
        if need_first:
            # s1's derivative w.r.t. x is the constant P1
            J_s = beta1 * np.broadcast_to(P1[None, :, :], (M, P1.shape[0], d))
            # z1's derivative w.r.t. x: sigma'(a) * W1.
            J_z = dz1_eff[:, :, None] * W1[None, :, :]
            J   = J_s + J_z
        else:
            J = None

        # 2nd-order
        if need_second:
            # In the first layer, t1 is linear in x, so t1_ij = 0; only sigma'' * (t_i t_j) remains
            # Here t_i = W1[:, i], shared across all samples
            w_outer = W1[None, :, :, None] * W1[None, :, None, :]
            H = d21_eff[:, :, None, None] * w_outer
        else:
            H = None

        # 3rd-order
        if need_third:
            # First layer: t1 is linear in x, so t1_ij = 0 and t1_ijk = 0; only sigma''' * t_i t_j t_k remains.
            # t_i = W1[:,i]
            J_lin_W1 = np.broadcast_to(W1[None, :, :], (M, W1.shape[0], d))  # (M, p1, d)
            t_i = J_lin_W1  # alias
            # T_z[m, o, i, j, k] = d31_eff[m,o] * t_i[m,o,i] * t_i[m,o,j] * t_i[m,o,k]
            T = (
                d31_eff[:, :, None, None, None]
                * t_i[:, :, :, None, None]
                * t_i[:, :, None, :, None]
                * t_i[:, :, None, None, :]
            )
        else:
            T = None

        A_list.append(A)
        if need_first:  J_list.append(J)
        if need_second: H_list.append(H)
        if need_third:  T_list.append(T)

        # ---------- deeper layers ----------
        L = len(self.layer_widths)
        for l in range(1, L):
            W, b, P = self.weight_list[l], self.bias_list[l], self.proj_list[l]
            alpha = self.shape_list[l]
            beta  = float(self.beta_list[l])

            t  = A @ W.T + b[None, :]
            a  = t * alpha[None, :]
            z, dz, d2, d3 = self._act_all(a, self.nlin_type)

            dz_eff = dz * alpha[None, :]
            d2_eff = d2 * (alpha[None, :]**2)
            d3_eff = d3 * (alpha[None, :]**3)

            s = beta * (A @ P.T)
            A = s + z

            if need_first:
                # t_i = (d/dx) (A_prev @ W^T) = J_prev * W
                J_lin_W = np.einsum('mpd,op->mod', J, W, optimize=True)
                J_lin_P = np.einsum('mpd,op->mod', J, P, optimize=True)
                J = beta * J_lin_P + dz_eff[:, :, None] * J_lin_W

            if need_second:
                # t_ij = H_prev * W
                H_lin_W = np.einsum('mpab,op->moab', H, W, optimize=True)
                H_lin_P = np.einsum('mpab,op->moab', H, P, optimize=True)
                # outer = t_i t_j
                outer   = J_lin_W[:, :, :, None] * J_lin_W[:, :, None, :]
                # H = s part + z part
                H = beta * H_lin_P \
                    + dz_eff[:, :, None, None] * H_lin_W \
                    + d2_eff[:, :, None, None] * outer

            if need_third:
                # t_ijk = T_prev * W
                # s_ijk = T_prev * P
                T_lin_W = np.einsum('mpabc,op->moabc', T, W, optimize=True)
                T_lin_P = np.einsum('mpabc,op->moabc', T, P, optimize=True)

                # 3rd order chain-rule:
                # σ'''(t) t_i t_j t_k
                # + σ''(t) [ t_ij t_k + t_ik t_j + t_jk t_i ]
                # + σ'(t) t_ijk
                M_, p_out, d_ = J_lin_W.shape
                T_z = np.zeros((M_, p_out, d_, d_, d_), dtype=A.dtype)

                for i in range(d_):
                    ti = J_lin_W[:, :, i]
                    for j in range(d_):
                        tj = J_lin_W[:, :, j]
                        t_ij = H_lin_W[:, :, i, j]
                        for k in range(d_):
                            tk   = J_lin_W[:, :, k]
                            t_ik = H_lin_W[:, :, i, k]
                            t_jk = H_lin_W[:, :, j, k]
                            t_ijk = T_lin_W[:, :, i, j, k]

                            cubic_term = d3_eff * (ti * tj * tk)
                            sigma2_term = d2_eff * (t_ij * tk + t_ik * tj + t_jk * ti)
                            sigma1_term = dz_eff * t_ijk
                            T_z[:, :, i, j, k] = cubic_term + sigma2_term + sigma1_term

                T = beta * T_lin_P + T_z

            A_list.append(A)
            if need_first:  J_list.append(J)
            if need_second: H_list.append(H)
            if need_third:  T_list.append(T)

        # ---------- concatenate selected layers ----------
        if concat_layers is None:
            use_ids = [len(A_list) - 1]
        else:
            use_ids = []
            for tag in concat_layers:
                if isinstance(tag, str):
                    if tag.lower() != 'final':
                        raise ValueError(f"Unknown concat_layers tag '{tag}'")
                    use_ids.append(len(A_list) - 1)
                else:
                    idx = int(tag)
                    if not (0 <= idx < len(A_list)):
                        raise ValueError(f"concat_layers index {idx} out of range [0,{len(A_list)-1}]")
                    use_ids.append(idx)

        A_cat = np.concatenate([A_list[i] for i in use_ids], axis=1)

        def _cat_J_dim(dim):
            return np.concatenate([J_list[i][:, :, dim] for i in use_ids], axis=1)

        def _cat_H_ij(i, j):
            return np.concatenate([H_list[k][:, :, i, j] for k in use_ids], axis=1)

        def _cat_T_ijk(i, j, k):
            return np.concatenate([T_list[idx][:, :, i, j, k] for idx in use_ids], axis=1)

        out = {}

        def _append_const(mat, is_value):
            if not self.include_const:
                return mat
            if is_value:
                return np.concatenate([mat, np.ones((M, 1), float)], axis=1)
            else:
                return np.concatenate([mat, np.zeros((M, 1), float)], axis=1)

        for key in eval_list:
            if key == 'u':
                out[key] = _append_const(A_cat, is_value=True)

            elif len(key) == 2 and key[0] == 'u' and key[1:].isdigit():
                # 1st derivative: 'u0'
                if not need_first:
                    raise RuntimeError("Internal: first derivatives were not computed")
                dim = int(key[1]); assert 0 <= dim < d
                out[key] = _append_const(_cat_J_dim(dim), is_value=False)

            elif len(key) == 3 and key[0] == 'u' and key[1:].isdigit():
                # 2nd derivative: 'u00', 'u01', ...
                if not need_second:
                    raise RuntimeError("Internal: second derivatives were not computed")
                i = int(key[1]); j = int(key[2])
                assert 0 <= i < d and 0 <= j < d
                out[key] = _append_const(_cat_H_ij(i, j), is_value=False)

            elif len(key) == 4 and key[0] == 'u' and key[1:].isdigit():
                # 3rd derivative: 'u000', 'u012', ...
                if not need_third:
                    raise RuntimeError("Internal: third derivatives were not computed")
                i = int(key[1]); j = int(key[2]); k = int(key[3])
                assert 0 <= i < d and 0 <= j < d and 0 <= k < d
                out[key] = _append_const(_cat_T_ijk(i, j, k), is_value=False)

            else:
                raise ValueError(f"Unsupported eval item '{key}'")

        return out

    # ------------------------------------------
    # 2D geometric first-layer initialization
    # ------------------------------------------
    @staticmethod
    def _line_segment_len_in_square(w, b, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0):
        """Length of the intersection of line {x: w·x + b = 0} with [xmin,xmax]×[ymin,ymax]."""
        EPS = 1e-12
        w0, w1 = float(w[0]), float(w[1])
        pts = []
        if abs(w1) > EPS:
            for x in (xmin, xmax):
                y = -(w0 * x + b) / w1
                if y >= ymin - EPS and y <= ymax + EPS:
                    pts.append((x, np.clip(y, ymin, ymax)))
        if abs(w0) > EPS:
            for y in (ymin, ymax):
                x = -(w1 * y + b) / w0
                if x >= xmin - EPS and x <= xmax + EPS:
                    pts.append((np.clip(x, xmin, xmax), y))
        if not pts:
            return 0.0
        uniq = []
        for p in pts:
            if not any(abs(p[0]-q[0]) < 1e-9 and abs(p[1]-q[1]) < 1e-9 for q in uniq):
                uniq.append(p)
        if len(uniq) < 2:
            return 0.0
        P = np.asarray(uniq, float)
        d2 = 0.0
        for i in range(len(P)):
            for j in range(i+1, len(P)):
                dij = (P[i, 0] - P[j, 0])**2 + (P[i, 1] - P[j, 1])**2
                if dij > d2:
                    d2 = dij
        return float(np.sqrt(d2))

    @staticmethod
    def _Lmax_square_oriented_from_w(w):
        """For unit w, max chord length in [-1,1]^2 along that normal through the center."""
        EPS = 1e-12
        wx, wy = float(w[0]), float(w[1])
        nrm = np.hypot(wx, wy)
        if nrm < EPS:
            return 0.0
        wx /= nrm; wy /= nrm
        return 2.0 / max(abs(wx), abs(wy))

    def _make_first_layer_2d(self, B, shape, radius,
                             *, min_ratio, ratio_ref, ref_circle_R,
                             allow_tangent, symmetric_b,
                             max_tries_per_sample, rng,
                             min_angle_deg=5.0,
                             min_parallel_b_sep=0.05):
        """Build (W1, b1) with geometric filtering; then scale by 'shape'."""
        W_list, b_list = [], []
        tries = 0
        rejects = 0
        EPS = 1e-12
                
        cos_min = np.cos(np.deg2rad(min_angle_deg)) # angle < min_angle_deg <=> dot > cos_min
        
        def _length_ok(w, b):
            """ segement length ratio filtering """
            L_sq = self._line_segment_len_in_square(w, b, -1, 1, -1, 1)
            if ratio_ref == 'square_oriented':
                L_ref = self._Lmax_square_oriented_from_w(w)
            elif ratio_ref == 'circle':
                R = ref_circle_R
                d = abs(b)
                L_ref = 2.0 * np.sqrt(max(0.0, R * R - d * d))
            elif ratio_ref == 'box':
                L_ref = L_sq
            else:
                raise ValueError("ratio_ref must be 'square_oriented' | 'circle' | 'box'")
            if L_ref <= EPS:
                return False
            return (L_sq / L_ref) >= min_ratio if allow_tangent else (L_sq / L_ref) > min_ratio
        
        def _diversity_ok(w_new, b_new):
            """ Compare against the selected lines one by one, using a single-threshold rule. """
            if not W_list:
                return True
            dots = np.abs(np.dot(np.asarray(W_list), w_new))  # |w_new·w_old|, shape (k,)
            for dot, w_old, b_old in zip(dots, W_list, b_list):
                if dot > cos_min:
                    # Angle < min_angle_deg: nearly parallel; allow but should with larger normal separation.
                    s = 1.0 if (np.dot(w_new, w_old) >= 0.0) else -1.0
                    if abs(b_new - s * b_old) < min_parallel_b_sep:
                        return False
                # else: angle >= min_angle_deg, allow if angle is large enough
            return True
               
        while len(W_list) < B and tries < max_tries_per_sample * max(1, B):
            tries += 1
            w = rng.normal(size=2)
            nrm = np.linalg.norm(w)
            if nrm < EPS:
                rejects += 1
                continue
            w = w / nrm  # unit normal before applying 'shape'
            b = (2.0 * rng.random() - 1.0) * radius if symmetric_b else (rng.random() * radius)

            # segement length filtering first
            if not _length_ok(w, b):
                rejects += 1
                continue
            
            # diversity filtering (line by line)
            if not _diversity_ok(w, b):
                rejects += 1
                continue
            
            W_list.append(w)
            b_list.append(b)
        
        rejects = tries - len(W_list)
        assert tries >= len(W_list), f"INTERNAL: tries({tries}) < kept({len(W_list)})"
        
        if len(W_list) < B:
            msg = (f"[NeuralBasisCore] only generate {len(W_list)}/{B} effective lines "
                   f" (tried {tries}, reject {rejects})."
                   "should reasonable reduce min_angle_deg or min_parallel_b_sep")
            raise RuntimeError(msg)
        
        W0 = np.asarray(W_list, float) * float(shape)
        b0 = np.asarray(b_list, float) * float(shape)
        return W0, b0, tries, rejects
