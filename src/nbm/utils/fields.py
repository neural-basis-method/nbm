import numpy as np

def kappa_xy_constant(x, y, val=0.3*6.328):
    return np.ones_like(x) * val
