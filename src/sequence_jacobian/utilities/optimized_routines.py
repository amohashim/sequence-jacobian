"""Njitted routines to speed up some steps in backward iteration or aggregation"""

import numpy as np
from numba import njit


@njit
def setmin(x, xmin):
    """Set 2-dimensional array x where each row is ascending equal to equal to max(x, xmin)."""
    ni, nj = x.shape
    for i in range(ni):
        for j in range(nj):
            if x[i, j] < xmin:
                x[i, j] = xmin
            else:
                break


@njit
def within_tolerance_err(x1, x2, tol):
    """Efficiently test max(abs(x1-x2)) <= tol for arrays of same dimensions x1, x2."""
    y1 = x1.ravel()
    y2 = x2.ravel()

    for i in range(y1.shape[0]):
        err = np.abs(y1[i] - y2[i])
        if np.abs(y1[i] - y2[i]) > tol:
            return False, err
    return True, 0.0

@njit
def within_tolerance(x1, x2, tol):
    """Efficiently test max(abs(x1-x2)) <= tol for arrays of same dimensions x1, x2."""
    y1 = x1.ravel()
    y2 = x2.ravel()

    for i in range(y1.shape[0]):
        if np.abs(y1[i] - y2[i]) > tol:
            return False
    return True

@njit
def fast_aggregate(X, Y):
    """If X has dims (T, ...) and Y has dims (T, ...), do dot product for each T to get length-T vector.

    Identical to np.sum(X*Y, axis=(1,...,X.ndim-1)) but avoids costly creation of intermediates,
    useful for speeding up aggregation in td by factor of 4 to 5."""
    T = X.shape[0]
    Xnew = X.reshape(T, -1)
    Ynew = Y.reshape(T, -1)
    Z = np.empty(T)
    for t in range(T):
        Z[t] = Xnew[t, :] @ Ynew[t, :]
    return Z


# ---------- extra functions for testing convergence ---------- #

@njit
def max_abs_diff(x1, x2):
    """
    from Chat
    Return max(abs(x1-x2)) for arrays of same shape."""
    y1 = x1.ravel()
    y2 = x2.ravel()
    md = 0.0
    for i in range(y1.shape[0]):
        d = abs(y1[i] - y2[i])
        if d > md:
            md = d
    return md

@njit
def update_monotonicity(mask, count, m, e_prev, e_curr, eta):
    """
    from Chat
    Bitmask over last m steps: new_bit=1 if e_curr < (1-eta)*e_prev.
    Returns updated (mask, count). 'count' is popcount(mask) maintained in O(1).
    """
    # bit that will fall off when we shift (the MSB before shift)
    msb = (mask >> (m - 1)) & 1
    # did we improve by a meaningful relative margin?
    improved = 1 if e_curr < (1.0 - eta) * e_prev else 0
    # shift-in new bit; keep only m bits
    mask = ((mask << 1) & ((1 << m) - 1)) | improved
    # maintain running count without popcount()
    count = count - msb + improved
    return mask, count

@njit
def update_stagnation(e_min, stall, e_curr, delta):
    """
    from Chat
    Track best-so-far error and a stall counter.
    If e_curr < (1-delta)*e_min: new best → reset stall.
    Else: increment stall.
    """
    if e_curr < (1.0 - delta) * e_min:
        e_min = e_curr
        stall = 0
    else:
        stall += 1
    return e_min, stall



