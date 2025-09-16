import numpy as np
from numba import guvectorize

from ..blocks.het_block import het
from .. import interpolate

# def hh_init(b_grid, a_grid, z_grid, eis):
    
#     print("step 1")
#     Va = (0.6 + 1.1 * b_grid[:, np.newaxis] + a_grid) ** (-1 / eis) * np.ones((z_grid.shape[0], 1, 1))
#     Vb = (0.5 + b_grid[:, np.newaxis] + 1.2 * a_grid) ** (-1 / eis) * np.ones((z_grid.shape[0], 1, 1))
#     return Va, Vb

def check_VaVb_monotone_decreasing_in_a(Va, Vb, *, rtol=1e-9, atol=1e-12, strict=False, max_report=5):
    """
    Validate that (Va/Vb) - 1 is monotone decreasing along the last axis (a').
    
    Args:
        Va, Vb : arrays with shape (..., na) — typically (nz, nb, na).
        rtol, atol : relative/absolute tolerances for floating-point slop.
        strict : if True, require strictly decreasing (<); else allow flat (<=) within tolerance.
        max_report : how many violating slices to include in the error message if raising.
        
    Returns:
        ok : bool
        info : dict with diagnostics:
            - 'viol_mask': boolean array of shape (..., na-1) where True marks a local violation
            - 'num_viol_pairs': number of (z,b) (or higher-dim) slices with ≥1 violation
            - 'num_viol_points': total number of violating adjacent pairs along a'
            - 'worst_example': dict with indices and values for the largest violation
    """
    Va = np.asanyarray(Va, dtype=float)
    Vb = np.asanyarray(Vb, dtype=float)

    if Va.shape != Vb.shape:
        raise ValueError(f"Va and Vb must have the same shape, got {Va.shape} vs {Vb.shape}.")
    if Va.ndim < 1 or Va.shape[-1] < 2:
        return True, {'viol_mask': np.zeros(Va.shape[:-1] + (0,), dtype=bool),
                      'num_viol_pairs': 0, 'num_viol_points': 0, 'worst_example': None}
    if not np.all(np.isfinite(Va)) or not np.all(np.isfinite(Vb)):
        raise ValueError("Va or Vb contains non-finite values.")
    if np.any(Vb <= 0):
        raise ValueError("Vb must be positive to form Va/Vb safely for this check.")

    ratio = Va / Vb - 1.0  # shape (..., na)

    # Adjacent diffs along a' (last axis)
    d = np.diff(ratio, axis=-1)  # shape (..., na-1); want d <= tol (or < -tol if strict)

    # Pairwise tolerances based on neighbor magnitudes
    r_left = ratio[..., :-1]
    r_right = ratio[..., 1:]
    tol = atol + rtol * np.maximum(np.abs(r_left), np.abs(r_right))

    if strict:
        # strictly decreasing: d < -tol
        viol_mask = d >= -tol
        margin = d + tol  # larger = worse
    else:
        # non-increasing: d <= tol
        viol_mask = d > tol
        margin = d - tol  # larger = worse

    total_viol_points = int(np.count_nonzero(viol_mask))
    # Collapse all but last axis to count how many slices (…, na-1) have ≥1 violation
    has_viol_per_slice = np.any(viol_mask, axis=-1)
    num_viol_pairs = int(np.count_nonzero(has_viol_per_slice))

    worst_example = None
    if total_viol_points > 0:
        # Find the worst violation
        worst_flat = int(np.argmax(margin * viol_mask))
        worst_idx = np.unravel_index(worst_flat, margin.shape)
        # Map to indices (…, k) where k is the left point of the violating pair
        # Grab the values for reporting
        rL = r_left[worst_idx]
        rR = r_right[worst_idx]
        d_val = d[worst_idx]
        tol_val = tol[worst_idx]
        worst_example = {
            'indices': worst_idx,           # e.g. (z_idx, b_idx, k) if shape is (nz, nb, na-1)
            'r_left': float(rL),
            'r_right': float(rR),
            'diff': float(d_val),
            'tol': float(tol_val),
            'margin': float(margin[worst_idx]),
        }

    ok = (total_viol_points == 0)
    info = {
        'viol_mask': viol_mask,
        'num_viol_pairs': num_viol_pairs,
        'num_viol_points': total_viol_points,
        'worst_example': worst_example,
    }
    return ok, info


def assert_VaVb_monotone_decreasing_in_a(Va, Vb, *, rtol=1e-9, atol=1e-12, strict=False, max_report=5):
    """
    Assert variant of the check; raises AssertionError with a concise report if it fails.
    """
    ok, info = check_VaVb_monotone_decreasing_in_a(
        Va, Vb, rtol=rtol, atol=atol, strict=strict, max_report=max_report
    )
    if ok:
        return

    viol_mask = info['viol_mask']
    # Build human-readable summary of a few violating slices
    # Get indices of slices with any violation
    slice_idxs = np.argwhere(np.any(viol_mask, axis=-1))
    lines = []
    for n, idx in enumerate(slice_idxs[:max_report]):
        # For each slice, find the first violating step k
        # idx refers to all axes except the last; we need to find k on the last axis
        k = int(np.argmax(viol_mask[tuple(idx.tolist())]))
        idx_full = tuple(idx.tolist()) + (k,)
        lines.append(f"  slice {tuple(idx.tolist())}, step k={k} (a[k]→a[k+1]): "
                     f"Δ={(float(np.diff((Va/Vb - 1.0), axis=-1)[idx_full])):.3e} "
                     f" (tol≈{float((atol + rtol * np.maximum(np.abs((Va/Vb - 1.0)[..., :-1]),
                                                               np.abs((Va/Vb - 1.0)[..., 1:])))[idx_full]):.3e})")
    worst = info['worst_example']
    worst_line = ""
    if worst is not None:
        worst_line = (f"\nWorst violation at indices {worst['indices']}: "
                      f"r_left={worst['r_left']:.6g}, r_right={worst['r_right']:.6g}, "
                      f"Δ={worst['diff']:.3e}, tol≈{worst['tol']:.3e}")

    raise AssertionError(
        "Monotonicity check failed: (Va/Vb) - 1 must be "
        + ("strictly decreasing" if strict else "non-increasing")
        + f" along a'. Violating slices={info['num_viol_pairs']}, "
          f"violating pairs={info['num_viol_points']}.{worst_line}\n"
          + "\n".join(lines)
    )


def hh_init(a_grid, b_grid, z_grid, eis,
            theta_a=1.0, theta_b=0.15, phi_b=0.5, base=1.0):
    """
    Initial guess for envelopes V_a, V_b with (V_a / V_b)-1 strictly decreasing in a'
    for every (z,b'). Works for any positive EIS.

    Shapes returned: (nz, nb, na).
    """

    # normalize to [0,1] for stability across grid magnitudes
    a_norm = (a_grid - a_grid[0]) / (a_grid[-1] - a_grid[0] + 1e-12)   # (na,)
    b_norm = (b_grid - b_grid[0]) / (b_grid[-1] - b_grid[0] + 1e-12)   # (nb,)

    # denominators (kept >0 by base)
    den_Va = base + theta_a * a_norm[None, None, :] + theta_b * b_norm[None, :, None]   # (1, nb, na)
    den_Vb = base +                   0.0 * a_norm[None, None, :] +  phi_b * b_norm[None, :, None]   # (1, nb, na)

    # isoelastic mapping to "envelope-like" positive seeds
    Va = den_Va ** (-1.0 / eis)  # (1, nb, na)
    Vb = den_Vb ** (-1.0 / eis)  # (1, nb, na)

    # broadcast across z; same z-scaling for both cancels in the ratio (fine for a seed)
    Va = np.broadcast_to(Va, (len(z_grid), len(b_grid), len(a_grid))).copy()
    Vb = np.broadcast_to(Vb, (len(z_grid), len(b_grid), len(a_grid))).copy()
    
    assert_VaVb_monotone_decreasing_in_a(Va, Vb, strict=False)
    
    return Va, Vb


def adjustment_costs(a, a_grid, ra, chi0, chi1, chi2):
    chi = get_Psi_and_deriv(a, a_grid, ra, chi0, chi1, chi2)[0]
    return chi


def marginal_cost_grid(a_grid, ra, chi0, chi1, chi2):
    # precompute Psi1(a', a) on grid of (a', a) for steps 3 and 5
    Psi1 = get_Psi_and_deriv(a_grid[:, np.newaxis],
                             a_grid[np.newaxis, :], ra, chi0, chi1, chi2)[1]

    return Psi1

# policy and bacward order as in grid!
@het(exogenous='Pi', policy=['b', 'a'], backward=['Vb', 'Va'],
     hetinputs=[marginal_cost_grid], hetoutputs=[adjustment_costs], backward_init=hh_init)  
def hh(Va_p, Vb_p, a_grid, b_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2, Psi1):
    # === STEP 2: Wb(z, b', a') and Wa(z, b', a') ===
    # (take discounted expectation of tomorrow's value function)
    print("step 2")
    if (ra > rb) and not np.all(Va_p > Vb_p):
        print(f"r_a > r_b but not Va_p > Vb_p")
    # ADDITION
    print("# ------------------------------------------------------------ #")
    
    Wb = beta * Vb_p
    
    Wa = beta * Va_p
    W_ratio = Wa / Wb
    
    # ADDITION
    # print(f"W_ratio starting guess: {W_ratio[0, ]}")
    print(f"w_ratio slice: {W_ratio[0,:,1]}")
    # === STEP 3: a'(z, b', a) for UNCONSTRAINED ===
    print(f"psi1 shape {Psi1.shape}")
    print("step 3")
    # for each (z, b', a), linearly interpolate to find a' between gridpoints
    # satisfying optimality condition W_ratio == 1+Psi1

    i, pi = lhs_equals_rhs_interpolate(W_ratio, 1 + Psi1)
    
    print(f"for (y,b) = (0,0), lhs = w_ratio[0,0,:]")
    print(f"so lhs = {W_ratio[0,0,:]}")
    print(f"and rhs = {1 + Psi1}")
    
    print(f"lhs - rhs[:,0]: {W_ratio[0,0,:] - (1 + Psi1[:,0])}")
    
    print(f"lhs - rhs[0,0]: {W_ratio[0,0,:] - (1 + Psi1[0,0])}")
    print(f"lhs - rhs[1,0]: {W_ratio[0,0,:] - (1 + Psi1[1,0])}")
    print(f"lhs - rhs[2,0]: {W_ratio[0,0,:] - (1 + Psi1[2,0])}")
    
    print(f"psi1[:,0] {Psi1[:,0]}")
    print(f"i[0,:,1], pi[0,:,1]: {i[0,:,1], pi[0, :, 1]}")

    # ADDITION
    # print(f"(i, pi): {(i, pi)}")

    # use same interpolation to get Wb and then c
    a_endo_unc = interpolate.apply_coord(i, pi, a_grid)
    c_endo_unc = interpolate.apply_coord(i, pi, Wb) ** (-eis)

    # === STEP 4: b'(z, b, a), a'(z, b, a) for UNCONSTRAINED ===
    print("step 4")
    # solve out budget constraint to get b(z, b', a)
    b_endo = (c_endo_unc + a_endo_unc + addouter(-z_grid, b_grid, -(1 + ra) * a_grid)
              + get_Psi_and_deriv(a_endo_unc, a_grid, ra, chi0, chi1, chi2)[0]) / (1 + rb)

    # interpolate this b' -> b mapping to get b -> b', so we have b'(z, b, a)
    # and also use interpolation to get a'(z, b, a)
    # (note utils.interpolate.interpolate_coord and utils.interpolate.apply_coord work on last axis,
    #  so we need to swap 'b' to the last axis, then back when done)
    i, pi = interpolate.interpolate_coord(b_endo.swapaxes(1, 2), b_grid)
    a_unc = interpolate.apply_coord(i, pi, a_endo_unc.swapaxes(1, 2)).swapaxes(1, 2)
    b_unc = interpolate.apply_coord(i, pi, b_grid).swapaxes(1, 2)

    # === STEP 5: a'(z, kappa, a) for CONSTRAINED ===

    print("step 5")
    # for each (z, kappa, a), linearly interpolate to find a' between gridpoints
    # satisfying optimality condition W_ratio/(1+kappa) == 1+Psi1, assuming b'=0
    lhs_con = W_ratio[:, 0:1, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
    print(lhs_con.shape)
    i, pi = lhs_equals_rhs_interpolate(lhs_con, 1 + Psi1)
    
    print(f"lhs_constrained = {lhs_con[0,0,:]}")
    print(f"rhs: = {1 + Psi1}")
    
    print(f"lhs_constrained - rhs[:,0]: {lhs_con[0,0,:] - (1 + Psi1[:,0])}")
    
    print(f"lhs_constrained - rhs[0,0]: {lhs_con[0,0,:][0,0,:] - (1 + Psi1[0,0])}")
    print(f"lhs_constrained - rhs[1,0]: {lhs_con[0,0,:] - (1 + Psi1[1,0])}")
    print(f"lhs_constrained - rhs[2,0]: {lhs_con[0,0,:] - (1 + Psi1[2,0])}")
    
    print(f"psi1[:,0] {Psi1[:,0]}")
    print(f"i[0,:,1], pi[0,:,1]: {i[0,:,1], pi[0, :, 1]}")

    # use same interpolation to get Wb and then c
    a_endo_con = interpolate.apply_coord(i, pi, a_grid)
    c_endo_con = ((1 + k_grid[np.newaxis, :, np.newaxis]) ** (-eis)
                  * interpolate.apply_coord(i, pi, Wb[:, 0:1, :]) ** (-eis))

    # === STEP 6: a'(z, b, a) for CONSTRAINED ===

    # solve out budget constraint to get b(z, kappa, a), enforcing b'=0
    b_endo = (c_endo_con + a_endo_con
              + addouter(-z_grid, np.full(len(k_grid), b_grid[0]), -(1 + ra) * a_grid)
              + get_Psi_and_deriv(a_endo_con, a_grid, ra, chi0, chi1, chi2)[0]) / (1 + rb)

    # interpolate this kappa -> b mapping to get b -> kappa
    # then use the interpolated kappa to get a', so we have a'(z, b, a)
    # (utils.interpolate.interpolate_y does this in one swoop, but since it works on last
    #  axis, we need to swap kappa to last axis, and then b back to middle when done)
    a_con = interpolate.interpolate_y(b_endo.swapaxes(1, 2), b_grid,
                                      a_endo_con.swapaxes(1, 2)).swapaxes(1, 2)

    # === STEP 7: obtain policy functions and update derivatives of value function ===

    # combine unconstrained solution and constrained solution, choosing latter
    # when unconstrained goes below minimum b
    a, b = a_unc.copy(), b_unc.copy()
    b[b <= b_grid[0]] = b_grid[0]
    a[b <= b_grid[0]] = a_con[b <= b_grid[0]]

    # calculate adjustment cost and its derivative
    Psi, _, Psi2 = get_Psi_and_deriv(a, a_grid, ra, chi0, chi1, chi2)

    # solve out budget constraint to get consumption and marginal utility
    c = addouter(z_grid, (1 + rb) * b_grid, (1 + ra) * a_grid) - Psi - a - b
    uc = c ** (-1 / eis)
    uce = e_grid[:, np.newaxis, np.newaxis] * uc

    # update derivatives of value function using envelope conditions
    Va = (1 + ra - Psi2) * uc
    Vb = (1 + rb) * uc
    
    print("# ------------------------------------------------------------ #")
    exit()

    return Va, Vb, a, b, c, uce


'''Supporting functions for HA block'''

def get_Psi_and_deriv(ap, a, ra, chi0, chi1, chi2):
    """Adjustment cost Psi(ap, a) and its derivatives with respect to
    first argument (ap) and second argument (a)"""
    a_with_return = (1 + ra) * a
    a_change = ap - a_with_return
    abs_a_change = np.abs(a_change)
    sign_change = np.sign(a_change)

    adj_denominator = a_with_return + chi0
    core_factor = (abs_a_change / adj_denominator) ** (chi2 - 1)

    Psi = chi1 / chi2 * abs_a_change * core_factor
    Psi1 = chi1 * sign_change * core_factor
    Psi2 = -(1 + ra) * (Psi1 + (chi2 - 1) * Psi / adj_denominator)
    
    if (chi1 == 0.0) and not (np.all(Psi1 == 0.0) and np.all(Psi2 == 0.0)):
        print(Psi1)
        print(f"sum of psi1: {np.sum(Psi1)}")
        print(f"sum of psi2: {np.sum(Psi2)}")
        raise ValueError("chi1 was 0.0 but not all Psi1 or Psi2 were 0.0")

    return Psi, Psi1, Psi2


def matrix_times_first_dim(A, X):
    """Take matrix A times vector X[:, i1, i2, i3, ... , in] separately
    for each i1, i2, i3, ..., in. Same output as A @ X if X is 1D or 2D"""
    # flatten all dimensions of X except first, then multiply, then restore shape
    return (A @ X.reshape(X.shape[0], -1)).reshape(X.shape)


def addouter(z, b, a):
    """Take outer sum of three arguments: result[i, j, k] = z[i] + b[j] + a[k]"""
    return z[:, np.newaxis, np.newaxis] + b[:, np.newaxis] + a


@guvectorize(['void(float64[:], float64[:,:], uint32[:], float64[:])'], '(ni),(ni,nj)->(nj),(nj)')
def lhs_equals_rhs_interpolate(lhs, rhs, iout, piout):
    """
    Given lhs (i) and rhs (i,j), for each j, find the i such that

    lhs[i] > rhs[i,j] and lhs[i+1] < rhs[i+1,j]

    i.e. where given j, lhs == rhs in between i and i+1.

    Also return the pi such that

    pi*(lhs[i] - rhs[i,j]) + (1-pi)*(lhs[i+1] - rhs[i+1,j]) == 0

    i.e. such that the point at pi*i + (1-pi)*(i+1) satisfies lhs == rhs by linear interpolation.

    If lhs[0] < rhs[0,j] already, just return u=0 and pi=1.

    ***IMPORTANT: Assumes that solution i is monotonically increasing in j
    and that lhs - rhs is monotonically decreasing in i.***
    """

    ni, nj = rhs.shape
    assert len(lhs) == ni

    i = 0
    for j in range(nj):
        while True:
            if lhs[i] < rhs[i, j]:
                break
            elif i < nj - 1:
                i += 1
            else:
                break

        if i == 0:
            iout[j] = 0
            piout[j] = 1
        else:
            iout[j] = i - 1
            err_upper = rhs[i, j] - lhs[i]
            err_lower = rhs[i - 1, j] - lhs[i - 1]
            piout[j] = err_upper / (err_upper - err_lower)
            