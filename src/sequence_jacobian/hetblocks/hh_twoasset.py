import warnings

import numpy as np
from numba import guvectorize

from ..blocks.het_block import het
from .. import interpolate

# def hh_init(b_grid, a_grid, z_grid, eis):
# def hh_init(b_grid, a_grid, z_grid, eis):
    
#     print("step 1")
#     Va = (0.6 + 1.1 * b_grid[:, np.newaxis] + a_grid) ** (-1 / eis) * np.ones((z_grid.shape[0], 1, 1))
#     Vb = (0.5 + b_grid[:, np.newaxis] + 1.2 * a_grid) ** (-1 / eis) * np.ones((z_grid.shape[0], 1, 1))
#     return Va, Vb

def is_monotone_decreasing_in_a(Va, Vb, tol=1e-12, strict=False):
    """
    Return True iff Va/Vb is non-increasing (or strictly decreasing) along the last axis.
    Assumes Vb > 0.
    """
    ratio = Va / Vb
    d = np.diff(ratio, axis=-1)
    return np.all(d < -tol) if strict else np.all(d <= tol)

def assert_monotone_decreasing_in_a(Va, Vb, tol=1e-12, strict=False):
    if not is_monotone_decreasing_in_a(Va, Vb, tol=tol, strict=strict):
        raise AssertionError("Expected (Va/Vb) to be non-increasing along a' (last axis).")

def check_W_ratio_constant(W_ratio, ra, rb, chi1, *, rtol=1e-6, atol=1e-10, warn_only=False, tag=""):
    """If chi1==0, verify W_ratio is approximately (1+ra)/(1+rb) everywhere."""
    if chi1 != 0:
        return True
    target = (1.0 + ra) / (1.0 + rb)
    err = np.max(np.abs(W_ratio - target))
    ok = err <= (atol + rtol * abs(target))
    if not ok:
        msg = (f"[chi1=0] Expected W_ratio ≈ {(target):.12g} everywhere "
               f"(max abs err = {err:.3e}){(' ' + tag) if tag else ''}.")
        if warn_only:
            warnings.warn(msg, RuntimeWarning)
        else:
            raise AssertionError(msg)
    return ok


# Va, Vb shaped as (y, b, a)
# def hh_init(a_grid, b_grid, z_grid, eis):
#     A = a_grid[None, :]                  # (1, a)
#     B = b_grid[:, None]                  # (b, 1)
#     Va_2d = (1.0 + B + A) ** (-1.0 / eis)
#     Vb_2d = (1.0 + B + 0.5 * A) ** (-1.0 / eis)
#     Va = np.broadcast_to(Va_2d, (len(z_grid),) + Va_2d.shape).copy()
#     Vb = np.broadcast_to(Vb_2d, (len(z_grid),) + Vb_2d.shape).copy()
    
#     assert_monotone_decreasing_in_a(Va, Vb)
#     return Va, Vb

def hh_init(a_grid, b_grid, z_grid, eis,
            alpha=1.0, eps=0.5, gamma=1.0, delta=0.5,
            check_monotone=True,
            require_crossing_inside=True,
            strict_monotone=False):
    if not (delta * eps < 1.0):
        raise ValueError("delta*eps >= 1")
    if not (delta * alpha < gamma):
        raise ValueError("delta*alpha >= gamma")
    if np.isclose(delta, 1.0):
        raise ValueError("delta == 1")

    A = a_grid[None, :]              # (1, a)s
    B = b_grid[:, None]              # (b, 1)

    Va_2d = (alpha + eps * B + A) ** (-1.0 / eis)       # (b, a)
    Vb_2d = (gamma + B + delta * A) ** (-1.0 / eis)     # (b, a)

    Va = np.broadcast_to(Va_2d, (len(z_grid),) + Va_2d.shape).copy()
    Vb = np.broadcast_to(Vb_2d, (len(z_grid),) + Vb_2d.shape).copy()

    if check_monotone:
        if not is_monotone_decreasing_in_a(Va, Vb, strict=strict_monotone):
            print("Expected Va/Vb to be non-increasing along a.")
            raise AssertionError("Expected Va/Vb to be non-increasing along a.")

    return Va, Vb


def adjustment_costs(a, a_grid, ra, chi0, chi1, chi2):
    chi = get_Psi_and_deriv(a, a_grid, ra, chi0, chi1, chi2)[0]
    return chi


def marginal_cost_grid(a_grid, ra, chi0, chi1, chi2):
    # precompute Psi1(a', a) on grid of (a', a) for steps 3 and 5
    Psi1 = get_Psi_and_deriv(a_grid[:, np.newaxis],
                             a_grid[np.newaxis, :], ra, chi0, chi1, chi2)[1]

    return Psi1

@het(exogenous='Pi', policy=['b', 'a'], backward=['Vb', 'Va'],
     hetinputs=[marginal_cost_grid], hetoutputs=[adjustment_costs], backward_init=hh_init)  
def hh(Va_p, Vb_p, a_grid, b_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2, Psi1):
    # === STEP 2: Wb(z, b', a') and Wa(z, b', a') ===
    # (take discounted expectation of tomorrow's value function)
    # print("step 2")
    # if (ra > rb) and not np.all(Va_p > Vb_p):
    #     print(f"r_a > r_b but not Va_p > Vb_p")
    # # ADDITION
    # print("# ------------------------------------------------------------ #")
    
    Wb = beta * Vb_p
    
    Wa = beta * Va_p
    W_ratio = Wa / Wb

    # for each (z, b', a), linearly interpolate to find a' between gridpoints
    # satisfying optimality condition W_ratio == 1+Psi1

    i, pi = lhs_equals_rhs_interpolate(W_ratio, 1 + Psi1)

    # use same interpolation to get Wb and then c
    a_endo_unc = interpolate.apply_coord(i, pi, a_grid)
    c_endo_unc = interpolate.apply_coord(i, pi, Wb) ** (-eis)

    # === STEP 4: b'(z, b, a), a'(z, b, a) for UNCONSTRAINED ===
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

    # for each (z, kappa, a), linearly interpolate to find a' between gridpoints
    # satisfying optimality condition W_ratio/(1+kappa) == 1+Psi1, assuming b'=0
    lhs_con = W_ratio[:, 0:1, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
    i, pi = lhs_equals_rhs_interpolate(lhs_con, 1 + Psi1)
        
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
    
    # print("# ------------------------------------------------------------ #")

    return Va, Vb, a, b, c, uce

# policy and bacward order as in grid!
@het(exogenous='Pi', policy=['b', 'a'], backward=['Vb', 'Va'],
     hetinputs=[marginal_cost_grid], hetoutputs=[adjustment_costs], backward_init=hh_init)  
def hh_print(Va_p, Vb_p, a_grid, b_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2, Psi1):
    # === STEP 2: Wb(z, b', a') and Wa(z, b', a') ===
    # (take discounted expectation of tomorrow's value function)
    # print("step 2")
    # if (ra > rb) and not np.all(Va_p > Vb_p):
    #     print(f"r_a > r_b but not Va_p > Vb_p")
    # # ADDITION
    # print("# ------------------------------------------------------------ #")
    
    Wb = beta * Vb_p
    
    Wa = beta * Va_p
    W_ratio = Wa / Wb

    # ADDITION
    print(f"W_ratio starting guess: {W_ratio[0, ]}")
    print(f"w_ratio slice: {W_ratio[0,:,1]}")
    # # === STEP 3: a'(z, b', a) for UNCONSTRAINED ===
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

    print(f"for (y,b) = (0,0), lhs = w_ratio[0,0,:]")
    print(f"so lhs = {W_ratio[0,0,:]}")
    print(f"and rhs = {1 + Psi1}")
    
    print(f"lhs - rhs[:,0]: {W_ratio[0,0,:] - (1 + Psi1[:,0])}")
    
    print(f"lhs - rhs[0,0]: {W_ratio[0,0,:] - (1 + Psi1[0,0])}")
    print(f"lhs - rhs[1,0]: {W_ratio[0,0,:] - (1 + Psi1[1,0])}")
    print(f"lhs - rhs[2,0]: {W_ratio[0,0,:] - (1 + Psi1[2,0])}")
    
    print(f"psi1[:,0] {Psi1[:,0]}")
    print(f"i[0,:,1], pi[0,:,1]: {i[0,:,1], pi[0, :, 1]}")

    print(f"(i, pi): {(i, pi)}")

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
    print("5a")
    lhs_con = W_ratio[:, 0:1, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
    
    print("5b")
    i, pi = lhs_equals_rhs_interpolate(lhs_con, 1 + Psi1)
    
    # print(f"lhs_constrained = {lhs_con[0,0,:]}")
    # print(f"rhs: = {1 + Psi1}")
    
    # print(f"lhs_constrained - rhs[:,0]: {lhs_con[0,0,:] - (1 + Psi1[:,0])}")
    
    # print(f"lhs_constrained - rhs[0,0]: {lhs_con[0,0,:] - (1 + Psi1[0,0])}")
    # print(f"lhs_constrained - rhs[1,0]: {lhs_con[0,0,:] - (1 + Psi1[1,0])}")
    # print(f"lhs_constrained - rhs[2,0]: {lhs_con[0,0,:] - (1 + Psi1[2,0])}")
    
    # print(f"psi1[:,0] {Psi1[:,0]}")
    # print(f"i[0,:,1], pi[0,:,1]: {i[0,:,1], pi[0, :, 1]}")

    # use same interpolation to get Wb and then c
    print("5c")
    a_endo_con = interpolate.apply_coord(i, pi, a_grid)
    print(a_endo_con.shape)
    
    # print(f"a_endo_con[] = {a_endo_con[0,0,:]}")
    print("5d")
    c_endo_con = ((1 + k_grid[np.newaxis, :, np.newaxis]) ** (-eis)
                  * interpolate.apply_coord(i, pi, Wb[:, 0:1, :]) ** (-eis))

    # === STEP 6: a'(z, b, a) for CONSTRAINED ===
    print("step 6")
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
    print("step 7")
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
    
    # print("# ------------------------------------------------------------ #")

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
            