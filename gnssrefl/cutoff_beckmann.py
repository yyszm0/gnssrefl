import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------
# Beckmann–Spizzichino model utilities
# ----------------------------------------
def area_A(eps_rad, lam, h):
    """
    Approximate scattering area A using Fresnel zone:
        A ≈ π λ h / sin(ε)
    """
    return np.pi * lam * h / np.sin(eps_rad)


def g_from_sigma(sigma_h, eps_rad, lam):
    """
    Compute g = (4π σ_h sin ε / λ)^2
    """
    return (4.0 * np.pi * sigma_h * np.sin(eps_rad) / lam) ** 2


def S_series(g, m_max=80):
    """
    Compute S(g) = Σ_{m=1}^{∞} g^m / (m! * m)
    using a truncated series up to m_max.
    """
    # Assume scalar g for simplicity
    s = 0.0
    term = g  # g^1 / 1!
    s += term

    for m in range(2, m_max + 1):
        term *= g / m         # term = g^m / m!
        s += term / m         # add g^m / (m! * m)

    return s


def incoh(sigma_h, eps_rad, T, lam, h, m_max=80):
    """
    Compute incoherent component:

        incoh = (π T^2 / A) * Σ g^m / (m! * m)

    where
        A = A(eps_rad, lam, h)
        g = g_from_sigma(sigma_h, eps_rad, lam)
    """
    A = area_A(eps_rad, lam, h)
    g = g_from_sigma(sigma_h, eps_rad, lam)
    Sg = S_series(g, m_max=m_max)
    return (np.pi * T ** 2 / A) * Sg


# ----------------------------------------
# Solve incoh(sigma_h) = 1 for sigma_h by Newton's method
# ----------------------------------------
def solve_sigma_for_cutoff(
    eps_rad,
    T,
    lam,
    h,
    sigma_init=0.5,
    tol=1e-6,
    max_iter=50,
    m_max=80,
):
    """
    Solve incoh(sigma_h, eps_rad, T, lam, h) = 1
    using Newton's method for sigma_h.
    """
    sigma = float(sigma_init)

    for _ in range(max_iter):
        f = incoh(sigma, eps_rad, T, lam, h, m_max=m_max) - 1.0
        if abs(f) < tol:
            return sigma

        # Numerical derivative
        delta = max(1e-4, 1e-3 * abs(sigma))
        sigma_p = sigma + delta
        sigma_m = max(sigma - delta, 1e-8)

        f_p = incoh(sigma_p, eps_rad, T, lam, h, m_max=m_max) - 1.0
        f_m = incoh(sigma_m, eps_rad, T, lam, h, m_max=m_max) - 1.0
        fp = (f_p - f_m) / (sigma_p - sigma_m)

        if fp == 0 or not np.isfinite(fp):
            return np.nan

        sigma_new = sigma - f / fp

        # sigma_h must be positive
        if sigma_new <= 0:
            sigma_new = sigma / 2.0

        if abs(sigma_new - sigma) < tol:
            return sigma_new

        sigma = sigma_new

    # If not converged, return NaN
    return np.nan


# ----------------------------------------
# Grid computation and plotting
# ----------------------------------------
def precompute_sigma_grid(
    T_values,
    eps_deg_values,
    lam,
    h,
    sigma_init_guess=0.5,
    m_max=80,
):
    """
    Compute sigma_h satisfying incoh = 1 on a (T, ε) grid.

    Returns:
        sigma_grid[i, j] = sigma_h for T_values[i], eps_deg_values[j]
    """
    T_values = np.array(T_values, dtype=float)
    eps_deg_values = np.array(eps_deg_values, dtype=float)
    sigma_grid = np.full((len(T_values), len(eps_deg_values)), np.nan, dtype=float)

    for i, T in enumerate(T_values):
        # Use previous solution in the row as initial guess
        sigma_prev = sigma_init_guess
        for j, eps_deg in enumerate(eps_deg_values):
            eps_rad = np.deg2rad(eps_deg)
            sigma_init = sigma_prev
            sigma = solve_sigma_for_cutoff(
                eps_rad,
                T,
                lam,
                h,
                sigma_init=sigma_init,
                m_max=m_max,
            )
            sigma_grid[i, j] = sigma
            if not np.isnan(sigma):
                sigma_prev = sigma

    return sigma_grid


def plot_sigma_vs_epsilon(T_values, eps_deg_values, sigma_grid):
    """
    Plot sigma_h vs. epsilon for each T.
    """
    plt.figure()
    for i, T in enumerate(T_values):
        sigmas = sigma_grid[i, :]
        plt.plot(eps_deg_values, sigmas, label=f"T = {T:.1f} m")

    plt.xlabel("Cut-off elevation ε [deg]")
    plt.ylabel("σ_h [m]")
    plt.title("σ_h vs. cut-off elevation ε (incoh = 1)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------
# Grid-based bilinear interpolation: (T, ε) -> sigma_h
# ----------------------------------------
def _find_bracketing_indices(x_values, x):
    """
    Find indices (i0, i1) such that x_values[i0] <= x <= x_values[i1].
    """
    x_values = np.array(x_values, dtype=float)

    if x < x_values[0] or x > x_values[-1]:
        raise ValueError(f"value {x} is outside grid range [{x_values[0]}, {x_values[-1]}]")

    idx = np.searchsorted(x_values, x)
    if x == x_values[0]:
        return 0, 0
    if x == x_values[-1]:
        return len(x_values) - 1, len(x_values) - 1

    i1 = idx
    i0 = idx - 1
    return i0, i1


def bilinear_interpolate_sigma(
    T_query,
    eps_deg_query,
    T_values,
    eps_deg_values,
    sigma_grid,
):
    """
    Bilinear interpolation of sigma_h from a precomputed grid.
    """
    T_values = np.array(T_values, dtype=float)
    eps_deg_values = np.array(eps_deg_values, dtype=float)
    sigma_grid = np.array(sigma_grid, dtype=float)

    iT0, iT1 = _find_bracketing_indices(T_values, T_query)
    ie0, ie1 = _find_bracketing_indices(eps_deg_values, eps_deg_query)

    T0, T1 = T_values[iT0], T_values[iT1]
    e0, e1 = eps_deg_values[ie0], eps_deg_values[ie1]

    Q11 = sigma_grid[iT0, ie0]
    Q12 = sigma_grid[iT0, ie1]
    Q21 = sigma_grid[iT1, ie0]
    Q22 = sigma_grid[iT1, ie1]

    # Handle exact grid points
    if T0 == T1 and e0 == e1:
        return Q11
    if T0 == T1:
        w = (eps_deg_query - e0) / (e1 - e0)
        return (1 - w) * Q11 + w * Q12
    if e0 == e1:
        w = (T_query - T0) / (T1 - T0)
        return (1 - w) * Q11 + w * Q21

    # Bilinear interpolation
    t = (T_query - T0) / (T1 - T0)
    u = (eps_deg_query - e0) / (e1 - e0)

    sigma_interp = (
        (1 - t) * (1 - u) * Q11
        + (1 - t) * u * Q12
        + t * (1 - u) * Q21
        + t * u * Q22
    )
    return sigma_interp


def swh_from_T_eps(
    T_query,
    eps_deg_query,
    T_values,
    eps_deg_values,
    sigma_grid,
    factor=4.0,
):
    """
    Compute SWH from T and cut-off elevation epsilon using
    interpolated sigma_h and the relation SWH = factor * sigma_h.
    """
    sigma = bilinear_interpolate_sigma(
        T_query,
        eps_deg_query,
        T_values,
        eps_deg_values,
        sigma_grid,
    )
    return factor * sigma


# ----------------------------------------
# Main function (example usage as a standalone script)
# ----------------------------------------
if __name__ == "__main__":
    # Example physical parameters
    lam = 0.19029367  # GNSS signal wavelength [m], e.g., GPS L1
    h = 20.0          # Antenna height above sea surface [m]

    # 1. Define grid
    T_values = np.arange(1.0, 11.0, 1.0)     # T = 1–10 m
    eps_deg_values = np.linspace(5, 30, 26)  # ε = 5–30 deg

    # 2. Compute sigma_h satisfying incoh = 1 on the grid
    sigma_grid = precompute_sigma_grid(T_values, eps_deg_values, lam, h)

    # 3. Plot sigma_h vs. epsilon for each T
    plot_sigma_vs_epsilon(T_values, eps_deg_values, sigma_grid)

    # 4. Example: compute SWH for a specific (T, epsilon)
    T_test = 5.5     # [m]
    eps_test = 12.3  # [deg]
    try:
        swh_val = swh_from_T_eps(
            T_test,
            eps_test,
            T_values,
            eps_deg_values,
            sigma_grid,
            factor=4.0,   # SWH = 4 * sigma_h (assumed)
        )
        print(f"T = {T_test:.2f} m, ε = {eps_test:.2f} deg -> SWH ≈ {swh_val:.3f} m")
    except ValueError as e:
        print("Failed to interpolate SWH:", e)
