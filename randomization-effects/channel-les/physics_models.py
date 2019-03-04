# ===============================================================================
#
# Imports
#
# ===============================================================================
import numpy as np
import pandas as pd


# ===============================================================================
#
# Function definitions
#
# ===============================================================================
def smag(X):

    S, S_mag, _ = compute_derived_quantities(X)

    delta = np.sqrt(8 * np.pi / 200 * 3 * np.pi / 128)
    cs = 0.1
    nut = (cs * delta) ** 2 * np.sqrt(2.0) * S_mag ** 0.5
    tau_uv = 2 * nut * S[0, 1]

    return pd.DataFrame(tau_uv, columns=["tau_uv"], index=X.index)


# ===============================================================================
def wale(X):

    S, S_mag, Sd_mag = compute_derived_quantities(X)

    delta = np.sqrt(8 * np.pi / 200 * 3 * np.pi / 128)
    cw = 0.325
    nut = (cw * delta) ** 2 * (Sd_mag ** 1.5) / (S_mag ** 2.5 + Sd_mag ** 1.25)
    tau_uv = 2 * nut * S[0, 1]

    return pd.DataFrame(tau_uv, columns=["tau_uv"], index=X.index)


# ===============================================================================
def compute_derived_quantities(X):

    # gradient g_ij = du_i/dx_j
    g = np.empty((3, 3, X.shape[0]))
    # moment of gradient, g_ik g_kj tensor
    g2 = np.zeros((3, 3, X.shape[0]))
    # g_ij + g_ji tensor
    S = np.empty((3, 3, X.shape[0]))
    # tensor defined in the WALE paper
    Sd = np.empty((3, 3, X.shape[0]))

    # Populate the gradients in a convenient format
    g[0, 0, :] = X.dudx
    g[0, 1, :] = X.dudy
    g[0, 2, :] = X.dudz
    g[1, 0, :] = X.dvdx
    g[1, 1, :] = X.dvdy
    g[1, 2, :] = X.dvdz
    g[2, 0, :] = X.dwdx
    g[2, 1, :] = X.dwdy
    g[2, 2, :] = X.dwdz

    # Compute S_ij and g2
    for i in range(3):
        for j in range(3):
            S[i, j, ...] = 0.5 * (g[i, j, ...] + g[j, i, ...])
            for k in range(3):
                g2[i, j, ...] += g[i, k, ...] * g[k, j, ...]

    # Need trace of g2 for computing Sd
    g2_trace = g2[0, 0, ...] + g2[1, 1, ...] + g2[2, 2, ...]

    # Compute Sd tensor
    for i in range(3):
        for j in range(3):
            Sd[i, j, ...] = (
                0.5 * (g2[i, j, ...] + g2[j, i, ...]) - (i == j) * g2_trace / 3
            )

    # Compute magnitudes of S and Sd tensors
    S_mag = np.zeros_like(g2_trace)
    Sd_mag = np.zeros_like(g2_trace)
    for i in range(3):
        for j in range(3):
            S_mag += S[i, j, ...] ** 2
            Sd_mag += Sd[i, j, ...] ** 2

    return S, S_mag, Sd_mag
