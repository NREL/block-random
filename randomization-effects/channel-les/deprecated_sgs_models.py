import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Read filtered data in realspace from the precomupted file
data = np.load("scaled.npy")

# Ordering on the indices in the data files
vels = ["u", "v", "w"]
grads = ["dudx", "dudy", "dudz", "dvdx", "dvdy", "dvdz", "dwdx", "dwdy", "dwdz"]
taus = ["uu", "vv", "ww", "uv", "uw", "vw"]
fields = vels + grads + taus

# Get dimensions
_, ny, nz, nx, nfiles = data.shape

# Init tensors required for SGS computations

# gradient g_ij = du_i/dx_j
g = np.empty((3, 3, ny, nz, nx, nfiles))
# moment of gradient, g_ik g_kj tensor
g2 = np.zeros((3, 3, ny, nz, nx, nfiles))
# g_ij + g_ji tensor
S = np.empty((3, 3, ny, nz, nx, nfiles))
# tensor defined in the WALE paper
Sd = np.empty((3, 3, ny, nz, nx, nfiles))

# Populate the gradients in a convenient format from data file
g[0, 0] = data[fields.index("dudx"), ...]
g[0, 1] = data[fields.index("dudy"), ...]
g[0, 2] = data[fields.index("dudz"), ...]
g[1, 0] = data[fields.index("dvdx"), ...]
g[1, 1] = data[fields.index("dvdy"), ...]
g[1, 2] = data[fields.index("dvdz"), ...]
g[2, 0] = data[fields.index("dwdx"), ...]
g[2, 1] = data[fields.index("dwdy"), ...]
g[2, 2] = data[fields.index("dwdz"), ...]

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
        Sd[i, j, ...] = 0.5 * (g2[i, j, ...] + g2[j, i, ...]) - (i == j) * g2_trace / 3

# Compute magnitudes of S and Sd tensors
S_mag = np.zeros_like(g2_trace)
Sd_mag = np.zeros_like(g2_trace)
for i in range(3):
    for j in range(3):
        S_mag += S[i, j, ...] ** 2
        Sd_mag += Sd[i, j, ...] ** 2

# Smagonisky model
delta = 1530.0 / 5200.0
cs = 0.1
nu_smag = (cs * delta) ** 2 * np.sqrt(2.0) * S_mag ** 0.5
uv_smag = -2 * nu_smag * S[0, 1]

# WALE model
nu_wale = (
    10 * (cs * delta) ** 2 * (Sd_mag) ** (1.5) / (S_mag ** (2.5) + Sd_mag ** (1.25))
)
uv_wale = -2 * nu_wale * S[0, 1]


MSE_smag = ((uv_smag - data[fields.index("uv"), ...]) ** 2).mean()
MSE_wale = ((uv_wale - data[fields.index("uv"), ...]) ** 2).mean()
print("SMAG MSE: ", MSE_smag)
print("WALE MSE: ", MSE_wale)

C_smag = np.mean(
    (uv_smag - uv_smag.mean()) * (data[15, ...] - data[15, ...].mean())
) / (
    np.mean((uv_smag - uv_smag.mean()) ** 2) ** 0.5
    * np.mean((data[15, ...] - data[15, ...].mean()) ** 2) ** 0.5
)
C_wale = np.mean(
    (uv_wale - uv_wale.mean()) * (data[15, ...] - data[15, ...].mean())
) / (
    np.mean((uv_wale - uv_wale.mean()) ** 2) ** 0.5
    * np.mean((data[15, ...] - data[15, ...].mean()) ** 2) ** 0.5
)
print("SMAG Correlation: ", C_smag)
print("WALE Correlation: ", C_wale)


# Save prediction plots
plt.figure()
plt.plot(
    data[fields.index("uv"), 768, ...].flatten(),
    uv_wale[768, ...].flatten(),
    marker="x",
    linestyle="none",
)
plt.plot([-2, 1.5], [-2, 1.5], "--")
plt.xlabel(r"$\tau_{12}$ data")
plt.ylabel(r"$\tau_{12}$ prediction")
plt.savefig("WALE.pdf")

plt.figure()
plt.plot(
    data[fields.index("uv"), 768, ...].flatten(),
    uv_smag[768, ...].flatten(),
    marker="x",
    linestyle="none",
)
plt.plot([-2, 1.5], [-2, 1.5], "--")
plt.xlabel(r"$\tau_{12}$ data")
plt.ylabel(r"$\tau_{12}$ prediction")
plt.savefig("SMAG.pdf")
