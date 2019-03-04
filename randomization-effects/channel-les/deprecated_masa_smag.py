import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import functions

data = np.zeros((18, 12, 16, 10, 1))
vels = ["u", "v", "w"]
grads = ["dudx", "dudy", "dudz", "dvdx", "dvdy", "dvdz", "dwdx", "dwdy", "dwdz"]
taus = ["uu", "vv", "ww", "uv", "uw", "vw"]
fields = vels + grads + taus
_, ny, nz, nx, nfiles = data.shape
g = np.empty((3, 3, ny, nz, nx, nfiles))
g2 = np.zeros((3, 3, ny, nz, nx, nfiles))
S = np.empty((3, 3, ny, nz, nx, nfiles))
Sd = np.empty((3, 3, ny, nz, nx, nfiles))

x = np.linspace(0, 8 * np.pi, nx, endpoint=False)
y = np.linspace(-1, 1, ny, endpoint=False)
z = np.linspace(0, 3 * np.pi, nz, endpoint=False)
YY, ZZ, XX = np.meshgrid(y, z, x, indexing="ij")

# # Fill in velocities
# data[fields.index('u'),:,:,:,0] = np.sin(XX) * np.cos(YY) * np.cos(ZZ/1.5)
# data[fields.index('v'),:,:,:,0] = np.cos(XX) * np.sin(YY) * np.cos(ZZ/1.5)
# data[fields.index('w'),:,:,:,0] = -3*np.cos(XX) * np.cos(YY) * np.sin(ZZ/1.5)

# # Fill in gradients
# data[fields.index('dudx'),:,:,:,0] = np.cos(XX) * np.cos(YY) * np.cos(ZZ/1.5)
# data[fields.index('dvdx'),:,:,:,0] = -np.sin(XX) * np.sin(YY) * np.cos(ZZ/1.5)
# data[fields.index('dwdx'),:,:,:,0] = 3*np.sin(XX) * np.cos(YY) * np.sin(ZZ/1.5)
# data[fields.index('dudy'),:,:,:,0] = np.sin(XX) * -np.sin(YY) * np.cos(ZZ/1.5)
# data[fields.index('dvdy'),:,:,:,0] = np.cos(XX) * np.cos(YY) * np.cos(ZZ/1.5)
# data[fields.index('dwdy'),:,:,:,0] = -3*np.cos(XX) * -np.sin(YY) * np.sin(ZZ/1.5)
# data[fields.index('dudz'),:,:,:,0] = np.sin(XX) * np.cos(YY) * -np.sin(ZZ/1.5)/1.5
# data[fields.index('dvdz'),:,:,:,0] = np.cos(XX) * np.sin(YY) * -np.sin(ZZ/1.5)/1.5
# data[fields.index('dwdz'),:,:,:,0] = -3*np.cos(XX) * np.cos(YY) * np.cos(ZZ/1.5)/1.5

# Fill in velocities
data[fields.index("u"), :, :, :, 0] = 0.5 * XX ** 2
data[fields.index("v"), :, :, :, 0] = 0.5 * YY ** 2
data[fields.index("w"), :, :, :, 0] = 0.5 * ZZ ** 2

# Fill in gradients
data[fields.index("dudx"), :, :, :, 0] = XX
data[fields.index("dvdx"), :, :, :, 0] = 0
data[fields.index("dwdx"), :, :, :, 0] = 0
data[fields.index("dudy"), :, :, :, 0] = 0
data[fields.index("dvdy"), :, :, :, 0] = YY
data[fields.index("dwdy"), :, :, :, 0] = 0
data[fields.index("dudz"), :, :, :, 0] = 0
data[fields.index("dvdz"), :, :, :, 0] = 0
data[fields.index("dwdz"), :, :, :, 0] = ZZ

g[0, 0] = data[fields.index("dudx"), ...]
g[0, 1] = data[fields.index("dudy"), ...]
g[0, 2] = data[fields.index("dudz"), ...]
g[1, 0] = data[fields.index("dvdx"), ...]
g[1, 1] = data[fields.index("dvdy"), ...]
g[1, 2] = data[fields.index("dvdz"), ...]
g[2, 0] = data[fields.index("dwdx"), ...]
g[2, 1] = data[fields.index("dwdy"), ...]
g[2, 2] = data[fields.index("dwdz"), ...]
for i in range(3):
    for j in range(3):
        S[i, j, ...] = 0.5 * (g[i, j, ...] + g[j, i, ...])
        for k in range(3):
            g2[i, j, ...] += g[i, k, ...] * g[k, j, ...]
g2_trace = g2[0, 0, ...] + g2[1, 1, ...] + g2[2, 2, ...]
for i in range(3):
    for j in range(3):
        Sd[i, j, ...] = 0.5 * (g2[i, j, ...] + g2[j, i, ...]) - (i == j) * g2_trace / 3
S_mag = np.zeros_like(g2_trace)
Sd_mag = np.zeros_like(g2_trace)
for i in range(3):
    for j in range(3):
        S_mag += S[i, j, ...] ** 2
        Sd_mag += Sd[i, j, ...] ** 2

# Smagonisky
delta = 1530.0 / 5200.0
cs = 0.1
nu_smag = (cs * delta) ** 2 * np.sqrt(2.0) * S_mag ** 0.5
