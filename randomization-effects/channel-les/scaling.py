import functions
import numpy as np
import h5py
import time

Re_tau = 5185.897
nu = 8e-6

# Read true means file to get Y, u_mean and w_mean
truemeans = functions.read_true_means()
Y = truemeans["Y"]
ny = len(Y)
nx = 200
nz = 128
LMFx = 4
LMFz = 1.5
k_plus_cutoff = 2.0 * np.pi / 1530.0

# Create a circular mask
nxh = int(nx / 2)
circular_mask = np.zeros([nz, nxh + 1, ny])
kx_plus = np.fft.rfftfreq(nx) * nx / LMFx / Re_tau
kz_plus = np.fft.fftfreq(nz) * nz / LMFz / Re_tau
for j in range(nz):
    for i in range(nxh + 1):
        if k_plus_cutoff ** 2 - (kx_plus[i] ** 2 + kz_plus[j] ** 2) > 0:
            circular_mask[j, i] = 1.0

# NOTE: path might change based on where you have the data file
# either change to the absolute path of where its stored or create an appropriate symlink
path = "../200-128/"

#  Read velocity/stress data from files
read_start = time.perf_counter()
data = functions.read_data(path, nx, ny, nz, truemeans, nu, circular_mask)
read_end = time.perf_counter()
print("read time: ", read_end - read_start)

# Create a constant scaled version of the raw data in realspace
scale_start = time.perf_counter()
scaled = functions.scale_data(data)
scale_end = time.perf_counter()
print("scale time: ", scale_end - scale_start)

# Save the scaled data to a npy file for quick learning on it
save_start = time.perf_counter()
vels = ["u", "v", "w"]
grads = ["dudx", "dudy", "dudz", "dvdx", "dvdy", "dvdz", "dwdx", "dwdy", "dwdz"]
taus = ["uu", "vv", "ww", "uv", "uw", "vw"]
fields = vels + grads + taus
nfiles = data["u"].shape[-1]
save = np.empty((18, ny, nz, nx, nfiles))
for i in range(18):
    save[i, :] = scaled[fields[i]]
np.save("scaled", save)
save_end = time.perf_counter()
print("save time: ", save_end - save_start)
