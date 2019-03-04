import numpy as np
import h5py
import glob
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
from scipy import interpolate as interp
from scipy import optimize as op

# The data files are stored just for the fluctuating variables
# Need to read the true means from a separate file
def read_true_means():
    True_Mean = h5py.File("../../data/True_mean_R5200.h5", "r")
    truemeans = {}
    truemeans["u"] = True_Mean["U_mean"][()]
    truemeans["w"] = True_Mean["W_mean"][()]
    truemeans["Y"] = True_Mean["LABS_COL"][()]
    True_Mean.close()
    return truemeans


# Reads field in wavespace, applies an optional mask for filtering and transforms to realspace
def filt_real_field(nx, ny, nz, hfile, field, mask=1):
    nxh = int(nx / 2)
    nzh = int(nz / 2)
    # Read in just the relevant data from wavespace
    filt = np.zeros([nz, nxh + 1, ny], dtype="complex")
    filt[0:nzh, :nxh, :].real = hfile[field][0:nzh, 0:nxh, :ny][..., 0]
    filt[0:nzh, :nxh, :].imag = hfile[field][0:nzh, 0:nxh, :ny][..., 1]
    filt[nzh + 1 :, :nxh, :].real = hfile[field][-nzh + 1 :, 0:nxh, :ny][..., 0]
    filt[nzh + 1 :, :nxh, :].imag = hfile[field][-nzh + 1 :, 0:nxh, :ny][..., 1]
    # Convert to realspace
    filt = np.fft.irfft(np.fft.ifft(filt * mask, axis=0), axis=1) * nx * nz
    # Transpose so the axes are (y,z,x)
    return filt.transpose((2, 0, 1))


# Reads data from wavespace, filters it according to the cirular filter defined and converts to realspace
def read_data(path, nx, ny, nz, truemeans, nu, mask=1):
    fields = [
        "u",
        "v",
        "w",
        "uu",
        "vv",
        "ww",
        "uv",
        "uw",
        "vw",
        "dudy",
        "dwdy",
        "dudx",
        "dwdz",
    ]
    vels = ["u", "v", "w"]
    wall = ["dudy", "dwdy"]
    taumap = {
        "uu": ["u", "u"],
        "vv": ["v", "v"],
        "ww": ["w", "w"],
        "uv": ["u", "v"],
        "uw": ["u", "w"],
        "vw": ["v", "w"],
    }
    derivs = ["dudx", "dudy", "dwdz", "dwdy"]
    filenames = sorted(glob.glob(path + "*.h5"))
    dfilenames = {}
    for field in derivs:
        dfilenames[field] = sorted(glob.glob(path + "/dudy/" + field + "_*"))
    nfiles = len(filenames)
    nfiles = 1
    data = {}
    for field in fields:
        if field in wall:
            data["w" + field] = np.empty((2, nz, nx, nfiles))
            data[field] = np.empty((ny, nz, nx, nfiles))
        else:
            data[field] = np.empty((ny, nz, nx, nfiles))
    data["dvdy"] = np.empty((ny, nz, nx, nfiles))
    data["u_tau"] = np.empty((2, nz, nx, nfiles))
    for fnum in range(nfiles):
        print(filenames[fnum])
        hfile = h5py.File(filenames[fnum], "r")
        dfiles = {}
        for field in derivs:
            dfiles[field] = h5py.File(dfilenames[field][fnum], "r")

        # Read bspline matrices
        if fnum == 0:
            D0 = hfile["LABS_D0_FULL"].value
            D1 = hfile["LABS_D1_FULL"].value
        # Read u,v,w
        for field in vels:
            data[field][:, :, :, fnum] = filt_real_field(nx, ny, nz, hfile, field, mask)
        # Read stored derivatives
        for field in derivs:
            data[field][:, :, :, fnum] = filt_real_field(
                nx, ny, nz, dfiles[field], field, mask
            )
        # Get the wall dudy and dwdy for u_tau
        for field in wall:
            data["w" + field][0, :, :, fnum] = data[field][0, :, :, fnum]
            data["w" + field][1, :, :, fnum] = data[field][-1, :, :, fnum]
        # Read the u_iu_j terms and compute tau_ij
        for field in taumap.keys():
            data[field][:, :, :, fnum] = -filt_real_field(
                nx, ny, nz, hfile, field, mask
            ) + np.multiply(
                data[taumap[field][0]][:, :, :, fnum],
                data[taumap[field][1]][:, :, :, fnum],
            )
        # Add the true mean value for u and w
        for i in range(ny):
            for field in ["u", "w"]:
                data[field][:, :, :, fnum][i, :] = (
                    data[field][:, :, :, fnum][i, :] + truemeans[field][i]
                )
        for j in range(nz):
            for i in range(nx):
                # Do the computations for dvdy
                data["dvdy"][:, j, i, fnum] = D1.dot(
                    np.linalg.solve(D0, data["v"][:, j, i, fnum])
                )
        # Close the handle for the h5 file
        hfile.close()
        for field in derivs:
            dfiles[field].close()
    # Compute u_tau
    data["u_tau"] = (
        np.square(data["wdudy"]) + np.square(data["wdwdy"])
    ) ** 0.25 * nu ** 0.5
    # Fill missing wall parallel derivatives
    data["dudz"] = oget_z_deriv(data["u"])
    data["dvdx"] = oget_x_deriv(data["v"])
    data["dvdz"] = oget_z_deriv(data["v"])
    data["dwdx"] = oget_x_deriv(data["w"])
    print("Finished reading data for %d time steps" % (nfiles))
    return data


# Just divides velocities by u_tau and SGS terms by u_tau^2
def scale_data(data):
    scaled = {}
    vels = ["u", "v", "w"]
    grads = ["dudx", "dudy", "dudz", "dvdx", "dvdy", "dvdz", "dwdx", "dwdy", "dwdz"]
    taus = ["uu", "vv", "ww", "uv", "uw", "vw"]
    u_tau = data["u_tau"].mean()
    for field in vels + grads:
        scaled[field] = data[field] / u_tau
    for field in taus:
        scaled[field] = data[field] / u_tau ** 2
    return scaled


# Computes derivative of given field spectrally in x direction
def oget_x_deriv(field):
    (ny, nz, nx, n) = field.shape
    fourier = np.fft.fft(field, axis=2)
    L = 4.0
    waveno = np.fft.fftfreq(nx) * nx / L
    deriv_mult = np.tile(1.0j * waveno, (ny, nz, 1))
    for fnum in range(n):
        fourier[:, :, :, fnum] = fourier[:, :, :, fnum] * deriv_mult
    return np.real(np.fft.ifft(fourier, axis=2))


# Computes derivative of given field spectrally in z direction
def oget_z_deriv(field):
    (ny, nz, nx, n) = field.shape
    fourier = np.fft.fft(field, axis=1)
    L = 1.5
    waveno = np.fft.fftfreq(nz) * nz / L
    deriv_mult = np.tile(1.0j * waveno, (ny, 1))
    for fnum in range(n):
        for i in range(nx):
            fourier[:, :, i, fnum] = fourier[:, :, i, fnum] * deriv_mult
    return np.real(np.fft.ifft(fourier, axis=1))


# Transform into y-plus units
def yplus_transform(y):
    Re_tau = 5200
    return np.where(y <= 0, (1 + y) * Re_tau, (1 - y) * Re_tau)
