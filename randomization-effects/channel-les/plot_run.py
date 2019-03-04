import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===============================================================================
#
# Some defaults variables
#
# ===============================================================================
cmap = [
    "#EE2E2F",
    "#008C48",
    "#185AA9",
    "#F47D23",
    "#662C91",
    "#A21D21",
    "#B43894",
    "#010202",
]
dashseq = [
    (None, None),
    [10, 5],
    [10, 4, 3, 4],
    [3, 3],
    [10, 4, 3, 4, 3, 4],
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3],
]
markertype = ["s", "d", "o", "p", "h"]


# ===============================================================================
#
# Function definitions
#
# ===============================================================================
def plot_prediction(x, y, m, name="scatter"):
    plt.figure()
    plt.scatter(y.tau_uv, m.tau_uv, alpha=0.05, s=15, marker=markertype[0])
    p = plt.plot(
        [y.tau_uv.min(), y.tau_uv.max()],
        [y.tau_uv.min(), y.tau_uv.max()],
        color=cmap[-1],
        lw=2,
    )
    p[0].set_dashes(dashseq[0])
    plt.xlabel(r"$\tau_{12}$ data")
    plt.ylabel(r"$\tau_{12}$ prediction")
    plt.savefig(name + ".png", format="png", dpi=300)
    plt.close()


# ===============================================================================
def plot_histogram(x, y, m, name="hist"):

    plt.figure()
    vals = (y.tau_uv - m.tau_uv) / y.tau_uv.std()
    plt.hist(vals, bins=200, range=(-2, 2), density=True)
    plt.xlabel(r"$\frac{\mathrm{data} - \mathrm{prediction}}{\sigma_{\mathrm{data}}}$")
    plt.ylabel("probability")
    plt.savefig(name + ".pdf")

    plt.figure()
    plt.hist(y.tau_uv, bins=200, range=(-2, 2), density=True)
    plt.xlabel(r"$\tau_{12}$")
    plt.ylabel("probability")
    plt.savefig(name + "_tau.pdf")


# ===============================================================================
def plot_conditionals(x, y, m, name="conditional"):

    f, ax = plt.subplots(4, 3, figsize=(12, 12))
    nbins = 32
    for idx, col in enumerate(x):
        means, bin_edges, binnumber = stats.binned_statistic(
            x[col], y.tau_uv, statistic="mean", bins=nbins
        )
        bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        means_model, bin_edges, binnumber = stats.binned_statistic(
            x[col], m.tau_uv, statistic="mean", bins=nbins
        )
        idx2d = np.unravel_index(idx, (4, 3))

        ax[idx2d].plot(bins, means, "ob")
        ax[idx2d].plot(bins, means_model, "sr")
        ax[idx2d].set_xlabel(col)
        ax[idx2d].set_ylabel("tau_uv")

    plt.savefig(name + ".pdf")


# ===============================================================================
#
# Main
#
# ===============================================================================
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Simple plotting script")
    parser.add_argument(
        "-s" "--show", dest="show", help="Show plots", action="store_true"
    )
    parser.add_argument(
        "-r",
        "--rundir",
        help="Directory containing model and predictions",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    # Load data
    x_test = pd.read_pickle(os.path.join(args.rundir, "xtest.gz"))
    y_test = pd.read_pickle(os.path.join(args.rundir, "ytest.gz"))
    m_test = pd.read_pickle(os.path.join(args.rundir, "mtest.gz"))

    # Plot data
    plot_prediction(x_test, y_test, m_test, name=os.path.join(args.rundir, "scatter"))
    plot_histogram(x_test, y_test, m_test, name=os.path.join(args.rundir, "hist"))
    plot_conditionals(x_test, y_test, m_test, name=os.path.join(args.rundir, "cond"))

    if args.show:
        plt.show()
