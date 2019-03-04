# ===============================================================================
#
# Imports
#
# ===============================================================================
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.stats.kde import gaussian_kde
import functions as functions
import gen_data as gen_data
import utilities as utilities
import physics_models as physics_models

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

col_latex = {
    "u": "u",
    "v": "v",
    "w": "w",
    "dudx": r"\nicefrac{\partial u}{\partial x}",
    "dudy": r"\nicefrac{\partial u}{\partial y}",
    "dudz": r"\nicefrac{\partial v}{\partial z}",
    "dvdx": r"\nicefrac{\partial v}{\partial x}",
    "dvdy": r"\nicefrac{\partial v}{\partial y}",
    "dvdz": r"\nicefrac{\partial v}{\partial z}",
    "dwdx": r"\nicefrac{\partial w}{\partial x}",
    "dwdy": r"\nicefrac{\partial w}{\partial y}",
    "dwdz": r"\nicefrac{\partial w}{\partial z}",
}


# ===============================================================================
#
# Function definitions
#
# ===============================================================================
def plot_predictions(fdirs, do_physics=True):
    """Plot predictions"""

    fname = "predictions.pdf"
    pdf_space = np.linspace(-2, 2, 100)
    nbins = 32

    with PdfPages(fname) as pdf:
        plt.close("all")
        plt.rc("text", usetex=True)
        plt.rc("text.latex", preamble=r"\usepackage{nicefrac}")

        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)

        fig_ypc = plt.figure("ypc")
        ax_ypc = fig_ypc.add_subplot(111)

        fig_yc = plt.figure("yc")
        ax_yc = fig_yc.add_subplot(111)

        for k, fdir in enumerate(fdirs):

            # Load data
            scaler = joblib.load(os.path.join(fdirs[fdir], "scaler.pkl"))
            x = pd.read_pickle(os.path.join(fdirs[fdir], "xtest.gz"))
            x_unscaled = pd.DataFrame(
                scaler.inverse_transform(x), columns=x.columns, index=x.index
            )
            y = pd.read_pickle(os.path.join(fdirs[fdir], "ytest.gz"))
            c = pd.read_pickle(os.path.join(fdirs[fdir], "ctest.gz"))
            m = pd.read_pickle(os.path.join(fdirs[fdir], "mtest.gz"))
            ystd = y.tau_uv.std()
            m_error = (y.tau_uv - m.tau_uv) / ystd

            # Physics
            if do_physics:
                w = physics_models.wale(x_unscaled)
                w_error = (y.tau_uv - w.tau_uv) / ystd
                kde = gaussian_kde(w_error)
                w_pkde = kde(pdf_space)

            # PDF of the error
            kde = gaussian_kde(m_error)
            m_pkde = kde(pdf_space)

            # Scatter plot
            fig0 = plt.figure(0)
            fig0.clf()
            ax0 = fig0.add_subplot(111)
            # ax0.hexbin(y.tau_uv, m.tau_uv, gridsize=50)
            ax0.scatter(
                y.tau_uv,
                m.tau_uv,
                c=cmap[k],
                alpha=0.05,
                s=15,
                marker=markertype[k],
                rasterized=True,
            )
            ax0.plot(
                [y.tau_uv.min(), y.tau_uv.max()],
                [y.tau_uv.min(), y.tau_uv.max()],
                lw=1,
                color=cmap[-1],
            )
            ax0.set_xlabel(r"$\tau_{12}$", fontsize=22, fontweight="bold")
            ax0.set_ylabel(r"$\tau_{12}^m$", fontsize=22, fontweight="bold")
            plt.setp(ax0.get_xmajorticklabels(), fontsize=18, fontweight="bold")
            plt.setp(ax0.get_ymajorticklabels(), fontsize=18, fontweight="bold")
            fig0.subplots_adjust(bottom=0.15)
            fig0.subplots_adjust(left=0.17)
            pdf.savefig(dpi=300)

            # Plot PDF of the error
            plt.figure(1)
            p = ax1.plot(pdf_space, m_pkde, lw=2, color=cmap[k])
            p[0].set_dashes(dashseq[k])
            if do_physics:
                p = ax1.plot(pdf_space, w_pkde, lw=2, color=cmap[-2], label="WALE")

            # Plot conditionals
            for idx, col in enumerate(x_unscaled):
                means, bin_edges, binnumber = stats.binned_statistic(
                    x_unscaled[col], y.tau_uv, statistic="mean", bins=nbins
                )
                means_model, _, _ = stats.binned_statistic(
                    x_unscaled[col], m.tau_uv, statistic="mean", bins=bin_edges
                )
                bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])

                plt.figure(col)
                p = plt.plot(
                    bins, means_model, lw=1, c=cmap[k], ms=5, marker=markertype[k]
                )
                p[0].set_dashes(dashseq[k])
                plt.plot(bins, means, lw=1, color=cmap[-1])
                if do_physics:
                    means_wale, _, _ = stats.binned_statistic(
                        x_unscaled[col], w.tau_uv, statistic="mean", bins=bin_edges
                    )
                    plt.plot(
                        bins,
                        means_wale,
                        lw=1,
                        c=cmap[-2],
                        ms=5,
                        marker=markertype[-1],
                        label="WALE",
                    )

            # Plot y+ coordinate conditional
            yplus = functions.yplus_transform(c.y.values)
            bmin, bmax = 1e-1, 5200
            means, bin_edges, binnumber = stats.binned_statistic(
                yplus,
                np.where(c.y <= 0, y.tau_uv, -y.tau_uv),
                statistic="mean",
                bins=np.logspace(np.log10(bmin), np.log10(bmax), nbins),
            )
            means_model, _, _ = stats.binned_statistic(
                yplus,
                np.where(c.y <= 0, m.tau_uv, -m.tau_uv),
                statistic="mean",
                bins=np.logspace(np.log10(bmin), np.log10(bmax), nbins),
            )
            bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            p = ax_ypc.plot(
                bins,
                means_model,
                lw=1,
                c=cmap[k],
                ms=5,
                marker=markertype[k],
                label="DNN",
            )
            p[0].set_dashes(dashseq[k])
            ax_ypc.plot(bins, means, lw=1, color=cmap[-1], label="DNS")
            if do_physics:
                means_wale, _, _ = stats.binned_statistic(
                    yplus,
                    np.where(c.y <= 0, w.tau_uv, -w.tau_uv),
                    statistic="mean",
                    bins=np.logspace(np.log10(bmin), np.log10(bmax), nbins),
                )
                ax_ypc.plot(
                    bins,
                    means_wale,
                    lw=1,
                    c=cmap[-2],
                    ms=5,
                    marker=markertype[-1],
                    label="WALE",
                )

            # Plot y coordinate conditional
            means, bin_edges, binnumber = stats.binned_statistic(
                1 - np.fabs(c.y),
                np.where(c.y <= 0, y.tau_uv, -y.tau_uv),
                statistic="mean",
                bins=nbins,
            )
            means_model, _, _ = stats.binned_statistic(
                1 - np.fabs(c.y),
                np.where(c.y <= 0, m.tau_uv, -m.tau_uv),
                statistic="mean",
                bins=bin_edges,
            )
            bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            p = ax_yc.plot(
                bins,
                means_model,
                lw=1,
                c=cmap[k],
                ms=5,
                marker=markertype[k],
                label="DNN",
            )
            p[0].set_dashes(dashseq[k])
            ax_yc.plot(bins, means, lw=1, color=cmap[-1], label="DNS")
            if do_physics:
                means_wale, _, _ = stats.binned_statistic(
                    1 - np.fabs(c.y),
                    np.where(c.y <= 0, w.tau_uv, -w.tau_uv),
                    statistic="mean",
                    bins=bin_edges,
                )
                ax_yc.plot(
                    bins,
                    means_wale,
                    lw=1,
                    c=cmap[-2],
                    ms=5,
                    marker=markertype[-1],
                    label="WALE",
                )

        plt.figure(1)
        ax1.set_xlabel(
            r"$\nicefrac{(\tau_{12} - \tau_{12}^m)}{\sigma_{\tau_{12}}}$",
            fontsize=22,
            fontweight="bold",
        )
        ax1.set_ylabel(
            r"$P(\nicefrac{(\tau_{12} - \tau_{12}^m)}{\sigma_{\tau_{12}}})$",
            fontsize=22,
            fontweight="bold",
        )
        plt.setp(ax1.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax1.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        ax1.set_xticks([-2, -1, 0, 1, 2])
        ax1.set_yscale("log")
        fig1.subplots_adjust(bottom=0.15)
        fig1.subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)

        for idx, col in enumerate(x):
            plt.figure(col)
            ax = plt.gca()
            ax.set_xlabel(
                r"${0:s}$".format(col_latex[col]), fontsize=22, fontweight="bold"
            )
            ax.set_ylabel(
                r"$\langle \tau_{12} | " + r"{:s}".format(col_latex[col]) + r"\rangle$",
                fontsize=22,
                fontweight="bold",
            )
            plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
            plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
            plt.gcf().subplots_adjust(bottom=0.15)
            plt.gcf().subplots_adjust(left=0.17)
            pdf.savefig(dpi=300)

        plt.figure("ypc")
        ax_ypc.set_xlabel(r"$y^+$", fontsize=22, fontweight="bold")
        ax_ypc.set_ylabel(
            r"$\langle \tau_{12} | y^+ \rangle$", fontsize=22, fontweight="bold"
        )
        # ax_ypc.set_ylim(-0.5, 0.5)
        ax_ypc.set_xscale("log")
        plt.setp(ax_ypc.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax_ypc.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gcf().subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)

        plt.figure("yc")
        ax_yc.set_xlabel(r"$y$", fontsize=22, fontweight="bold")
        ax_yc.set_ylabel(
            r"$\langle \tau_{12} | y \rangle$", fontsize=22, fontweight="bold"
        )
        plt.setp(ax_yc.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax_yc.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gcf().subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)


# ===============================================================================
def plot_accuracies(fdirs, legend=False):
    """Plot accuracy vs batch size"""

    fname = "accuracies.pdf"
    lst = []
    for k, fdir in enumerate(fdirs):

        # Load data
        history = utilities.load_history(os.path.join(fdirs[fdir], "history.pkl"))
        parameters = utilities.load_parameters(os.path.join(fdirs[fdir], "run.log"))

        df = pd.DataFrame(
            np.array(
                [history["mean_squared_error"], history["val_mean_squared_error"]]
            ).T,
            columns=["train_mse", "test_mse"],
        )
        df["epoch"] = df.index
        df["runid"] = k
        df = df.join(pd.DataFrame(parameters, index=[k]), on=["runid"])
        lst.append(df)

    df = pd.concat(lst)
    df.sort_values(by=["batch_size", "runid"], inplace=True)

    # Plot accuracy
    with PdfPages(fname) as pdf:
        plt.close("all")
        plt.rc("text", usetex=True)
        plt.rc("text.latex", preamble=r"\usepackage{nicefrac}")

        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(111)

        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)

        subdf = df[df.epoch == df.epochs - 1]
        for k, name in enumerate(["shuffled", "sorted", "block"]):
            group = subdf[subdf.permutation == name]

            plt.figure(0)
            ax0.plot(
                group.batch_size,
                group.train_mse,
                lw=1,
                c=cmap[k],
                marker=markertype[k],
                ms=5,
                label=group.permutation.iloc[0],
            )

            plt.figure(1)
            ax1.plot(
                group.batch_size,
                group.test_mse,
                lw=1,
                c=cmap[k],
                marker=markertype[k],
                ms=5,
                label=group.permutation.iloc[0],
            )

        # Format plots
        plt.figure(0)
        if legend:
            lgd = ax0.legend()
        ax0.set_xlabel(r"$n_b$", fontsize=22, fontweight="bold")
        ax0.set_ylabel(r"$\epsilon_t$", fontsize=22, fontweight="bold")
        ax0.set_xscale("log")
        plt.setp(ax0.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax0.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig0.subplots_adjust(bottom=0.14)
        fig0.subplots_adjust(left=0.16)
        pdf.savefig(dpi=300)

        plt.figure(1)
        if legend:
            lgd = ax1.legend()
        ax1.set_xlabel(r"$n_b$", fontsize=22, fontweight="bold")
        ax1.set_ylabel(r"$\epsilon_v$", fontsize=22, fontweight="bold")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        plt.setp(ax1.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax1.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig1.subplots_adjust(bottom=0.14)
        fig1.subplots_adjust(left=0.16)
        pdf.savefig(dpi=300)


# ========================================================================
def jensen_shannon_divergence(p, q):
    """
    This will be part of scipy as some point.
    See https://github.com/scipy/scipy/pull/8295
    We use this implementation for now: https://stackoverflow.com/questions/15880133/jensen-shannon-divergence

    :param p: PDF (normalized to 1)
    :type p: array
    :param q: PDF (normalized to 1)
    :type q: array
    """
    eps = 1e-13
    M = np.clip(0.5 * (p + q), eps, None)
    return 0.5 * (stats.entropy(p, M) + stats.entropy(q, M))


# ===============================================================================
def compute_batch_jsd(fdir):
    """Compute batch JSD"""

    # Load data
    parameters = utilities.load_parameters(os.path.join(fdir, "run.log"))
    x_train = pd.read_pickle(os.path.join(parameters["datadir"], "xtrain.gz"))
    y_train = pd.read_pickle(os.path.join(parameters["datadir"], "ytrain.gz"))

    # Setup
    ymin, ymax = y_train.tau_uv.min(), y_train.tau_uv.max()
    abs_max = np.fabs([ymin, ymax]).max()
    nbins = 64
    bins = np.linspace(-abs_max, abs_max, nbins + 1)
    nbatch = int(np.ceil(y_train.shape[0] / parameters["batch_size"]))

    # Get PDF of all data
    pdf, _ = np.histogram(y_train.tau_uv, bins=bins, density=True)

    # Compute JSD difference
    jsds = np.zeros((nbatch, 2))
    previous_batch_pdf = pdf.copy()
    for k, (_, batch_y) in enumerate(
        gen_data.generator(
            x_train,
            y_train,
            parameters["batch_size"],
            parameters["permutation"] == "shuffle",
            rtype=parameters["rebalance"],
        )
    ):

        # Exit if we exceed the number of batches in an epoch
        if k >= nbatch:
            break

        # Compute JSD between overall PDF and previous batch PDF
        batch_pdf, _ = np.histogram(batch_y.tau_uv, bins=bins, density=True)
        jsds[k, 0] = jensen_shannon_divergence(pdf, batch_pdf)
        jsds[k, 1] = jensen_shannon_divergence(previous_batch_pdf, batch_pdf)
        previous_batch_pdf = batch_pdf.copy()

    # Save JSDs
    df = pd.DataFrame(jsds, columns=["base", "difference"])
    df.to_pickle(os.path.join(fdir, "jsds.gz"))
    return df


# ===============================================================================
def plot_batch_jsds(fdirs, legend=False):
    """Plot batch JSD"""

    fname = "jsds.pdf"
    pdf_space = np.linspace(0, np.log(2), 200)
    with PdfPages(fname) as pdf:
        plt.close("all")
        plt.rc("text", usetex=True)
        plt.rc("text.latex", preamble=r"\usepackage{nicefrac}")

        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(111)

        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)

        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)

        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot(111)

        fig4 = plt.figure(4)
        ax4 = fig4.add_subplot(111)

        fig5 = plt.figure(5)
        ax5 = fig5.add_subplot(111)

        for k, fdir in enumerate(fdirs):

            # Load data
            df = pd.read_pickle(os.path.join(fdirs[fdir], "jsds.gz"))
            kde_base = gaussian_kde(df.base)
            kde_difference = gaussian_kde(df.difference)

            nslice = df.shape[0] // 50
            means_base = df.base.rolling(window=nslice, center=True).mean().values
            means_difference = (
                df.difference.rolling(window=nslice, center=True).mean().values
            )

            plt.figure(0)
            ax0.plot(
                df.base,
                ls="None",
                c=cmap[k],
                marker=markertype[k],
                ms=3,
                alpha=0.05,
                rasterized=True,
            )
            ax0.plot(means_base, c=cmap[-1], zorder=10, lw=3, rasterized=True)
            p = ax0.plot(means_base, c=cmap[k], zorder=10, lw=1, rasterized=True)
            p[0].set_dashes(dashseq[k])

            plt.figure(1)
            p = ax1.plot(means_base, c=cmap[k], lw=2, rasterized=True)
            p[0].set_dashes(dashseq[k])

            plt.figure(2)
            p = ax2.plot(pdf_space, kde_base(pdf_space), lw=2, c=cmap[k])
            p[0].set_dashes(dashseq[k])

            plt.figure(3)
            ax3.plot(
                df.difference,
                ls="None",
                c=cmap[k],
                marker=markertype[k],
                ms=3,
                alpha=0.05,
                rasterized=True,
            )
            ax3.plot(means_difference, c=cmap[-1], zorder=10, lw=3, rasterized=True)
            p = ax3.plot(means_difference, c=cmap[k], zorder=10, lw=1, rasterized=True)
            p[0].set_dashes(dashseq[k])

            plt.figure(4)
            p = ax4.plot(means_difference, c=cmap[k], lw=2, rasterized=True)
            p[0].set_dashes(dashseq[k])

            plt.figure(5)
            p = ax5.plot(pdf_space, kde_difference(pdf_space), lw=2, c=cmap[k])
            p[0].set_dashes(dashseq[k])

        # Format plots
        plt.figure(0)
        if legend:
            lgd = ax0.legend()
        ax0.set_xlabel(r"$b$", fontsize=22, fontweight="bold")
        ax0.set_ylabel(r"$J(Y, Y_b)$", fontsize=22, fontweight="bold")
        plt.setp(ax0.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax0.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        ax0.set_xticks([0, 40000, 80000, 120000])
        fig0.subplots_adjust(bottom=0.14)
        fig0.subplots_adjust(left=0.16)
        pdf.savefig(dpi=300)

        plt.figure(1)
        if legend:
            lgd = ax1.legend()
        ax1.set_xlabel(r"$b$", fontsize=22, fontweight="bold")
        ax1.set_ylabel(r"$J(Y, Y_b)$", fontsize=22, fontweight="bold")
        plt.setp(ax1.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax1.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        ax1.set_xticks([0, 40000, 80000, 120000])
        ax1.set_ylim(0, np.log(2))
        fig1.subplots_adjust(bottom=0.14)
        fig1.subplots_adjust(left=0.16)
        pdf.savefig(dpi=300)

        plt.figure(2)
        if legend:
            lgd = ax2.legend()
        ax2.set_xlabel(r"$J(Y, Y_b)$", fontsize=22, fontweight="bold")
        ax2.set_ylabel(r"$P(J(Y, Y_b))$", fontsize=22, fontweight="bold")
        plt.setp(ax2.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax2.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig2.subplots_adjust(bottom=0.14)
        fig2.subplots_adjust(left=0.16)
        pdf.savefig(dpi=300)

        plt.figure(3)
        if legend:
            lgd = ax3.legend()
        ax3.set_xlabel(r"$b$", fontsize=22, fontweight="bold")
        ax3.set_ylabel(r"$J(Y_{b-1}, Y_b)$", fontsize=22, fontweight="bold")
        plt.setp(ax3.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax3.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        ax3.set_xticks([0, 40000, 80000, 120000])
        fig3.subplots_adjust(bottom=0.14)
        fig3.subplots_adjust(left=0.16)
        pdf.savefig(dpi=300)

        plt.figure(4)
        if legend:
            lgd = ax4.legend()
        ax4.set_xlabel(r"$b$", fontsize=22, fontweight="bold")
        ax4.set_ylabel(r"$J(Y_{b-1}, Y_b)$", fontsize=22, fontweight="bold")
        plt.setp(ax4.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax4.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        ax4.set_xticks([0, 40000, 80000, 120000])
        ax4.set_ylim(0, 0.15)
        fig4.subplots_adjust(bottom=0.14)
        fig4.subplots_adjust(left=0.16)
        pdf.savefig(dpi=300)

        plt.figure(5)
        if legend:
            lgd = ax5.legend()
        ax5.set_xlabel(r"$J(Y_{b-1}, Y_b)$", fontsize=22, fontweight="bold")
        ax5.set_ylabel(r"$P(J(Y_{b-1}, Y_b))$", fontsize=22, fontweight="bold")
        plt.setp(ax5.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax5.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig5.subplots_adjust(bottom=0.14)
        fig5.subplots_adjust(left=0.16)
        pdf.savefig(dpi=300)
