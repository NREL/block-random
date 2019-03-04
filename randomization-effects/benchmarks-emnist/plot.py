# ===============================================================================
#
# Imports
#
# ===============================================================================
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import pickle


# ===============================================================================
#
# Global variables
#
# ===============================================================================
datasets = ["fashion", "digits", "letters", "byclass", "balanced", "mnist"]
batch_sizes_base = [32, 64, 128, 256, 512, 1024, 2048]
orders = ["shuffled", "block", "sorted"]
nc = {
    "fashion": 10,
    "digits": 10,
    "balanced": 47,
    "mnist": 10,
    "letters": 26,
    "byclass": 62,
    "bymerge": 47,
}
n_samples = {
    "fashion": 60000,
    "digits": 240000,
    "balanced": 112800,
    "mnist": 60000,
    "letters": 124800,
    "byclass": 697932,
    "bymerge": 697932,
}

batch_size_dataset = {
    "fashion": [32, 64, 128, 256, 512, 1024, 2048, 4096, 5120],
    "digits": [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 20480],
    "balanced": [32, 64, 128, 256, 512, 1024, 1536, 2048],
    # 'balanced': [32, 64, 128, 256, 512, 1024, 1536, 2048, 2400, 3200, 6400],
    "mnist": [32, 64, 128, 256, 512, 1024, 2048, 4096, 5120],
    "letters": [32, 64, 128, 256, 512, 1024, 2048, 4096],
    "byclass": batch_sizes_base + [4096],
    "bymerge": batch_sizes_base + [4096],
}

colors = {
    "fashion": "#EE2E2F",
    "digits": "#008C48",
    "letters": "#185AA9",
    "byclass": "#F47D23",
    "balanced": "#662C91",
    "mnist": "#A21D21",
    "bymerge": "#B43894",
}

markers = {
    "fashion": "s",
    "digits": "d",
    "letters": "o",
    "byclass": "p",
    "balanced": "h",
    "mnist": "x",
    "bymerge": "+",
}

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

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{nicefrac}")


# ===============================================================================
#
# Function definitions
#
# ===============================================================================
def plot_results_by_algorithm(results, dataset, order, save_path):
    plt.figure()
    batch_sizes = batch_size_dataset[dataset]
    # batch_sizes = batch_sizes_base
    for batch_size in batch_sizes:
        plt.plot(
            results[dataset][batch_size][order]["val_acc"],
            label="Batch size: %d" % batch_size,
        )
    ax = plt.gca()
    plt.xlabel("Epochs", fontsize=22, fontweight="bold")
    # plt.ylim(ymin=0.99, ymax=1.0)
    plt.ylabel("Validation Accuracy", fontsize=22, fontweight="bold")
    plt.legend(loc="best")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.17)
    plt.savefig("%s/%s-%s.pdf" % (save_path, dataset, order))


# ===============================================================================
def plot_results_by_ratio(results, save_path, legend=False):
    fname = "%s/summary.pdf" % save_path
    with PdfPages(fname) as pdf:
        plt.close("all")

        lst = []
        for dataset in datasets:
            samples_per_class = float(n_samples[dataset]) / nc[dataset]
            batch_sizes = batch_size_dataset[dataset]
            ratios = np.zeros(len(batch_sizes))
            val_shuffled = np.zeros_like(ratios)
            val_block = np.zeros_like(ratios)
            val_sorted = np.zeros_like(ratios)
            for i in range(len(batch_sizes)):
                ratios[i] = batch_sizes[i] / samples_per_class
                val_shuffled[i] = results[dataset][batch_sizes[i]]["shuffled"][
                    "val_acc"
                ][-1]
                val_sorted[i] = results[dataset][batch_sizes[i]]["sorted"]["val_acc"][
                    -1
                ]
                val_block[i] = results[dataset][batch_sizes[i]]["block"]["val_acc"][-1]

            df = pd.DataFrame(
                {
                    "batch": batch_sizes,
                    "ratio": ratios,
                    "shuffled": val_shuffled,
                    "sorted": val_sorted,
                    "block": val_block,
                }
            )
            df["dataset"] = dataset
            lst.append(df)

            fig = plt.figure(0)
            plt.semilogx(
                df.ratio,
                df.block,
                color=colors[dataset],
                marker=markers[dataset],
                label=dataset,
            )
            ax = plt.gca()
            ax.axhline(y=np.max(val_shuffled), color=colors[dataset], linestyle="--")

        df = pd.concat(lst)

        # Compare accuracies
        subdf = df[df.batch == 64]
        ind = np.arange(subdf.shape[0])
        width = 0.2
        fig1 = plt.figure(1)
        ax = plt.gca()
        ax.barh(ind, subdf.block, width, color=cmap[2])
        ax.barh(ind + width, subdf.sorted, width, color=cmap[1])
        ax.barh(ind + 2 * width, subdf.shuffled, width, color=cmap[0])
        ax.set(
            yticks=ind + width,
            yticklabels=subdf.dataset,
            ylim=[2 * width - 1, len(subdf)],
        )

        # Format and save
        fig = plt.figure(0)
        ax = plt.gca()
        if legend:
            plt.legend(loc="best")
        plt.ylabel(r"$\alpha$", fontsize=22, fontweight="bold")
        plt.xlabel(r"$\nicefrac{n_b}{n_c}$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig.subplots_adjust(bottom=0.15)
        fig.subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)

        fig1 = plt.figure(1)
        ax = plt.gca()
        plt.xlabel(r"$\alpha$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig1.subplots_adjust(bottom=0.15)
        fig1.subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)


# ===============================================================================
#
# Main
#
# ===============================================================================
if __name__ == "__main__":

    # Set data and plot paths
    data_path = "./batch_size_study/"
    save_path = "./batch_size_study/"

    results = {}
    for dataset in datasets:
        results[dataset] = {}
        batch_sizes = batch_size_dataset[dataset]
        for batch_size in batch_sizes:
            fname = "%s/%s-%d.pkl" % (data_path, dataset, batch_size)
            with open(fname, "rb") as fr:
                res = pickle.load(fr)
            results[dataset][batch_size] = res
        for order in orders:
            plot_results_by_algorithm(results, dataset, order, save_path)
    plot_results_by_ratio(results, save_path)
