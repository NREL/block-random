import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utilities as utilities


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
        help="Directory containing the runs to compare",
        type=str,
        required=True,
    )
    parser.add_argument("--col", help="Column axis", type=str, default="n_layers")
    parser.add_argument("--row", help="Row axis", type=str, default="learning_rate")
    parser.add_argument("--hue", help="Hue axis", type=str, default="n_nodes")
    parser.add_argument("--style", help="Style axis", type=str, default="n_nodes")
    args = parser.parse_args()

    # Get all subdirectories
    fdirs = next(os.walk(args.rundir))[1]

    lst = []
    for k, fdir in enumerate(fdirs):
        history = utilities.load_history(os.path.join(args.rundir, fdir, "history.pkl"))
        parameters = utilities.load_parameters(
            os.path.join(args.rundir, fdir, "run.log")
        )
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

    palette = sns.color_palette(n_colors=len(np.unique(df[args.hue])))

    plt.figure(0)
    ax = sns.relplot(
        x="epoch",
        y="train_mse",
        facet_kws={"sharey": "row"},
        col=args.col,
        row=args.row,
        hue=args.hue,
        style=args.style,
        palette=palette,
        kind="line",
        estimator=None,
        linewidth=2,
        data=df,
    )
    # ax.set(ylim=(0.12, 0.25))
    plt.savefig("train_histories.pdf")

    plt.figure(1)
    ax = sns.relplot(
        x="epoch",
        y="test_mse",
        facet_kws={"sharey": "row"},
        col=args.col,
        row=args.row,
        hue=args.hue,
        style=args.style,
        palette=palette,
        kind="line",
        estimator=None,
        linewidth=2,
        data=df,
    )
    # ax.set(ylim=(0.12, 0.25))
    plt.savefig("test_histories.pdf")

    if args.show:
        plt.show()
