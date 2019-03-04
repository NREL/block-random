import os
import argparse
import itertools
import numpy as np
import pandas as pd
import joblib
from sklearn.utils import shuffle
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
import functions as functions


# ===============================================================================
# Generate data sets
def gen_training(data, yname="tau_uv", ptype="shuffled", test_size=0.05, bwidth=16):

    if ptype not in ["block", "sorted", "shuffled"]:
        raise ValueError("Unkown ptype, please choose from block, sorted or shuffled")

    # Get the data
    x = data[: len(get_xnames()), :]
    y = data[len(get_xnames()) + get_ynames().index(yname), :]

    # Coordinates
    coords = np.asarray(
        np.meshgrid(
            functions.read_true_means()["Y"],
            np.linspace(0, 2 * np.pi * 1.5, 128, endpoint=False),
            np.linspace(0, 2 * np.pi * 4, 200, endpoint=False),
            indexing="ij",
        )
    )

    # Shapes and sizes
    blen = bwidth ** 3
    x = np.squeeze(x)
    y = np.squeeze(y)
    N = x.shape
    ranges = [range(0, N[1], bwidth), range(0, N[2], bwidth), range(0, N[3], bwidth)]

    # Get data blocks
    # TODO: redo this to avoid appending to lists.
    nblocks = np.prod([len(x) for x in ranges])
    lst_xblocks = []
    lst_yblocks = []
    lst_cblocks = []
    for cnt, (i, j, k) in enumerate(itertools.product(ranges[0], ranges[1], ranges[2])):

        xblk = np.s_[:, i : i + bwidth, j : j + bwidth, k : k + bwidth]
        yblk = np.s_[i : i + bwidth, j : j + bwidth, k : k + bwidth]

        lst_xblocks.append(x[xblk].reshape(N[0], -1))
        lst_yblocks.append(y[yblk].flatten())
        lst_cblocks.append(coords[xblk].reshape((3, -1)))

    ntrain = int((1 - test_size) * nblocks)

    # Pick random blocks for test cases, but remove them
    # from the training set
    idx = np.arange(nblocks)
    idx_test = np.random.permutation(nblocks)[ntrain:]
    idx_train = np.delete(idx, idx_test)

    if ptype == "block":
        # randomize the training blocks
        np.random.shuffle(idx_train)

    # Construct the dataframes
    x_train = pd.DataFrame(
        np.concatenate([lst_xblocks[idx].T for idx in idx_train], axis=0),
        columns=get_xnames(),
    )
    x_test = pd.DataFrame(
        np.concatenate([lst_xblocks[idx].T for idx in idx_test], axis=0),
        columns=get_xnames(),
    )
    y_train = pd.DataFrame(
        np.concatenate([lst_yblocks[idx] for idx in idx_train], axis=0), columns=[yname]
    )
    y_test = pd.DataFrame(
        np.concatenate([lst_yblocks[idx] for idx in idx_test], axis=0), columns=[yname]
    )
    c_train = pd.DataFrame(
        np.concatenate([lst_cblocks[idx].T for idx in idx_train], axis=0),
        columns=get_cnames(),
    )
    c_test = pd.DataFrame(
        np.concatenate([lst_cblocks[idx].T for idx in idx_test], axis=0),
        columns=get_cnames(),
    )

    # Index using block id
    x_train.index = np.repeat(idx_train, blen)[: x_train.shape[0]]
    x_test.index = np.repeat(idx_test, blen)[: x_test.shape[0]]
    y_train.index = x_train.index
    y_test.index = x_test.index
    c_train.index = x_train.index
    c_test.index = x_test.index

    if ptype == "shuffled":
        # Fully shuffle the training dataframes
        x_train, y_train, c_train = shuffle(x_train, y_train, c_train, random_state=0)

    return x_train, x_test, y_train, y_test, c_train, c_test


# ========================================================================
def get_xnames():
    return [
        "u",
        "v",
        "w",
        "dudx",
        "dudy",
        "dudz",
        "dvdx",
        "dvdy",
        "dvdz",
        "dwdx",
        "dwdy",
        "dwdz",
    ]


# ========================================================================
def get_ynames():
    return ["tau_uu", "tau_vv", "tau_ww", "tau_uv", "tau_uw", "tau_vw"]


# ========================================================================
def get_cnames():
    return ["y", "z", "x"]


# ===============================================================================
# Resample based on classes
def resample(classes):
    n_classes = len(np.unique(classes))
    nsamples_per_class = np.int(len(classes) / n_classes)
    samples = np.zeros(nsamples_per_class * n_classes)
    for k, c in enumerate(np.unique(classes)):
        idx = np.where(classes == c)[0].tolist()
        samples[
            k * nsamples_per_class : (k + 1) * nsamples_per_class
        ] = np.random.choice(idx, nsamples_per_class, replace=True)

    return samples


# ===============================================================================
# Rebalance data
def rebalance(x, y, rtype="kmeans"):

    # Resample training set through KMeans clustering
    if rtype == "kmeans":
        n_clusters = 25
        km = KMeans(n_clusters=n_clusters, random_state=0).fit(y)
        classes = km.predict(y)
        samples = resample(classes)

    elif rtype == "uniform":
        hist, bin_edges = np.histogram(y.tau_uv, bins=200)
        classes = np.digitize(y.tau_uv, bin_edges)
        samples = resample(classes)

    else:
        print(
            "Unkown rebalance type, please choose from kmeans, or uniform. Skipping rebalance"
        )

    return samples


# ===============================================================================
# Generator
def generator(x, y, batch_size, shuffle, rtype=None):

    if shuffle:
        np.random.seed(28)
        idx = np.random.permutation(len(x))
        x = x.iloc[idx]
        y = y.iloc[idx]

    while True:
        for i in range(0, len(x), batch_size):
            x_batch = x.iloc[i : i + batch_size]
            y_batch = y.iloc[i : i + batch_size]

            # Reshuffle if we reach the end of the list
            if (i + batch_size > len(x)) and shuffle:
                idx = np.random.permutation(len(x))
                x = x.iloc[idx]
                y = y.iloc[idx]

            # Rebalance the batch if desired
            if rtype is not None:
                samples = rebalance(x_batch, y_batch, rtype=rtype)
                x_batch = x_batch.iloc[samples]
                y_batch = y_batch.iloc[samples]

            yield (x_batch, y_batch)


# ===============================================================================
#
# Main
#
# ===============================================================================
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate data")
    parser.add_argument(
        "-o",
        "--odir",
        help="Output directory (in data directory)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-p",
        "--permutation",
        help="Permutation type on training data",
        type=str,
        default="shuffled",
    )
    parser.add_argument(
        "--rebalance",
        dest="rebalance",
        help="Rebalance entire training data set [kmeans | uniform]",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-n", "--nsamples", help="Number of samples to keep", type=int, default=None
    )
    parser.add_argument("-b", "--bwidth", help="Block width", type=int, default=16)
    args = parser.parse_args()

    # Deterministic results in this randomization study!
    np.random.seed(28)

    # Directories to save
    datadir = os.path.join("data", args.odir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    # Read data from the saved npy file
    data = np.load("../../data/scaled.npy")
    # data = np.load("/lwork/prakash/scaled.npy")

    # Generate the training
    yname = "tau_uv"
    x_train, x_test, y_train, y_test, c_train, c_test = gen_training(
        data, yname=yname, ptype=args.permutation, bwidth=args.bwidth
    )

    # Rebalance the training data (optional)
    if args.rebalance is not None:
        samples = rebalance(x_train, y_train, rtype=args.rebalance)
        x_train = x_train.iloc[samples]
        y_train = y_train.iloc[samples]
        c_train = c_train.iloc[samples]

    # Subset the data if you want
    if args.nsamples is not None:
        x_train = x_train.iloc[: args.nsamples, :]
        y_train = y_train.iloc[: args.nsamples, :]
        c_train = c_train.iloc[: args.nsamples, :]
        x_test = x_test.iloc[: args.nsamples, :]
        y_test = y_test.iloc[: args.nsamples, :]
        c_test = c_test.iloc[: args.nsamples, :]

    # Scale the data using the RobustScaler
    scaler = RobustScaler()
    scaler.fit(x_train)
    scaler.transform(x_train)
    scaler.transform(x_test)

    # Save the data, permutation type, and the scaler
    joblib.dump(scaler, os.path.join(datadir, "scaler.pkl"))
    x_train.to_pickle(os.path.join(datadir, "xtrain.gz"))
    x_test.to_pickle(os.path.join(datadir, "xtest.gz"))
    y_train.to_pickle(os.path.join(datadir, "ytrain.gz"))
    y_test.to_pickle(os.path.join(datadir, "ytest.gz"))
    c_train.to_pickle(os.path.join(datadir, "ctrain.gz"))
    c_test.to_pickle(os.path.join(datadir, "ctest.gz"))
    with open(os.path.join(datadir, "permutation.txt"), "w") as f:
        f.write(args.permutation)
