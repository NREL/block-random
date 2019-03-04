"""
Use filtered channel data to test performance of sgs predictionand study SGD convergence vs randomization
"""

# ===============================================================================
#
# Imports
#
# ===============================================================================
import os
import time
import datetime
from datetime import timedelta
from shutil import copyfile
import argparse
import logging
import tensorflow as tf
from tensorflow.python.client import timeline
from keras import backend as K
import keras
from keras.models import Sequential, load_model
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    LeakyReLU,
)
from keras.optimizers import RMSprop, adam
from keras.callbacks import TensorBoard
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
import joblib
import gen_data as gen_data


# ===============================================================================
#
# Function definitions
#
# ===============================================================================
# Function to train a feed forward neural network with tanh activation function
def train_model_ff(
    x_train,
    y_train,
    x_test,
    y_test,
    n_layers,
    n_nodes,
    logdir,
    logger,
    shuffle=False,
    epochs=50,
    learning_rate=1e-4,
    batch_size=128,
    trace=False,
    restart=None,
    rebalance=None,
):
    if restart is not None:
        model = load_model(os.path.join(logdir, "model.h5"))
    else:
        model = Sequential()
        model.add(Dense(n_nodes, input_shape=(x_train.shape[1],)))
        model.add(LeakyReLU())
        for i in range(n_layers):
            model.add(Dense(n_nodes))
            model.add(Activation("tanh"))
        model.add(Dense(1))

        run_options = None
        run_metadata = None
        if trace:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

        model.compile(
            loss="mse",
            optimizer=adam(lr=learning_rate),
            metrics=["mse"],
            options=run_options,
            run_metadata=run_metadata,
        )
    model.summary(print_fn=logger.info)

    tensorboard = TensorBoard(log_dir=logdir)

    # Model fitting
    history = model.fit_generator(
        gen_data.generator(x_train, y_train, batch_size, shuffle, rtype=rebalance),
        epochs=epochs,
        steps_per_epoch=np.ceil(x_train.shape[0] / batch_size),
        verbose=1,
        validation_data=(x_test, y_test),
        shuffle=shuffle,
        callbacks=[tensorboard],
    )
    score = model.evaluate(x_test, y_test, verbose=0)
    logger.info("Test loss: {0:.16f}".format(score[0]))
    logger.info("Test accuracy {0:.16f}".format(score[1]))

    # Save trace
    if trace:
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        tname = os.path.join(logdir, "timeline.json")
        with open(tname, "w") as f:
            f.write(ctf)

    # Append histories if this is a restart
    if restart is not None:
        fname = os.path.join(logdir, "history.pkl")
        with open(fname, "rb") as f:
            old = pickle.load(f)

        for key in old:
            history.history[key] = old[key] + history.history[key]

    return (model, history)


# ===============================================================================
# Random forest for regression
def do_rf_fit(x_train, y_train, x_test, y_test):
    rf = RandomForestRegressor(
        n_estimators=100, random_state=42, verbose=True, oob_score=True, max_depth=30
    )
    rf.fit(x_train, y_train)
    joblib.dump(rf, "/scratch/pmohan/rf_uv.pkl")
    rf.score(x_test, y_test)
    preds = np.empty((n_test, 2))
    preds[:, 0] = y_test
    preds[:, 1] = rf.predict(x_test)
    print("MSE: ", ((preds[:, 1] - y_test) ** 2).mean())
    np.save("output/pred_uv_rf", preds)


# ===============================================================================
def read_restart(args):
    fname = os.path.join(args.restart, "run.log")
    with open(fname, "r") as f:
        for line in f:
            if "n_layers" in line:
                args.layers = int(line.split()[-1])
            elif "n_nodes" in line:
                args.nodes = int(line.split()[-1])
            elif "datadir" in line:
                datadir = line.split()[-1]
            elif "logdir" in line:
                logdir = line.split()[-1]

    # Backup old log file
    bname = os.path.join(
        args.restart,
        "bkp_{}.log".format(datetime.datetime.now().strftime("%b%d_%H-%M-%S")),
    )
    copyfile(fname, bname)

    return args, datadir, logdir


# ========================================================================
def init_logger(fname):
    logger = logging.getLogger()

    # Clear any handlers that the logger may have
    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        filename=fname,
        filemode="w",
    )

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # set a format which is simpler for console use
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")

    # tell the handler to use this format
    console.setFormatter(formatter)

    # add the handler to the root logger
    logger.addHandler(console)

    return logger


# ===============================================================================
#
# Main
#
# ===============================================================================
if __name__ == "__main__":

    # Timer
    start = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run channel SGS learning")
    parser.add_argument(
        "--ldir",
        help="Load data from this directory (relative to data directory)",
        type=str,
        default="shuffled-small",
    )
    parser.add_argument(
        "-o", "--odir", help="Directory to save the output", type=str, default="runs"
    )
    parser.add_argument(
        "--restart", help="Restart using directory", type=str, default=None
    )
    parser.add_argument(
        "--use_gpu",
        dest="use_gpu",
        help="Use a GPU to perform computations",
        action="store_true",
    )
    parser.add_argument(
        "--trace",
        dest="trace",
        help="Profile the run (requires CUPTI)",
        action="store_true",
    )
    parser.add_argument(
        "--precision", help="Precision of computation", type=str, default="float32"
    )
    parser.add_argument("-e", "--epochs", help="Number of epochs", type=int, default=50)
    parser.add_argument(
        "--learning_rate", help="Learning rate for optimizer", type=float, default=1e-4
    )
    parser.add_argument("-b", "--batch_size", help="Batch size", type=int, default=128)
    parser.add_argument("-l", "--layers", help="Number of layers", type=int, default=4)
    parser.add_argument(
        "-n", "--nodes", help="Number of nodes per layer", type=int, default=30
    )
    parser.add_argument(
        "--rebalance",
        dest="rebalance",
        help="Rebalance each batch [kmeans | uniform]",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    # Deterministic results in this randomization study!
    np.random.seed(28)

    # Setup device
    num_cores = 1
    if args.use_gpu:
        num_GPU = 1
        num_CPU = 1
    if not args.use_gpu:
        num_CPU = 1
        num_GPU = 0

    config = tf.ConfigProto(
        intra_op_parallelism_threads=num_cores,
        inter_op_parallelism_threads=num_cores,
        allow_soft_placement=True,
        device_count={"CPU": num_CPU, "GPU": num_GPU},
    )
    session = tf.Session(config=config)
    K.set_session(session)

    # Directories
    datadir = os.path.join("data", args.ldir)
    with open(os.path.join(datadir, "permutation.txt")) as f:
        permutation = f.readline()
    model_string = "%d-%d-%s" % (args.layers, args.nodes, permutation)
    logdir = os.path.join(
        args.odir, datetime.datetime.now().strftime("%b%d_%H-%M-%S_") + model_string
    )

    # If restarting, populate arguments from restart directory
    if args.restart is not None:
        args, datadir, logdir = read_restart(args)

    # Load data
    x_train = pd.read_pickle(os.path.join(datadir, "xtrain.gz"))
    y_train = pd.read_pickle(os.path.join(datadir, "ytrain.gz"))
    x_test = pd.read_pickle(os.path.join(datadir, "xtest.gz"))
    y_test = pd.read_pickle(os.path.join(datadir, "ytest.gz"))
    save = True
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Set precision
    K.set_floatx(args.precision)
    x_train.astype(dtype=args.precision)
    y_train.astype(dtype=args.precision)
    x_test.astype(dtype=args.precision)
    y_test.astype(dtype=args.precision)

    # DNN setup
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]

    # Log
    logname = os.path.join(logdir, "run.log")
    logger = init_logger(logname)
    logger.info("Training with the following:")
    logger.info("  epochs = {}".format(args.epochs))
    logger.info("  batch_size = {}".format(args.batch_size))
    logger.info("  learning_rate = {}".format(args.learning_rate))
    logger.info("  n_train = {}".format(n_train))
    logger.info("  n_test = {}".format(n_test))
    logger.info("  n_layers = {}".format(args.layers))
    logger.info("  n_nodes = {}".format(args.nodes))
    logger.info("  rebalance = {}".format(args.rebalance))
    logger.info("  permutation = {}".format(permutation))
    logger.info("  datadir = {}".format(datadir))
    logger.info("  logdir = {}".format(logdir))

    # DNN for regression with n_layers and n_nodes
    model, history = train_model_ff(
        x_train,
        y_train,
        x_test,
        y_test,
        args.layers,
        args.nodes,
        logdir,
        logger,
        permutation == "shuffled",
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        trace=args.trace,
        restart=args.restart,
        rebalance=args.rebalance,
    )

    # Save the model, history and predictions
    if save:
        model.save(os.path.join(logdir, "model.h5"))
        with open(os.path.join(logdir, "history.pkl"), "wb") as fw:
            pickle.dump(history.history, fw)

        src = os.path.relpath(datadir, start=logdir)
        try:
            os.symlink(
                os.path.join(src, "scaler.pkl"), os.path.join(logdir, "scaler.pkl")
            )
            os.symlink(
                os.path.join(src, "xtrain.gz"), os.path.join(logdir, "xtrain.gz")
            )
            os.symlink(
                os.path.join(src, "ytrain.gz"), os.path.join(logdir, "ytrain.gz")
            )
            os.symlink(
                os.path.join(src, "ctrain.gz"), os.path.join(logdir, "ctrain.gz")
            )
            os.symlink(os.path.join(src, "xtest.gz"), os.path.join(logdir, "xtest.gz"))
            os.symlink(os.path.join(src, "ytest.gz"), os.path.join(logdir, "ytest.gz"))
            os.symlink(os.path.join(src, "ctest.gz"), os.path.join(logdir, "ctest.gz"))
        except FileExistsError:
            pass
        m_train = pd.DataFrame(
            model.predict(x_train), columns=y_train.columns, index=y_train.index
        )
        m_test = pd.DataFrame(
            model.predict(x_test), columns=y_train.columns, index=y_test.index
        )
        m_train.to_pickle(os.path.join(logdir, "mtrain.gz"))
        m_test.to_pickle(os.path.join(logdir, "mtest.gz"))

    end = time.time() - start
    logger.info(
        "Elapsed total time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
