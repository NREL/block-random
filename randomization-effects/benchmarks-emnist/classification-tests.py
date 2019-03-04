"""
Use the EMNIST datasets to test for SGD convergence vs randomization
"""

# ===============================================================================
#
# Imports
#
# ===============================================================================
from __future__ import print_function

import os
import argparse
import keras
from keras.datasets import mnist, fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, adam
import scipy.io
import numpy as np
import pickle
import multiprocessing

# ===============================================================================
#
# Function definitions
#
# ===============================================================================
# Function to load emnist datasets
def load_emnist(data="byclass"):
    emnist = scipy.io.loadmat("../../data/emnist-dataset/matlab/emnist-%s.mat" % data)

    # load training dataset
    x_train = emnist["dataset"][0][0][0][0][0][0]
    x_train = x_train.astype(np.float32)
    # load training labels
    y_train = emnist["dataset"][0][0][0][0][0][1]
    num_classes = np.unique(y_train).size
    # load test dataset
    x_test = emnist["dataset"][0][0][1][0][0][0]
    x_test = x_test.astype(np.float32)
    # load test labels
    y_test = emnist["dataset"][0][0][1][0][0][1]
    # normalize
    x_train /= 255
    x_test /= 255
    # reshape using matlab order
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28, order="A")
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28, order="A")
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    # for some reason, letters has y from 1-26, change to 0-25
    if data == "letters":
        y_train = y_train - 1
        y_test = y_test - 1

    return (num_classes, (x_train, y_train), (x_test, y_test))


# ===============================================================================
# Function to load the keras mnist dataset
def load_mnist(data="digits"):
    if data == "fashion":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    num_classes = np.unique(y_train).size
    return (num_classes, (x_train, y_train), (x_test, y_test))


# ===============================================================================
# Function to train a feed forward neural network with dropout regularization
def train_model_ff(x_train, y_train, x_test, y_test, shuffle=False):
    model = Sequential()
    model.add(Dense(512, activation="relu", input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))

    model.summary()

    model.compile(
        loss="categorical_crossentropy", optimizer=adam(), metrics=["accuracy"]
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test),
        shuffle=shuffle,
    )
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return history


# ===============================================================================
# Function to train a CNN with dropout regularization
def train_model_cnn(
    num_classes, x_train, y_train, x_test, y_test, shuffle=False, batch_size=128
):
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    model = Sequential()
    model.add(
        Conv2D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
        )
    )
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.summary()

    model.compile(
        loss="categorical_crossentropy", optimizer=adam(), metrics=["accuracy"]
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test),
        shuffle=shuffle,
    )
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return (model, history)


# ===============================================================================
# Function to do custom batch ordering
def custom_batch_order(x_train, y_train, order="block", batch_size=128):
    # Option to just return the dataset as is
    if order == "shuffled":
        return (x_train, y_train)

    # First sort by class
    ind = np.lexsort(y_train.T)
    y_train_new = y_train[ind]
    x_train_new = x_train[ind]
    if order == "sorted":
        return (x_train_new, y_train_new)

    # Custom batching on sorted classes
    # Split sorted data into blocks (same size as the mini-batch)
    # Shuffle these batches around
    # So, every mini-batch iteration will still just see one class, but the next iteration will
    # most likely have a different class
    batches = np.arange(np.ceil(y_train.shape[0] / batch_size))
    np.random.shuffle(batches)
    new_idx = np.array(
        [np.arange(i * batch_size, (i + 1) * batch_size) for i in batches], dtype=int
    ).flatten()
    new_idx = new_idx[new_idx < y_train.shape[0]]
    return (x_train_new[new_idx], y_train_new[new_idx])


# ===============================================================================
def get_results(dataset, save_path, batch_size=128):
    # Load data
    if dataset == "fashion":
        num_classes, (x_train, y_train), (x_test, y_test) = load_mnist("fashion")
    else:
        num_classes, (x_train, y_train), (x_test, y_test) = load_emnist(dataset)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    orders = ["shuffled", "block", "sorted"]
    histories = {}
    for ordering in orders:
        # Custom order data for study
        x_train_new, y_train_new = custom_batch_order(
            x_train, y_train, ordering, batch_size
        )
        # Finally, train the model, print scores, etc
        model, history = train_model_cnn(
            num_classes,
            x_train_new,
            y_train_new,
            x_test,
            y_test,
            ordering == "shuffled",
            batch_size,
        )
        histories[ordering] = history.history
    model_string = "%s/%s-%d" % (save_path, dataset, batch_size)
    model.save("%s.h5" % model_string)
    with open("%s.pkl" % (model_string), "wb") as fw:
        pickle.dump(histories, fw)
    return histories


# ===============================================================================
#
# Main
#
# ===============================================================================
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run classification problems")
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        help="""Dataset to run ["fashion", "digits", "letters", "byclass", "bymerge", "balanced", "mnist"]""",
        type=str,
        default=["fashion"],
    )
    args = parser.parse_args()

    # Deterministic results in this randomization study!
    np.random.seed(28)

    # Init params
    batch_sizes_base = [32, 64, 128, 256, 512, 1024, 2048]
    batch_size_dataset = {
        "fashion": [32, 64, 128, 256, 512, 1024, 2048, 4096, 5120],
        "digits": [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 20480],
        "balanced": [32, 64, 128, 256, 512, 1024, 1536, 2048],
        "mnist": [32, 64, 128, 256, 512, 1024, 2048, 4096, 5120],
        "letters": [32, 64, 128, 256, 512, 1024, 2048, 4096],
        "byclass": batch_sizes_base + [4096],
        "bymerge": batch_sizes_base + [4096],
    }

    epochs = 50

    save_path = "batch_size_study"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for dataset in args.datasets:
        for batch_size in batch_size_dataset[dataset]:
            res = get_results(dataset, save_path, batch_size)
