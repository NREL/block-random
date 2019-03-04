import matplotlib.pyplot as plt
import numpy as np
import pickle

datasets = ["fashion", "digits", "letters", "byclass", "bymerge", "balanced", "mnist"]
batch_size = 128


def plot_results(res, dataset, batch_size=128):
    plt.figure()
    plt.plot(res["shuffled"]["val_acc"], label="shuffled")
    plt.plot(res["block"]["val_acc"], label="batch-random")
    plt.plot(res["sorted"]["val_acc"], label="sorted")
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title("Training on dataset: %s with batch size: %d" % (dataset, batch_size))
    plt.savefig("%s.pdf" % dataset)
    plt.figure()
    plt.plot(res["shuffled"]["val_acc"], label="suffled")
    plt.plot(res["block"]["val_acc"], label="batch-random")
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title("Training on dataset: %s with batch size: %d" % (dataset, batch_size))
    plt.savefig("%s-zoomed.pdf" % dataset)


for dataset in datasets:
    fr = open("%s.pkl" % dataset, "rb")
    res = pickle.load(fr)
    fr.close()
    plot_results(res, dataset, batch_size)
plt.show()
