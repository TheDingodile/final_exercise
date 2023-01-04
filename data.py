import numpy as np
from os import listdir
import os
import helper
import torch
import matplotlib.pyplot as plt

def mnist():
    # Define a transform to normalize the data
    files = listdir("data/corruptmnist")
    train_in = np.empty(shape=(0, 28, 28))
    train_out = np.empty(shape=(0))

    test_in = np.empty(shape=(0, 28, 28))
    test_out = np.empty(shape=(0))
    for f in files:
        if f[0:5] == "train":
            train_in = np.concatenate((train_in, np.load("data/corruptmnist/" + f)['images']), axis=0)
            train_out = np.concatenate((train_out, np.load("data/corruptmnist/" + f)['labels']), axis=0)
        else:
            test_in = np.concatenate((test_in, np.load("data/corruptmnist/" + f)['images']), axis=0)
            test_out = np.concatenate((test_out, np.load("data/corruptmnist/" + f)['labels']), axis=0)
    return (train_in, train_out), (test_in, test_out)

mnist()
