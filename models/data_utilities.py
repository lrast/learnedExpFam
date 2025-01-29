# utilities for handling data

import numpy as np

import torch
from torch.utils.data import Subset


def exact_data_ratios(MNIST_dataset, ratios=torch.ones(10), percent_train=0.95):
    """ returns training and validation datasets with the exact ratios of 
        train and validation data.
    """
    # Step one: upsample to the total number that we need
    num_per_bin = torch.bincount(MNIST_dataset.targets)
    max_bin = num_per_bin.max()

    samples_to_balance = max_bin - num_per_bin
    samples_to_ratio = max_bin * ratios - max_bin
    samples_to_ratio = samples_to_ratio.type_as(samples_to_balance)

    num_to_sample = samples_to_balance + samples_to_ratio

    extra_inds = []

    for i in range(10):
        all_inds = torch.nonzero(MNIST_dataset.targets == i).squeeze()

        conditional_inds = np.random.choice(num_per_bin[i], size=(num_to_sample[i],))
        extra_inds.append(all_inds[conditional_inds])

    extra_inds.append(torch.arange(MNIST_dataset.targets.shape[0]))
    all_inds = torch.concat(extra_inds)

    # Step 2: split into training and validation sets
    targets = MNIST_dataset.targets[all_inds]
    total_per_bin = torch.bincount(targets)

    percent_train = 0.95

    train_inds = []
    val_inds = []

    for i in range(10):
        num_total = total_per_bin[i]
        cutoff = round(num_total.item() * percent_train)

        data_inds = torch.nonzero((targets == i)).squeeze()
        perm = torch.randperm(num_total)

        train_inds.append(all_inds[data_inds[perm[0:cutoff]]])
        val_inds.append(all_inds[data_inds[perm[cutoff:]]])

    train_inds = torch.concat(train_inds)
    val_inds = torch.concat(val_inds)

    train_set = Subset(MNIST_dataset, train_inds)
    val_set = Subset(MNIST_dataset, val_inds)

    return train_set, val_set

