import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
import itertools

class LabeledUnlabeledBatchSampler(Sampler):
    """Minibatch index sampler for labeled and unlabeled indices. 

    An epoch is one pass through the labeled indices.
    """
    def __init__(
            self, 
            labeled_idx, 
            unlabeled_idx, 
            labeled_batch_size, 
            unlabeled_batch_size):

        self.labeled_idx = labeled_idx
        self.unlabeled_idx = unlabeled_idx
        self.unlabeled_batch_size = unlabeled_batch_size
        self.labeled_batch_size = labeled_batch_size

        assert len(self.labeled_idx) >= self.labeled_batch_size > 0
        assert len(self.unlabeled_idx) >= self.unlabeled_batch_size > 0

    @property
    def num_labeled(self):
        return len(self.labeled_idx)

    def __iter__(self):
        labeled_iter = iterate_once(self.labeled_idx)
        unlabeled_iter = iterate_eternally(self.unlabeled_idx)
        return (
            labeled_batch + unlabeled_batch
            for (labeled_batch, unlabeled_batch)
            in  zip(batch_iterator(labeled_iter, self.labeled_batch_size),
                    batch_iterator(unlabeled_iter, self.unlabeled_batch_size))
        )

    def __len__(self):
        return len(self.labeled_idx) // self.labeled_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def batch_iterator(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip(*args)

def get_dataloaders(train_dataset, test_dataset, batch_size):
    train_batch_sampler = LabeledUnlabeledBatchSampler(train_dataset.labeled_ids,
                                                       train_dataset.unlabeled_ids,
                                                       batch_size//2, batch_size//2)
    trainloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size, shuffle=True, pin_memory=True)

    return trainloader, testloader

def get_test_dataloader(test_dataset, batch_size):
    return DataLoader(test_dataset, batch_size, shuffle=True, pin_memory=True)