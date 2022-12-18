import numpy as np
import itertools
import random
from torchvision.datasets import SVHN, MNIST, FashionMNIST, CIFAR10, CelebA, Omniglot
from torchvision.datasets import STL10, Food101, Caltech101, GTSRB, Flowers102, KMNIST, CIFAR100

from data_transforms import train_transforms, test_transforms

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

class TrainDataset():
    def __init__(self, config):
        self.config = config
        self.labeled_ids = []
        self.unlabeled_ids = []
        self.image_tensors = []
        self.labels = []
        self.counter = 0
    
    def prepare(self, indata_size=5000, outdata_size=1700, label_ratio=0.2):
        # Prepare OOD data
        for k in self.config['out_data']:
            dataset = self.config[k]['dataset']
            transforms = TransformTwice(self.config[k]['transforms'])
            n_outlabels = int(outdata_size * label_ratio)
            dataset_ids = np.random.choice(len(dataset), outdata_size)
            dataset_labeled_ids = set(np.random.choice(dataset_ids, n_outlabels))
            for idx in dataset_ids:
                img = dataset[idx][0]
                img_tensor = transforms(img)
                self.image_tensors.append(img_tensor)
                if idx in dataset_labeled_ids:
                    self.labeled_ids.append(self.counter)
                    self.labels.append(0)
                else:
                    self.unlabeled_ids.append(self.counter)
                    self.labels.append(-1)
                self.counter += 1
            print(len(self.labeled_ids))
            print(f'{k} dataset processed...')
        
        # Prepare ID data
        in_data = self.config['in_data']
        dataset = self.config[in_data]['dataset']
        transforms = TransformTwice(self.config[k]['transforms'])
        n_inlabels = int(indata_size * label_ratio)
        dataset_ids = np.random.choice(len(dataset), indata_size)
        dataset_labeled_ids = set(np.random.choice(dataset_ids, n_inlabels))
        for idx in dataset_ids:
            img = dataset[idx][0]
            img_tensor = transforms(img)
            self.image_tensors.append(img_tensor)
            if idx in dataset_labeled_ids:
                self.labeled_ids.append(self.counter)
                self.labels.append(1)
            else:
                self.unlabeled_ids.append(self.counter)
                self.labels.append(-1)
            self.counter += 1
        print(len(self.labeled_ids))
        print(f'{in_data} dataset processed...')
        
        random.shuffle(self.labeled_ids)
        random.shuffle(self.unlabeled_ids)
    
    def __len__(self):
        return len(self.image_tensors)
    
    def __getitem__(self, idx):
        return self.image_tensors[idx], self.labels[idx]


class TestDataset():
    def __init__(self, config: dict):
        self.config = config
        self.image_tensors = []
        self.labels = []
    
    def prepare(self, indata_size=600, outdata_size=200):
        # Prepare OOD data
        for k in self.config['out_data']:
            dataset = self.config[k]['dataset']
            transforms = self.config[k]['transforms']
            for i, (img, _) in enumerate(dataset):
                if i == outdata_size:
                    break
                img_tensor = transforms(img)
                self.image_tensors.append(img_tensor)
            self.labels += [0] * outdata_size
        
        # Prepare ID data
        in_data = self.config['in_data']
        dataset = self.config[in_data]['dataset']
        transforms = self.config[in_data]['transforms']
        for i, (img, _) in enumerate(dataset):
            if i == indata_size:
                break
            img_tensor = transforms(img)
            self.image_tensors.append(img_tensor)
        self.labels += [1] * indata_size
    
    def __len__(self):
        return len(self.image_tensors)
    
    def __getitem__(self, idx):
        return self.image_tensors[idx], self.labels[idx]


def get_dataset(dataset, data_dir, train=True):
    if dataset == 'mnist':
        if train:
            return MNIST(root=data_dir, download=True)
        else:
            return MNIST(root=data_dir, train=False, download=True) 
    if dataset == 'kmnist':
        if train:
            return KMNIST(root=data_dir, download=True)
        else:
            return KMNIST(root=data_dir, train=False, download=True)
    if dataset == 'fmnist':
        if train:
            return FashionMNIST(root=data_dir, download=True)
        else:
            return FashionMNIST(root=data_dir, train=False, download=True)
    if dataset == 'omniglot':
        if train:
            return Omniglot(root=data_dir, download = True)
        else:
            return Omniglot(root=data_dir, download = True)
    if dataset == 'cifar10':
        if train:
            return CIFAR10(root=data_dir, download=True)
        else:
            return CIFAR10(root=data_dir, train=False, download=True)
    if dataset == 'cifar100':
        if train:
            return CIFAR100(root=data_dir, download=True)
        else:
            return CIFAR100(root=data_dir, train=False, download=True)
    if dataset == 'stl10':
        if train:
            return STL10(root=data_dir, split='train', download=True)
        else:
            return STL10(root=data_dir, split='test', download=True)
    if dataset == 'svhn':
        if train:
            return SVHN(root=data_dir, split='train', download=True)
        else:
            return SVHN(root=data_dir, split='test', download=True)
    if dataset == 'celeba':
        if train:
            return CelebA(root=data_dir, download = True)
        else:
            return CelebA(root=data_dir, download = True)
    if dataset == 'tinyimagenet':
        if train:
            pass
        else:
            pass
    if dataset == 'imageneto':
        if train:
            pass
        else:
            pass


def get_train_config(in_data, out_data_list, root_dir):
    config = {}

    config['in_data'] = in_data
    config['out_data'] = out_data_list

    config[in_data] = {}
    config[in_data]['dataset'] = get_dataset(in_data, root_dir, train=True)
    config[in_data]['transforms'] = train_transforms[in_data]

    for out_data in out_data_list:
        config[out_data] = {}
        config[out_data]['dataset'] = get_dataset(out_data, root_dir, train=True)
        config[out_data]['transforms'] = train_transforms[out_data]
    
    return config

def get_test_config(in_data, out_data_list, root_dir):
    config = {}

    config['in_data'] = in_data
    config['out_data'] = out_data_list

    config[in_data] = {}
    config[in_data]['dataset'] = get_dataset(in_data, root_dir, train=False)
    config[in_data]['transforms'] = test_transforms[in_data]

    for out_data in out_data_list:
        config[out_data] = {}
        config[out_data]['dataset'] = get_dataset(out_data, root_dir, train=False)
        config[out_data]['transforms'] = test_transforms[out_data]
    
    return config