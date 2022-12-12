#!/usr/bin/env python
import sys
import os
import flow_ssl
from torchvision.datasets import SVHN, MNIST, FashionMNIST, CIFAR10, CelebA, Omniglot
from torchvision.datasets import STL10, Food101, Caltech101, GTSRB, Flowers102, KMNIST, CIFAR100

import torchvision
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from experiments.train_flows.utils import norm_util
from torch.utils.data.sampler import Sampler


import numpy as np
import itertools
import random
import argparse

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


from experiments.train_flows.utils import train_utils, optim_util
from scipy.spatial.distance import cdist
from experiments.train_flows.utils import shell_util

from tqdm import tqdm
import torch.nn as nn
import math
import torch.nn.init as init
import torch.nn.functional as F


from flow_ssl.distributions import SSLGaussMixture
from flow_ssl import FlowLoss
from tensorboardX import SummaryWriter





parser = argparse.ArgumentParser(description='Run custom flow datasets for OOD')

parser.add_argument('--indata_size', type=int, default=50000)
parser.add_argument('--outdata_size', type=int, default=18000)
parser.add_argument('--label_ratio', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--save_freq', type=int, default=2)
parser.add_argument('--eval_freq', type=int, default=2)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--use_ldam', action='store_true')
parser.add_argument('--root_path')

parser.add_argument('--in_data', default="cifar")
parser.add_argument('--out_data', default="svhn,fashionmnist,mnist")

parser.add_argument('--test_out_data', default=None) #could be extra as well

parser.add_argument('--test_indata_size', type=int, default=900)
parser.add_argument('--test_outdata_size', type=int, default=300)

args = parser.parse_args()

sys.path.append(os.path.join(args.root_path, 'flowgmm-public'))
data_dir = os.path.join(args.root_path, 'data')

indata_size = args.indata_size
outdata_size = args.outdata_size
label_ratio = args.label_ratio
batch_size = args.batch_size

in_data = args.in_data
out_data = args.out_data.split(',')

if not args.test_out_data:
    test_out_data = out_data
else:
    test_out_data = args.test_out_data.split(',')
test_indata_size = args.test_indata_size
test_outdata_size = args.test_outdata_size

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


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

class Dataset():
    def __init__(self, config):
        self.config = config
        self.labeled_ids = []
        self.unlabeled_ids = []
        self.image_tensors = []
        self.labels = []
        self.counter = 0
    
    def prepare(self, in_data='cifar', out_data=['mnist', 'svhn', 'fashionmnist'], indata_size=5000, outdata_size=1700, label_ratio=0.2):
        # Prepare OOD data
        for k in out_data:
            dataset = config[k]['dataset']
            transforms = TransformTwice(config[k]['transforms'])
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
        dataset = config[in_data]['dataset']
        transforms = TransformTwice(config[k]['transforms'])
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


class SLDataset():
    def __init__(self, config: dict):
        self.config = config
        self.image_tensors = []
        self.labels = []
    
    def prepare(self, in_data='cifar', out_data=['mnist', 'svhn', 'fashionmnist'], indata_size=600, outdata_size=200):
        print(out_data)
        # Prepare OOD data
        for k in out_data:
            dataset = config[k]['dataset']
            transforms = config[k]['transforms']
            for i, (img, _) in enumerate(dataset):
                if i == outdata_size:
                    break
                img_tensor = transforms(img)
                self.image_tensors.append(img_tensor)
            self.labels += [0] * outdata_size
        
        # Prepare ID data
        dataset = config[in_data]['dataset']
        transforms = config[in_data]['transforms']
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

# convert all images to the same size as the in-data distribution
# in most cases it will be 3X32X32

transform_to_three_channel = transforms.Lambda(lambda img: img.expand(3,*img.shape[1:]))

svhn_transforms = transforms.Compose([
                    # transforms.Grayscale(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Resize((32,32)),
                    transform_to_three_channel,
                ])

mnist_transforms = transforms.Compose([
                   transforms.RandomCrop(32, padding=4),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Resize((32,32)),
                   transform_to_three_channel
                ])

fashionmnist_transforms = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Resize((32,32)),
                    transform_to_three_channel
                ])

cifar_transforms = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    # transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Resize((32,32)),
                    transform_to_three_channel
                ])

food_transforms = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    # transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Resize((32,32)),
                    transform_to_three_channel
                ])

celeb_transforms = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    # transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Resize((32,32)),
                    transform_to_three_channel
                ])
stl10_transforms = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    # transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Resize((32,32)),
                    transform_to_three_channel
                ])
german_sign_transforms = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    # transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Resize((32,32)),
                    transform_to_three_channel
                ])
caltech_transforms = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    # transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Resize((32,32)),
                    transform_to_three_channel
                ])

flowers_transforms = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    # transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Resize((32,32)),
                    transform_to_three_channel
                ])
# load train datasets from pytorch

svhn_dataset = SVHN(root=data_dir, split='train', download=True)
mnist_dataset = MNIST(root=data_dir, download=True)
kmnist_dataset = KMNIST(root=data_dir, download=True)
fashionmnist_dataset = FashionMNIST(root=data_dir, download=True)
cifar_dataset = CIFAR10(root=data_dir, download=True)
cifar100_dataset = CIFAR100(root=data_dir, download=True)
stl10_dataset = STL10(root=data_dir, split='train', download=True)
food_dataset = Food101(root=data_dir, download = True)
# celeb_dataset = CelebA(root=data_dir, download = True)
flowers_dataset = Flowers102(root=data_dir, download = True)
caltech_dataset = Caltech101(root=data_dir, download = True)
german_sign_dataset = GTSRB(root=data_dir, download = True)
omniglot_dataset = Omniglot(root=data_dir, download = True)



# load test datasets from pytorch

svhn_test_dataset = SVHN(root=data_dir, split='test', download=True)
mnist_test_dataset = MNIST(root=data_dir, train=False, download=True)
kmnist_test_dataset = KMNIST(root=data_dir, train=False, download=True)
fashionmnist_test_dataset = FashionMNIST(root=data_dir, train=False, download=True)
cifar_test_dataset = CIFAR10(root=data_dir, train=False, download=True)
cifar100_test_dataset = CIFAR100(root=data_dir, train=False, download=True)
stl10_test_dataset = STL10(root=data_dir, split='test', download=True)
food_test_dataset = Food101(root=data_dir, split='test', download = True)
# celeb_dataset = CelebA(root=data_dir, download = True)
flowers_test_dataset = Flowers102(root=data_dir, split='test', download = True)
caltech_test_dataset = Caltech101(root=data_dir, download = True)
german_sign_test_dataset = GTSRB(root=data_dir, split='test', download = True)
omniglot_test_dataset = Omniglot(root=data_dir, download = True)

# create train config, to be used in the 
# training dataset, composed of images from various sources

config = {}

config['svhn'] = {}
config['svhn']['dataset'] = svhn_dataset
config['svhn']['transforms'] = svhn_transforms

config['mnist'] = {}
config['mnist']['dataset'] = mnist_dataset
config['mnist']['transforms'] = mnist_transforms

config['kmnist'] = {}
config['kmnist']['dataset'] = kmnist_dataset
config['kmnist']['transforms'] = mnist_transforms

config['fashionmnist'] = {}
config['fashionmnist']['dataset'] = fashionmnist_dataset
config['fashionmnist']['transforms'] = fashionmnist_transforms

config['cifar'] = {}
config['cifar']['dataset'] = cifar_dataset
config['cifar']['transforms'] = cifar_transforms

config['cifar100'] = {}
config['cifar100']['dataset'] = cifar100_dataset
config['cifar100']['transforms'] = cifar_transforms

config['food'] = {}
config['food']['dataset'] = food_dataset
config['food']['transforms'] = food_transforms

config['stl10'] = {}
config['stl10']['dataset'] = stl10_dataset
config['stl10']['transforms'] = stl10_transforms

config['flowers'] = {}
config['flowers']['dataset'] = flowers_dataset
config['flowers']['transforms'] = flowers_transforms

config['caltech'] = {}
config['caltech']['dataset'] = caltech_dataset
config['caltech']['transforms'] = caltech_transforms

config['german_sign'] = {}
config['german_sign']['dataset'] = german_sign_dataset
config['german_sign']['transforms'] = german_sign_transforms

config['omniglot'] = {}
config['omniglot']['dataset'] = omniglot_dataset
config['omniglot']['transforms'] = german_sign_transforms

# write test_config, to be used in the test data

test_config = {}

test_config['svhn'] = {}
test_config['svhn']['dataset'] = svhn_test_dataset
test_config['svhn']['transforms'] = svhn_transforms

test_config['mnist'] = {}
test_config['mnist']['dataset'] = mnist_test_dataset
test_config['mnist']['transforms'] = mnist_transforms

test_config['kmnist'] = {}
test_config['kmnist']['dataset'] = kmnist_test_dataset
test_config['kmnist']['transforms'] = mnist_transforms

test_config['fashionmnist'] = {}
test_config['fashionmnist']['dataset'] = fashionmnist_test_dataset
test_config['fashionmnist']['transforms'] = fashionmnist_transforms

test_config['cifar'] = {}
test_config['cifar']['dataset'] = cifar_test_dataset
test_config['cifar']['transforms'] = cifar_transforms

test_config['cifar100'] = {}
test_config['cifar100']['dataset'] = cifar100_test_dataset
test_config['cifar100']['transforms'] = cifar_transforms

test_config['food'] = {}
test_config['food']['dataset'] = food_test_dataset
test_config['food']['transforms'] = food_transforms

test_config['stl10'] = {}
test_config['stl10']['dataset'] = stl10_test_dataset
test_config['stl10']['transforms'] = stl10_transforms

test_config['flowers'] = {}
test_config['flowers']['dataset'] = flowers_test_dataset
test_config['flowers']['transforms'] = flowers_transforms

test_config['caltech'] = {}
test_config['caltech']['dataset'] = caltech_test_dataset
test_config['caltech']['transforms'] = caltech_transforms

test_config['german_sign'] = {}
test_config['german_sign']['dataset'] = german_sign_test_dataset
test_config['german_sign']['transforms'] = german_sign_transforms

test_config['omniglot'] = {}
test_config['omniglot']['dataset'] = omniglot_test_dataset
test_config['omniglot']['transforms'] = german_sign_transforms

# define LDAM loss for skewed data

cls_num_list = [outdata_size, indata_size]

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

# create custom train and test dataset based on our config

# out_data = ['mnist', 'svhn', 'fashionmnist', 'stl10','food', 'caltech','flowers','german_sign']

ds = Dataset(config)
ds.prepare(in_data=in_data, out_data=out_data, indata_size=indata_size, outdata_size=outdata_size, label_ratio=label_ratio)

test_ds = SLDataset(test_config)
test_ds.prepare(in_data=in_data, out_data=test_out_data, indata_size=test_indata_size, outdata_size=test_outdata_size)


# create dataloaders for train and test dataset

train_batch_sampler = LabeledUnlabeledBatchSampler(ds.labeled_ids, ds.unlabeled_ids, batch_size//2, batch_size//2)
trainloader = torch.utils.data.DataLoader(ds, batch_sampler=train_batch_sampler, pin_memory=True)
testloader = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=True, pin_memory=True)

total_labels = len(ds.labeled_ids)
# test dataloaders

for batch in trainloader:
    print(batch[0][0].shape)
    print(batch[0][1].shape)
    print(batch[1].shape)
    break

for batch in testloader:
    print(len(batch))
    print(batch[0].get_device())
    print(batch[0].shape)
    print(batch[1].shape)
    break


# decide based on channels of dataset

#if in_data in ['mnist','fashionmnist','svhn']:
#    img_shape = (1, 32, 32)
#    flow = 'MNISTResidualFlow'
#    model_cfg = getattr(flow_ssl, flow)
#    net = model_cfg(in_channels=img_shape[0], num_classes=2)

#else:
img_shape = (3, 32, 32)
flow = 'ResidualFlow'
model_cfg = getattr(flow_ssl, flow)
net = model_cfg(in_channels=img_shape[0], num_classes=2)

if flow in ["iCNN3d", "iResnetProper","SmallResidualFlow","ResidualFlow","MNISTResidualFlow"]:
    net = net.flow

print(args)
print(img_shape)
print(flow)

# initialize means and priors

means = 'random'
means_r = 1.0
cov_std = 1.0
# img_shape = (1, 32, 32)
device = 'cuda'
n_classes = 2

net = net.to(device)
r = means_r
cov_std = torch.ones((n_classes)) * cov_std
cov_std = cov_std.to(device)
means = train_utils.get_means(means, num_means=n_classes, r=means_r, trainloader=trainloader, 
                        shape=img_shape, device=device, net=net)
means_init = means.clone().detach()

#print("Means:", means)
#print("Cov std:", cov_std)
means_np = means.cpu().numpy()
#print("Pairwise dists:", cdist(means_np, means_np))

means_trainable = True
covs_trainable = True
weights_trainable = True

if means_trainable:
    print("Using learnable means")
    means = torch.tensor(means_np, requires_grad=True, device=device)

prior = SSLGaussMixture(means, device=device)
prior.weights.requires_grad = weights_trainable
prior.inv_cov_stds.requires_grad = covs_trainable
loss_fn = FlowLoss(prior)
sup_loss_fn = LDAMLoss(cls_num_list, s=15)


param_groups = norm_util.get_param_groups(net, 0.0, norm_suffix='weight_g')
optimizer = optim.Adam(param_groups, lr=5e-4, weight_decay=1e-2)
opt_gmm = optim.Adam([prior.means, prior.weights, prior.inv_cov_stds], lr=1e-4, weight_decay=1e-2)

log_directory = f'./{in_data}_{",".join(out_data)}_{label_ratio}'
print(log_directory)
os.mkdir(log_directory)
writer = SummaryWriter(log_dir=log_directory)
device = 'cuda' if torch.cuda.is_available() and len([0]) > 0 else 'cpu'
start_epoch = 0


"""
Train function for the flow
"""

def train(epoch, net, trainloader, device, optimizer, opt_gmm, loss_fn,
          label_weight, max_grad_norm, consistency_weight,
          writer, sup_loss_fn, use_unlab=True,  acc_train_all_labels=False,
          ):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = shell_util.AverageMeter()
    loss_unsup_meter = shell_util.AverageMeter()
    loss_nll_meter = shell_util.AverageMeter()
    loss_consistency_meter = shell_util.AverageMeter()
    jaclogdet_meter = shell_util.AverageMeter()
    acc_meter = shell_util.AverageMeter()
    acc_all_meter = shell_util.AverageMeter()
    with tqdm(total=2*total_labels) as progress_bar:
        for (x1, x2), y in trainloader:

            x1 = x1.to(device)
            if not acc_train_all_labels:
                y = y.to(device)
            else:
                y, y_all_lab = y[:, 0], y[:, 1]
                y = y.to(device)
                y_all_lab = y_all_lab.to(device)

            labeled_mask = (y != NO_LABEL)

            optimizer.zero_grad()
            opt_gmm.zero_grad()

            with torch.no_grad():
                x2 = x2.to(device)
                # z21 = net(x2)
                # z22 = net(transform(x2))
                # y22 = classify(z22)
                # l_cons = FlowLoss(z21, y22)
                # print(x2.get_device(), next(net.parameters()).is_cuda)
                z2 = net(x2)
                z2 = z2.detach()
                pred2 = loss_fn.prior.classify(z2.reshape((len(z2), -1)))

            z1 = net(x1)
            sldj = net.logdet()

            z_all = z1.reshape((len(z1), -1))
            z_labeled = z_all[labeled_mask]
            y_labeled = y[labeled_mask]

            logits_all = loss_fn.prior.class_logits(z_all)
            logits_labeled = logits_all[labeled_mask]
            if args.use_ldam:
                loss_nll = sup_loss_fn(logits_labeled, y_labeled)
            else:
                loss_nll = F.cross_entropy(logits_labeled, y_labeled)
            # print(loss_nll)
            # loss_nll = loss_nll.mean()
            # print(loss_nll)

            if use_unlab:
                loss_unsup = loss_fn(z1, sldj=sldj)
                # print(loss_unsup)
                loss = loss_nll * label_weight + loss_unsup
            else:
                loss_unsup = torch.tensor([0.])
                loss = loss_nll

            # consistency loss
            loss_consistency = loss_fn(z1, sldj=sldj, y=pred2)
            loss = loss + loss_consistency * consistency_weight

            loss.backward()
            optim_util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()
            opt_gmm.step()

            preds_all = torch.argmax(logits_all, dim=1)
            preds = preds_all[labeled_mask]
            acc = (preds == y_labeled).float().mean().item()
            if acc_train_all_labels:
                acc_all = (preds_all == y_all_lab).float().mean().item()
            else:
                acc_all = acc

            acc_meter.update(acc, x1.size(0))
            acc_all_meter.update(acc_all, x1.size(0))
            loss_meter.update(loss.item(), x1.size(0))
            loss_unsup_meter.update(loss_unsup.item(), x1.size(0))
            loss_nll_meter.update(loss_nll.item(), x1.size(0))
            jaclogdet_meter.update(sldj.mean().item(), x1.size(0))
            loss_consistency_meter.update(loss_consistency.item(), x1.size(0))

            progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=optim_util.bits_per_dim(x1, loss_unsup_meter.avg),
                                     acc=acc_meter.avg,
                                     acc_all=acc_all_meter.avg)
            progress_bar.update(y_labeled.size(0))

    x1_img = torchvision.utils.make_grid(x1[:10], nrow=2 , padding=2, pad_value=255)
    x2_img = torchvision.utils.make_grid(x2[:10], nrow=2 , padding=2, pad_value=255)
    writer.add_image("data/x1", x1_img)
    writer.add_image("data/x2", x2_img)

    writer.add_scalar("train/loss", loss_meter.avg, epoch)
    writer.add_scalar("train/loss_unsup", loss_unsup_meter.avg, epoch)
    writer.add_scalar("train/loss_nll", loss_nll_meter.avg, epoch)
    writer.add_scalar("train/jaclogdet", jaclogdet_meter.avg, epoch)
    writer.add_scalar("train/acc", acc_meter.avg, epoch)
    writer.add_scalar("train/acc_all", acc_all_meter.avg, epoch)
    writer.add_scalar("train/bpd", optim_util.bits_per_dim(x1, loss_unsup_meter.avg), epoch)
    writer.add_scalar("train/loss_consistency", loss_consistency_meter.avg, epoch)


def test_classifier(epoch, net, testloader, device, loss_fn, writer=None, postfix="",
                    show_classification_images=False, confusion=False):
    net.eval()
    loss_meter = shell_util.AverageMeter()
    jaclogdet_meter = shell_util.AverageMeter()
    acc_meter = shell_util.AverageMeter()
    all_pred_labels = []
    all_xs = []
    all_ys = []
    all_zs = []
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, y in testloader:
                all_xs.append(x.data.numpy())
                all_ys.append(y.data.numpy())
                x = x.to(device)
                y = y.to(device)
                z = net(x)
                all_zs.append(z.cpu().data.numpy())
                sldj = net.logdet()
                loss = loss_fn(z, y=y, sldj=sldj)
                loss_meter.update(loss.item(), x.size(0))
                jaclogdet_meter.update(sldj.mean().item(), x.size(0))

                preds = loss_fn.prior.classify(z.reshape((len(z), -1)))
                preds = preds.reshape(y.shape)
                all_pred_labels.append(preds.cpu().data.numpy())
                acc = (preds == y).float().mean().item()
                acc_meter.update(acc, x.size(0))

                progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=optim_util.bits_per_dim(x, loss_meter.avg),
                                     acc=acc_meter.avg)
                progress_bar.update(x.size(0))
    all_pred_labels = np.hstack(all_pred_labels)
    all_xs = np.vstack(all_xs)
    all_zs = np.vstack(all_zs)
    all_ys = np.hstack(all_ys)

    if writer is not None:
        writer.add_scalar("test/loss{}".format(postfix), loss_meter.avg, epoch)
        writer.add_scalar("test/acc{}".format(postfix), acc_meter.avg, epoch)
        writer.add_scalar("test/bpd{}".format(postfix), optim_util.bits_per_dim(x, loss_meter.avg), epoch)
        writer.add_scalar("test/jaclogdet{}".format(postfix), jaclogdet_meter.avg, epoch)

        for cls in range(np.max(all_pred_labels)+1):
            num_imgs_cls = (all_pred_labels==cls).sum()
            writer.add_scalar("test_clustering/num_class_{}_{}".format(cls,postfix), 
                    num_imgs_cls, epoch)
            if num_imgs_cls == 0:
                writer.add_scalar("test_clustering/num_class_{}_{}".format(cls,postfix), 
                    0., epoch)
                continue
            writer.add_histogram('label_distributions/num_class_{}_{}'.format(cls,postfix), 
                    all_ys[all_pred_labels==cls], epoch)

            writer.add_histogram(
                'distance_distributions/num_class_{}'.format(cls),
                torch.norm(torch.tensor(all_zs[all_pred_labels==cls]) - loss_fn.prior.means[cls].cpu(), p=2, dim=1),
                epoch
            )

            if show_classification_images:
                images_cls = all_xs[all_pred_labels==cls][:10]
                images_cls = torch.from_numpy(images_cls).float()
                images_cls_concat = torchvision.utils.make_grid(
                        images_cls, nrow=2, padding=2, pad_value=255)
                writer.add_image("test_clustering/class_{}".format(cls), 
                        images_cls_concat)

        if confusion:
            fig = plt.figure(figsize=(8, 8))
            cm = confusion_matrix(all_ys, all_pred_labels)
            cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
            sns.heatmap(cm, annot=True, cmap=plt.cm.Blues)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            conf_img = torch.tensor(np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep=''))
            conf_img = torch.tensor(conf_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))).transpose(0, 2).transpose(1, 2)
            writer.add_image("confusion", conf_img, epoch)



def linear_rampup(final_value, epoch, num_epochs, start_epoch=0):
    t = (epoch - start_epoch + 1) / num_epochs
    if t > 1:
        t = 1.
    return t * final_value

def sample(net, prior, batch_size, cls, device, sample_shape):
    """Sample from RealNVP model.
    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    with torch.no_grad():
        if cls is not None:
            z = prior.sample((batch_size,), gaussian_id=cls)
        else:
            z = prior.sample((batch_size,))
        x = net.inverse(z)

        return x

# Start training

NO_LABEL = -1
schedule = None
n_epochs = args.n_epochs
lr = 5e-4
lr_gmm = 1e-4
consistency_weight = 0.01
consistency_rampup = 1
label_weight = 1.0
max_grad_norm = 100.0
# need to feed this as args
ckptdir = f'./{in_data}_{",".join(out_data)}_{label_ratio}'
save_freq = args.save_freq
eval_freq = args.eval_freq
confusion = True
num_samples = 50


for epoch in range(start_epoch, n_epochs):
    cons_weight = linear_rampup(consistency_weight, epoch, consistency_rampup, start_epoch)
    
    writer.add_scalar("hypers/learning_rate", lr, epoch)
    writer.add_scalar("hypers/learning_rate_gmm", lr_gmm, epoch)
    writer.add_scalar("hypers/consistency_weight", cons_weight, epoch)

    train(epoch, net, trainloader, device, optimizer, opt_gmm, loss_fn,
          label_weight, max_grad_norm, cons_weight,
          writer, sup_loss_fn, use_unlab=True)

    # Save checkpoint
    if (epoch % save_freq == 0):
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'means': prior.means,
        }
        os.makedirs(ckptdir, exist_ok=True)
        torch.save(state, os.path.join(ckptdir, str(epoch)+'.pt'))

    # Save samples and data
    if epoch % eval_freq == 0:
        test_classifier(epoch, net, testloader, device, loss_fn, writer, confusion=confusion)
        # if args.swa:
        #     optimizer.swap_swa_sgd() 
        #     print("updating bn")
        #     SWA.bn_update(bn_loader, net)
        #     utils.test_classifier(epoch, net, testloader, device, loss_fn, 
        #             writer, postfix="_swa")

        z_means = prior.means
        data_means = net.inverse(z_means)
        z_mean_imgs = torchvision.utils.make_grid(
                z_means.reshape((n_classes, *img_shape)), nrow=2)
        data_mean_imgs = torchvision.utils.make_grid(
                data_means.reshape((n_classes, *img_shape)), nrow=2)
        writer.add_image("z_means", z_mean_imgs, epoch)
        writer.add_image("data_means", data_mean_imgs, epoch)

        means_np = prior.means.detach().cpu().numpy()
        fig = plt.figure(figsize=(8, 8))
        sns.heatmap(cdist(means_np, means_np))
        img_data = torch.tensor(np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep=''))
        img_data = torch.tensor(img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))).transpose(0, 2).transpose(1, 2)
        writer.add_image("mean_dists", img_data, epoch)

        for i in range(n_classes):
            writer.add_scalar("train_gmm/weight/{}".format(i), F.softmax(prior.weights)[i], epoch)

        for i in range(n_classes):
            writer.add_scalar("train_gmm/cov/{}".format(i), F.softplus(prior.inv_cov_stds[i])**2, epoch)

        for i in range(n_classes):
            writer.add_scalar("train_gmm/mean_dist_init/{}".format(i), torch.norm(prior.means[i]-means_init[i], 2), epoch)

        images = []
        for i in range(n_classes):
            images_cls = sample(net, loss_fn.prior, num_samples // n_classes,
                                      cls=i, device=device, sample_shape=img_shape)
            images.append(images_cls)
            images_cls_concat = torchvision.utils.make_grid(
                    images_cls, nrow=2, padding=2, pad_value=255)
            writer.add_image("samples/class_"+str(i), images_cls_concat)
        images = torch.cat(images)
        os.makedirs(os.path.join(ckptdir, 'samples'), exist_ok=True)
        images_concat = torchvision.utils.make_grid(images, nrow=num_samples //  n_classes , padding=2, pad_value=255)
        os.makedirs(ckptdir, exist_ok=True)
        torchvision.utils.save_image(images_concat, 
                                    os.path.join(ckptdir, 'samples/epoch_{}.png'.format(epoch)))
