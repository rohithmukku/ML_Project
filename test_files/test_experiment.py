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
from sklearn.metrics import auc, precision_recall_curve, roc_curve


f = open("details.txt","a")

parser = argparse.ArgumentParser(description='Run custom flow datasets for OOD')

parser.add_argument('--indata_size', type=int, default=5000)
parser.add_argument('--outdata_size', type=int, default=5000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--use_ldam', action='store_true')
parser.add_argument('--model_path')
parser.add_argument('--root_path')

parser.add_argument('--in_data', default="svhn")
parser.add_argument('--out_data', default="mnist")

args = parser.parse_args()

sys.path.append(os.path.join(args.root_path, 'flowgmm-public'))
data_dir = os.path.join(args.root_path, 'data')

indata_size = args.indata_size
outdata_size = args.outdata_size
batch_size = args.batch_size

in_data = args.in_data
out_data = args.out_data.split(',')

model_path = args.model_path

f.write(f"Indata: {in_data} Outdata: {','.join(out_data)}\n")

class SLDataset():
    def __init__(self, config: dict):
        self.config = config
        self.image_tensors = []
        self.labels = []

    def prepare(self, in_data='cifar', out_data=['mnist', 'svhn', 'fashionmnist'], indata_size=600, outdata_size=200):
        print(out_data)
        # Prepare OOD data
        for k in out_data:
            dataset = self.config[k]['dataset']
            transforms = self.config[k]['transforms']
            for i, (img, _) in enumerate(dataset):
                if i == outdata_size:
                    break
                img_tensor = transforms(img)
                self.image_tensors.append(img_tensor)
            self.labels += [0] * outdata_size

        # Prepare ID data
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


transform_to_three_channel = transforms.Lambda(lambda img: img.expand(3,*img.shape[1:]))

all_transforms = transforms.Compose([
                    # transforms.Grayscale(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Resize((32,32)),
                    transform_to_three_channel,
                ])

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


test_config = {}

test_config['svhn'] = {}
test_config['svhn']['dataset'] = svhn_test_dataset
test_config['svhn']['transforms'] = all_transforms

test_config['mnist'] = {}
test_config['mnist']['dataset'] = mnist_test_dataset
test_config['mnist']['transforms'] = all_transforms

test_config['kmnist'] = {}
test_config['kmnist']['dataset'] = kmnist_test_dataset
test_config['kmnist']['transforms'] = all_transforms

test_config['fashionmnist'] = {}
test_config['fashionmnist']['dataset'] = fashionmnist_test_dataset
test_config['fashionmnist']['transforms'] = all_transforms

test_config['cifar'] = {}
test_config['cifar']['dataset'] = cifar_test_dataset
test_config['cifar']['transforms'] = all_transforms

test_config['cifar100'] = {}
test_config['cifar100']['dataset'] = cifar100_test_dataset
test_config['cifar100']['transforms'] = all_transforms

test_config['food'] = {}
test_config['food']['dataset'] = food_test_dataset
test_config['food']['transforms'] = all_transforms

test_config['stl10'] = {}
test_config['stl10']['dataset'] = stl10_test_dataset
test_config['stl10']['transforms'] = all_transforms

test_config['flowers'] = {}
test_config['flowers']['dataset'] = flowers_test_dataset
test_config['flowers']['transforms'] = all_transforms

test_config['caltech'] = {}
test_config['caltech']['dataset'] = caltech_test_dataset
test_config['caltech']['transforms'] = all_transforms

test_config['german_sign'] = {}
test_config['german_sign']['dataset'] = german_sign_test_dataset
test_config['german_sign']['transforms'] = all_transforms

test_config['omniglot'] = {}
test_config['omniglot']['dataset'] = omniglot_test_dataset
test_config['omniglot']['transforms'] = all_transforms

test_ds = SLDataset(test_config)
test_ds.prepare(in_data=in_data, out_data=out_data, indata_size=indata_size, outdata_size=outdata_size)
testloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True, pin_memory=True)

device = 'cuda' if torch.cuda.is_available() and len([0]) > 0 else 'cpu'
img_shape = (3, 32, 32)
flow = 'ResidualFlow'
model_cfg = getattr(flow_ssl, flow)
net = model_cfg(in_channels=img_shape[0], num_classes=2)
net = net.flow
net = net.to(device)

def test(net, testloader, device, loss_fn):
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
#                 jaclogdet_meter.update(sldj.mean().item(), x.size(0))

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
    return all_ys, all_pred_labels


checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint['net'], strict=False)
test_epoch = checkpoint['epoch']

means = checkpoint['means']
prior = SSLGaussMixture(means, device=device)
loss_fn = FlowLoss(prior)

all_ys, all_pred_labels = test(net, testloader, "cuda", loss_fn)

def auroc(preds, labels, pos_label=1):
    """Calculate and return the area under the ROC curve using unthresholded predictions on the data and a binary true label.

    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    pos_label: label of the positive class (1 by default)
    """
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)
    f.write(f"False positive Rate: {fpr}\n") 
    f.write(f"True positive Rate:, {tpr}\n") 
    score = auc(fpr, tpr)
    f.write(f"AUCROC: {score}\n")

auroc(all_pred_labels, all_ys)
f.close()
