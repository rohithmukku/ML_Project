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

from datasets import TestDataset
from datasets import get_test_config
from dataloaders import get_test_dataloader
from utils import StatsMeter


parser = argparse.ArgumentParser(description='Run custom flow datasets for OOD')

parser.add_argument('--data_size', type=int, default=5000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--use_ldam', action='store_true')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--root_path', type=str, default='/scratch/rm5708/ml/ML_Project')

parser.add_argument('--in_data', default="svhn")
parser.add_argument('--out_data', default="mnist")

args = parser.parse_args()

sys.path.append(os.path.join(args.root_path, 'flowgmm-public'))
data_dir = os.path.join(args.root_path, 'data')

indata_size = args.data_size
outdata_size = args.data_size
batch_size = args.batch_size

in_data = args.in_data
out_data_list = args.out_data.split(',')

model_path = args.model_path

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
    stats_meter = StatsMeter()
    all_pred_labels = []
    # all_scores = []
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

                preds = loss_fn.prior.classify(z.reshape((len(z), -1)))
                preds = preds.reshape(y.shape)
                # scores = scores.reshape(y.shape)
                all_pred_labels.append(preds.cpu().data.numpy())
                # all_scores.append(scores.cpu().data.numpy())
                acc = (preds == y).float().mean().item()
                acc_meter.update(acc, x.size(0))

                progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=optim_util.bits_per_dim(x, loss_meter.avg),
                                     acc=acc_meter.avg)
                progress_bar.update(x.size(0))

    all_pred_labels = np.hstack(all_pred_labels)
    # all_scores = np.hstack(all_scores)
    all_xs = np.vstack(all_xs)
    all_zs = np.vstack(all_zs)
    all_ys = np.hstack(all_ys)

    return stats_meter.calculate_stats(all_pred_labels, all_ys)
    

checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint['net'], strict=False)
test_epoch = checkpoint['epoch']

means = checkpoint['means']
prior = SSLGaussMixture(means, score_factor=800, device=device)
loss_fn = FlowLoss(prior)

for out_data in out_data_list:
    test_config = get_test_config(in_data, [out_data], data_dir)
    test_dataset = TestDataset(test_config)
    test_dataset.prepare(indata_size=indata_size, outdata_size=outdata_size)
    testloader = get_test_dataloader(test_dataset, batch_size)
    auroc, auprc, fpr95 = test(net, testloader, "cuda", loss_fn)
    print(f"{in_data} (in-distribution) vs {out_data} (out-distribution)")
    print(f"AUROC: f{auroc}\tAURPC: {auprc}\tFPR@95: {fpr95}")

