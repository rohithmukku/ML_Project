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

from datasets import TestDataset
from datasets import get_test_config
from dataloaders import get_test_dataloader
from utils import StatsMeter

device = 'cuda' if torch.cuda.is_available() and len([0]) > 0 else 'cpu'

def test(net, testloader, device, loss_fn, writer=None):
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
    in_dist_log_probs = []
    out_dist_log_probs = []
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, y in testloader:
                all_xs.append(x.data.numpy())
                all_ys.append(y.data.numpy())
                in_dist_mask = (y == 1)
                out_dist_mask = (y == 0)
                x = x.to(device)
                y = y.to(device)
                z = net(x)
                all_zs.append(z.cpu().data.numpy())
                sldj = net.logdet()
                loss = loss_fn(z, y=y, sldj=sldj)
                in_dist_log_probs.append(loss_fn.prior.class_logits(z[in_dist_mask]).cpu().data.numpy())
                out_dist_log_probs.append(loss_fn.prior.class_logits(z[out_dist_mask]).cpu().data.numpy())
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
    in_dist_log_probs = np.concatenate((in_dist_log_probs))
    out_dist_log_probs = np.concatenate((out_dist_log_probs))

    if writer:
        writer.add_histogram('Test In-Distribution (class 0)', in_dist_log_probs[:, 0])
        writer.add_histogram('Test In-Distribution (class 1)', in_dist_log_probs[:, 1])
        writer.add_histogram('Test Out-Distribution (class 0)', out_dist_log_probs[:, 0])
        writer.add_histogram('Test Out-Distribution (class 1)', out_dist_log_probs[:, 1])

    return stats_meter.calculate_stats(all_pred_labels, all_ys), in_dist_log_probs, out_dist_log_probs

def test_experiments(data_dir, in_data, out_data_list, data_size, batch_size, net, means, writer=None):
    prior = SSLGaussMixture(means, device=device)
    loss_fn = FlowLoss(prior)
    in_log_probs_list = []
    out_log_probs_list = []
    for out_data in out_data_list:
        test_config = get_test_config(in_data, [out_data], data_dir)
        test_dataset = TestDataset(test_config)
        test_dataset.prepare(indata_size=data_size, outdata_size=data_size)
        testloader = get_test_dataloader(test_dataset, batch_size)
        (auroc, auprc, fpr95), in_dist_log_probs, out_dist_log_probs = test(net, testloader, "cuda", loss_fn, writer)
        in_log_probs_list.append(in_dist_log_probs)
        out_log_probs_list.append(out_dist_log_probs)
        print(f"{in_data} (in-distribution) vs {out_data} (out-distribution)")
        print(f"AUROC: f{auroc}\tAURPC: {auprc}\tFPR@95: {fpr95}")

    return in_log_probs_list, out_log_probs_list
