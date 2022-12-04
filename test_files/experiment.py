#!/usr/bin/env python
import sys
import os
import flow_ssl
from torchvision.datasets import SVHN, MNIST, FashionMNIST, CIFAR10

import torchvision.transforms as transforms
import torch
from torch.utils.data.sampler import Sampler
import numpy as np
import itertools
import random
import argparse


from experiments.train_flows.utils import train_utils, optim_util

from scipy.spatial.distance import cdist

from experiments.train_flows.utils import shell_util

from tqdm.notebook import tqdm
import torch.nn as nn
import math
import torch.nn.init as init
import torch.nn.functional as F

from flow_ssl.distributions import SSLGaussMixture
from flow_ssl import FlowLoss
from tensorboardX import SummaryWriter

root = "/mnt/c/Users/gyane/Projects/"

sys.path.append(os.path.join(root, 'flowgmm-public'))

data_dir = os.path.join(root, 'data')

parser = argparse.ArgumentParser(description='Run custom flow datasets for OOD')

parser.add_argument('--indata_size', type=int, default=50000)
parser.add_argument('--outdata_size', type=int, default=18000)
parser.add_argument('--label_ratio', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--in_data', default="mnist")
parser.add_argument('--out_data', default="svhn,fashionmnist,cifar")

parser.add_argument('--test_out_data', default="svhn,fashionmnist,cifar") #could be extra as well

parser.add_argument('--test_indata_size', type=int, default=10000)
parser.add_argument('--test_outdata_size', type=int, default=3000)

args = parser.parse_args()


indata_size = args.indata_size
outdata_size = args.outdata_size
label_ratio = args.label_ratio
batch_size = args.batch_size

in_data = args.in_data
out_data = set(args.out_data.split(','))

test_out_data = set(args.test_out_data.split(','))
test_indata_size = args.test_indata_size
test_outdata_size = args.test_outdata_size

print(args)

class Dataset():
    def __init__(self, config: dict):
        self.config = config
        self.labeled_ids = []
        self.unlabeled_ids = []
        self.image_tensors = []
        self.labels = []
    
    def prepare(self, in_data='mnist', out_data=None, indata_size=50000, outdata_size=18000, label_ratio=0.2):
        print(in_data)
        print(out_data)
        # Prepare OOD data
        for k in out_data:
            dataset = config[k]['dataset']
            transforms = config[k]['transforms']
            start_id = len(self.labels)
            end_id = start_id + int(label_ratio * outdata_size)
            for i, (img, _) in enumerate(dataset):
                if i == outdata_size:
                    break
                img_tensor = transforms(img)
                self.image_tensors.append(img_tensor)
            self.labels += [0] * (int(label_ratio * outdata_size))
            self.labels += [-1] * (int((1 - label_ratio) * outdata_size))
            self.labeled_ids += range(start_id, end_id)
            self.unlabeled_ids += range(end_id, len(self.labels))
        
        # print(self.labeled_ids)
        
        # Prepare ID data
        dataset = config[in_data]['dataset']
        transforms = config[in_data]['transforms']
        start_id = len(self.labels)
        end_id = start_id + int(label_ratio * indata_size)
        for i, (img, _) in enumerate(dataset):
            if i == indata_size:
                break
            img_tensor = transforms(img)
            self.image_tensors.append(img_tensor)
        self.labels += [1] * (int(label_ratio * indata_size))
        self.labels += [-1] * (int((1 - label_ratio) * indata_size))
        self.labeled_ids += range(start_id, end_id)
        self.unlabeled_ids += range(end_id, len(self.labels))
        
        random.shuffle(self.labeled_ids)
        random.shuffle(self.unlabeled_ids)
    
    def __len__(self):
        return len(self.image_tensors)
    
    def __getitem__(self, idx):
        labeled_id = self.labeled_ids[idx % len(self.labeled_ids)]
        unlabeled_id = self.unlabeled_ids[idx % len(self.unlabeled_ids)]
        return self.image_tensors[labeled_id], self.image_tensors[unlabeled_id], self.labels[labeled_id]


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

        print(len(self.labeled_idx), self.labeled_batch_size)
        assert len(self.labeled_idx) >= self.labeled_batch_size >= 0
        assert len(self.unlabeled_idx) >= self.unlabeled_batch_size >= 0

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




svhn_transforms = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Resize((32,32))
                ])

mnist_transforms = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Resize((32,32))
                ])

fashionmnist_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((32,32))
                ])

cifar_transforms = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Resize((32,32))
                ])

svhn_dataset = SVHN(root=data_dir, split='train', download=True)
mnist_dataset = MNIST(root=data_dir, download=True)
fashionmnist_dataset = FashionMNIST(root=data_dir, download=True)
cifar_dataset = CIFAR10(root=data_dir, download=True)

svhn_test_dataset = SVHN(root=data_dir, split='test', download=True)
mnist_test_dataset = MNIST(root=data_dir, train=False, download=True)
fashionmnist_test_dataset = FashionMNIST(root=data_dir, train=False, download=True)
cifar_test_dataset = CIFAR10(root=data_dir, train=False, download=True)


config = {}

config['svhn'] = {}
config['svhn']['dataset'] = svhn_dataset
config['svhn']['transforms'] = svhn_transforms

config['mnist'] = {}
config['mnist']['dataset'] = mnist_dataset
config['mnist']['transforms'] = mnist_transforms

config['fashionmnist'] = {}
config['fashionmnist']['dataset'] = fashionmnist_dataset
config['fashionmnist']['transforms'] = fashionmnist_transforms

config['cifar'] = {}
config['cifar']['dataset'] = cifar_dataset
config['cifar']['transforms'] = cifar_transforms


test_config = {}

test_config['svhn'] = {}
test_config['svhn']['dataset'] = svhn_test_dataset
test_config['svhn']['transforms'] = svhn_transforms

test_config['mnist'] = {}
test_config['mnist']['dataset'] = mnist_test_dataset
test_config['mnist']['transforms'] = mnist_transforms

test_config['fashionmnist'] = {}
test_config['fashionmnist']['dataset'] = fashionmnist_test_dataset
test_config['fashionmnist']['transforms'] = fashionmnist_transforms

test_config['cifar'] = {}
test_config['cifar']['dataset'] = cifar_test_dataset
test_config['cifar']['transforms'] = cifar_transforms


ds = Dataset(config)
ds.prepare(in_data=in_data, out_data=out_data, indata_size=indata_size, outdata_size=outdata_size, label_ratio=label_ratio)

test_ds = Dataset(test_config)
test_ds.prepare(in_data=in_data, out_data=test_out_data, indata_size=test_indata_size, outdata_size=test_outdata_size, label_ratio=1)


train_batch_sampler = LabeledUnlabeledBatchSampler(ds.labeled_ids, ds.unlabeled_ids, batch_size//2, batch_size//2)
test_batch_sampler = LabeledUnlabeledBatchSampler(test_ds.labeled_ids, test_ds.unlabeled_ids, batch_size//2, 0)

trainloader = torch.utils.data.DataLoader(ds, batch_sampler=train_batch_sampler, pin_memory=True)
testloader = torch.utils.data.DataLoader(test_ds, batch_sampler=test_batch_sampler, pin_memory=True)


for batch in trainloader:
    print(batch[0].get_device())
    print(batch[1].shape)
    print(batch[2].shape)
    break


img_shape = (1, 32, 32)
flow = 'MNISTResidualFlow'
model_cfg = getattr(flow_ssl, flow)
net = model_cfg(in_channels=img_shape[0], num_classes=2)


if flow in ["iCNN3d", "iResnetProper","SmallResidualFlow","ResidualFlow","MNISTResidualFlow"]:
    net = net.flow


means = 'random'
means_r = 1.0
cov_std = 1.0
img_shape = (1, 32, 32)
device = 'cuda'
n_classes = 2

net = net.to(device)
r = means_r
cov_std = torch.ones((n_classes)) * cov_std
cov_std = cov_std.to(device)
means = train_utils.get_means(means, num_means=n_classes, r=means_r, trainloader=trainloader, 
                        shape=img_shape, device=device, net=net)
means_init = means.clone().detach()

print("Means:", means)
print("Cov std:", cov_std)
means_np = means.cpu().numpy()
print("Pairwise dists:", cdist(means_np, means_np))

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


from experiments.train_flows.utils import norm_util
import torch.optim as optim

param_groups = norm_util.get_param_groups(net, 0.0, norm_suffix='weight_g')

optimizer = optim.Adam(param_groups, lr=1e-3)
opt_gmm = optim.Adam([prior.means, prior.weights, prior.inv_cov_stds], lr=1e-4, weight_decay=0.)





writer = SummaryWriter(log_dir='./')
device = 'cuda' if torch.cuda.is_available() and len([0]) > 0 else 'cpu'
start_epoch = 0


total_labels = len(ds.labeled_ids)


def train(epoch, net, trainloader, device, optimizer, opt_gmm, loss_fn,
          label_weight, max_grad_norm, consistency_weight,
          writer, use_unlab=True,  acc_train_all_labels=False,
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
    with tqdm(total=total_labels) as progress_bar:
        for x1, x2, y in trainloader:

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
            loss_nll = F.cross_entropy(logits_labeled, y_labeled)

            if use_unlab:
                loss_unsup = loss_fn(z1, sldj=sldj)
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
    writer.add_scalar("train/bpd", utils.bits_per_dim(x1, loss_unsup_meter.avg), epoch)
    writer.add_scalar("train/loss_consistency", loss_consistency_meter.avg, epoch)



NO_LABEL = -1
schedule = None
n_epochs = 10
lr = 5e-4
lr_gmm = 1e-4
consistency_weight = 1.0
consistency_rampup = 1
label_weight = 1.0
max_grad_norm = 100.0
save_freq = 2
ckptdir = './'
eval_freq = 1
confusion = True
num_samples = 50

def linear_rampup(final_value, epoch, num_epochs, start_epoch=0):
    t = (epoch - start_epoch + 1) / num_epochs
    if t > 1:
        t = 1.
    return t * final_value

for epoch in range(start_epoch, n_epochs):
    cons_weight = linear_rampup(consistency_weight, epoch, consistency_rampup, start_epoch)
    
    writer.add_scalar("hypers/learning_rate", lr, epoch)
    writer.add_scalar("hypers/learning_rate_gmm", lr_gmm, epoch)
    writer.add_scalar("hypers/consistency_weight", cons_weight, epoch)

    train(epoch, net, trainloader, device, optimizer, opt_gmm, loss_fn,
          label_weight, max_grad_norm, cons_weight,
          writer, use_unlab=True)

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
        train_utils.test_classifier(epoch, net, testloader, device, loss_fn, writer, confusion=confusion)
        # if args.swa:
        #     optimizer.swap_swa_sgd() 
        #     print("updating bn")
        #     SWA.bn_update(bn_loader, net)
        #     utils.test_classifier(epoch, net, testloader, device, loss_fn, 
        #             writer, postfix="_swa")

        z_means = prior.means
        data_means = net.module.inverse(z_means)
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
            images_cls = utils.sample(net, loss_fn.prior, num_samples // n_classes,
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


s = iter(trainloader)