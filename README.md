# ML_Project
ML Course Project

Report can be found [here](report/ML_report.pdf)

## Experiment - 3

Results for MNIST family of datasets.

| InD | Train OOD | Test OOD | AUROC | AUPRC | FPR@95 |
| ------ | ------ | ------ | ------ | ------ | ------ |
| MNIST | KMNIST<br>FMNIST | FMNIST<br>KMNIST<br>Omniglot | 0.99<br>0.98<br>0.99 | 0.99<br>0.98<br>0.99 | 0.002<br>0.02<br>0.0 |
| MNIST | FMNIST<br>Omniglot | FMNIST<br>KMNIST<br>Omniglot | 0.98<br>0.89<br>0.99 | 0.98<br>0.91<br>0.99 | 0.02<br>0.2<br>0.007 |
| MNIST | Omniglot<br>KMNIST | FMNIST<br>KMNIST<br>Omniglot | 0.91<br>0.90<br>0.99 | 0.92<br>0.91<br>0.99 | 0.15<br>0.18<br>0.004 |
| FMNIST | MNIST<br>KMNIST | MNIST<br>KMNIST<br>Omniglot | 0.52<br>0.5<br>0.5 | 0.75<br>0.75<br>0.75 | 0.90<br>0.94<br>0.94 |
| FMNIST | KMNIST<br>Omniglot | MNIST<br>KMNIST<br>Omniglot | 0.74<br>0.76<br>0.99 | 0.83<br>0.84<br>0.99 | 0.49<br>0.44<br>0.0 |
| FMNIST | Omniglot<br>MNIST | MNIST<br>KMNIST<br>Omniglot | 0.73<br>0.52<br>0.99 | 0.82<br>0.75<br>0.99 | 0.51<br>0.91<br>0.0 |

Results for datasets similar to CIFAR-10.

| InD | Train OOD | Test OOD | AUROC | AUPRC | FPR@95 |
| ------ | ------ | ------ | ------ | ------ | ------ |
| CIFAR10 | CIFAR100<br>SVHN | CIFAR100<br>SVHN<br>STL10 | 0.51<br>0.86<br>0.53 | 0.74<br>0.89<br>0.74 | 0.92<br>0.23<br>0.89 |
| CIFAR10 | SVHN<br>STL10 | CIFAR100<br>SVHN<br>STL10 | 0.52<br>0.65<br>0.5 | 0.59<br>0.82<br>0.57 | 0.94<br>0.92<br>0.95 |
| CIFAR10 | STL10<br>CIFAR100 | CIFAR100<br>SVHN<br>STL10 | 0.52<br>0.61<br>0.50 | 0.60<br>0.71<br>0.58 | 0.94<br>0.93<br>0.94 |
| CIFAR100 | CIFAR10<br>SVHN | CIFAR10<br>SVHN<br>STL10 | 0.5<br>0.5<br>0.5 | 0.74<br>0.74<br>0.74 | 0.94<br>0.94<br>0.95 |
| CIFAR100 | SVHN<br>STL10 | CIFAR10<br>SVHN<br>STL10 | 0.52<br>0.72<br>0.27 | 0.63<br>0.85<br>0.53 | 0.94<br>0.90<br>0.99 |
| CIFAR100 | STL10<br>CIFAR10 | CIFAR10<br>SVHN<br>STL10 | 0.51<br>0.59<br>0.34 | 0.59<br>0.69<br>0.49 | 0.94<br>0.93<br>0.97 |
| SVHN | CIFAR10<br>CIFAR100 | CIFAR10<br>CIFAR100<br>STL10 | 0.91<br>0.89<br>0.99 | 0.92<br>0.91<br>0.99 | 0.17<br>0.20<br>0.003 |
| SVHN | CIFAR100<br>STL10 | CIFAR10<br>CIFAR100<br>STL10 | 0.83<br>0.82<br>0.83 | 0.90<br>0.89<br>0.91 | 0.85<br>0.85<br>0.84 |
| SVHN | STL10<br>CIFAR10 | CIFAR10<br>CIFAR100<br>STL10 | 0.91<br>0.90<br>0.93 | 0.94<br>0.93<br>0.97 | 0.61<br>0.62<br>0.60 |