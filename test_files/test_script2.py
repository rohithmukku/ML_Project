import os

paths = ["svhn_fashionmnist,cifar100,food,flowers,german_sign,caltech_0.001/40.pt","svhn_fashionmnist,cifar100,food,flowers,german_sign,caltech_0.01/40.pt"
 ,"svhn_kmnist,fashionmnist,cifar100,food,flowers,german_sign,caltech_0.1/18.pt", "svhn_mnist,fashionmnist,cifar100,food,flowers,german_sign,caltech_0.1/40.pt"]

for path in paths:
    print(path.split('_')[-1])
    command = f"python -W ignore test_experiment.py --root_path=/scratch/gg2501/ML_Project/ --model_path={path} --in_data=svhn  --out_data=cifar --indata_size=2000 --outdata_size=2000"
    os.system(command)
