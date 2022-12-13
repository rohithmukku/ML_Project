import os

experiments = {}
experiments["svhn"] = "omniglot,mnist,german_sign"
experiments["fashionmnist"] = "mnist,kmnist,omniglot"
experiments["mnist"] = "kmnist,omniglot,svhn"
#experiment["cifar"] = ["svhn","flowers","food"]

paths = {}
paths["svhn"] = "svhn_mnist,fashionmnist,cifar100,food,flowers,german_sign,caltech_0.1/48.pt"
paths["cifar"] = "svhn_mnist,fashionmnist,cifar100,food,flowers,german_sign,caltech_0.1/48.pt"
paths["fashionmnist"] = "fashionmnist_kmnist,svhn,mnist,cifar,cifar100,food,flowers,german_sign_0.1/28.pt"
paths["mnist"] = "mnist_fashionmnist,cifar100,food,flowers,german_sign,caltech,stl10_0.1/8.pt"

for experiment in experiments:
    print(experiment, experiments[experiment])
    command = f"python -W ignore test_experiment.py --root_path=/scratch/gg2501/ML_Project/ --model_path={paths[experiment]} --in_data={experiment}  --out_data={experiments[experiment]} --indata_size=9000 --outdata_size=3000"
    os.system(command)
