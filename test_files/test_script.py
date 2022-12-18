import os

experiments = {}
#experiments["svhn"] = ["cifar","mnist"]
#experiments["fashionmnist"] = "mnist,kmnist,omniglot"
experiments["mnist"] = ["omniglot","fashionmnist"]
#experiment["cifar"] = ["svhn","flowers","food"]

paths = {}
paths["svhn"] = "svhn_mnist,fashionmnist,cifar100,food,flowers,german_sign,caltech_0.1/48.pt"
paths["cifar"] = "svhn_mnist,fashionmnist,cifar100,food,flowers,german_sign,caltech_0.1/48.pt"
paths["fashionmnist"] = "fashionmnist_kmnist,svhn,mnist,cifar,cifar100,food,flowers,german_sign_0.1/28.pt"
paths["mnist"] = "mnist_kmnist,svhn,food,flowers,german_sign,svhn,caltech_0.1/26.pt"
for in_data in experiments:
    for out_data in experiments[in_data]:
        command = f"python -W ignore test_experiment.py --root_path=/scratch/gg2501/ML_Project/ --model_path={paths[in_data]} --in_data={in_data}  --out_data={out_data} --indata_size=9000 --outdata_size=9000"
        os.system(command)
