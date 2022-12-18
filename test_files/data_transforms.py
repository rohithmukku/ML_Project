import torchvision.transforms as transforms

# convert all images to the same size as the in-data distribution
# in most cases it will be 3X32X32
transform_to_three_channel = transforms.Lambda(lambda img: img.expand(3,*img.shape[1:]))

train_transforms = {}
test_transforms = {}

# Single channel
train_transforms['mnist'] = transforms.Compose([
                   transforms.RandomCrop(32, padding=4),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Resize((32,32)),
                   transform_to_three_channel
                ])

train_transforms['kmnist'] = transforms.Compose([
                   transforms.RandomCrop(32, padding=4),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Resize((32,32)),
                   transform_to_three_channel
                ])

train_transforms['fmnist'] = transforms.Compose([
                   transforms.RandomCrop(32, padding=4),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Resize((32,32)),
                   transform_to_three_channel
                ])

train_transforms['omniglot'] = transforms.Compose([
                   transforms.RandomCrop(32, padding=4),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Resize((32,32)),
                   transform_to_three_channel
                ])

# Three channel
train_transforms['svhn'] = transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Resize((32,32)),
                        ])

train_transforms['cifar10'] = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    # transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Resize((32,32)),
                ])

train_transforms['cifar100'] = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    # transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Resize((32,32)),
                ])

train_transforms['celeba'] = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    # transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Resize((32,32)),
                ])

train_transforms['stl10'] = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    # transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Resize((32,32)),
                ])

# Test Data

# Single channel
test_transforms['mnist'] = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Resize((32,32)),
                   transform_to_three_channel
                ])

test_transforms['kmnist'] = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Resize((32,32)),
                   transform_to_three_channel
                ])

test_transforms['fmnist'] = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Resize((32,32)),
                   transform_to_three_channel
                ])

test_transforms['omniglot'] = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Resize((32,32)),
                   transform_to_three_channel
                ])

# Three channel
test_transforms['svhn'] = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Resize((32,32)),
                ])

test_transforms['cifar10'] = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Resize((32,32)),
                ])

test_transforms['cifar100'] = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Resize((32,32)),
                ])

test_transforms['celeba'] = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Resize((32,32)),
                ])

test_transforms['stl10'] = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Resize((32,32)),
                ])
