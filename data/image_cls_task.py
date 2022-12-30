# encoding=utf8
import torchvision
import os

from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

abs_dir_path = os.path.dirname(os.path.abspath(__file__))

def get_CIFAR10_data_loader(args, device_id):

    # 1. Transform
    norm_mean = (0.5, 0.5, 0.5)
    norm_std = (0.5, 0.5, 0.5)

    train_transform = Compose([RandomResizedCrop(args.img_size), ToTensor(), Normalize(norm_mean, norm_std)])
    val_transform = Compose([Resize(args.img_size), ToTensor(), Normalize(norm_mean, norm_std)])

    # 2. Dataset
    root_path = os.path.join(abs_dir_path, 'cifar10')
    train_dataset = torchvision.datasets.CIFAR10(root=root_path, train=True, download=True, transform=train_transform)
    val_dataset = torchvision.datasets.CIFAR10(root=root_path, train=False, download=True, transform=val_transform)

    # 3. Sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=device_id)
    val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=device_id)

    # 4. Loader
    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=args.test_batch_size,
        num_workers=2,
        pin_memory=True
    )
    return train_loader, val_loader

def get_ILSVRC2012_data_loader(args, device_id):

    # 1. Transform
    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)
    train_transform = Compose(
        [
            RandomResizedCrop(args.img_size),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(norm_mean, norm_std)
        ]
    )
    val_transform = Compose(
        [
            Resize(args.img_size),
            CenterCrop(args.img_size),
            ToTensor(),
            Normalize(norm_mean, norm_std)
        ]
    )

    # 2. Dataset
    root_path = os.path.join(abs_dir_path, 'ILSVRC2012')
    train_dataset = torchvision.datasets.ImageNet(
        root=root_path,
        split='train',
        transform=train_transform
    )
    val_dataset = torchvision.datasets.ImageNet(
        root=root_path,
        split='val',
        transform=val_transform
    )

    # 3. Sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=device_id)
    val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=device_id)

    # 4. Loader
    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=args.test_batch_size,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader
