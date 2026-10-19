from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np

from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .augmentations import IMAGENET_MEAN, IMAGENET_STD, TwoCropsTransform, build_transform


class SSLDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label


def get_base_dataset(name: str, root: str, split: str, download: bool = True):
    name = name.lower()
    root = str(Path(root).expanduser())
    if name == "cifar10":
        return datasets.CIFAR10(root, train=(split == "train"), download=download)
    if name == "cifar100":
        return datasets.CIFAR100(root, train=(split == "train"), download=download)
    if name == "stl10":
        return datasets.STL10(root, split=split, download=download)
    raise ValueError(f"Unsupported dataset: {name}")


def data_fingerprint(base):
    """Hash CIFAR images in logical order, matching the recorded experiment splits."""
    digest = hashlib.sha256()
    for start in range(0, len(base.data), 1000):
        digest.update(memoryview(np.ascontiguousarray(base.data[start:start + 1000])))
    digest.update(np.asarray(base.targets, dtype='<i8').tobytes())
    return digest.hexdigest()


def validate_split(split, base, name):
    if name not in {'cifar10', 'cifar100'} or split.get('dataset') != name:
        raise ValueError('Validation splits require the matching CIFAR dataset')
    labels = np.asarray(base.targets)
    train, val = np.asarray(split['train']), np.asarray(split['validation'])
    if train.dtype.kind not in 'iu' or val.dtype.kind not in 'iu':
        raise ValueError('Split indices must be integers')
    if (len(labels), len(train), len(val)) != (50000, 45000, 5000):
        raise ValueError('Expected a 45000/5000 partition of 50000 images')
    if not np.array_equal(np.sort(np.concatenate([train, val])), np.arange(50000)):
        raise ValueError('Split contains duplicate, missing or out-of-range indices')
    classes = 10 if name == 'cifar10' else 100
    for ids in (train, val):
        if not np.array_equal(np.bincount(labels[ids], minlength=classes),
                              np.full(classes, len(ids) // classes)):
            raise ValueError('Split is not stratified')
    if split['data_sha256'] != data_fingerprint(base):
        raise ValueError('Dataset differs from the saved split')


def create_split(name, base, seed=0):
    if name not in {'cifar10', 'cifar100'}:
        raise ValueError('Validation splitting is supported for CIFAR only')
    labels = np.asarray(base.targets)
    rng = np.random.default_rng(seed)
    train, val = [], []
    classes = 10 if name == 'cifar10' else 100
    for label in range(classes):
        ids = rng.permutation(np.flatnonzero(labels == label)).tolist()
        val.extend(ids[:5000 // classes])
        train.extend(ids[5000 // classes:])
    split = dict(dataset=name, seed=seed, train=train, validation=val,
                 data_sha256=data_fingerprint(base))
    validate_split(split, base, name)
    return split


def split_digest(split):
    return hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()


def make_datasets(
    name: str,
    root: str,
    augmentation_types: str = "default",
    size: int | None = None,
    use_unlabeled_stl10: bool = False,
    split: dict | None = None,
):
    name = name.lower()
    if size is None:
        size = 96 if name == "stl10" else 32

    train_base = get_base_dataset(name, root, "train")
    if split is not None:
        if use_unlabeled_stl10:
            raise ValueError('Unlabeled STL-10 cannot use a CIFAR validation split')
        validate_split(split, train_base, name)
        from torch.utils.data import Subset
        train = Subset(train_base, split['train'])
        val = Subset(train_base, split['validation'])
        eval_tf = transforms.Compose([
            transforms.Resize((size, size)), transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return (SSLDataset(train, TwoCropsTransform(build_transform(size, augmentation_types))),
                SSLDataset(train, eval_tf), SSLDataset(val, eval_tf))
    ssl_base = train_base
    if name == "stl10" and use_unlabeled_stl10:
        unlabeled = get_base_dataset(name, root, "unlabeled")
        ssl_base = _ConcatImageDatasets(train_base, unlabeled)

    ssl = SSLDataset(
        ssl_base,
        TwoCropsTransform(build_transform(size, augmentation_types)),
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_eval = get_base_dataset(name, root, "train")
    test_eval = get_base_dataset(name, root, "test")
    train_eval.transform = eval_tf
    test_eval.transform = eval_tf
    return ssl, train_eval, test_eval


class _ConcatImageDatasets(Dataset):
    def __init__(self, *datasets_):
        self.datasets = datasets_
        self.offsets = []
        total = 0
        for ds in datasets_:
            self.offsets.append(total)
            total += len(ds)
        self.total = total

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        for ds, offset in reversed(list(zip(self.datasets, self.offsets))):
            if index >= offset:
                image, label = ds[index - offset]
                return image, -1 if label is None else label
        raise IndexError(index)
