from __future__ import annotations

from pathlib import Path

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


def make_datasets(
    name: str,
    root: str,
    augmentation_types: str = "default",
    size: int | None = None,
    use_unlabeled_stl10: bool = False,
):
    name = name.lower()
    if size is None:
        size = 96 if name == "stl10" else 32

    train_base = get_base_dataset(name, root, "train")
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
