from __future__ import annotations

import random

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GaussianBlur:
    def __init__(self, kernel_size: int, sigma=(0.1, 2.0), p: float = 0.5):
        self.blur = transforms.GaussianBlur(kernel_size, sigma=sigma)
        self.p = float(p)

    def __call__(self, image):
        return self.blur(image) if random.random() < self.p else image


def _blur_kernel(size: int) -> int:
    # Lightly's SimCLR transform uses a Gaussian kernel derived from roughly
    # 10% of the image size. torchvision requires an odd positive integer.
    kernel_size = int(0.1 * size)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    return max(3, kernel_size)


def build_transform(size: int, augmentation_types: str = "default", color_strength: float = 0.5):
    """Build the augmentation combinations exposed by the original release.

    The paper names five augmentation families:
      a: random resized crop
      b: Gaussian blur
      c: color dropping (grayscale)
      d: color distortion
      e: horizontal flip

    ``default`` matches the original repository's default Lightly collate
    configuration: SimCLR-style crop/flip/color-distortion/grayscale with
    Gaussian blur disabled for the 32x32 setting.
    """
    aliases = {
        "default": "acde",
        "color": "cd",
    }
    augmentation_types = aliases.get(augmentation_types, augmentation_types)
    if not augmentation_types or not set(augmentation_types) <= set("abcde"):
        raise ValueError("augmentation_types must be one of default/color or a combination of a,b,c,d,e")

    ops = [transforms.RandomResizedCrop(size, scale=(0.08, 1.0))]
    ops.append(transforms.RandomHorizontalFlip(p=0.5)) if "e" in augmentation_types else None
    if "d" in augmentation_types:
        strength = float(color_strength)
        ops.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        0.8 * strength,
                        0.8 * strength,
                        0.8 * strength,
                        0.2 * strength,
                    )
                ],
                p=0.8,
            )
        )
    if "c" in augmentation_types:
        ops.append(transforms.RandomGrayscale(p=0.2))
    if "b" in augmentation_types:
        ops.append(GaussianBlur(_blur_kernel(size), p=0.5))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


class TwoCropsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        return self.transform(image), self.transform(image)
