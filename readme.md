# AMCL (v2.0)

*This release removes outdated dependencies from AMCL and updates the implementation and training pipeline for improved compatibility and stability.*


## Structure

```text
cl/
├── amcl/
│   ├── augmentations.py       # original-release augmentation families
│   ├── backbones.py           # CIFAR/STL ResNet-18 stem
│   ├── datasets.py            # CIFAR-10/100 and STL-10
│   ├── evaluation.py          # kNN evaluation
│   ├── losses.py              # baseline + AMCL objectives
│   ├── models.py              # SimCLR/MoCo/SimSiam/Barlow Twins
│   └── train.py               # training loop
├── scripts/
│   ├── smoke_test.py          # forward/backward checks for all methods
│   └── training_smoke.py      # multi-step optimizer checks for all methods
├── tests/
│   └── test_amcl.py           # numerical/equation regression tests
├── eval_linear.py              # frozen-encoder linear evaluation
├── train.py                    # CLI entry point
├── requirements.txt
└── pyproject.toml
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For CUDA, install a PyTorch/torchvision build appropriate for the target CUDA runtime first, then install this package.

## Verification before release

The repository includes two levels of automated verification.

### Unit tests

```bash
pytest -q
```

These tests cover:

- the stationary point of $\Omega(\tau)$;
- temperature bounds and inverse-sigmoid direction;
- finite baseline losses;
- AMCL NT-Xent with float16 inputs, verifying the loss is evaluated in float32;
- the SimCLR negative set containing $2B-2$ negatives;
- AMCL MoCo queue loss and gradients;
- SimSiam and Barlow Twins baseline/AMCL losses;
- Barlow Twins fixed-batch-size enforcement.

### End-to-end model smoke test

```bash
python scripts/smoke_test.py
```

This runs forward, loss, backward, and parameter-gradient checks for both baseline and AMCL versions of SimCLR, MoCo, SimSiam, and Barlow Twins. The MoCo path also verifies that the queue can be updated after backpropagation.

### Multi-step optimizer test

```bash
python scripts/training_smoke.py
```

This runs three optimizer steps for every released training path and checks loss, gradients, and parameters for NaN/Inf after every step.

For the complete CLI orchestration and checkpoint-write path, run `python scripts/train_loop_smoke.py`; this is separate because it exercises four full ResNet training loops.

The smoke tests use small synthetic batches so they remain fast. They are numerical/implementation tests.

## Training

The top-level command is:

```bash
python train.py --help
```

The default dataset preprocessing is consistent with the original training script's Lightly SimCLR configuration: ImageNet normalization, random resized crop, horizontal flip, color distortion and grayscale, with blur disabled in the `default` configuration.

### SimCLR + AMCL

```bash
python train.py \
  --dataset cifar10 \
  --method simclr \
  --amcl \
  --epochs 1000 \
  --batch-size 512 \
  --heads 3 \
  --topk 100 \
  --beta 1e-4 \
  --aug default \
  --output ./runs/cifar10_simclr_amcl
```

### MoCo + AMCL

```bash
python train.py \
  --dataset cifar10 \
  --method moco \
  --amcl \
  --epochs 1000 \
  --batch-size 512 \
  --heads 3 \
  --queue-size 4096 \
  --momentum 0.99 \
  --topk 100 \
  --beta 1e-4 \
  --aug default \
  --output ./runs/cifar10_moco_amcl
```

### SimSiam + AMCL

```bash
python train.py \
  --dataset cifar10 \
  --method simsiam \
  --amcl \
  --epochs 1000 \
  --batch-size 512 \
  --heads 3 \
  --beta 1e-4 \
  --aug default \
  --output ./runs/cifar10_simsiam_amcl
```

### Barlow Twins + AMCL

```bash
python train.py \
  --dataset cifar10 \
  --method barlow_twins \
  --amcl \
  --epochs 1000 \
  --batch-size 512 \
  --heads 3 \
  --beta 1e-6 \
  --aug default \
  --output ./runs/cifar10_barlow_amcl
```

For CIFAR-100, replace `cifar10` by `cifar100`. For STL-10, use `--dataset stl10`; the input size is automatically set to 96. To use the STL-10 unlabeled set during SSL pretraining, add `--stl10-unlabeled`.

## Augmentation families

The implementation exposes the five families used in the paper:

| Flag | Transformation |
|---|---|
| `a` | random resized crop |
| `b` | random Gaussian blur |
| `c` | color dropping / random grayscale |
| `d` | color distortion |
| `e` | random horizontal flip |

Examples:

```bash
python train.py --dataset cifar10 --method simclr --amcl --aug a
python train.py --dataset cifar10 --method simclr --amcl --aug abc
python train.py --dataset cifar10 --method simclr --amcl --aug abcde
```

`default` is the original repository's default SimCLR-style augmentation configuration (`a+c+d+e` with blur probability set to zero). `color` is provided as the compact `c+d` color-only combination.

## Implementation details

### SimCLR / NT-Xent

For each head, Eq. (2.1) is implemented as the temperature-weighted positive term plus a hard-negative term and the corresponding positive/negative temperature regularization. In the practical Top-$k$ version, the implementation selects the top-$k$ similarities and gathers the matching adaptive temperatures by the returned indices.

For a SimCLR batch of size $B$, each anchor uses **$2B-2$ negatives**: the other samples from both augmented views. Both directions between the two views are evaluated, and the per-head objectives are summed as specified by the multi-head formulation.

### MoCo

The MoCo path uses a momentum key encoder and a FIFO feature queue. AMCL applies the Eq. (2.1) structure to the queue negatives. The queue is updated **after** the optimizer has completed backpropagation for the current step. This ordering is important for autograd correctness.

### SimSiam

The SimSiam path uses a three-layer projection MLP with a final non-affine batch normalization and a two-layer predictor, matching the baseline structure. Stop-gradient is applied to the target branch. AMCL then applies the two pair-adaptive positive temperatures from Eq. (2.2).

### Barlow Twins

Eq. (2.3) requires a temperature for every diagonal correlation and every off-diagonal channel pair. The implementation therefore constructs temperatures from the full batch-wise channel vectors

$$
z_{l:}=[z_{l1},\ldots,z_{lN}]^T,
$$

rather than collapsing a channel to a scalar mean. The same temperature map is shared across all heads.

Because $\phi$ operates on vectors of length $N$, **AMCL Barlow Twins requires a fixed training batch size**. The data loader uses `drop_last=True`, and the loss explicitly checks the configured batch size to prevent a silent mismatch. The Barlow Twins coefficient is `lambda=5e-3`, as in the paper.


## Linear evaluation

A frozen-encoder linear probe is provided for the small datasets:

```bash
python eval_linear.py \
  --checkpoint ./runs/cifar10_simclr_amcl/last.pt \
  --dataset cifar10 \
  --method simclr \
  --epochs 100 \
  --batch-size 512
```

The evaluator reconstructs the model from the checkpoint metadata, including whether AMCL was enabled and the saved model configuration. For MoCo, the trained query encoder is used as the downstream encoder.

## Checkpoints

Each training run writes:

```text
runs/<name>/
├── config.json
├── history.json
└── last.pt
```

The checkpoint contains the model state, the resolved command-line configuration, the completed epoch, and the global optimizer-step count.

## Citation

You can cite the following paper for the use of this work:

```bibtex
@inproceedings{wang2024adaptive,
  title={Adaptive multi-head contrastive learning},
  author={Wang, Lei and Koniusz, Piotr and Gedeon, Tom and Zheng, Liang},
  booktitle={European Conference on Computer Vision},
  pages={404--421},
  year={2024},
  organization={Springer}
}
```

### Acknowledgment
This code is based on [MIFA-Lab/contrastive2021](https://github.com/MIFA-Lab/contrastive2021). 
