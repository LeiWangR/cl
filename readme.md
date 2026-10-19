# AMCL (v3.0)

> This update extends the original release with experiments on four self-supervised methods: SimCLR, MoCo, SimSiam and Barlow Twins, using CIFAR-10 and CIFAR-100. It evaluates the effect of adding AMCL and documents the implementation details, training configurations and evaluation procedures.

*This release removes outdated dependencies from AMCL and updates the implementation and training pipeline for improved compatibility and stability.*

## Structure

```text
cl-main/
├── amcl/
│   ├── __init__.py
│   ├── augmentations.py       # SSL augmentation families
│   ├── backbones.py           # CIFAR/STL ResNet-18 stem
│   ├── datasets.py            # CIFAR-10/100 and STL-10
│   ├── evaluation.py          # kNN evaluation
│   ├── losses.py              # baseline + AMCL objectives
│   ├── models.py              # SimCLR/MoCo/SimSiam/Barlow Twins
│   └── train.py               # SSL training and checkpoint selection
├── scripts/
│   ├── smoke_test.py          # forward/backward checks for all methods
│   ├── training_smoke.py      # multi-step optimizer checks
│   ├── train_loop_smoke.py    # training-loop and checkpoint checks
│   └── verify_release.sh     # release verification
├── tests/
│   └── test_amcl.py           # numerical/equation regression tests
├── runs/
│   ├── index.json             # recorded experiment results
│   └── <dataset>_<method>_<variant>/  # checkpoints and experiment metadata
├── eval_linear.py            # linear-probe training and test evaluation
├── train.py                  # CLI entry point
├── requirements.txt
├── pyproject.toml
└── readme.md
```

## Table

| Dataset | Method | Baseline (%) | +AMCL (%) |
|---|---|---:|---:|
| CIFAR-10 | SimCLR | 90.56 | 90.16 |
| CIFAR-10 | MoCo | 88.98 | 89.76 |
| CIFAR-10 | SimSiam | 86.44 | 87.04 |
| CIFAR-10 | Barlow Twins | 88.40 | 89.37 |
| CIFAR-100 | SimCLR | 55.70 | 61.17 |
| CIFAR-100 | MoCo | 57.24 | 60.26 |
| CIFAR-100 | SimSiam | 51.87 | 54.02 |
| CIFAR-100 | Barlow Twins | 56.63 | 60.21 |

## Installation

Python 3.10 or newer is required. For CUDA, install a compatible PyTorch/torchvision build in the environment before installing this package.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Recorded environment references include PyTorch `2.5.1+cu121` / torchvision `0.20.1+cu121` in the CUDA 12.1 dependency snapshot, and PyTorch `2.8.0+cu128` / torchvision `0.23.0+cu128` for the RunPod MoCo runs. The package supports broader version ranges; these do not guarantee identical results across environments.

## Verification before release

Verification covers loss equations, model gradients, optimizer updates and checkpoint writing.

### Unit tests

```bash
python -m pip install 'pytest>=7,<10'
python -m pytest -q
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

Run `python scripts/train_loop_smoke.py` to check the training loop and checkpoint writing for all four AMCL methods using small synthetic batches.

The smoke tests use small synthetic batches so they remain fast. They are numerical/implementation tests.

## Training

```bash
python train.py --help
```

The commands below use the AMCL configurations behind the table: CIFAR ResNet-18, `abcde` SSL augmentation, batch size 512, seed 0, SGD with momentum 0.9, cosine scheduling, and gradient clipping at 1.0. CUDA training uses AMP. `--tau-min` is eta; `--tau-max` is eta + iota.

SimCLR, MoCo and Barlow Twins use a stratified 45k/5k train/validation split (split seed 0) and select `best.pt` by validation kNN every 50 epochs. SimSiam uses all 50k training images and `last.pt` at epoch 1000. These are the recorded hyperparameters; exact retraining scores can vary across environments.

The table reports official-test linear-probe top-1 accuracy. Baselines use default hyperparameters, all 50k training images and the final encoder; their classifiers use 100 epochs, learning rate 0.1 and zero weight decay. The +AMCL results use tuned configurations for SimCLR, MoCo and Barlow Twins, and the default configuration for SimSiam. Training data size, checkpoint selection and tuning budgets therefore differ between these results.

Use an empty `--output` directory when retraining; existing checkpoints are protected from overwriting. To evaluate downloaded models without training, use the saved-classifier commands under **Test**.

### SimCLR + AMCL

**CIFAR-10**

```bash
python train.py \
  --dataset cifar10 --data-root ./data/cifar10 \
  --method simclr --amcl \
  --epochs 1000 --batch-size 512 --heads 3 \
  --proj-dim 128 --temperature-hidden-dim 128 \
  --lr 0.14745122424701196 --weight-decay 0.000695436678101201 \
  --beta 0.05534140428409663 --tau-min 0.001095234112164022 --tau-max 0.0066507674582755305 \
  --topk 150 \
  --amp-initial-scale 65536 \
  --aug abcde --seed 0 --num-workers 4 \
  --grad-clip 1.0 --eval-k 200 \
  --validation-size 5000 --split-seed 0 \
  --checkpoint-selection best --eval-every 50 \
  --output ./runs/cifar10_simclr_amcl
```

**CIFAR-100**

```bash
python train.py \
  --dataset cifar100 --data-root ./data/cifar100 \
  --method simclr --amcl \
  --epochs 1000 --batch-size 512 --heads 3 \
  --proj-dim 128 --temperature-hidden-dim 128 \
  --lr 0.18034549067292038 --weight-decay 0.00011692142617847786 \
  --beta 0.017407998232692574 --tau-min 0.0017179731776498439 --tau-max 0.003391856732513295 \
  --topk 150 \
  --amp-initial-scale 65536 \
  --aug abcde --seed 0 --num-workers 4 \
  --grad-clip 1.0 --eval-k 200 \
  --validation-size 5000 --split-seed 0 \
  --checkpoint-selection best --eval-every 50 \
  --output ./runs/cifar100_simclr_amcl
```

### MoCo + AMCL

**CIFAR-10**

```bash
python train.py \
  --dataset cifar10 --data-root ./data/cifar10 \
  --method moco --amcl \
  --epochs 1000 --batch-size 512 --heads 3 \
  --proj-dim 128 --temperature-hidden-dim 128 \
  --lr 0.08 --weight-decay 0.0005 \
  --beta 1e-05 --tau-min 0.03 --tau-max 0.07 \
  --topk 100 \
  --queue-size 4096 --momentum 0.99 \
  --temperature-lr-scale 0.01 --loader-seed 0 \
  --amp-initial-scale 65536 \
  --aug abcde --seed 0 --num-workers 8 \
  --grad-clip 1.0 --eval-k 200 \
  --validation-size 5000 --split-seed 0 \
  --checkpoint-selection best --eval-every 50 \
  --output ./runs/cifar10_moco_amcl
```

**CIFAR-100**

```bash
python train.py \
  --dataset cifar100 --data-root ./data/cifar100 \
  --method moco --amcl \
  --epochs 1000 --batch-size 512 --heads 3 \
  --proj-dim 128 --temperature-hidden-dim 128 \
  --lr 0.08 --weight-decay 0.0005 \
  --beta 1e-05 --tau-min 0.03 --tau-max 0.07 \
  --topk 100 \
  --queue-size 4096 --momentum 0.99 \
  --temperature-lr-scale 0.01 --loader-seed 0 \
  --amp-initial-scale 65536 \
  --aug abcde --seed 0 --num-workers 8 \
  --grad-clip 1.0 --eval-k 200 \
  --validation-size 5000 --split-seed 0 \
  --checkpoint-selection best --eval-every 50 \
  --output ./runs/cifar100_moco_amcl
```

### SimSiam + AMCL

**CIFAR-10**

```bash
python train.py \
  --dataset cifar10 --data-root ./data/cifar10 \
  --method simsiam --amcl \
  --epochs 1000 --batch-size 512 --heads 3 \
  --proj-dim 2048 --temperature-hidden-dim 128 \
  --lr 0.05 --weight-decay 0.0005 \
  --beta 0.0001 --tau-min 1e-05 --tau-max 2.0 \
  --amp-initial-scale 65536 \
  --aug abcde --seed 0 --num-workers 4 \
  --grad-clip 1.0 --eval-k 200 \
  --validation-size 0 --checkpoint-selection last --eval-every 10 \
  --output ./runs/cifar10_simsiam_amcl
```

**CIFAR-100**

```bash
python train.py \
  --dataset cifar100 --data-root ./data/cifar100 \
  --method simsiam --amcl \
  --epochs 1000 --batch-size 512 --heads 3 \
  --proj-dim 2048 --temperature-hidden-dim 128 \
  --lr 0.05 --weight-decay 0.0005 \
  --beta 0.0001 --tau-min 1e-05 --tau-max 2.0 \
  --amp-initial-scale 65536 \
  --aug abcde --seed 0 --num-workers 4 \
  --grad-clip 1.0 --eval-k 200 \
  --validation-size 0 --checkpoint-selection last --eval-every 10 \
  --output ./runs/cifar100_simsiam_amcl
```

### Barlow Twins + AMCL

**CIFAR-10**

```bash
python train.py \
  --dataset cifar10 --data-root ./data/cifar10 \
  --method barlow_twins --amcl \
  --epochs 1000 --batch-size 512 --heads 3 \
  --proj-dim 2048 --temperature-hidden-dim 128 \
  --lr 0.15747629669629123 --weight-decay 0.0009125113726879362 \
  --beta 5.998610317933858e-08 --tau-min 0.8148026273230112 --tau-max 0.8173512413218031 \
  --amp-initial-scale 1 \
  --aug abcde --seed 0 --num-workers 4 \
  --grad-clip 1.0 --eval-k 200 \
  --validation-size 5000 --split-seed 0 \
  --checkpoint-selection best --eval-every 50 \
  --output ./runs/cifar10_barlow_twins_amcl
```

**CIFAR-100**

```bash
python train.py \
  --dataset cifar100 --data-root ./data/cifar100 \
  --method barlow_twins --amcl \
  --epochs 1000 --batch-size 512 --heads 3 \
  --proj-dim 2048 --temperature-hidden-dim 128 \
  --lr 0.0467032388809851 --weight-decay 0.000299501049315049 \
  --beta 8.419801315457466e-07 --tau-min 0.7967992115257492 --tau-max 0.8052621498554488 \
  --amp-initial-scale 1 \
  --aug abcde --seed 0 --num-workers 4 \
  --grad-clip 1.0 --eval-k 200 \
  --validation-size 5000 --split-seed 0 \
  --checkpoint-selection best --eval-every 50 \
  --output ./runs/cifar100_barlow_twins_amcl
```

### Hyperparameter selection

SimCLR used Hyperopt random search with 200 encoder candidates per dataset, including the default configuration (search seed 20261007). Log-uniform ranges were lr `[0.01, 0.2]`, weight decay `[1e-6, 1e-3]`, eta and iota `[1e-5, 2]`, and beta `[1e-5, 10]`; top-k was selected from `{100, 150}`. Candidates were ranked by final-epoch linear validation accuracy after fitting a 100-epoch probe to each validation-kNN-selected encoder.

For the winning SimCLR configuration, classifier search covered lr `{0.01, 0.03, 0.1, 0.3, 1.0}`, weight decay `{0, 1e-6, 1e-5, 1e-4}`, and training horizons `{50, 100, 200}` on both `best.pt` and `last.pt`: 240 runs across both datasets. Classifiers were selected by validation accuracy, with the earliest epoch retained on ties.

The MoCo configuration was selected on CIFAR-10 pilot validation and transferred unchanged to CIFAR-100. Barlow Twins configurations were selected by linear validation accuracy. Neither used classifier hyperparameter search. These are our experimental settings, not an exact replication of the paper's search protocol.

## Test

These commands freeze the newly trained encoder, fit a linear classifier, and evaluate the official 10k-image test set. Linear inputs use deterministic Resize, ToTensor and ImageNet normalization. All classifiers use SGD with momentum 0.9, cosine scheduling, batch size 512 and seed 0.

The evaluator reconstructs the encoder from checkpoint metadata and freezes its parameters and BatchNorm statistics. MoCo uses the trained query encoder.

SimCLR selects the classifier by validation accuracy over 200 epochs. MoCo and Barlow Twins use the final classifier at epoch 100; SimSiam also uses epoch 100, fitted on all 50k training images. Runs with a validation split load it automatically from `split.json` beside the encoder checkpoint.

The archived MoCo encoders have `protocol.json` instead of `split.json`. When fitting a classifier from those checkpoints, add `--split-file ./runs/<dataset>_simclr_amcl/split.json`; these saved splits have identical train/validation indices. Direct evaluation with `--test-only` does not require a split file.

### SimCLR + AMCL

**CIFAR-10**

```bash
python eval_linear.py \
  --checkpoint ./runs/cifar10_simclr_amcl/best.pt \
  --dataset cifar10 --data-root ./data/cifar10 --method simclr \
  --epochs 200 --lr 1.0 --weight-decay 0 \
  --batch-size 512 --num-workers 4 --seed 0 \
  --selection best --eval-split test \
  --output ./runs/cifar10_simclr_amcl/linear_evaluation
```

**CIFAR-100**

```bash
python eval_linear.py \
  --checkpoint ./runs/cifar100_simclr_amcl/best.pt \
  --dataset cifar100 --data-root ./data/cifar100 --method simclr \
  --epochs 200 --lr 0.3 --weight-decay 1e-05 \
  --batch-size 512 --num-workers 4 --seed 0 \
  --selection best --eval-split test \
  --output ./runs/cifar100_simclr_amcl/linear_evaluation
```

### MoCo + AMCL

**CIFAR-10**

```bash
python eval_linear.py \
  --checkpoint ./runs/cifar10_moco_amcl/best.pt \
  --dataset cifar10 --data-root ./data/cifar10 --method moco \
  --epochs 100 --lr 0.1 --weight-decay 0 \
  --batch-size 512 --num-workers 4 --seed 0 \
  --selection last --eval-split test \
  --cache-features \
  --output ./runs/cifar10_moco_amcl/linear_evaluation
```

**CIFAR-100**

```bash
python eval_linear.py \
  --checkpoint ./runs/cifar100_moco_amcl/best.pt \
  --dataset cifar100 --data-root ./data/cifar100 --method moco \
  --epochs 100 --lr 0.1 --weight-decay 0 \
  --batch-size 512 --num-workers 4 --seed 0 \
  --selection last --eval-split test \
  --cache-features \
  --output ./runs/cifar100_moco_amcl/linear_evaluation
```

### SimSiam + AMCL

**CIFAR-10**

```bash
python eval_linear.py \
  --checkpoint ./runs/cifar10_simsiam_amcl/last.pt \
  --dataset cifar10 --data-root ./data/cifar10 --method simsiam \
  --epochs 100 --lr 0.1 --weight-decay 0 \
  --batch-size 512 --num-workers 4 --seed 0 \
  --selection last --eval-split test \
  --output ./runs/cifar10_simsiam_amcl/linear_evaluation
```

**CIFAR-100**

```bash
python eval_linear.py \
  --checkpoint ./runs/cifar100_simsiam_amcl/last.pt \
  --dataset cifar100 --data-root ./data/cifar100 --method simsiam \
  --epochs 100 --lr 0.1 --weight-decay 0 \
  --batch-size 512 --num-workers 4 --seed 0 \
  --selection last --eval-split test \
  --output ./runs/cifar100_simsiam_amcl/linear_evaluation
```

### Barlow Twins + AMCL

**CIFAR-10**

```bash
python eval_linear.py \
  --checkpoint ./runs/cifar10_barlow_twins_amcl/best.pt \
  --dataset cifar10 --data-root ./data/cifar10 --method barlow_twins \
  --epochs 100 --lr 0.1 --weight-decay 0 \
  --batch-size 512 --num-workers 4 --seed 0 \
  --selection last --eval-split test \
  --output ./runs/cifar10_barlow_twins_amcl/linear_evaluation
```

**CIFAR-100**

```bash
python eval_linear.py \
  --checkpoint ./runs/cifar100_barlow_twins_amcl/best.pt \
  --dataset cifar100 --data-root ./data/cifar100 --method barlow_twins \
  --epochs 100 --lr 0.1 --weight-decay 0 \
  --batch-size 512 --num-workers 4 --seed 0 \
  --selection last --eval-split test \
  --output ./runs/cifar100_barlow_twins_amcl/linear_evaluation
```

### Evaluate a saved classifier directly

The recorded SimCLR, MoCo and Barlow Twins AMCL runs include `linear_probe.pt`, containing both the selected encoder and classifier. Use `--test-only` to evaluate without retraining; replace the dataset and method below to select another recorded run.

```bash
python eval_linear.py \
  --checkpoint ./runs/cifar10_simclr_amcl/linear_probe.pt \
  --dataset cifar10 --data-root ./data/cifar10 --method simclr \
  --batch-size 512 --num-workers 4 --seed 0 \
  --test-only \
  --output ./runs/test_results/cifar10_simclr_amcl
```

The recorded baselines and SimSiam AMCL runs contain encoder checkpoints only. To evaluate those, fit a classifier from `runs/<dataset>_<method>_<variant>/last.pt` for 100 epochs with `--lr 0.1 --weight-decay 0 --selection last`, batch size 512 and seed 0.

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

Random resized crop is always applied, even when `a` is omitted. `default` uses crop, grayscale, color distortion and horizontal flip (`a+c+d+e`), without blur. `color` adds grayscale and color distortion (`c+d`) to the crop. All configurations finish with ToTensor and ImageNet normalization.

## Implementation details

### SimCLR / NT-Xent

For each head, Eq. (2.1) is implemented as the temperature-weighted positive term plus a hard-negative term and the corresponding positive/negative temperature regularization. In the practical Top-$k$ version, the implementation selects the top-$k$ similarities and gathers the matching adaptive temperatures by the returned indices.

For a SimCLR batch of size $B$, each anchor uses **$2B-2$ negatives**: the other samples from both augmented views. Both directions between the two views are evaluated, and the per-head objectives are summed as specified by the multi-head formulation.

### MoCo

MoCo uses a **one-way query-to-key/queue loss**: each query is compared with its positive key and the queue negatives. The key encoder and projector receive no gradients; their parameters follow the query branch through momentum updates. AMCL learns pair-adaptive temperatures for these comparisons. The queue is updated only after a successful optimizer step, so AMP overflow skips do not enqueue keys.

### SimSiam

The SimSiam path uses a three-layer projection MLP with a final non-affine batch normalization and a two-layer predictor, matching the baseline structure. Stop-gradient is applied to the target branch. AMCL then applies the two pair-adaptive positive temperatures from Eq. (2.2).

### Barlow Twins

Eq. (2.3) requires a temperature for every diagonal correlation and every off-diagonal channel pair. The implementation therefore constructs temperatures from the full batch-wise channel vectors

$$
z_{l:}=[z_{l1},\ldots,z_{lN}]^T,
$$

rather than collapsing a channel to a scalar mean. The same temperature map is shared across all heads.

Because $\phi$ operates on vectors of length $N$, **AMCL Barlow Twins requires a fixed training batch size**. The data loader uses `drop_last=True`, and the loss explicitly checks the configured batch size to prevent a silent mismatch. The Barlow Twins coefficient is `lambda=5e-3`, as in the paper.

## Checkpoints

**Pretrained checkpoints:** [Download from Google Drive](https://drive.google.com/drive/folders/1-0Ve-pzCx9NtUMmiWPJGOTn6cCRHrpTZ?usp=drive_link). Extract the archive into the repository root to populate `runs/`.

New training runs write:

```text
runs/<name>/
├── config.json                 # resolved training configuration
├── history.json                # losses, kNN scores and AMP statistics
├── last.pt                     # latest completed epoch
├── split.json                  # when a validation split is used
├── best.pt                     # when validation kNN selection is enabled
└── checkpoint_selection.json   # selected encoder epoch and validation score
```

New training checkpoints include model, optimizer, scheduler and AMP scaler states, configuration, epoch, optimizer-step count and RNG states. Earlier recorded checkpoints may contain fewer fields.

Linear-classifier training writes `linear_probe.pt`, `history.json` and `test_result.json` or `validation_result.json` to its output directory, plus `split.json` when applicable. `linear_probe.pt` contains the frozen encoder, trained classifier and selection metadata; use it with `--test-only` for inference.

Recorded experiment scores are indexed in `runs/index.json`. Each experiment's `metadata.json` records checkpoint sources and SHA-256 checksums.

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
