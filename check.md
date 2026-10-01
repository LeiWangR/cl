# v2.0 check

## Unit tests

```text
pytest -q
```

## Forward/Backward checks

```text
python scripts/smoke_test.py
```

Verified finite forward losses and finite gradients for baseline and AMCL variants of:

- SimCLR
- MoCo
- SimSiam
- Barlow Twins

The MoCo check also verifies the queue update after backward.

## Multi-step optimization checks

```text
python scripts/training_smoke.py
```

Three SGD steps were run for every baseline/AMCL path with finite loss, gradient, and parameter checks after every update.

## Actual CLI training-loop checks

```text
python scripts/train_loop_smoke.py
```

The real `amcl.train.train()` function was exercised for two optimizer steps for all four AMCL methods using an in-memory synthetic dataset. 



