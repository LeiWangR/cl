# CL

This code is based on [MIFA-Lab/contrastive2021](https://github.com/MIFA-Lab/contrastive2021).

## Model

- SimCLR
- Barlow Twins
- MoCo
- SimSiam

## Augmentation

- Random Cropping
- Random Gaussian Blur
- Color Dropping
- Color Distortion
- Random Horizontal Flipping

## Installation
```bash
python -m venv venv                 # create a virtual environment named venv
source venv/bin/activate            # activate the environment
pip install -r requirements.txt     # install the dependencies
```

## Sample code

```console
├── cl
    ├── readme.md                  // descriptions about the repo
    ├── requirements.txt           // dependencies
    ├── src
        ├── models.py              // models
        ├── util.py                // utilities
    ├── train_eval.py              // training and evaluation script
```

## Citation

If you find our work useful in your research, please consider citing:

```
@article{wang2023adaptive,
  title={Adaptive multi-head contrastive learning},
  author={Wang, Lei and Koniusz, Piotr and Gedeon, Tom and Zheng, Liang},
  journal={arXiv preprint arXiv:2310.05615},
  year={2023}
}
```

