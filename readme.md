# CL

## Installation
```bash
python -m venv venv                 # create a virtual environment named venv
source venv/bin/activate            # activate the environment
pip install -r requirements.txt     # install the dependencies
```

## Description

```console
├── cl
    ├── readme.md                  // descriptions about the repo
    ├── requirements.txt           // dependencies
    ├── src
        ├── models.py              // models
        ├── util.py                // utilities
    ├── train_eval.py              // training and evaluation script
```

## Model

- SimCLR
- MoCo
- SimSiam
- Barlow Twins

## Augmentation

- Random cropping
- Random gaussian blur
- Color dropping
- Color distortion
- Random horizontal flipping

## Citation

You can cite the following paper for the use of this work:

```
@inproceedings{wang2024adaptive,
  title={Adaptive multi-head contrastive learning},
  author={Wang, Lei and Koniusz, Piotr and Gedeon, Tom and Zheng, Liang},
  booktitle={European Conference on Computer Vision},
  year={2024},
  organization={Springer}
}
```
### Acknowledgment
This code is based on [MIFA-Lab/contrastive2021](https://github.com/MIFA-Lab/contrastive2021).

