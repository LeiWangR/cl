# Adaptive Multi-head Contrastive Learning (AMCL)

## 1. Summary

### 1.1: Motivation

In (a), each column denotes positive (green dots) or negative instances (red dots) with corresponding similarity measures. Additional augmentations can cause positive samples to appear dissimilar and occasionally make negative samples seem similar. The table in (a) shows the original similarity measure (in gray) and the similarity scores from our method (in black).

(b)-(d): for traditional contrastive learning methods, when increasing the number of augmentations from 1 to 5, similarities of more positive pairs drop below 0.5, causing more significant overlapping regions between histograms of positive (orange) and negative (blue) pairs. 

In comparison, our multi-head approach (e)-(g) yields better separation of positive and negative sample pairs as more augmentation types are used, e.g., (g) vs. (d).

![Alt Text](https://github.com/LeiWangR/cl/blob/main/images/motivation.png)

### 1.2: Model comparison

A comparison of (a) the standard constant-temperature, single-head approach and (b)–(c) our adaptive temperature, single- and multi-head approaches. 

In each subfigure, the first light blue trapezoid represents the base encoder, the second light blue trapezoid signifies the MLP projection head, and the third light orange trapezoid denotes the shared MLP layer for learning the temperature parameters. In (c), the projection head is replicated C times to capture diverse image content. 

For better visualization, we set C = 3 for simplicity. The architecture of the MLP projection head remains unchanged; however, the weights are learned independently. For a given image pair, each projection head produces a pair of feature vectors, which are later used for learning the pair-adaptive, head-wise temperature. 

The learned temperatures, along with the projected features, are seamlessly incorporated into our Adaptive Multi-head Contrastive Learning (AMCL) loss function.

![Alt Text](https://github.com/LeiWangR/cl/blob/main/images/comparison.png)

### 1.3: Loss comparison

- Standard contrastive learning methods and their loss functions:

![Alt Text](https://github.com/LeiWangR/cl/blob/main/images/table1.png)

- Loss functions for applying AMCL to widely used contrastive learning frameworks involve introducing C heads and a regularization term (highlighted in green):

![Alt Text](https://github.com/LeiWangR/cl/blob/main/images/table2.png)


## 2. Structure

```console
├── cl
    ├── readme.md                  // descriptions about the repo
    ├── requirements.txt           // dependencies
    ├── src
        ├── models.py              // models
        ├── util.py                // utilities
    ├── train_eval.py              // training and evaluation script
```


## 3. Installation

Follow these steps to set up the environment:
```bash
python -m venv venv                 # create a virtual environment named venv
source venv/bin/activate            # activate the environment
pip install -r requirements.txt     # install the dependencies
```

## 4. Model & data augmentation

Our approach, Adaptive Multi-Head Contrastive Learning (AMCL), can be applied to and experimentally enhances several popular contrastive learning methods such as:

- SimCLR
- MoCo
- SimSiam
- Barlow Twins

We consider five different types of transformations for data augmentations:

- Random cropping
- Random gaussian blur
- Color dropping
- Color distortion
- Random horizontal flipping

## 5. Train & eval
```
# parameters
model=simclr
epochs=800
augs=default
num_runs=10
timestamp=`date '+%s'`

# train and eval
python train_eval.py --model=${model} --epochs=${epochs} --augs=${augs} --num_runs=${num_runs} | tee ./results/"${timestamp}_${model}_epochs_${epochs}_augs_${augs}".txt
```

Pretrained models for CIFAR-10 and CIFAR-100 can be downloaded [here](123).


## 6. Citation

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

