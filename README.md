# Kaggle competition "SenNet + HOA - Hacking the Human Vasculature in 3D" solution

## The problem

### Dataset

TODO: Describe the kidney IDs and shortnames

## My solutions

### Backbones and libraries

### 2D

### 2.5D

## Results

### Table

|     Model      | Train dataset |  Test dataset  | Test avg 2D dice score | Test Surface dice score | Public LB  | Private LB |
|----------------|---------------|----------------|------------------------|-------------------------|------------|------------|
| Submission V47 |   1d          |      3s        |       0.589            |        0.593            |   0.585    |   0.491    |


V47:
- Train resolution: 512x512
- Train augmentation: my_aug_v2b
- Loss: focal loss
- Test resolution: 768x864
- Test augmentation: tta_5max
- Threshold 0.4 


### 2D comparison

Image tiles descriptions:
- Top left: original image
- Top middle: ground truth
- Top right: comparison prediction Vs ground truth overlay on image
- Bottom left: raw prediction
- Bottom middle: binarized prediction with threshold
- Bottom right: comparison

Colors:
- Green: true positives
- Blue: false negatives
- Red: false positives

#### kidney_3 sparse ground truth Vs model V47

Slice 170/1035 - Dice score 0.42 (a lot of false negatives)

![Slice 170 kidney 3 space V47](./docs/v47_v3s_0169_dice_score_0.42.png "v47_v3s_0169")

Slice 785/1035 - Dice score 0.80

![Slice 785 kidney 3 sparse V47](./docs/v47_v3s_0784_dice_score_0.80.png "v47_v3s_0784")

Comment: the model works well with small vessels, but it completely ignores
the big one on the left. This is more evident with the 3D visualization below.

### 3D comparison

The 3D points clouds are rescaled to 10% for faster processing

Colors:
- Green: true positives
- Blue: false negatives
- Red: false positives

#### kidney_3 sparse ground truth Vs model V47

Here we can see where the false negatives are concentrated

![Kidney sparse 3 GT Vs V47](./docs/kidney_3_sparse_label_vs_v47.gif "Kidney sparse 3 GT Vs V47")

TODO: Add an example with dice loss to highlight the different behavior on raw values