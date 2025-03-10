# Boosting the visual interpretability of CLIP via adversarial fine-tuning

Implementation for ICLR 2025 paper [Boosting the visual interpretability of CLIP via adversarial fine-tuning](https://openreview.net/forum?id=khuIvzxPRp)
 by [Shizhan Gong](https://peterant330.github.io/), [Haoyu Lei](lh218.github.io), [Qi Dou](https://www.cse.cuhk.edu.hk/~qdou/), and [Farzan Farnia](https://www.cse.cuhk.edu.hk/~farnia/)

## Sample Results
![sample_image](assets/image.png)
Figure. (a,b) Comparison of Simple Gradients/Grad-Cam between CLIP w/wo AFT. AFT greatly
improves the visual quality. (c) Evaluation of Simple Gradients on out-of-distribution dataset. (d)
Evaluation of Simple Gradients with linear probing. The improvements of visual interpretability
stem from AFT can transfer across datasets and to different tasks.

## Setup
We recommend to install the environment through
```
pip install -r requirements.txt
```

## Training

Please use the following code for adversarial fine-tuning.

```

```

## Interpretation
Please refer to for generating simple gradient map and GradCAM for CLIP model.


## Pre-trained Checkpoint
Our pretrained checkpoint can be downloaded through [one-drive]().

## Bibtex

If you find this work helpful, you can cite our paper as follows:

```
@inproceedings{
gong2025boosting,
title={Boosting the visual interpretability of CLIP via adversarial fine-tuning},
author={Shizhan Gong and Haoyu LEI and Qi Dou and Farzan Farnia},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=khuIvzxPRp}
}
```

## Acknowledgement

Our work is based on the codebase of the previous work including [OpenCLIP](https://github.com/mlfoundations/open_clip),
[RobustVLM](https://github.com/chs20/RobustVLM), [CLIP-Dissect](https://github.com/Trustworthy-ML-Lab/CLIP-dissect),
[VLM-visualizer](https://github.com/zjysteven/VLM-Visualizer), and [CLIP-benchmark](https://github.com/LAION-AI/CLIP_benchmark).


## Contact
For any questions, please contact <a href="mailto:szgong22@cse.cuhk.edu.hk">szgong22@cse.cuhk.edu.hk</a>
