# MetaGAD (IEEE DSAA 2024)
This repo is the official implementation of MetaGAD: "[MetaGAD: Meta Representation Adaptation for Few-Shot Graph Anomaly Detection](https://arxiv.org/abs/2305.10668)".

![image](https://github.com/user-attachments/assets/5c6cc949-b4df-429c-a97c-866c88bf4ec9)

## Contributions
• We study a new problem of few-shot graph anomaly detection; <br/>
• We propose a novel meta-learning approach to learn to transfer node representations from self-supervised tasks to assist supervised tasks with little labeled anomalies; <br/>
• We conduct extensive experiments on six real-world datasets with synthetically injected anomalies and organic anomalies. The experimental results demonstrate the effectiveness of the proposed approach MetaGAD for graph anomaly detection.

## Getting Started
### Environment
* python             3.10.8
* torch              1.13.0
* numpy              1.23.4
* scipy              1.9.3
* pandas             1.5.2

### Run
To get the result of Table 2 and Table 4, run the following scripts in a terminal as follows:

Cora dataset:

`python run.py --dataset injected_cora --detector_lr 5e-4 --adaptor_lr 5e-4 --pos_weight 0.5 --num_epoch 5500 --num_run 3`

Citeseer dataset:

`python run.py --dataset injected_citeseer --detector_lr 5e-4 --adaptor_lr 5e-3 --pos_weight 1 --num_epoch 6500 --num_run 3`

Amazon Photo dataset:

`python run.py --dataset injected_amazon_photo --detector_lr 1e-4 --adaptor_lr 1e-2 --pos_weight 5 --num_epoch 3500  --num_run 3`

Wiki dataset:

`python run.py --dataset wiki --detector_lr 5e-4 --adaptor_lr 5e-4 --pos_weight 0.1 --num_epoch 8000 --num_run 3`

Amazon Review dataset:

`python run.py --dataset amazon_review --detector_lr 5e-4 --adaptor_lr 5e-4 --pos_weight 1 --num_epoch 6000 --num_run 3`

Yelpchi dataset:

`python run.py --dataset yelpchi --detector_lr 5e-4 --adaptor_lr 5e-4 --pos_weight 0.6 --num_epoch 15000 --num_run 3`

## Cite
If you find this repository useful for your work, please consider citing the paper as follows:

```bibtex
@article{xu2023metagad,
  title={MetaGAD: Learning to Meta Transfer for Few-shot Graph Anomaly Detection},
  author={Xu, Xiongxiao and Ding, Kaize and Chen, Canyu and Shu, Kai},
  journal={arXiv preprint arXiv:2305.10668},
  year={2023}
}
```
