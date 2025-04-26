# GraphPAR
Source code for WWW 2024 paper "[Endowing Pre-trained Graph Models with Provable Fairness](https://arxiv.org/pdf/2402.12161)"

![image-20240302114357020](http://img.dreamcodecity.cn/img/image-20240302114357020.png)

# Requirements
- Python 3.8
- PyTorch 2.1.1
- torch_geometric 2.4.0
- My operating system is Ubuntu 18.04.5 with one GPU (GeForce RTX 3090) and CPU (Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz)

Create a conda (see Anaconda or Miniconda) environment with the required packages:
```sh
conda env create -f environment.yml
```



# Datasets
We have uploaded the original data of Income, Credit, Pokec_z and Pokec_n in the `data/` folder.

# Pretrain GNNs
With DGI:
```sh
python infomax.py --dataset credit --activation leakyrelu --hidden_dim 18 --num_layer 2 --pre_epochs 2000 --pre_lr 0.001 --weight_decay 1e-05
python infomax.py --dataset pokec_z --activation leakyrelu --hidden_dim 24 --num_layer 2 --pre_epochs 2000 --pre_lr 0.001 --weight_decay 0.0
python infomax.py --dataset pokec_n --activation leakyrelu --hidden_dim 24 --num_layer 2 --pre_epochs 2000 --pre_lr 0.001 --weight_decay 0.0
```

# GraphPAR
With RandAT:
```sh
python train_GraphPAR.py --dataset credit --activation leakyrelu --hidden_dim 18 --num_layer 2 --pre_epochs 2000 --pre_lr 0.001 --weight_decay 1e-05 --perturb_epsilon 0.7 --adv_loss_weight 0 --random_attack_num_samples 500 --tune_epochs 1000 --data_aug
python train_GraphPAR.py --dataset pokec_z --activation leakyrelu --hidden_dim 24 --num_layer 2 --pre_epochs 2000 --pre_lr 0.001 --weight_decay 0.0 --perturb_epsilon 0.5 --adv_loss_weight 0 --random_attack_num_samples 500 --tune_epochs 1000 --data_aug
python train_GraphPAR.py --dataset pokec_n --activation leakyrelu --hidden_dim 24 --num_layer 2 --pre_epochs 2000 --pre_lr 0.001 --weight_decay 0.0 --perturb_epsilon 0.5 --adv_loss_weight 0 --random_attack_num_samples 500 --tune_epochs 500 --data_aug
```

With MinMAx:
```sh
python train_GraphPAR.py --dataset credit --activation leakyrelu --hidden_dim 18 --num_layer 2 --pre_epochs 2000 --pre_lr 0.001 --weight_decay 1e-05 --perturb_epsilon 0.5 --adv_loss_weight 0.1 --random_attack_num_samples 20 --tune_epochs 1000
python train_GraphPAR.py --dataset pokec_z --activation leakyrelu --hidden_dim 24 --num_layer 2 --pre_epochs 2000 --pre_lr 0.001 --weight_decay 0.0 --perturb_epsilon 0.5 --adv_loss_weight 0.8 --random_attack_num_samples 100 --tune_epochs 1000
python train_GraphPAR.py --dataset pokec_n --activation leakyrelu --hidden_dim 24 --num_layer 2 --pre_epochs 2000 --pre_lr 0.001 --weight_decay 0.0 --perturb_epsilon 0.5 --adv_loss_weight 0.1 --random_attack_num_samples 20 --tune_epochs 1000
```


# Reference

```
@inproceedings{zhang2024endowing,
  title={Endowing Pre-trained Graph Models with Provable Fairness},
  author={Zhang, Zhongjian and Zhang, Mengmei and Yu, Yue and Yang, Cheng and Liu, Jiawei and Shi, Chuan},
  booktitle={Proceedings of the ACM on Web Conference 2024},
  pages={1045--1056},
  year={2024}
}
```
