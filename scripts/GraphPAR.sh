# Infomax RandAT
python ../src/train_GraphPAR.py --dataset credit --activation leakyrelu --hidden_dim 18 --num_layer 2 --pre_epochs 2000 --pre_lr 0.001 --weight_decay 1e-05 --perturb_epsilon 0.7 --adv_loss_weight 0 --random_attack_num_samples 500 --tune_epochs 1000 --data_aug
python ../src/train_GraphPAR.py --dataset pokec_z --activation leakyrelu --hidden_dim 24 --num_layer 2 --pre_epochs 2000 --pre_lr 0.001 --weight_decay 0.0 --perturb_epsilon 0.5 --adv_loss_weight 0 --random_attack_num_samples 500 --tune_epochs 1000 --data_aug
python ../src/train_GraphPAR.py --dataset pokec_n --activation leakyrelu --hidden_dim 24 --num_layer 2 --pre_epochs 2000 --pre_lr 0.001 --weight_decay 0.0 --perturb_epsilon 0.5 --adv_loss_weight 0 --random_attack_num_samples 500 --tune_epochs 500 --data_aug

# Infomax Minmax
python ../src/train_GraphPAR.py --dataset credit --activation leakyrelu --hidden_dim 18 --num_layer 2 --pre_epochs 2000 --pre_lr 0.001 --weight_decay 1e-05 --perturb_epsilon 0.5 --adv_loss_weight 0.1 --random_attack_num_samples 20 --tune_epochs 1000
python ../src/train_GraphPAR.py --dataset pokec_z --activation leakyrelu --hidden_dim 24 --num_layer 2 --pre_epochs 2000 --pre_lr 0.001 --weight_decay 0.0 --perturb_epsilon 0.5 --adv_loss_weight 0.8 --random_attack_num_samples 100 --tune_epochs 1000
python ../src/train_GraphPAR.py --dataset pokec_n --activation leakyrelu --hidden_dim 24 --num_layer 2 --pre_epochs 2000 --pre_lr 0.001 --weight_decay 0.0 --perturb_epsilon 0.5 --adv_loss_weight 0.1 --random_attack_num_samples 20 --tune_epochs 1000