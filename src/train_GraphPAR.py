import argparse
import warnings

import numpy as np
import torch.cuda as cuda

from GraphPAR import GraphPAR
from utils import setup_seed, get_path_to

warnings.filterwarnings('ignore')


def get_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # =============================== Pre-trained Graph Model ===============================
    parser.add_argument('--seed', type=int, default=15)
    parser.add_argument('--pre_train', type=str, default='infomax')
    parser.add_argument('--dataset', type=str, default='pokec_z', choices=['income', 'credit', 'pokec_z', 'pokec_n'])
    parser.add_argument('--device', type=int, default=1 if cuda.is_available() else 'cpu')
    parser.add_argument('--activation', type=str, default='leakyrelu', choices=['relu', 'leakyrelu', 'prelu'])
    parser.add_argument('--hidden_dim', type=int, default=24, choices=[18, 24])
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--pre_epochs', type=int, default=2000)
    parser.add_argument('--pre_lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters')

    # =============================== Fairness ===============================
    parser.add_argument('--perturb_epsilon', type=float, default=0.5)
    # parser.add_argument('--data_aug', type=bool, default=False)
    parser.add_argument('--data_aug', action="store_true", help="whether use data augmentation")
    parser.add_argument('--adv_loss_weight', type=float, default=0.8)
    parser.add_argument('--random_attack_num_samples', type=int, default=100)
    parser.add_argument('--tune_epochs', type=int, default=1000)
    parser.add_argument('--cls_sigma', type=float, default=1)
    parser.add_argument('--perf_alpha', type=float, default=1)

    # =============================== Provable Fairness ===============================
    # center smooth
    parser.add_argument('--adapter_alpha', type=float, default=0.01,
                        help="With probability at least 1 - alpha either abstain or return the prediction")
    parser.add_argument('--adapter_n', type=int, default=10000,
                        help="Number of samples required by the encoder for certification")
    parser.add_argument('--adapter_n0', type=int, default=10000)
    # random smooth
    parser.add_argument('--cls_alpha', type=float, default=0.001,
                        help="With probability at least 1 - alpha either abstain or return the prediction")
    parser.add_argument('--cls_n', type=int, default=100000,
                        help="Number of samples required by the classifier for prediction and certification")
    parser.add_argument('--cls_n0', type=int, default=2000)
    return parser.parse_args()


def main():
    params = get_params()
    acc_array, f1_array, equality_array, parity_array = np.array([]), np.array([]), np.array([]), np.array([])
    seeds = [11, 13, 15, 17, 19]
    for seed in seeds:
        print("=" * 25, f"seed={seed}", "=" * 25)
        params.seed = seed
        setup_seed(params.seed)
        print("Arguments: %s " % ",".join([("%s=%s" % (k, v)) for k, v in params.__dict__.items()]))
        file_name = (f"{params.dataset}_{params.seed}_{params.activation}_hidden-dim({params.hidden_dim})_"
                     f"num-layer({params.num_layer})_epochs({params.pre_epochs})_lr({params.pre_lr})_weight_decay({params.weight_decay})")
        model_path = get_path_to('saved_models')
        pgm_file = f'{model_path}/pretrain/infomax/{file_name}_weights.pt'
        print("Load the PGM from:", pgm_file)
        model = GraphPAR(params, pgm_file)
        acc, f1, equality, parity = model.train_robust_adapter()
        # model.train_robust_classifier()
        # model.certify()
        acc_array = np.append(acc_array, acc)
        f1_array = np.append(f1_array, f1)
        equality_array = np.append(equality_array, equality)
        parity_array = np.append(parity_array, parity)
    print(f"Acc(↑):{acc_array.mean():.2f}±{acc_array.std():.2f}, F1(↑):{f1_array.mean():.2f}±{f1_array.std():.2f}, "
          f"DP(↓):{parity_array.mean():.2f}±{parity_array.std():.2f}, EO(↓):{equality_array.mean():.2f}±{equality_array.std():.2f}")


if __name__ == '__main__':
    main()
