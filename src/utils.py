import argparse
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.spatial import distance_matrix
from torch_geometric.data import Data


def empty_folder(p: Path):
    """Delete the contents of a folder."""
    for child_path in p.iterdir():
        if child_path.is_file():
            child_path.unlink()
        elif child_path.is_dir():
            shutil.rmtree(child_path)


def get_or_create_path(p: Path, empty_dir: bool = False) -> Path:
    """Create a folder if it does not already exist."""
    if not p.is_dir():
        p.mkdir(parents=True)
    else:
        if empty_dir:
            empty_folder(p)
    return p


def get_path_to(dir_name: str) -> Path:
    """Returns the path to fair-vision/data/."""
    assert dir_name in ['data', 'saved_models']
    return get_or_create_path(get_root_path() / dir_name)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_root_path() -> Path:
    """Returns the path to fair-vision/."""
    return Path(__file__).resolve().parent.parent


def calculate_parameter_count(optimizer):
    optimizer_params = sum(p.numel() for p in optimizer.param_groups[0]['params'])
    print(f"The number of tuning parameters: {optimizer_params}")


def load_data(params: argparse.Namespace) -> Data:
    dataset_dir = get_path_to('data')
    adj, features, labels, idx_train, idx_valid, idx_test, sens, sens_idx, raw_data_info = load(dataset_dir,
                                                                                                params.dataset)
    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)

    data = Data(x=features.to(params.device), edge_index=edge_index.to(params.device), input_dim=features.shape[1],
                idx_train_list=idx_train, idx_valid_list=idx_valid, idx_test_list=idx_test, y=labels.to(params.device),
                sens=sens.to(params.device), adj=adj)
    return data


@torch.no_grad()
def compute_attribute_vectors_avg_diff(embed: torch.Tensor, sens: torch.Tensor, sample=False,
                                       norm=False) -> torch.Tensor:
    pos_mask = (sens == 1).long()
    neg_mask = 1 - pos_mask

    if sample:
        neg_sample_num = min(pos_mask.sum(), neg_mask.sum()).item()
        np.random.seed(0)
        negative_indices = torch.nonzero(neg_mask).squeeze(1).detach().cpu().numpy()
        random_sample_negative_indices = np.random.choice(negative_indices, neg_sample_num, replace=False)
        neg_mask = torch.zeros_like(neg_mask)
        neg_mask[random_sample_negative_indices] = 1

    cnt_pos = pos_mask.count_nonzero().item()
    cnt_neg = neg_mask.count_nonzero().item()
    z_pos_per_attribute = torch.sum(embed * pos_mask.unsqueeze(1), 0)
    z_neg_per_attribute = torch.sum(embed * neg_mask.unsqueeze(1), 0)
    attr_vector = ((z_pos_per_attribute / cnt_pos) - (z_neg_per_attribute / cnt_neg))
    l2_norm = math.sqrt((attr_vector * attr_vector).sum())
    if norm is True:
        attr_vector /= l2_norm
    return attr_vector


def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
    return parity.item(), equality.item()


def load(path_root, dataset):
    raw_data_info = None
    if dataset == 'credit':
        sens_attr = "Age"  # column number after feature process is 1
        sens_idx = 1
        predict_attr = 'NoDefaultNextMonth'
        label_number = 6000
        path_credit = path_root / "credit"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(dataset, sens_attr, predict_attr,
                                                                                path=path_credit,
                                                                                label_number=label_number)
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features
    elif dataset.split('_')[0] == 'pokec':
        if dataset == 'pokec_z':
            dataset = 'region_job'
        elif dataset == 'pokec_n':
            dataset = 'region_job_2'
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        label_number = 6000
        sens_idx = 3
        path_pokec = path_root / "pokec"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec(dataset, sens_attr, predict_attr,
                                                                               path=path_pokec,
                                                                               label_number=label_number, )
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features
    elif dataset == 'income':
        sens_attr = "race"  # column number after feature process is 1
        sens_idx = 8
        predict_attr = 'income'
        label_number = 1000
        path_income = path_root / "income"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_income(dataset, sens_attr, predict_attr,
                                                                                path=path_income,
                                                                                label_number=label_number)
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features
    else:
        print('Invalid dataset name!!')
        exit(0)

    num_class = len(labels.unique())
    sens_1 = sum(sens == 1).item()
    sens_0 = sum(sens == 0).item()
    train_sens_1 = sum(sens[idx_train] == 1).item()
    train_sens_0 = sum(sens[idx_train] == 0).item()
    val_sens_1 = sum(sens[idx_val] == 1).item()
    val_sens_0 = sum(sens[idx_val] == 0).item()
    test_sens_1 = sum(sens[idx_test] == 1).item()
    test_sens_0 = sum(sens[idx_test] == 0).item()

    label_1 = sum(labels == 1).item()
    label_0 = sum(labels == 0).item()
    train_label_1 = sum(labels[idx_train] == 1).item()
    train_label_0 = sum(labels[idx_train] == 0).item()
    val_label_1 = sum(labels[idx_val] == 1).item()
    val_label_0 = sum(labels[idx_val] == 0).item()
    test_label_1 = sum(labels[idx_test] == 1).item()
    test_label_0 = sum(labels[idx_test] == 0).item()

    print(f"Dataset: {dataset}.", "node count:", features.shape[0], ' feature dim:', features.shape[1],
          ' num class:', num_class, ' label_1 num:', label_1, ' label_0 num:', label_0)
    print("positive sens num:", sens_1, " negative sens num:", sens_0)
    print('train size:', idx_train.shape[0], ' positive sens num:', train_sens_1, ' negative sens num:', train_sens_0,
          ' label_1 num:', train_label_1, ' label_0 num:', train_label_0)
    print('valid size:', idx_val.shape[0], ' positive sens num:', val_sens_1, ' negative sens num:', val_sens_0,
          ' label_1 num:', val_label_1, ' label_0 num:', val_label_0)
    print('test size:', idx_test.shape[0], ' positive sens num:', test_sens_1, ' negative sens num:', test_sens_0,
          ' label_1 num:', test_label_1, ' label_0 num:', test_label_0)

    return adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx, raw_data_info


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1


def load_credit(dataset, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="./dataset/credit/",
                label_number=3000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')

    #    # Normalize MaxBillAmountOverLast6Months
    #    idx_features_labels['MaxBillAmountOverLast6Months'] = (idx_features_labels['MaxBillAmountOverLast6Months']-idx_features_labels['MaxBillAmountOverLast6Months'].mean())/idx_features_labels['MaxBillAmountOverLast6Months'].std()
    #
    #    # Normalize MaxPaymentAmountOverLast6Months
    #    idx_features_labels['MaxPaymentAmountOverLast6Months'] = (idx_features_labels['MaxPaymentAmountOverLast6Months'] - idx_features_labels['MaxPaymentAmountOverLast6Months'].mean())/idx_features_labels['MaxPaymentAmountOverLast6Months'].std()
    #
    #    # Normalize MostRecentBillAmount
    #    idx_features_labels['MostRecentBillAmount'] = (idx_features_labels['MostRecentBillAmount']-idx_features_labels['MostRecentBillAmount'].mean())/idx_features_labels['MostRecentBillAmount'].std()
    #
    #    # Normalize MostRecentPaymentAmount
    #    idx_features_labels['MostRecentPaymentAmount'] = (idx_features_labels['MostRecentPaymentAmount']-idx_features_labels['MostRecentPaymentAmount'].mean())/idx_features_labels['MostRecentPaymentAmount'].std()
    #
    #    # Normalize TotalMonthsOverdue
    #    idx_features_labels['TotalMonthsOverdue'] = (idx_features_labels['TotalMonthsOverdue']-idx_features_labels['TotalMonthsOverdue'].mean())/idx_features_labels['TotalMonthsOverdue'].std()

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_pokec(dataset, sens_attr, predict_attr, path="../dataset/pokec/", label_number=6000):
    # print('Loading {} dataset from {}'.format(dataset,path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    # header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    # labels[labels == -1] = 0
    labels[labels > 1] = 1

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_income(dataset, sens_attr="race", predict_attr="income", path="../data/income/", label_number=1000):  # 1000
    print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]

    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    # print('building edge relationship complete')
    idx_map = np.array(idx_map)

    return idx_map
