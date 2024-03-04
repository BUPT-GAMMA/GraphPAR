import argparse
import warnings

import numpy as np
import torch
import torch.cuda as cuda
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import optim
from torch_geometric.nn import DeepGraphInfomax

from src.models import GNN, Classifier
from src.utils import load_data, setup_seed, calculate_parameter_count, fair_metric, get_path_to

warnings.filterwarnings('ignore')


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


def train(params, data, pgm_file):
    PGM = GNN(data.input_dim, params.hidden_dim, params.num_layer, params.activation)
    model = DeepGraphInfomax(hidden_channels=params.hidden_dim, encoder=PGM,
                             summary=lambda h, *args, **kwargs: torch.sigmoid(h.mean(dim=0)), corruption=corruption)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.pre_lr, weight_decay=params.weight_decay)
    calculate_parameter_count(optimizer)
    model = model.to(params.device)
    best_loss = 10000
    for epoch in range(1, params.pre_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(data.x, data.edge_index)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        if loss < best_loss:
            model.eval()
            torch.save(PGM.state_dict(), pgm_file)
            best_loss = loss
        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{params.pre_epochs}, Loss: {loss:.4f}')


def loop(classifier, optimizer, embeddings, labels, sens, mode="train"):
    logits = classifier(embeddings)
    loss = F.binary_cross_entropy_with_logits(logits, labels.unsqueeze(1).float())
    if mode == "train":
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    predictions = (logits > 0).type_as(labels)
    acc = accuracy_score(y_true=labels.cpu().detach().numpy(), y_pred=predictions.cpu().detach().numpy())
    f1 = f1_score(y_true=labels.cpu().detach().numpy(), y_pred=predictions.cpu().detach().numpy())
    parity, equality = fair_metric(predictions.cpu().detach().numpy(), labels.cpu().detach().numpy(),
                                   sens.cpu().detach().numpy())
    return acc * 100, f1 * 100, equality * 100, parity * 100, loss


def test_pgm(data, params, pgm_file):
    PGM = GNN(data.input_dim, params.hidden_dim, params.num_layer, params.activation).to(params.device)
    PGM.load_state_dict(torch.load(pgm_file, map_location=torch.device(params.device)))
    PGM.eval()
    pgm_embeddings = PGM(data.x, data.edge_index).clone().detach()
    idx_train, idx_val, idx_test = data.idx_train_list, data.idx_valid_list, data.idx_test_list
    sens_train, sens_val, sens_test = data.sens[idx_train], data.sens[idx_val], data.sens[idx_test]
    train_embeddings, valid_embeddings, test_embeddings = pgm_embeddings[idx_train], pgm_embeddings[idx_val], \
        pgm_embeddings[idx_test]
    classifier = Classifier(input_dim=params.hidden_dim, num_classes=1).to(params.device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.01, weight_decay=params.weight_decay)
    best_perf = -1
    test_acc, test_f1, test_equality, test_parity = -1, -1, -1, -1
    for epoch in range(1000):
        classifier.train()
        _ = loop(classifier, optimizer, train_embeddings, data.y[idx_train], sens_train)
        classifier.eval()
        valid_acc, valid_f1, valid_equality, valid_parity, valid_loss = loop(classifier, optimizer, valid_embeddings,
                                                                             data.y[idx_val], sens_val, mode="eval")
        valid_perf = (valid_acc + valid_f1)
        if valid_perf > best_perf and epoch > 20:
            best_perf = valid_perf
            test_acc, test_f1, test_equality, test_parity, _ = loop(classifier, optimizer, test_embeddings,
                                                                    data.y[idx_test], sens_test, mode="eval")
            print(
                f"Epoch:{epoch}, Acc:{valid_acc:.4f}, F1: {valid_f1:.4f}, DP: {valid_parity:.4f}, EO: {valid_equality:.4f}")
    print(f"Test Acc:{test_acc:.4f}, F1: {test_f1:.4f}, DP: {test_parity:.4f}, EO: {test_equality:.4f}")
    return test_acc, test_f1, test_equality, test_parity


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--pre_train', type=str, default='infomax')
    parser.add_argument('--device', type=int, default=1 if cuda.is_available() else 'cpu')
    parser.add_argument('--dataset', type=str, default='pokec_z', choices=['income', 'credit', 'pokec_z', 'pokec_n'])
    parser.add_argument('--hidden_dim', type=int, default=24)
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--pre_epochs', type=int, default=2000)
    parser.add_argument('--pre_lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--activation', type=str, default='leakyrelu', choices=['relu', 'leakyrelu', 'prelu'])
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
        print("Save the PGM to:", pgm_file)
        data = load_data(params)
        train(params, data, pgm_file)
        acc, f1, equality, parity = test_pgm(data, params, pgm_file)
        acc_array = np.append(acc_array, acc)
        f1_array = np.append(f1_array, f1)
        equality_array = np.append(equality_array, equality)
        parity_array = np.append(parity_array, parity)
    print(f"Acc(↑):{acc_array.mean():.2f}±{acc_array.std():.2f}, F1(↑):{f1_array.mean():.2f}±{f1_array.std():.2f}, "
          f"DP(↓):{parity_array.mean():.2f}±{parity_array.std():.2f}, EO(↓):{equality_array.mean():.2f}±{equality_array.std():.2f}")


if __name__ == '__main__':
    main()
