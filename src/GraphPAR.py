# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   GraphPAR.py
@Time    :   2023/9/3 21:31
@Author  :   zhongjian zhang
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import optim
from tqdm import tqdm

from models.model import GNN, Adapter, Classifier
from certification.center_smoothing import CenterSmoothing
from certification.certification_utils import CertConsts, CertificationStatistics
from certification.classifier_smoothing import ClassifierSmoothing
from utils import calculate_parameter_count, compute_attribute_vectors_avg_diff, load_data, fair_metric, \
    get_path_to


class GraphPAR:
    def __init__(self, params, pgm_file):
        self.device = params.device
        self.dataset = params.dataset
        self.seed = params.seed
        self.data = load_data(params)

        # pre-trained graph model
        self.pre_train = params.pre_train
        self.hidden_dim = params.hidden_dim
        self.weight_decay = params.weight_decay
        self.PGM = GNN(self.data.input_dim, params.hidden_dim, params.num_layer, params.activation).to(params.device)
        self.PGM.load_state_dict(torch.load(pgm_file, map_location=torch.device(params.device)))

        # fairness
        if params.data_aug is True and params.adv_loss_weight == 0:
            self.model_name = "RandAT"
        elif params.data_aug is False and params.adv_loss_weight > 0:
            self.model_name = "MinMax"
        elif params.data_aug is False and params.adv_loss_weight == 0:
            self.model_name = "Naive"
        else:
            print("Unknown setting!")
            exit()
        self.tune_lr = 0.01
        self.perf_alpha = 0 if params.adv_loss_weight == 0 and params.data_aug is False else params.perf_alpha
        self.data_aug = params.data_aug
        self.perturb_epsilon = params.perturb_epsilon
        self.random_attack_num_samples = params.random_attack_num_samples
        self.adv_loss_weight = params.adv_loss_weight
        self.tune_epochs = params.tune_epochs
        self.cls_sigma = params.cls_sigma
        self.y = self.data.y
        self.idx_train, self.idx_val, self.idx_test = self.data.idx_train_list, self.data.idx_valid_list, self.data.idx_test_list
        self.sens_train, self.sens_val, self.sens_test = self.data.sens[self.idx_train], self.data.sens[self.idx_val], \
            self.data.sens[self.idx_test]
        self.PGM.eval()
        pgm_embeddings = self.PGM(self.data.x, self.data.edge_index).clone().detach()
        self.train_embeddings, self.valid_embeddings, self.test_embeddings = pgm_embeddings[self.idx_train], \
            pgm_embeddings[self.idx_val], pgm_embeddings[self.idx_test]
        self.sens_attr_vector = compute_attribute_vectors_avg_diff(self.train_embeddings, self.sens_train)
        self.adapter = Adapter(input_dim=params.hidden_dim, activation=params.activation).to(params.device)
        self.classifier = Classifier(input_dim=params.hidden_dim, num_classes=1).to(params.device)
        parameters = list(self.classifier.parameters()) + list(self.adapter.parameters())
        self.optimizer = optim.Adam(parameters, lr=self.tune_lr, weight_decay=self.weight_decay)

        # provable fairness
        # center smooth parameters
        self.adapter_sigma = params.perturb_epsilon / 2
        self.adapter_alpha = params.adapter_alpha
        self.adapter_n = params.adapter_n
        self.adapter_n0 = params.adapter_n0
        # random smooth parameters
        self.cls_alpha = params.cls_alpha
        self.cls_n = params.cls_n
        self.cls_n0 = params.cls_n0

    def loop(self, embeddings, label, sens, mode="train"):
        if self.data_aug is False:
            logits = self.classifier(self.adapter(embeddings))
            class_loss = F.binary_cross_entropy_with_logits(logits, label.unsqueeze(1).float())
        else:
            assert self.adv_loss_weight == 0
            noisy_embeds, y_repeated = self.augment_data(embeddings, label)
            train_embed_combined = torch.cat([embeddings, noisy_embeds])
            y_targets = torch.cat([label, y_repeated])
            logits = self.classifier(self.adapter(train_embed_combined))
            class_loss = F.binary_cross_entropy_with_logits(logits, y_targets.unsqueeze(1).float())
        if self.adv_loss_weight > 0:
            assert self.data_aug is False
            self.adapter.eval()
            train_embeddings_adv = self.get_adv_examples(embeddings)
            if mode == "train":
                self.adapter.train()
            adv_loss = self.calc_loss(embeddings, train_embeddings_adv).mean()
        else:
            adv_loss = 0
        total_loss = class_loss + self.adv_loss_weight * adv_loss
        if mode == "train":
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        self.adapter.eval(), self.classifier.eval()
        logits = self.classifier(self.adapter(embeddings))
        predictions = (logits > 0).type_as(label)
        acc = accuracy_score(y_true=label.cpu().detach().numpy(), y_pred=predictions.cpu().detach().numpy())
        f1 = f1_score(y_true=label.cpu().detach().numpy(), y_pred=predictions.cpu().detach().numpy())
        parity, equality = fair_metric(predictions.cpu().detach().numpy(), label.cpu().detach().numpy(),
                                       sens.cpu().detach().numpy())
        return acc * 100, f1 * 100, equality * 100, parity * 100, total_loss

    def calc_loss(self, embed: torch.Tensor, embed_adv: torch.Tensor) -> torch.Tensor:
        z_embed_adv = self.adapter(embed_adv)
        z_embed = self.adapter(embed)
        l_2 = torch.linalg.norm(z_embed - z_embed_adv, ord=2, dim=1)
        return l_2

    def train_robust_adapter(self):
        calculate_parameter_count(self.optimizer)
        model_path = get_path_to('saved_models')
        adapter_file = f'{model_path}/adapter/{self.pre_train}/{self.dataset}_{self.seed}_{self.model_name}_weights.pt'
        classifier_file = f'{model_path}/classifier/{self.pre_train}/{self.dataset}_{self.seed}_{self.model_name}_weights.pt'
        best_perf, test_acc, test_f1, test_equality, test_parity = -1, -1, -1, -1, -1
        for epoch in range(1, self.tune_epochs):
            # train
            self.adapter.train(), self.classifier.train()
            _ = self.loop(self.train_embeddings, self.y[self.idx_train], self.sens_train, mode="train")
            # eval
            self.adapter.eval(), self.classifier.eval()
            valid_acc, valid_f1, valid_equality, valid_parity, valid_loss = self.loop(self.valid_embeddings,
                                                                                      self.y[self.idx_val],
                                                                                      self.sens_val, mode="eval")
            valid_perf = (valid_acc + valid_f1) - self.perf_alpha * (valid_equality + valid_parity)
            if valid_perf > best_perf and epoch > 50:
                best_perf = valid_perf
                test_acc, test_f1, test_equality, test_parity, _ = self.loop(self.test_embeddings,
                                                                             self.y[self.idx_test], self.sens_test,
                                                                             mode="eval")
                print(
                    f"Epoch:{epoch}, Acc:{valid_acc:.4f}, F1: {valid_f1:.4f}, DP: {valid_parity:.4f}, EO: {valid_equality:.4f}")
                torch.save(self.adapter.state_dict(), adapter_file)
                torch.save(self.classifier.state_dict(), classifier_file)
        print(f"Test Acc:{test_acc:.4f}, F1: {test_f1:.4f}, DP: {test_parity:.4f}, EO: {test_equality:.4f}")
        return test_acc, test_f1, test_equality, test_parity

    def train_robust_classifier(self):
        """adversarial train classifier for maximizes the number of nodes"""
        model_path = get_path_to('saved_models')
        adapter_file = f'{model_path}/adapter/{self.pre_train}/{self.dataset}_{self.seed}_{self.model_name}_weights.pt'
        classifier_file = f'{model_path}/classifier/{self.pre_train}/{self.dataset}_{self.seed}_{self.model_name}_weights.pt'
        self.adapter.load_state_dict(torch.load(adapter_file, map_location=torch.device(self.device)))
        self.classifier.load_state_dict(torch.load(classifier_file, map_location=torch.device(self.device)))
        optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
        self.PGM.eval(), self.adapter.eval()
        adapter_embedding = self.adapter(self.PGM(self.data.x, self.data.edge_index)).clone().detach()
        train_embeddings, valid_embeddings, test_embeddings = adapter_embedding[self.idx_train], adapter_embedding[
            self.idx_val], adapter_embedding[self.idx_test]

        for epoch in range(0, 100):
            self.classifier.train()
            noise = torch.randn_like(train_embeddings, device=train_embeddings.device) * self.cls_sigma
            train_embed_adv = train_embeddings + noise
            logits = self.classifier(train_embed_adv)
            loss = F.binary_cross_entropy_with_logits(logits, self.y[self.idx_train].unsqueeze(1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        robust_classifier_file = f'{model_path}/adv_classifier/{self.pre_train}/{self.dataset}_{self.seed}_{self.model_name}_weights.pt'
        torch.save(self.classifier.state_dict(), robust_classifier_file)

    def certify(self):
        model_path = get_path_to('saved_models')
        adapter_file = f'{model_path}/adapter/{self.pre_train}/{self.dataset}_{self.seed}_{self.model_name}_weights.pt'
        robust_classifier_file = f'{model_path}/adv_classifier/{self.pre_train}/{self.dataset}_{self.seed}_{self.model_name}_weights.pt'
        self.adapter.load_state_dict(torch.load(adapter_file, map_location=torch.device(self.device)))
        self.classifier.load_state_dict(torch.load(robust_classifier_file, map_location=torch.device(self.device)))
        self.adapter.eval(), self.classifier.eval()
        center_smoothing = CenterSmoothing(self.sens_attr_vector, self.adapter, self.adapter_sigma)
        classifier_smoothing = ClassifierSmoothing(self.classifier, 1, self.cls_sigma)
        certification_statistics = CertificationStatistics()
        sample_index = 0
        time.sleep(1)
        pbar = tqdm(total=len(self.idx_test), ncols=100)
        for x, y, sens in zip(self.test_embeddings, self.y[self.idx_test], self.sens_test):
            certification_data = {CertConsts.SAMPLE_INDEX: sample_index}
            # center smooth
            start_time = time.time()
            center, r1, smoothing_error = center_smoothing.certify(x, self.perturb_epsilon)
            center_smoothing_time = time.time() - start_time
            if center is None:
                center = np.array([])
                assert r1 == CenterSmoothing.ABSTAIN
            certification_data.update({CertConsts.CSM_RADIUS: r1, CertConsts.CSM_ERROR: smoothing_error,
                                       CertConsts.CSM_TIME: center_smoothing_time,
                                       CertConsts.GROUND_TRUTH_LABEL: y.item()})
            sample_index += 1
            if r1 >= 0:
                # random smooth
                start_time = time.time()
                random_pred, r2 = classifier_smoothing.certify(center.unsqueeze(0), self.cls_n0, self.cls_n,
                                                               self.cls_alpha)
                random_smoothing_time = time.time() - start_time
                if r1 < r2:
                    r1_le_r2 = True
                else:
                    r1_le_r2 = False
                certification_data.update(
                    {CertConsts.SMOOTHED_CLS_PRED: random_pred, CertConsts.CERTIFIED_CLS_RADIUS: r2,
                     CertConsts.CERTIFIED_FAIRNESS: (random_pred != ClassifierSmoothing.ABSTAIN and r1_le_r2),
                     CertConsts.CERTIFIED_FAIRNESS_AND_ACCURACY: (random_pred == y.item() and r1_le_r2),
                     CertConsts.COHEN_SMOOTHING_TIME: random_smoothing_time})
            else:
                certification_data.update({CertConsts.SMOOTHED_CLS_PRED: ClassifierSmoothing.ABSTAIN})
                r2 = -1
                random_pred = torch.tensor(-1)
            certification_statistics.add(certification_data)
            pbar.set_postfix(r1=r1, r2=r2, random_pred=random_pred, label=y.item())
            pbar.update()
        certification_statistics.report()

    @torch.no_grad()
    def get_adv_examples(self, embed: torch.Tensor) -> torch.Tensor:
        noisy_emb_all = []
        losses_all = []
        for _ in range(self.random_attack_num_samples):
            noisy_emb = embed.clone()
            sens_attr_vector_repeated = torch.repeat_interleave(self.sens_attr_vector.unsqueeze(0), embed.shape[0],
                                                                dim=0)
            coeffs = (2 * torch.rand(embed.shape[0], 1, device=self.device) - 1) * self.perturb_epsilon
            noisy_emb += sens_attr_vector_repeated * coeffs
            noisy_emb_all.append(noisy_emb)
            loss = self.calc_loss(embed, noisy_emb)
            losses_all.append(loss.clone().detach())
        losses_all = torch.stack(losses_all, dim=1)
        _, idx = torch.max(losses_all, dim=1)
        adv_examples = []
        for i, sample_idx in enumerate(idx.cpu().tolist()):
            adv_examples.append(noisy_emb_all[sample_idx][i])
        return torch.stack(adv_examples, 0)

    @torch.no_grad()
    def augment_data(self, embed: torch.Tensor, y: torch.Tensor):
        assert y.dim() == 1 and self.sens_attr_vector is not None
        y_repeated = y.repeat_interleave(self.random_attack_num_samples)
        assert embed.dim() == 2 and embed.size(0) == y.size(0)
        noisy_latents = embed.repeat_interleave(self.random_attack_num_samples, dim=0).clone().detach()
        coeffs = (2 * torch.rand(noisy_latents.shape[0], 1, device=noisy_latents.device) - 1) * self.perturb_epsilon
        noisy_latents += self.sens_attr_vector * coeffs
        return noisy_latents, y_repeated
