from EXTRUST.utils.mutils import *
from EXTRUST.utils.inits import prepare
from EXTRUST.utils.loss import EnvLoss
from EXTRUST.utils.metric import Metric
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch.nn import functional as F
from torch import nn
from pathlib import Path
from tqdm import tqdm

import os
import sys
import time
import torch
import numpy as np
import torch.optim as optim
import pandas as pd
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
Path(BASE_DIR, 'checkpoint').mkdir(exist_ok=True)
Path(BASE_DIR, 'saved_model').mkdir(exist_ok=True)


class NeibSampler:
    def __init__(self, graph, num_nodes, nb_size, include_self=False):
        n = num_nodes
        assert 0 <= min(graph.nodes()) and max(graph.nodes()) < n
        if include_self:
            nb_all = torch.zeros(n, nb_size + 1, dtype=torch.int64)
            nb_all[:, 0] = torch.arange(0, n)
            nb = nb_all[:, 1:]
        else:
            nb_all = torch.zeros(n, nb_size, dtype=torch.int64)
            nb = nb_all
        popkids = []
        for v in range(graph.number_of_nodes()):
            nb_v = sorted(graph.neighbors(v))
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v)))
                nb[v] = torch.LongTensor(nb_v)
            else:
                popkids.append(v)
        self.include_self = include_self
        self.g, self.nb_all, self.pk = graph, nb_all, popkids

    def to(self, dev):
        self.nb_all = self.nb_all.to(dev)
        return self

    def sample(self):
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all
        nb_size = nb.size(1)
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)
        for i, v in enumerate(self.pk):
            pk_nb[i] = np.random.choice(sorted(self.g.neighbors(v)), nb_size)
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        return self.nb_all


class Runner(object):
    def __init__(self, args, model, gru, hldeconfounder, cvae, data, writer=None, **kwargs):
        seed_everything(args.seed)
        self.args = args
        self.data = data
        self.model = model
        self.gru = gru
        self.cvae = cvae
        self.hldeconfounder = hldeconfounder
        self.writer = writer
        self.len = len(data["train"]["edge_index_list"])
        self.len_train = self.len - args.testlength - args.vallength
        self.len_val = args.vallength
        self.len_test = args.testlength
        self.nbsz = args.nbsz
        self.n_factors = args.n_factors
        self.delta_d = args.delta_d
        self.d = self.n_factors * self.delta_d
        self.interv_size_ratio = args.interv_size_ratio

        if not args.use_node_feature:
            print('not use node feature!')
            self.x = None
        else:
            x = data["x"].to(args.device).clone().detach()
            self.x = [x for _ in range(self.len)] if len(x.shape) <= 2 else x

        if args.use_spacial_module:
            if "x_link" not in data:
                print("No edge feature found in data. Do you forget to run `scripts/data_prepare/add_feature.py`?")
                exit()
            self.x_link = data["x_link"].to(args.device)
            self.edge_index_link = data["edge_index_link"].to(args.device)
            self.edge_weight_link = data["edge_weight_link"].to(args.device)
            self.edge_index = data["edge_index"].to(args.device)
        else:
            print("not use spacial module!")

        if not args.use_gru_module:
            print("not use spacial module!")

        if not args.use_interv_module:
            print("not use intervention module!")

        self.edge_index_list_pre = [
            data["train"]["edge_index_list"][ix].long().to(args.device)
            for ix in range(self.len)
        ]
        neighbors_all = []
        for t in range(self.len):
            graph_data = Data(edge_index=self.edge_index_list_pre[t])
            graph = to_networkx(graph_data)
            sampler = NeibSampler(graph, args.num_nodes, self.nbsz)
            neighbors = sampler.sample().to(args.device)
            neighbors_all.append(neighbors)
        self.neighbors_all = torch.stack(neighbors_all).to(args.device)

        self.loss = EnvLoss(args)
        print("total length: {}, test length: {}".format(self.len, args.testlength))
        print("use threshold:", args.threshold)

    def cal_fact_rank(self, x_all):
        times = len(x_all)
        n = len(x_all[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        x_all = x_all.view(times, n, k, delta_d)
        x_all_trans = x_all.permute(1, 2, 0, 3)
        points = torch.var(x_all_trans, dim=[2, 3])
        rank = torch.argsort(points, 0, descending=True)
        return rank

    def cal_fact_var(self, x_all):
        times = len(x_all)
        n = len(x_all[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        x_all_trans = x_all.permute(1, 2, 0, 3)
        points = torch.var(x_all_trans, dim=[2, 3]).view(n, k)
        return points

    def intervention(self, x_all):
        times = len(x_all)
        n = len(x_all[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        m = int(self.interv_size_ratio * n)
        indices = torch.randperm(n)[:m]
        x_m = x_all[:, indices, :].view(times, m, k, delta_d)
        mask = self.cal_mask(x_m)
        mask_expand = torch.repeat_interleave(mask, delta_d, dim=1).view(m, k * delta_d)
        mask_expand = torch.stack([mask_expand] * times)

        saved_env = self.saved_env(x_m, mask)
        sampled_env = self.gen_env(saved_env)

        x_m = x_m.view(times, m, k * delta_d)
        embeddings_interv = x_m * mask_expand + sampled_env * (1 - mask_expand)
        x_all[:, indices, :] = embeddings_interv.to(torch.float32)
        return x_all, indices

    def intervention_faster(self, x_all):
        times = len(x_all)
        n = len(x_all[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        x_all = x_all.view(times, n, k, delta_d)
        mask = self.cal_mask_faster(x_all)
        mask_expand = torch.repeat_interleave(mask, delta_d, dim=1).view(n, k * delta_d)
        mask_expand = torch.stack([mask_expand] * times)

        saved_env = self.saved_env(x_all, mask)
        sampled_env = self.gen_env(saved_env)

        x_all = x_all.view(times, n, k * delta_d)
        embeddings_interv = x_all * mask_expand + sampled_env * (1 - mask_expand)
        embeddings_interv = embeddings_interv.to(torch.float32)
        return embeddings_interv

    def intervention_final(self, x_all_original):
        x_all = x_all_original.clone()
        times = len(x_all)
        n = len(x_all[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        m = int(self.interv_size_ratio * n)
        indices = torch.randperm(n)[:m]
        x_m = x_all[:, indices, :].view(times, m, k, delta_d)
        mask = self.cal_mask_faster(x_m)
        mask_expand = torch.repeat_interleave(mask, delta_d, dim=1).view(m, k * delta_d)
        mask_expand = torch.stack([mask_expand] * times)

        saved_env = self.saved_env(x_m, mask)
        sampled_env = self.gen_env(saved_env)

        x_m = x_m.view(times, m, k * delta_d)
        embeddings_interv = x_m * mask_expand + sampled_env * (1 - mask_expand)
        x_all[:, indices, :] = embeddings_interv.to(torch.float32)
        return x_all, indices

    def cal_mask(self, x_m):
        times = len(x_m)
        m = len(x_m[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        var = self.cal_fact_var(x_m).cpu().detach().numpy()

        def split_array(arr):
            n = len(arr)
            total_sum = sum(arr)
            dp = [[False for _ in range(total_sum + 1)] for __ in range(n + 1)]
            dp[0][0] = True
            for i in range(1, n + 1):
                for j in range(total_sum + 1):
                    if j >= arr[i - 1]:
                        dp[i][j] = dp[i - 1][j] or dp[i - 1][j - arr[i - 1]]
                    else:
                        dp[i][j] = dp[i - 1][j]
            min_diff = float("inf")
            for j in range(total_sum // 2, -1, -1):
                if dp[n][j]:
                    min_diff = total_sum - 2 * j
                    break
            return min_diff

        def process_matrix(matrix):
            matrix = adjust_matrix(matrix)
            n, k = matrix.shape
            result = np.zeros((n, k))
            for i in range(n):
                row = matrix[i]
                min_diff = split_array(row)
                avg = sum(row) / k
                for j in range(k):
                    if row[j] <= avg - min_diff / 2:
                        result[i][j] = 1
                    else:
                        result[i][j] = 0
                if np.sum(result[i]) == 0:
                    index = np.argmin(result[i])
                    result[i][index] = 1
            return result

        def adjust_matrix(matrix):
            for row in matrix:
                while np.min(row) < 1:
                    row *= 10
            matrix_min = matrix.min(axis=1).astype(int)
            matrix_min = np.expand_dims(matrix_min, axis=1)
            matrix_min = np.tile(matrix_min, (1, len(matrix[1])))
            matrix = matrix - matrix_min

            for row in matrix:
                while np.min(row) < 1:
                    row *= 10

            return matrix.astype(int)

        mask = process_matrix(var)
        return torch.from_numpy(mask).to(self.args.device)

    def cal_mask_faster(self, x_m):
        times = len(x_m)
        m = len(x_m[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        var = self.cal_fact_var(x_m)

        def max_avg_diff_index(sorted_tensor):
            n, k = sorted_tensor.shape
            result = np.zeros(n)
            for i in range(n):
                row = sorted_tensor[i]
                max_diff = 0
                max_index = 0
                for j in range(1, k):
                    avg1 = sum(row[:j]) / j
                    avg2 = sum(row[j:]) / (k - j)
                    diff = abs(avg1 - avg2)
                    if diff <= max_diff:
                        break
                    if diff > max_diff:
                        max_diff = diff
                        max_index = j
                result[i] = max_index - 1
            return result

        var_sorted = torch.sort(var, dim=1)
        var_sorted_index = var_sorted.indices
        indices = max_avg_diff_index(var_sorted.values).astype(int)
        for i in range(var.shape[0]):
            sort_indices = var_sorted_index[i]
            values = var[i, sort_indices]
            mask = torch.zeros_like(values)
            mask[: indices[i] + 1] = 1
            var[i, sort_indices] = mask

        return var

    def saved_env(self, x_all, mask):
        times = len(x_all)
        n = len(x_all[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        m = mask.shape[0]
        x_all = x_all.view(times, n, d)
        mask_env = 1 - mask
        mask_expand = torch.repeat_interleave(mask_env, delta_d, dim=1).view(
            m, k * delta_d
        )
        mask_expand = torch.stack([mask_expand] * times)

        extract_env = (
            (x_all * mask_expand).view(times, n, k, delta_d).permute(2, 0, 1, 3)
        )
        extract_env = extract_env.view(k, times * n, delta_d)
        extract_env = extract_env[:, torch.randperm(times * n), :]
        for i in range(k):
            zero_rows = (extract_env[i].sum(dim=1) == 0).nonzero(as_tuple=True)[0]
            non_zero_rows = (extract_env[i].sum(dim=1) != 0).nonzero(as_tuple=True)[0]
            if len(non_zero_rows) > 0:
                replacement_rows = non_zero_rows[
                    torch.randint(0, len(non_zero_rows), (len(zero_rows),))
                ]
                extract_env[i][zero_rows] = extract_env[i][replacement_rows]

        return extract_env.view(times, n, d)

    def gen_env(self, extract_env):
        times = len(extract_env)
        n = len(extract_env[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        n_gen = int(self.args.gen_ratio * n)

        z = torch.randn(n_gen * k, self.args.d_for_cvae).to(self.args.device)
        y = torch.ones(n_gen, k)
        for i in range(k):
            y[:, i : i + 1] = y[:, i : i + 1] * i
        y_T = y.transpose(0, 1)
        y = (F.one_hot(y_T.reshape(-1).to(torch.int64))).to(self.args.device)
        gen_env = self.cvae.decode(z, y).view(n_gen, k * delta_d)

        random_indices = torch.randperm(n)[:n_gen]
        extract_env[:, random_indices] = gen_env
        return extract_env.view(times, n, d)

    def loss_cvae(self, recon, x, mu, log_std) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="sum")
        kl_loss = -0.5 * (1 + 2 * log_std - mu.pow(2) - torch.exp(2 * log_std))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_loss
        return loss

    def train(self, epoch, data):
        args = self.args
        self.model.train()
        optimizer = self.optimizer

        embeddings = self.model(
            [
                data["edge_index_list"][ix].long().to(args.device)
                for ix in range(self.len)
            ],
            self.x,
            self.neighbors_all,
        )

        if args.use_gru_module:
            embeddings = self.gru(embeddings)

        if args.use_interv_module:
            y = torch.ones(len(embeddings[0]) * len(embeddings), args.n_factors)
            for i in range(args.n_factors):
                y[:, i : i + 1] = y[:, i : i + 1] * i
            y_T = y.transpose(0, 1)
            y = (F.one_hot(y_T.reshape(-1).to(torch.int64))).to(args.device)

            embeddings_view = embeddings.view(
                len(embeddings), len(embeddings[0]), args.n_factors, args.delta_d
            )
            embeddings_trans = embeddings_view.permute(2, 0, 1, 3)
            x_flatten = torch.flatten(embeddings_trans, start_dim=0, end_dim=2)
            recon, mu, log_std = self.cvae(x_flatten, y)
            cvae_loss = self.loss_cvae(recon, x_flatten, mu, log_std) / (
                len(embeddings[0] * len(embeddings[0]))
            )
        else:
            cvae_loss = 0

        if args.use_spacial_module:
            spacial_embeddings = self.hldeconfounder(embeddings,
                                             self.x_link,
                                             self.edge_index_link,
                                             self.edge_weight_link,
                                             self.edge_index)
            final_embeddings = embeddings + spacial_embeddings
        else:
            final_embeddings = embeddings

        train_metric = Metric(args)
        val_metric = Metric(args)
        test_metric = Metric(args)
        for t in range(self.len - 1):
            z = final_embeddings[t]
            _, pos_edge, neg_edge = prepare(data, t + 1)[:3]
            result = self.loss.predict(z, pos_edge, neg_edge, self.model.edge_decoder)
            if t < self.len_train - 1:
                train_metric.add_result(result)
            elif t < self.len_train + self.len_val - 1:
                val_metric.add_result(result)
            else:
                test_metric.add_result(result)

        device = final_embeddings[0].device
        edge_index = []
        pos_edge_index_all = []
        neg_edge_index_all = []
        edge_label = []
        tsize = []
        for t in range(self.len_train - 1):
            z = final_embeddings[t]
            pos_edge_index = prepare(data, t + 1)[0]
            if args.dataset == "yelp":
                neg_edge_index = bi_negative_sampling(
                    pos_edge_index, args.num_nodes, args.shift
                )
            else:
                neg_edge_index = negative_sampling(
                    pos_edge_index,
                    num_neg_samples=pos_edge_index.size(1) * args.sampling_times,
                )
            edge_index.append(torch.cat([pos_edge_index, neg_edge_index], dim=-1))
            pos_edge_index_all.append(pos_edge_index)
            neg_edge_index_all.append(neg_edge_index)
            pos_y = z.new_ones(pos_edge_index.size(1)).to(device)
            neg_y = z.new_zeros(neg_edge_index.size(1)).to(device)
            edge_label.append(torch.cat([pos_y, neg_y], dim=0))
            tsize.append(pos_edge_index.shape[1] * 2)

        edge_label = torch.cat(edge_label, dim=0)

        def generate_edge_index_interv(pos_edge_index_all, neg_edge_index_all, indices):
            edge_label = []
            edge_index = []
            pos_edge_index_interv = pos_edge_index_all.copy()
            neg_edge_index_interv = neg_edge_index_all.copy()
            index = indices.cpu().numpy()
            for t in range(self.len_train - 1):
                mask_pos = np.logical_and(
                    np.isin(pos_edge_index_interv[t].cpu()[0], index),
                    np.isin(pos_edge_index_interv[t].cpu()[1], index),
                )
                pos_edge_index = pos_edge_index_interv[t][:, mask_pos]
                mask_neg = np.logical_and(
                    np.isin(neg_edge_index_interv[t].cpu()[0], index),
                    np.isin(neg_edge_index_interv[t].cpu()[1], index),
                )
                neg_edge_index = neg_edge_index_interv[t][:, mask_neg]
                pos_y = z.new_ones(pos_edge_index.size(1)).to(device)
                neg_y = z.new_zeros(neg_edge_index.size(1)).to(device)
                edge_label.append(torch.cat([pos_y, neg_y], dim=0))
                edge_index.append(torch.cat([pos_edge_index, neg_edge_index], dim=-1))
            edge_label = torch.cat(edge_label, dim=0)
            return edge_label, edge_index

        def cal_y(embeddings, decoder):
            preds = torch.tensor([]).to(device)
            for t in range(self.len_train - 1):
                z = embeddings[t]
                pred = decoder(z, edge_index[t])
                preds = torch.cat([preds, pred])
            return preds

        def cal_y_interv(embeddings, decoder, edge_index_interv, indices):
            index = indices.cpu().numpy()
            preds = torch.tensor([]).to(device)
            for t in range(self.len_train - 1):
                z = embeddings[t]
                pred = decoder(z, edge_index_interv[t])
                preds = torch.cat([preds, pred])
            return preds

        criterion = torch.nn.BCELoss()

        def cal_loss(y, label):
            return criterion(y, label)

        pred_y = cal_y(final_embeddings, self.model.edge_decoder)
        main_loss = cal_loss(pred_y, edge_label)

        if args.use_interv_module:
            intervention_times = args.n_intervene
            env_loss = torch.tensor([]).to(device)
            for i in range(intervention_times):
                embeddings_interv, indices = self.intervention_final(embeddings)
                edge_label_interv, edge_index_interv = generate_edge_index_interv(
                    pos_edge_index_all, neg_edge_index_all, indices
                )
                pred_y_interv = cal_y_interv(
                    embeddings_interv, self.model.edge_decoder, edge_index_interv, indices
                )
                env_loss = torch.cat(
                    [env_loss, cal_loss(pred_y_interv, edge_label_interv).unsqueeze(0)]
                )
            var_loss = torch.var(env_loss)
        else:
            var_loss = 0

        alpha = args.alpha
        beta = args.beta

        if epoch % args.every_epoch == 0:
            loss = main_loss + alpha * var_loss + beta * cvae_loss
        else:
            loss = main_loss + alpha * var_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        average_epoch_loss = loss.item()

        return average_epoch_loss, train_metric.get_scores(), val_metric.get_scores(), test_metric.get_scores()

    def run(self):
        args = self.args
        min_epoch = args.min_epoch
        max_patience = args.patience
        patience = 0
        self.optimizer = optim.Adam(
            [p for n, p in self.model.named_parameters() if "ss" not in n],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        t_total0 = time.time()
        max_indicator_score = 0
        max_train_scores = {}
        max_val_scores = {}
        max_test_scores = {}

        with tqdm(range(1, args.max_epoch + 1)) as bar:
            for epoch in bar:
                t0 = time.time()
                epoch_losses, train_scores, val_scores, test_scores = self.train(
                    epoch, self.data["train"]
                )
                average_epoch_loss = epoch_losses

                if val_scores[args.early_stopping_indicator] > max_indicator_score:
                    max_indicator_score = val_scores[args.early_stopping_indicator]

                    max_train_scores = train_scores
                    max_val_scores = val_scores
                    max_test_scores = test_scores

                    test_results = self.test(epoch, self.data["test"])

                    patience = 0

                    filepath = BASE_DIR + "/checkpoint/" + self.args.dataset + ".pth"
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "cvae_state_dict": self.cvae.state_dict(),
                            "gru_state_dict": self.gru.state_dict(),
                            "hldeconfounder_state_dict": self.hldeconfounder.state_dict(),
                        },
                        filepath,
                    )

                else:
                    patience += 1
                    if epoch > min_epoch and patience > max_patience:
                        break
                if epoch == 1 or epoch % self.args.log_interval == 0:
                    print(
                        "Epoch:{}, Loss: {:.4f}, Time: {:.3f}".format(
                            epoch, average_epoch_loss, time.time() - t0
                        )
                    )
                    def format_score(score_dict):
                        return {k: f'{v:.4f}' for k, v in score_dict.items()}

                    print(
                        f"Current: Epoch:{epoch}, Train: {format_score(train_scores)}, Val Scores: {format_score(val_scores)}, Test Scores: {format_score(test_scores)}"
                    )

                    print(
                        f"Train: Epoch:{test_results[0]}, Train Scores:{format_score(max_train_scores)}, Val Scores: {format_score(max_val_scores)}, Test Scores: {format_score(max_test_scores)}"
                    )
                    print(
                        f"Test: Epoch:{test_results[0]}, Train Scores:{format_score(test_results[1])}, Val Scores: {format_score(test_results[2])}, Test Scores: {format_score(test_results[3])}"
                    )

        args.device = args.device.type
        return {
            "dataset_config": self.data['config'],
            "args": vars(args),
            "total_time": time.time() - t_total0,
            "total_epoch": epoch,
            "train_train_scores": max_train_scores,
            "train_test_scores": max_test_scores,
            "test_train_scores": test_results[1],
            "test_test_scores": test_results[3],
        }

    def test(self, epoch, data):
        args = self.args

        embeddings = self.model(
            [
                data["edge_index_list"][ix].long().to(args.device)
                for ix in range(self.len)
            ],
            self.x,
            self.neighbors_all,
        )
        if args.use_gru_module:
            embeddings = self.gru(embeddings)
        if args.use_spacial_module:
            spacial_embeddings = self.hldeconfounder(embeddings,
                                                     self.x_link,
                                                     self.edge_index_link,
                                                     self.edge_weight_link,
                                                     self.edge_index)
            embeddings += spacial_embeddings

        train_metric = Metric(args)
        val_metric = Metric(args)
        test_metric = Metric(args)
        for t in range(self.len - 1):
            z = embeddings[t]
            edge_index, pos_edge, neg_edge = prepare(data, t + 1)[:3]
            if is_empty_edges(neg_edge):
                continue
            result = self.loss.predict(z, pos_edge, neg_edge, self.model.edge_decoder)
            if t < self.len_train - 1:
                train_metric.add_result(result)
            elif t < self.len_train + self.len_val - 1:
                val_metric.add_result(result)
            else:
                test_metric.add_result(result)

        return [
            epoch,
            train_metric.get_scores(),
            val_metric.get_scores(),
            test_metric.get_scores(),
        ]

    def re_run(self):
        args = self.args
        data = self.data["test"]

        self.optimizer = optim.Adam(
            [p for n, p in self.model.named_parameters() if "ss" not in n],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        filepath = BASE_DIR + "/saved_model/" + self.args.dataset + ".pth"
        checkpoint = torch.load(filepath)

        self.cvae.load_state_dict(checkpoint["cvae_state_dict"])
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()
        embeddings = self.model(
            [
                data["edge_index_list"][ix].long().to(args.device)
                for ix in range(self.len)
            ],
            self.x,
            self.neighbors_all,
        )

        train_metric = Metric(args)
        val_metric = Metric(args)
        test_metric = Metric(args)
        for t in range(self.len - 1):
            z = embeddings[t]
            edge_index, pos_edge, neg_edge = prepare(data, t + 1)[:3]
            if is_empty_edges(neg_edge):
                continue
            result = self.loss.predict(z, pos_edge, neg_edge, self.model.edge_decoder)
            if t < self.len_train - 1:
                train_metric.add_result(result)
            elif t < self.len_train + self.len_val - 1:
                val_metric.add_result(result)
            else:
                test_metric.add_result(result)

        test_res = [
            0,
            train_metric.get_scores(),
            val_metric.get_scores(),
            test_metric.get_scores(),
        ]
        print('train scores:', test_res[1])
        print('val scores:', test_res[2])
        print('test scores:', test_res[3])
        return test_res
