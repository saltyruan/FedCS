import logging
import random
from matplotlib import pyplot as plt
import torch
import copy
import wandb
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from alg.fedavg import fedavg

from util.modelsel import modelsel
from util.traineval import train, test
from alg.core.comm import communication
from util.draw import heatmap, annotate_heatmap


class fedavg_nt(fedavg):
    def __init__(self, args):
        super(fedavg_nt, self).__init__(args)

    def set_client_weight(self):
        preckpt = '/home/featurize/work/PersonalizationFL-new/new_cks_balance/fed_selectall_medmnist_base_0.1_0.1_non_iid_dirichlet_0.6_0/base'
        i = 0
        for model in self.client_model:
            model.load_state_dict(torch.load(preckpt)['client_model_' + str(i)])
            i = i + 1
        del preckpt

    def distill_wlocals(self, public_dataset, args, a_iter):
        """
        save local prediction for one-shot distillation
        """
        distill_loader = DataLoader(dataset=public_dataset, batch_size=args.batch, shuffle=False)
        total_logits = []
        print(a_iter)
        if(a_iter>1):
            self.last_logot=self.total_logits
        for i, (images, _, idx) in enumerate(distill_loader):
            # import ipdb; ipdb.set_trace()
            images = images.cuda()
            batch_logits = []
            for n in range(len(self.client_model)):
                tmodel = copy.deepcopy(self.client_model[n])
                logits = tmodel(images)
                batch_logits.append(logits.detach())
                del tmodel
            # 其中nl表示客户端模型的数量，nb表示批次大小，ncls表示类别数量。
            batch_logits = torch.stack(batch_logits).cpu()  # (nl, nb, ncls)
            total_logits.append(batch_logits)
        self.total_logits = torch.cat(total_logits, dim=1).permute(1, 0, 2)  # (nsample, nl, ncls)
        del total_logits, batch_logits

    def get_per_logits_soft(self, args, iter):
        # 获得每个本地客户端的代表语义[nsample, nclient, nclass] ->[nclinet, nclass]
        total_logits = self.total_logits.mean(dim=0)

        # 客户端数量
        client_num = total_logits.shape[0]
        # 获得客户端之间的相似度矩阵 [nclient, nclient]
        weight_m = np.zeros((client_num, client_num))
        for i in range(client_num):
            for j in range(client_num):
                if i==j:
                    weight_m[i, j] = 1
                else:
                    weight_m[i, j] = torch.cosine_similarity(torch.nn.functional.softmax(total_logits[i], dim=0),
                        torch.nn.functional.softmax(total_logits[j], dim=0), dim=0)
        selected_clusters = {i: [] for i in range(client_num)}
        for i in range(client_num):
            for j in range(client_num):
                # 等概率随机采样rand
                rand = random.random()
                if rand < weight_m[i, j]:
                    selected_clusters[i].append(j)
        # 获得传送给每个客户的logits
        num_cluster = [len(v) for k, v in selected_clusters.items()]
        print(f"当前客户端分组为{selected_clusters}")

        weightmatrix = []
        for k, v in selected_clusters.items():
            selectN = v
            gap = torch.from_numpy(weight_m[k][v])
            if num_cluster[k]==1:
                localweight = [1]
            else:
                localweight = torch.nn.functional.softmax(gap, dim=0)

            weight_one = torch.zeros(client_num)
            n = 0
            for m in selectN:
                weight_one[m] = localweight[n]
                n = n + 1
            weightmatrix.append(weight_one)
        weightmatrix = torch.stack(weightmatrix, dim=0)
        self.client_weight = weightmatrix
        print(self.client_weight)