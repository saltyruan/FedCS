# coding=utf-8
from alg.fedavg import fedavg
import random
import torch
import copy
import numpy as np
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from util.traineval import train, train_prox


class fedprox_nt(fedavg):
    def __init__(self, args):
        super(fedprox_nt, self).__init__(args)

    def client_train(self, c_idx, dataloader, round):
        server_model = copy.deepcopy(self.client_model[c_idx]).to(self.args.device)
        if round > 0:
            train_loss, train_acc = train_prox(
                self.args, self.client_model[c_idx], server_model, dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        else:
            train_loss, train_acc = train(
                self.client_model[c_idx], dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        return train_loss, train_acc

    def distill_wlocals(self, public_dataset, args, a_iter):
        """
        save local prediction for one-shot distillation
        """
        distill_loader = DataLoader(
        dataset=public_dataset, batch_size=args.batch, shuffle=False, 
        num_workers=8, pin_memory=True, sampler=None)
        self.distil_loader = distill_loader
        total_logits = []
        for i, (images, _, idx) in enumerate(self.distil_loader):
            # import ipdb; ipdb.set_trace()
            images = images.cuda()
            batch_logits = []
            for n in range(len(self.client_model)):
                tmodel = copy.deepcopy(self.client_model[n])
                logits = tmodel(images)
                batch_logits.append(logits.detach())
                del tmodel
            batch_logits = torch.stack(batch_logits).cpu()#(nl, nb, ncls)
            total_logits.append(batch_logits)
        self.total_logits = torch.cat(total_logits,dim=1).permute(1,0,2) #(nsample, nl, ncls)
        del total_logits, batch_logits
        self.distil_loader = DataLoader(
            dataset=public_dataset, batch_size=args.batch, shuffle=True, 
            num_workers=8, pin_memory=True, sampler=None)


    def get_per_logits_soft(self, args, iter):
        # 获得每个本地客户端的代表语义[nsample, nclient, nclass] [nclinet, nclass]
        alp = torch.full((args.n_clients, args.n_clients), args.alp)
        total_logits = self.total_logits.mean(dim=0)
        # 获得客户端之间的相似度矩阵 [nclient, nclient]
        client_num = total_logits.shape[0]
        weight_m = np.zeros((client_num, client_num))
        selected_clusters = {i: [] for i in range(client_num)}
        for i in range(client_num):
            for j in range(client_num):
                if i == j:
                    weight_m[i, j] = 1
                else:
                    weight_m[i, j] = torch.cosine_similarity(torch.nn.functional.softmax(total_logits[i], dim=0), 
                                        torch.nn.functional.softmax(total_logits[j], dim=0), dim=0)
        for i in range(client_num):
            for j in range(client_num):
                # 等概率随机采样rand
                rand = random.random()
                if rand < weight_m[i,j]:
                    selected_clusters[i].append(j)
        weight_m_average = np.mean(weight_m, axis=1) # 是使用平均值、最小值或其他形式暂定
        wandb.log({"client0": weight_m_average[0], "epoch":iter})
        wandb.log({"client1": weight_m_average[1], "epoch":iter})
        wandb.log({"client2": weight_m_average[2], "epoch":iter})
        wandb.log({"client10": weight_m_average[10], "epoch":iter})
        wandb.log({"client19": weight_m_average[19], "epoch":iter})
        # 获得传送给每个客户的logits
        num_cluster = [len(v) for k,v in selected_clusters.items()]
        weightmatrix = []
        for k,v in selected_clusters.items():
            selectN = v
            total_logits = self.total_logits.permute(1,0,2).cuda()  #nlocal*nsamples*ncls
            total_logits = total_logits[torch.tensor(selectN)] #nlocal*nsamples*ncls
            gap = torch.from_numpy(weight_m[k][v])
            if num_cluster[k] == 1:
                localweight = [1]
            else:
                localweight = torch.nn.functional.softmax(gap, dim=0)
                # localweight = gap / gap.sum()
                # localweight = [1/num_cluster[k] for _ in v]
            weight_one = torch.zeros(client_num)
            n = 0
            for m in selectN:
                weight_one[m] = localweight[n]
                n = n + 1
            weightmatrix.append(weight_one)
        weightmatrix = torch.stack(weightmatrix, dim=0)
        if iter < 1:
            self.client_weight = weightmatrix
        else:
            client_weight_old = self.client_weight
            client_weight_new = client_weight_old * alp + weightmatrix * (1-alp)
            self.client_weight = client_weight_new
