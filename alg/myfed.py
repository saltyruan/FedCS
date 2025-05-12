import random
from cgi import maxlen

import torch
import copy
import wandb
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import deque

from util.log import init_loggers
from util.modelsel import modelsel
from util.traineval import train, test
from util.log import loger
from alg.core.comm import communication
from alg.SAM import SAM
from scipy.stats import beta

class myfed(nn.Module):
    def __init__(self,args):
        super(myfed, self).__init__()
        self.server_model,self.client_model ,self.client_weight= modelsel(args,args.device)
        self.optimizers = [optim.SGD(params=self.client_model[idx].parameters(
        ), lr=args.lr) for idx in range(args.n_clients)]
        self.loss_fun=nn.CrossEntropyLoss()
        self.args=args
        self.total_logits=[]
        self.group=[]
        self.history_client = [deque(maxlen = 3) for idx in range(args.n_clients)]
        self.iterations = [deque(maxlen = 3) for idx in range(args.n_clients)]

    def client_train(self,client_idx,dataloader,round=None):
        train_loss,train_acc=train(
            self.client_model[client_idx], dataloader, self.optimizers[client_idx], self.loss_fun, self.args.device
        )
        # self.history_client[client_idx].append(self.client_model[client_idx].state_dict())
        return train_loss,train_acc

    def client_eval(self, c_idx, dataloader):
        train_loss, train_acc = test(
            self.client_model[c_idx], dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc

    def server_aggre(self,iter = None):
        # for i,model in enumerate(self.client_model):
        #     self.history_client[i].append(model)
            # self.iterations[i].append(iter)
        # print(f"保存模型版本号：\n{self.iterations[0][-1]}")
        for i,model in enumerate(self.client_model):
            self.history_client[i].append(copy.deepcopy(model))
        self.server_model, self.client_model = communication(
            self.args, self.server_model, self.client_model, self.client_weight, None,self.history_client, iter)



    def compute_client_logits(self,public_dataset):
        disstill_loader = DataLoader(
            public_dataset,
            batch_size=self.args.batch,
            shuffle=False,
        )
        total_logits = []
        for i , (images, _ ,idx) in enumerate(disstill_loader):
            imags = images.cuda()
            batch_logits = []
            for n in range(len(self.client_model)):
                tmodel = copy.deepcopy(self.client_model[n])
                logits = tmodel(imags)
                batch_logits.append(logits.detach())
                del tmodel
            batch_logits = torch.stack(batch_logits).cpu()
            total_logits.append(batch_logits)
        self.total_logits = torch.cat(total_logits, dim=1).permute(1, 0, 2)  # (nsample, nl, ncls)
        del total_logits, batch_logits

    def compute_client_group(self ,iter):
        init_loggers(self.args)
        mean_logit = self.total_logits.mean(dim=0)

        client_num = mean_logit.shape[0]
        print(f"参与计算分组客户端数量为{client_num}个")
        weight_matrix = np.zeros((client_num, client_num))
        for i in range(client_num):
            for j in range(client_num):
                if i != j:
                    weight_matrix[i, j] = torch.cosine_similarity(
                        torch.nn.functional.softmax(mean_logit[i], dim=0),
                        torch.nn.functional.softmax(mean_logit[j], dim=0),
                        dim=0)
                else:
                    weight_matrix[i, j] = 1

        loger("tacc",f"余弦相似矩阵：\n{weight_matrix}")

        selected_clusters = {i: [] for i in range(client_num)}
        for i in range(client_num):
            for j in range(client_num):
                # 等概率随机采样rand
                rand = beta.rvs(2,5)
                if rand < weight_matrix[i, j]:
                    selected_clusters[i].append(j)
        num_cluster = [len(v) for k, v in selected_clusters.items()]
        loger("tacc",f"当前客户端分组为{selected_clusters}")
        # client_weight = []
        # for k, v in selected_clusters.items():
        #     selectN = v
        #     gap = torch.from_numpy(weight_matrix[k][v])
        #     if num_cluster[k]==1:
        #         localweight = [1]
        #     else:
        #         localweight = torch.nn.functional.softmax(gap, dim=0)
        #
        #     weight_one = torch.zeros(client_num)
        #     n = 0
        #     for m in selectN:
        #         weight_one[m] = localweight[n]
        #         n = n + 1
        #     client_weight.append(weight_one)
        # client_weight = torch.stack(client_weight, dim=0)
        # self.client_weight = client_weight
        # loger("tacc",f"当前客户端权重为：\n{client_weight}")
        client_weight = [[0 for j in range(client_num)]for i in range(client_num)]


        # 填充矩阵
        for i in selected_clusters:
            for j in selected_clusters[i]:
                if 0 <= i < 20 and 0 <= j < 20:  # 确保索引在有效范围内
                    client_weight[i][j] = 1 / len(selected_clusters[i])
        self.client_weight = client_weight
        loger("tacc",f"当前客户端权重为：\n{self.client_weight}")





