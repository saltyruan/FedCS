import os
import copy
import math
import random
import numpy as np
import torch
import wandb
from alg.clusterfed import DistillKL

from alg.fedavg import fedavg
from util.traineval import pretrain_model
from torch.utils.data import DataLoader


class fedbn_nt(fedavg):
    def __init__(self, args):
        super(fedbn_nt, self).__init__(args)

    # def get_countN(self, args, train_dataset):
    #     nclass = args.num_classes
    #     countN = np.zeros((args.n_clients, nclass))
    #     for n in range(args.n_clients):
    #         for m in range(nclass):
    #             indices = train_dataset[n].indices
    #             countN[n][m] = (train_dataset[n].data.targets[indices] == m).sum()
    #     self.countN = torch.tensor(countN).cuda() 
    #     del countN


    
    def set_client_weight(self, train_dataset, args):
        # self.get_countN(args, train_dataset)
        os.makedirs('./checkpoint/'+'pretrained/', exist_ok=True)
        preckpt = './checkpoint/'+'pretrained/' + \
            self.args.dataset+'_'+str(self.args.batch)
        self.pretrain_model = copy.deepcopy(
            self.server_model).to(self.args.device)
        if not os.path.exists(preckpt):
            pretrain_model(self.args, self.pretrain_model,
                           preckpt, self.args.device)
        for model in self.client_model:
            model.load_state_dict(torch.load(preckpt)['state'])
        del preckpt

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
        alp = torch.full((args.n_clients, args.n_clients), args.alp)
        # 获得每个本地客户端的代表语义[nsample, nclient, nclass] [nclinet, nclass]
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
        # wandb.log({"client0": weight_m_average[0], "epoch":iter})
        # wandb.log({"client1": weight_m_average[1], "epoch":iter})
        # wandb.log({"client2": weight_m_average[2], "epoch":iter})
        # wandb.log({"client10": weight_m_average[10], "epoch":iter})
        # wandb.log({"client19": weight_m_average[19], "epoch":iter})
        # weight_avg = np.mean(weight_m, axis=1)
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

    def get_per_logits(self, args, iter):
        ratio = [0.5]
        # 获得每个本地客户端的代表语义[nsample, nclient, nclass] [nclinet, nclass]
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
                    weight_m[i, j] = torch.cosine_similarity(torch.nn.functional.softmax(total_logits[i]), 
                                        torch.nn.functional.softmax(total_logits[j]), dim=0)
        for i in range(client_num):
            for j in range(client_num):
                # 获得每个客户的有益信息聚类群
                sort_weight = np.sort(weight_m[i])
                if iter >= len(ratio):
                    th = sort_weight[int(client_num*ratio[-1])-1]
                else:
                    th = sort_weight[int(client_num*ratio[iter])-1]
                if weight_m[i, j] >= th:
                # if weight_m[i, j] > np.mean(weight_m[i]):
                    selected_clusters[i].append(j)
        # 获得传送给每个客户的logits
        num_cluster = [len(v) for k,v in selected_clusters.items()]
        weightmatrix = []
        for k,v in selected_clusters.items():
            selectN = v
            total_logits = self.total_logits.permute(1,0,2).cuda()  #nlocal*nsamples*ncls
            total_logits = total_logits[torch.tensor(selectN)] #nlocal*nsamples*ncls
            criterion_div = DistillKL() 
            loss_div_list = [] 
            for j in selectN: 
                loss_div = criterion_div(self.total_logits.permute(1,0,2)[k], 
                                        self.total_logits.permute(1,0,2)[j])
                loss_div_list.append(loss_div)
            loss_div_list = torch.stack(loss_div_list, dim=0)
            if num_cluster[k] == 1:
                localweight = [1]
            else:
                localweight = (1.0 - torch.nn.functional.softmax(loss_div_list, dim=0)) / (num_cluster[k] - 1)
                # localweight = [1/num_cluster[k] for _ in v]
            weight_one = torch.zeros(20)
            n = 0
            for m in selectN:
                weight_one[m] = localweight[n]
                n = n + 1
            weightmatrix.append(weight_one)
        weightmatrix = torch.stack(weightmatrix, dim=0)
        self.client_weight = weightmatrix

    
    def get_per_logits_mean(self, args, iter):
        # selected_clusters = {i: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] for i in range(20)}
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
                    weight_m[i, j] = torch.cosine_similarity(torch.nn.functional.softmax(total_logits[i]), 
                                        torch.nn.functional.softmax(total_logits[j]), dim=0)
        for i in range(client_num):
            for j in range(client_num):
                # 等概率随机采样rand
                rand = random.random()
                if rand < weight_m[i,j]:
                    selected_clusters[i].append(j)
        # 获得传送给每个客户的logits
        num_cluster = [len(v) for k,v in selected_clusters.items()]
        print(f"当前客户端分组为{selected_clusters}")
        weightmatrix = []
        for k,v in selected_clusters.items():
            selectN = v
            total_logits = self.total_logits.permute(1,0,2).cuda()  #nlocal*nsamples*ncls
            total_logits = total_logits[torch.tensor(selectN)] #nlocal*nsamples*ncls
            # 按照数据的方式
            countN = self.countN[selectN]
            totalN = countN.sum(dim=1)
            localweight = totalN/(totalN.sum(dim=0) + 1e-6)
            if num_cluster[k] == 1:
                localweight = [1]
            else:
                localweight = totalN/(totalN.sum(dim=0) + 1e-6)
            # criterion_div = DistillKL() 
            # loss_div_list = [] 
            # for j in selectN: 
            #     loss_div = criterion_div(self.total_logits.permute(1,0,2)[k], 
            #                             self.total_logits.permute(1,0,2)[j])
            #     loss_div_list.append(loss_div)
            # loss_div_list = torch.stack(loss_div_list, dim=0)
            # if num_cluster[k] == 1:
            #     localweight = [1]
            # else:
            #     localweight = (1.0 - torch.nn.functional.softmax(loss_div_list, dim=0)) / (num_cluster[k] - 1)
                # localweight = [1/num_cluster[k] for _ in v]
            weight_one = torch.zeros(20)
            n = 0
            for m in selectN:
                weight_one[m] = localweight[n]
                n = n + 1
            weightmatrix.append(weight_one)
        weightmatrix = torch.stack(weightmatrix, dim=0)
        self.client_weight = weightmatrix


