from http import client
import logging
import os
import copy
import math
import random
from typing_extensions import Self
import numpy as np
import torch

from alg.fedavg import fedavg
from util.traineval import pretrain_model
from torch.utils.data import DataLoader


class clusterfed(fedavg):
    def __init__(self, args):
        super(clusterfed, self).__init__(args)

    def set_client_weight(self, train_dataset, args):
        self.get_countN(args, train_dataset)
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


    def get_countN(self, args, train_dataset):
        nclass = args.num_classes
        countN = np.zeros((args.n_clients, nclass))
        for n in range(args.n_clients):
            for m in range(nclass):
                indices = train_dataset[n].indices
                if args.dataset in ['medmnistC', 'medmnist', 'medmnistA', 'covid']:
                    countN[n][m] = (train_dataset[n].data.targets[indices] == m).sum()
                elif args.dataset in ['pacs']:
                    countN[n][m] = (train_dataset[n].dataset.targets[indices] == m).sum()
        self.countN = torch.tensor(countN).cuda() 
        del countN



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


    def get_per_logits(self, args, iter):
        ratio = [0.6]
        # 获得每个本地客户端的特征表示[nsample, nclient, nclass] [nclinet, nclass]
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
        ensemble_logits_all = []
        for k,v in selected_clusters.items():
            selectN = v
            total_logits = self.total_logits.permute(1,0,2).cuda()  #nlocal*nsamples*ncls
            total_logits = total_logits[torch.tensor(selectN)] #nlocal*nsamples*ncls
            if not iter:
                countN = self.countN[selectN]
                localweight = countN/(countN.sum(dim=0) + 1e-6) # nlocal*nclass
                localweight = localweight.unsqueeze(dim=1)# nlocal*1*ncls
            else:
                criterion_div = DistillKL() 
                loss_div_list = [] 
                for j in selectN: 
                    loss_div = criterion_div(self.total_logits.permute(1,0,2)[k], self.total_logits.permute(1,0,2)[j], is_ca=True)
                    loss_div_list.append(loss_div)
                loss_div_list = torch.stack(loss_div_list, dim=0)
                localweight = (1.0 - torch.nn.functional.softmax(loss_div_list, dim=0)) / (client_num - 1)
                localweight = localweight.unsqueeze(dim=2).cuda()
            ensemble_logits = (total_logits*localweight).sum(dim=0) #nsamples*ncls
            ensemble_logits_all.append(ensemble_logits)
        return ensemble_logits_all, num_cluster

    def get_per_logits_soft(self, args, iter):
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
        # 获得传送给每个客户的logits
        num_cluster = [len(v) for k,v in selected_clusters.items()]
        ensemble_logits_all = []
        for k,v in selected_clusters.items():
            selectN = v
            total_logits = self.total_logits.permute(1,0,2).cuda()  #nlocal*nsamples*ncls
            total_logits = total_logits[torch.tensor(selectN)] #nlocal*nsamples*ncls
            criterion_div = DistillKL() 
            loss_div_list = [] 
            for j in selectN: 
                loss_div = criterion_div(self.total_logits.permute(1,0,2)[k], 
                                        self.total_logits.permute(1,0,2)[j], is_ca=True)
                loss_div_list.append(loss_div)
            loss_div_list = torch.stack(loss_div_list, dim=0)
            localweight = (1.0 - torch.nn.functional.softmax(loss_div_list/args.T, dim=0)) / (client_num - 1)
            localweight = localweight.unsqueeze(dim=2).cuda()
            ensemble_logits = (total_logits*localweight).sum(dim=0) #nsamples*ncls
            ensemble_logits_all.append(ensemble_logits)
        return ensemble_logits_all, num_cluster


    def get_per_logits_mean(self, args, iter):
        selected_clusters = {i: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] for i in range(20)}
        # total_logits = self.total_logits.mean(dim=0)
        # # 获得客户端之间的相似度矩阵 [nclient, nclient]
        # client_num = total_logits.shape[0]
        # weight_m = np.zeros((client_num, client_num))
        # selected_clusters = {i: [] for i in range(client_num)}
        # for i in range(client_num):
        #     for j in range(client_num):
        #         if i == j:
        #             weight_m[i, j] = 1
        #         else:
        #             weight_m[i, j] = torch.cosine_similarity(torch.nn.functional.softmax(total_logits[i]), 
        #                                 torch.nn.functional.softmax(total_logits[j]), dim=0)
        # for i in range(client_num):
        #     for j in range(client_num):
        #         # 等概率随机采样rand
        #         rand = random.random()
        #         if rand < weight_m[i,j]:
        #             selected_clusters[i].append(j)
        # 获得传送给每个客户的logits
        num_cluster = [len(v) for k,v in selected_clusters.items()]
        ensemble_logits_all = []
        for k,v in selected_clusters.items():
            selectN = v
            total_logits = self.total_logits.permute(1,0,2).cuda()  #nlocal*nsamples*ncls
            total_logits = total_logits[torch.tensor(selectN)] #nlocal*nsamples*ncls
            # 按照数据的方式
            # countN = self.countN[selectN]
            # totalN = countN.sum(dim=1)
            # localweight = totalN/(totalN.sum(dim=0) + 1e-6) # nlocal*nclass
            # localweight = localweight.unsqueeze(dim=1).unsqueeze(dim=2)# nlocal*1*1
            # ensemble_logits = (total_logits*localweight).sum(dim=0) #nsamples*ncls
            # 平均的方式
            # ensemble_logits = total_logits.mean(dim=0) #nsamples*ncls
            # 获得权重
            criterion_div = DistillKL() 
            loss_div_list = [] 
            for j in selectN: 
                loss_div = criterion_div( self.total_logits.permute(1,0,2)[k], 
                                        self.total_logits.permute(1,0,2)[j], is_ca=True)
                loss_div_list.append(loss_div)
            loss_div_list = torch.stack(loss_div_list, dim=0)
            localweight = (1.0 - torch.nn.functional.softmax(loss_div_list, dim=0)/args.T) / (20 - 1)
            localweight = localweight.unsqueeze(dim=2).cuda()
            ensemble_logits = (total_logits*localweight).sum(dim=0) #nsamples*ncls
            ensemble_logits_all.append(ensemble_logits)
        return ensemble_logits_all, num_cluster



    def client_distill(self, model, ensemble_logits):
        self.client_model[model].train()
        loss_all = 0
        criterion = torch.nn.L1Loss(reduce=True)
        for i, (images, _, idx) in enumerate(self.distil_loader):
            data = images.cuda().float()
            output = self.client_model[model](data)
            ensemble_logit = ensemble_logits[idx].cuda()
            loss = criterion(output, ensemble_logit)
            loss_all += loss.item()

            self.optimizers[model].zero_grad()
            loss.backward()
            self.optimizers[model].step()

        return loss_all / len(self.distil_loader)
    
    def client_train_mem(self, c_idx, train_loader, wi):
        if wi == 0:
            self.distill_model = copy.deepcopy(self.client_model[c_idx])
            self.distill_model.eval()
        self.client_model[c_idx].train()
        loss_fun = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.L1Loss(reduce=True)
        loss_all = 0
        for data, target in train_loader:
            data = data.cuda().float()
            target = target.cuda().long()
            s_output = self.client_model[c_idx](data)
            t_output = self.distill_model(data)
            loss_cls = loss_fun(s_output, target)
            loss_mem = criterion(s_output, t_output)
            loss = loss_cls + loss_mem
            loss_all += loss.item()

            self.optimizers[c_idx].zero_grad()
            loss.backward()
            self.optimizers[c_idx].step()

        return loss_all / len(train_loader)


class DistillKL(torch.nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T=4):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, is_ca=False):
        p_s = torch.nn.functional.log_softmax(y_s/self.T, dim=1)
        p_t = torch.nn.functional.softmax(y_t/self.T, dim=1)
        if is_ca: 
            loss = (torch.nn.KLDivLoss(reduction='none')(p_s, p_t) * (self.T**2)).sum(-1)
        else:
            loss = torch.nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (self.T**2)
        return loss