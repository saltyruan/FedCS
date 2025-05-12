import random
import torch
import copy

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from util.modelsel import modelsel
from util.traineval import train, test
from alg.core.comm import communication
from alg.SAM import SAM


class fedavg(torch.nn.Module):
    def __init__(self, args):
        super(fedavg, self).__init__()
        self.server_model, self.client_model, self.client_weight = modelsel(args, args.device)
        if args.transfer:
            # base_optimizer = optim.SGD
            # self.optimizers =[SAM(params=[p for name, p in self.client_model[idx].named_parameters()
            #     if 'linear' in name or 'fc' in name], base_optimizer=base_optimizer,
            #     lr=args.lr) for idx in range(args.n_clients)]
            self.optimizers = [optim.SGD(params=[p for name, p in self.client_model[idx].named_parameters()
                                                 if 'linear' in name or 'fc' in name], lr=args.lr) for idx in
                               range(args.n_clients)]
        else:
            self.optimizers = [optim.SGD(params=self.client_model[idx].parameters(
            ), lr=args.lr) for idx in range(args.n_clients)]

            # print(self.optimizers)
        #     self.optimizers = [SAM]
        self.loss_fun = nn.CrossEntropyLoss()
        self.args = args


    def select_client(self, args, a_iter):
        m = max(int(args.participation_rate * args.n_clients), 1)
        if args.participation_rate < 1:
            selected_user = np.random.choice(range(args.n_clients), m, replace=False)
        # for i in selected_user:
        #     self.client_model.append(copy.deepcopy(self.client_model_all[i]).to(args.device))
        self.selected_user = np.sort(selected_user)


    def client_train(self, c_idx, dataloader, round=None):
        train_loss, train_acc = train(
            self.client_model[c_idx], dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        return train_loss, train_acc

    def server_aggre(self):
        self.server_model, self.client_model = communication(
            self.args, self.server_model, self.client_model, self.client_weight)

    def client_eval(self, c_idx, dataloader):
        train_loss, train_acc = test(
            self.client_model[c_idx], dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc

    def server_eval(self, dataloader):
        train_loss, train_acc = test(
            self.server_model, dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc

    def client_train_finetune(self, c_idx, dataloader, round=None):
        train_loss, train_acc = train(
            self.client_model[c_idx], dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        return train_loss, train_acc

    def set_client_weight(self):
        if self.args.dataset != 'cifar10':
            preckpt_root = './pre/' + self.args.pre_dataset
            pretrained_model = copy.deepcopy(self.server_model).to(self.args.device)
            preckpt = preckpt_root + '/best.pt'
            for model in self.client_model:
                a = torch.load(preckpt)['state']
                model.load_state_dict(a)
            del pretrained_model

