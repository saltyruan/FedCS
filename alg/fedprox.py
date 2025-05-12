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


class fedprox(fedavg):
    def __init__(self, args):
        super(fedprox, self).__init__(args)

    def client_train(self, c_idx, dataloader, round):
        server_model = copy.deepcopy(self.client_model[c_idx]).to(self.args.device)
        if round > 0 :
            train_loss, train_acc = train_prox(
                self.args, self.client_model[c_idx], server_model, dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        else:
            train_loss, train_acc = train(
                self.client_model[c_idx], dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        return train_loss, train_acc