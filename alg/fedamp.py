import copy
from alg.fedavg import fedavg
import math
import numpy as np
import torch

from util.traineval import train, train_amp

class fedamp(fedavg):
    def __init__(self,args):
        super(fedamp, self).__init__(args)

    def client_train(self, c_idx, dataloader, round):
        server_model = copy.deepcopy(self.client_model[c_idx]).to(self.args.device)
        if round > 0:
            train_loss, train_acc = train_amp(
                self.args, self.client_model[c_idx], server_model, dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        else:
            train_loss, train_acc = train(
                self.client_model[c_idx], dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        return train_loss, train_acc
    
    def get_weight_matrix_heuristic(self, args, iter):
        beta = args.beta
        client_num = self.args.n_clients
        weight_m = torch.zeros((client_num, client_num))
        for i in range(client_num):
            for j in range(client_num):
                if i == j:
                    weight_m[i, j] = 0
                else:
                    cos_sim = torch.tensor(0., device=args.device)
                    z = 0
                    for w, w_t in zip(self.client_model[i].parameters(), self.client_model[j].parameters()):
                        cos_sim += torch.cosine_similarity(w.view(-1),w_t.view(-1),dim=0)
                        z += 1
                    # cos_sim = torch.cosine_similarity(weight_flatten(self.client_model[i]), 
                    #                     weight_flatten(self.client_model[j]), dim=0)
                    cos_sim = cos_sim / z
                    weight_m[i, j] = torch.exp(cos_sim)
        weight = weight_m / torch.sum(weight_m, dim=1).unsqueeze(dim=1)
        weight_matrix = weight * (1 - beta)
        for i in range(client_num):
            weight_matrix[i, i] = beta
        self.client_weight = weight_matrix

    def get_weight_matrix(self, args, iter):
        alphaK = args.alphaK
        client_num = self.args.n_clients
        weight_m = torch.zeros((client_num, client_num))
        for i in range(client_num):
            for j in range(client_num):
                if i == j:
                    weight_m[i, j] = 0
                else:
                    sub = (weight_flatten(self.client_model[i]) - weight_flatten(self.client_model[j])).view(-1)
                    sub = torch.dot(sub, sub)
                    weight_m[i, j] = alphaK * e(sub)
        
        for i in range(client_num):
            weight_m[i, i] = 1 - torch.sum(weight_m[i])

        self.client_weight = weight_m

def weight_flatten(model):
    params = []
    for u in model.parameters():
        params.append(u.view(-1))
    params = torch.cat(params)
    return params



def e(x):
    sigma = 1e-1
    return math.exp(-x/sigma)/sigma