import random
import torch
import copy
import wandb
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from alg.fedavg import fedavg
import torch.nn.init as init

from util.modelsel import modelsel
from util.traineval import train, test
from alg.core.comm import communication_cfl


class fedicfa(fedavg):
    def __init__(self, args):
        super(fedicfa, self).__init__(args)

    def set_server_cluster(self, args):
        self.w_glob_per_cluster = []
        for k in range(args.cluster_num):
            self.server_model.apply(weight_init)
            self.w_glob_per_cluster.append(copy.deepcopy(self.server_model))

    def get_cluster(self, args, test_loaders):
        selected_clusters = {i: [] for i in range(args.cluster_num)}
        w_locals_clusters = [[] for _ in range(args.cluster_num)]
        client_num = args.n_clients
        for i in range(client_num):
            acc_select = []
            for k in range(args.cluster_num):
                self.client_model[i].load_state_dict(self.w_glob_per_cluster[k].state_dict())
                test_loss, test_acc = self.client_eval(i, test_loaders[i])
                acc_select.append(test_acc)
            idx_cluster = np.argmax(acc_select)
            self.client_model[i].load_state_dict(self.w_glob_per_cluster[idx_cluster].state_dict())
            selected_clusters[idx_cluster].append(i)
        
        args.selected_clusters = selected_clusters
        

    def server_aggre(self):
        self.w_glob_per_cluster, self.client_model = communication_cfl(
            self.args, self.client_model, self.client_weight, self.w_glob_per_cluster)



def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    return 