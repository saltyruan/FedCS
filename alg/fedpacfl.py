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
from typing import Iterable 

from util.modelsel import modelsel
from util.traineval import train, test
from alg.core.comm import communication_cfl


class fedpacfl(fedavg):
    def __init__(self, args):
        super(fedpacfl, self).__init__(args)

    def set_server_cluster(self, args):
        self.w_glob_per_cluster = []
        for k in range(args.cluster_num):
            self.w_glob_per_cluster.append(copy.deepcopy(self.server_model))

    def get_cluster(self, args, train_loaders):
        traindata_cls_ratio = {}
        budget = 20
        for i in range(args.n_clients):
            total_sum = sum([f for _,f in args.train_record_class[i]])
            base = 1/len(args.train_record_class[i])
            temp_ratio = {}
            for k,v in args.train_record_class[i]:
                ss = v/total_sum
                temp_ratio[k] = (v/total_sum)
                if ss >= (base + 0.05): 
                    temp_ratio[k] = v
                    
            sub_sum = sum(list(temp_ratio.values()))
            for k in temp_ratio.keys():
                temp_ratio[k] = (temp_ratio[k]/sub_sum)*budget
            
            round_ratio = round_to(list(temp_ratio.values()), budget)
            cnt = 0 
            for k in temp_ratio.keys():
                temp_ratio[k] = round_ratio[cnt]
                cnt+=1
                
            traindata_cls_ratio[i] = temp_ratio
        U_clients = []
        for idx in range(args.n_clients):
            U_temp = []
            for label, count in args.train_record_class[idx]:
                indices = train_loaders[idx].dataset.indices
                localdata = train_loaders[idx].dataset.data.data[indices]
                mask = train_loaders[idx].dataset.data.targets[indices] == label
                local_ds1 = localdata[mask]
                local_ds1 = local_ds1.reshape(count, -1)
                local_ds1 = local_ds1.T
                if args.partition_data_ori == 'non_iid_dirichlet' or args.partition_data == 'non_iid_dirichlet':
                    K = traindata_cls_ratio[idx][label]
                else:
                    K = 5
                u1_temp, sh1_temp, vh1_temp = np.linalg.svd(local_ds1, full_matrices=False)
                u1_temp=u1_temp/np.linalg.norm(u1_temp, ord=2, axis=0)
                U_temp.append(u1_temp[:, 0:K])
            U_clients.append(copy.deepcopy(np.hstack(U_temp)))
        print(f'Shape of U: {U_clients[-1].shape}')
        adj_mat = calculating_adjacency(range(args.n_clients), U_clients)
        clusters = hierarchical_clustering(copy.deepcopy(adj_mat), thresh=args.thresh, linkage='average')
        selected_clusters = {i: [] for i in range(len(clusters))}
        for k in range(len(clusters)):
            for q in clusters[k]:
                selected_clusters[k].append(q)
        args.cluster_num = len(clusters)
        args.selected_clusters = selected_clusters
        

    def server_aggre(self):
        self.w_glob_per_cluster, self.client_model = communication_cfl(
            self.args, self.client_model, self.client_weight, self.w_glob_per_cluster)

def weight_flatten(model):
    params = []
    for u in model.parameters():
        params.append(u.view(-1))
    params = torch.cat(params)
    return params

def calculating_adjacency(clients_idxs, U): 
        
    nclients = len(clients_idxs)
    
    sim_mat = np.zeros([nclients, nclients])
    for idx1 in range(nclients):
        for idx2 in range(nclients):
            U1 = copy.deepcopy(U[clients_idxs[idx1]])
            U2 = copy.deepcopy(U[clients_idxs[idx2]])
            
            mul = np.clip(U1.T@U2 ,a_min =-1.0, a_max=1.0)
            sim_mat[idx1,idx2] = np.min(np.arccos(mul))*180/np.pi
           
    return sim_mat

def error_gen(actual, rounded):
    divisor = np.sqrt(1.0 if actual < 1.0 else actual)
    return abs(rounded - actual) ** 2 / divisor

def round_to(percents, budget=100):
    if not np.isclose(sum(percents), budget):
        raise ValueError
    n = len(percents)
    rounded = [int(x) for x in percents]
    up_count = budget - sum(rounded)
    errors = [(error_gen(percents[i], rounded[i] + 1) - error_gen(percents[i], rounded[i]), i) for i in range(n)]
    rank = sorted(errors)
    for i in range(up_count):
        rounded[rank[i][1]] += 1
    return rounded

def hierarchical_clustering(A, thresh=1.5, linkage='maximum'):
    '''
    Hierarchical Clustering Algorithm. It is based on single linkage, finds the minimum element and merges
    rows and columns replacing the minimum elements. It is working on adjacency matrix. 
    
    :param: A (adjacency matrix), thresh (stopping threshold)
    :type: A (np.array), thresh (int)
    
    :return: clusters
    '''
    label_assg = {i: i for i in range(A.shape[0])}
    
    step = 0
    while A.shape[0] > 1:
        np.fill_diagonal(A,-np.NINF)
        #print(f'step {step} \n {A}')
        step+=1
        ind=np.unravel_index(np.argmin(A, axis=None), A.shape)

        if A[ind[0],ind[1]]>thresh:
            print('Breaking HC')
            break
        else:
            np.fill_diagonal(A,0)
            if linkage == 'maximum':
                Z=np.maximum(A[:,ind[0]], A[:,ind[1]])
            elif linkage == 'minimum':
                Z=np.minimum(A[:,ind[0]], A[:,ind[1]])
            elif linkage == 'average':
                Z= (A[:,ind[0]] + A[:,ind[1]])/2
            
            A[:,ind[0]]=Z
            A[:,ind[1]]=Z
            A[ind[0],:]=Z
            A[ind[1],:]=Z
            A = np.delete(A, (ind[1]), axis=0)
            A = np.delete(A, (ind[1]), axis=1)

            if type(label_assg[ind[0]]) == list: 
                label_assg[ind[0]].append(label_assg[ind[1]])
            else: 
                label_assg[ind[0]] = [label_assg[ind[0]], label_assg[ind[1]]]

            label_assg.pop(ind[1], None)

            temp = []
            for k,v in label_assg.items():
                if k > ind[1]: 
                    kk = k-1
                    vv = v
                else: 
                    kk = k 
                    vv = v
                temp.append((kk,vv))

            label_assg = dict(temp)

    clusters = []
    for k in label_assg.keys():
        if type(label_assg[k]) == list:
            clusters.append(list(flatten(label_assg[k])))
        elif type(label_assg[k]) == int: 
            clusters.append([label_assg[k]])
            
    return clusters


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


def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x