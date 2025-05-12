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


class fedcfl(fedavg):
    def __init__(self, args):
        super(fedcfl, self).__init__(args)

    def set_server_cluster(self, args):
        self.w_glob_per_cluster = []
        for k in range(args.cluster_num):
            self.w_glob_per_cluster.append(copy.deepcopy(self.server_model))

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


    def get_cluster(self, args, test_loaders):
        client_num = args.n_clients
        weight_m = np.zeros((client_num, client_num))
        client_list = np.array(self.total_logits.mean(dim=0))
        # client_list = [weight_flatten(self.client_model[i]) for i in range(client_num)]
        if args.choose_cluter == 'pa':
            weight_m = calculating_adjacency([z for z in range(args.n_clients)], client_list)
        else:
            for i in range(client_num):
                for j in range(client_num):
                    if i == j:
                        weight_m[i, j] = 0
                    else:
                        weight_m[i, j] = torch.cosine_similarity(torch.nn.functional.softmax(torch.tensor(client_list[i]),dim=0), 
                                                                    torch.nn.functional.softmax(torch.tensor(client_list[j]),dim=0), dim=0)
                        #weight_m[i, j] = torch.cosine_similarity(torch.tensor(client_list[i]), torch.tensor(client_list[j]), dim=0)
        clusters = hierarchical_clustering(weight_m, thresh=4.9, linkage='average')
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
    params = np.array(torch.cat(params).detach().cpu())
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