# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import copy
import numpy as np

def calculate(model_a,model_b):
    total_divergence = 0.0
    for name in model_a:
        param_a = model_a[name].cpu().float()
        param_b = model_b[name].cpu().float()
        total_divergence += torch.norm(param_a - param_b).item()
    # print("差异值{}".format(total_divergence))
    return total_divergence

def communication(args, server_model, models, client_weights, layer_index=None,history_clients=None,iter=None):
    client_num=len(models)
    with torch.no_grad():
        if args.alg.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.alg.lower() in ['fedap', 'fedbn_nt']:
            tmpmodels=[]
            for i in range(client_num):
                tmpmodels.append(copy.deepcopy(models[i]).to(args.device))
            with torch.no_grad():
                for cl in range(client_num):
                    for key in server_model.state_dict().keys():
                        temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                        for client_idx in range(client_num):
                            temp += client_weights[cl,client_idx] * tmpmodels[client_idx].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)
                        if 'bn' not in key:
                            models[cl].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.alg.lower() in ['fedmysoft_nt','fedprox_nt', 'fedavg_nt', 'fedmlb_nt', 'fedamp', 'fedavg_proto', 'fedpre']:
            tmpmodels=[]
            for i in range(client_num):
                tmpmodels.append(copy.deepcopy(models[i]).to(args.device))
            with torch.no_grad():
                for cl in range(client_num):
                    for key in server_model.state_dict().keys():
                        if args.transfer:
                            if 'linear' in key or 'fc' in key:
                                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                                for client_idx in range(client_num):
                                    temp += client_weights[cl,client_idx] * tmpmodels[client_idx].state_dict()[key]
                                server_model.state_dict()[key].data.copy_(temp)
                                models[cl].state_dict()[key].data.copy_(server_model.state_dict()[key])
                        else:
                            temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                            for client_idx in range(client_num):
                                temp += client_weights[cl,client_idx] * tmpmodels[client_idx].state_dict()[key]
                            server_model.state_dict()[key].data.copy_(temp)
                            models[cl].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.alg.lower() in ['fedhper']:
            # for key in server_model.state_dict().keys():
            #     if 'num_batches_tracked' in key:
            #         server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
            #     else:
            #         if args.dataset == 'cifar10':
            #             if key in layer_index or 'linear' in key or 'fc' in key:
            #                 temp = torch.zeros_like(server_model.state_dict()[key])
            #                 for client_idx in range(len(client_weights)):
            #                     temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
            #                 server_model.state_dict()[key].data.copy_(temp)
            #                 for client_idx in range(len(client_weights)):
            #                     models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
            #         elif args.dataset == 'medmnist':
            #             if key in layer_index or 'linear' in key:
            #                 temp = torch.zeros_like(server_model.state_dict()[key])
            #                 for client_idx in range(len(client_weights)):
            #                     temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
            #                 server_model.state_dict()[key].data.copy_(temp)
            #                 for client_idx in range(len(client_weights)):
            #                     models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
            tmpmodels=[]
            for i in range(client_num):
                tmpmodels.append(copy.deepcopy(models[i]).to(args.device))
            with torch.no_grad():
                for cl in range(client_num):
                    for key in server_model.state_dict().keys():
                        if args.dataset == 'cifar10':
                            if key in layer_index or 'linear' in key or 'fc' in key:
                                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                                for client_idx in range(client_num):
                                    temp += client_weights[cl,client_idx] * tmpmodels[client_idx].state_dict()[key]
                                server_model.state_dict()[key].data.copy_(temp)
                                models[cl].state_dict()[key].data.copy_(server_model.state_dict()[key])
                        elif args.dataset == 'medmnist':
                            if key in layer_index or 'linear' in key:
                                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                                for client_idx in range(client_num):
                                    temp += client_weights[cl,client_idx] * tmpmodels[client_idx].state_dict()[key]
                                server_model.state_dict()[key].data.copy_(temp)
                                models[cl].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.alg.lower() in ['fedlama']:
            for key in server_model.state_dict().keys():
                if (key in layer_index) or ('bn' in key):
                    if 'num_batches_tracked' in key:
                        server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                    else:
                        temp = torch.zeros_like(server_model.state_dict()[key])
                        for client_idx in range(len(client_weights)):
                            temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)
        elif args.alg.lower() in ['myfed']:
            print(f"聚合中，当前轮次为{iter}")
            if iter>args.warm_up:
                # history_clients=history_clients
                tmpmodels = [[] for _ in range(client_num)]
                # 为主要客户端寻找辅助客户端
                for main_client in range(client_num):
                    # 遍历该客户端所在分组所有历史模型，找到与主要客户端最新版本相差最小的版本
                    for j in range(client_num):
                        best_idx = 2
                        min_divergence = 999999
                        if j == main_client:
                            tmpmodels[main_client].append(copy.deepcopy(history_clients[j][2]).to(args.device))
                            continue
                        for i in range(3):
                            divergence=calculate(history_clients[j][i].state_dict(),models[j].state_dict())
                            # print(divergence)
                            if divergence < min_divergence:
                                min_divergence=divergence
                                best_idx=i
                        # print(f"客户端{j}选择的版本为{best_idx},最终差异值为{min_divergence}")
                        tmpmodels[main_client].append(copy.deepcopy(history_clients[j][best_idx]).to(args.device))
            else:
                tmpmodels = []
                for i in range(client_num):
                    tmpmodels.append(copy.deepcopy(models[i]).to(args.device))
            #聚合阶段
            with torch.no_grad():
                for cl in range(client_num):
                    for key in server_model.state_dict().keys():
                        if args.transfer:
                            if 'linear' in key or 'fc' in key:
                                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                                for client_idx in range(client_num):
                                    if iter<=args.warm_up:
                                        temp += client_weights[cl][client_idx] * tmpmodels[client_idx].state_dict()[key]
                                    else:
                                        temp += client_weights[cl][client_idx] * tmpmodels[cl][client_idx].state_dict()[key]
                                server_model.state_dict()[key].data.copy_(temp)
                                models[cl].state_dict()[key].data.copy_(server_model.state_dict()[key])
                        else:
                            temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                            for client_idx in range(client_num):
                                if iter<=args.warm_up:
                                    temp += client_weights[cl][client_idx] * tmpmodels[client_idx].state_dict()[key]
                                else:
                                    temp += client_weights[cl][client_idx] * tmpmodels[cl][client_idx].state_dict()[key]
                            server_model.state_dict()[key].data.copy_(temp)
                            models[cl].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models

def communication_cfl(args, models, client_weights, w_glob_per_cluster):
    client_num=len(models)
    with torch.no_grad():
        if args.alg.lower() in ['fedicfa', 'fedourcfl', 'fedcfl','fedpacfl']:
            w_locals_clusters = [[] for _ in range(args.cluster_num)]
            for k,v in args.selected_clusters.items():
                for idx in v:
                    w_locals_clusters[k].append(models[idx])
            # 每个簇内执行FedAvg操作
            for k in range(args.cluster_num):
                # 获得temp
                record_class = args.record_class
                count_client = np.array([np.array(record_class[i])[:, 1].sum() for i in range(0, args.n_clients*3, 3)])
                client_weights = count_client[args.selected_clusters[k]] / count_client[args.selected_clusters[k]].sum()
                for key in w_glob_per_cluster[k].state_dict().keys():
                    if len(w_locals_clusters[k]) != 0:
                        if 'num_batches_tracked' in key:
                            w_glob_per_cluster[k].state_dict()[key].data.copy_(w_locals_clusters[k][0].state_dict()[key])
                        else:
                            temp = torch.zeros_like(w_glob_per_cluster[k].state_dict()[key])
                            i = 0
                            for client in w_locals_clusters[k]:
                                temp += client_weights[i] * client.state_dict()[key]
                                i += 1
                            w_glob_per_cluster[k].state_dict()[key].data.copy_(temp)
            for k,v in args.selected_clusters.items():
                for i in v:
                    models[i].load_state_dict(w_glob_per_cluster[k].state_dict())
    return w_glob_per_cluster, models

def communication_soft_cfl(args, models, selected_clusters, clinet_to_its, w_glob_per_cluster):  # w_glob_per_cluster: 是一个列表，包含了每个簇的全局模型。
    client_num = len(models)
    with torch.no_grad():
        if args.alg.lower() in ['fedmysoft']:
            for k in range(args.cluster_num):
                idk_cluster = selected_clusters[k]
                for key in w_glob_per_cluster[k].state_dict().keys():
                        if 'num_batches_tracked' in key:
                            max_tuple = max(idk_cluster, key=lambda x: x[1])
                            w_glob_per_cluster[k].state_dict()[key].data.copy_(
                                models[max_tuple[0]].state_dict()[key])
                        else:
                            temp = torch.zeros_like(w_glob_per_cluster[k].state_dict()[key])
                            for tuple in idk_cluster:
                                id_client = tuple[0]
                                id_weight = tuple[1]
                                temp += id_weight*models[id_client].state_dict()[key]
                            w_glob_per_cluster[k].state_dict()[key].data.copy_(temp)

            for i in range(client_num):
                idk_client=clinet_to_its[i]
                for key in models[i].state_dict().keys():
                    if 'num_batches_tracked' in key:
                        max_tuple = max(idk_client, key=lambda x: x[1])
                        models[i].state_dict()[key].data.copy_(
                            models[max_tuple[0]].state_dict()[key])
                    else:
                        temp = torch.zeros_like(models[i].state_dict()[key])
                        for tuple in idk_client:
                            id_cluster = tuple[0]
                            id_weight = tuple[1]
                            temp += id_weight*w_glob_per_cluster[id_cluster].state_dict()[key]
                        models[i].state_dict()[key].data.copy_(temp)
    return w_glob_per_cluster, models
    