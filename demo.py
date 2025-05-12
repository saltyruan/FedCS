import logging
import sys
import time
import numpy as np
import torch
import argparse

from datautil.prepare_data import *
from util.config import img_param_init, set_random_seed
from util.evalandprint import evalandprint, evalandprint_nosave
from alg import algs
from util.log import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)

    parser.add_argument('--alg', type=str, default='myfed')
    parser.add_argument('--datapercent', type=float,
                        default=0.1, help='data percent to use')
    #公共数据集
    parser.add_argument('--dis_datapercent', type=float,
                        default=0.01, help='distillation data percent to use')
    # medmnist医学数据集28*28的图像
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='[cifar | covid | medmnist | None]')
    parser.add_argument('--distillation_dataset', type=str, default='cifar10',
                        help='[cifar | covid | medmnistC | chest]')
    parser.add_argument('--pre_dataset', type=str, default='cifar10',
                        help='[vlcs | pacs | officehome | pamap | covid | medmnistC | chest]')
    parser.add_argument('--root_dir', type=str,
                        default='./data/', help='数据集路径')
    parser.add_argument('--save_path', type=str,
                        default='./Layer-wise/', help='模型保存路径')
    parser.add_argument('--device', type=str,
                        default='cuda', help='[cuda | cpu]')
    # 批大小
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    # 通信轮次300默认
    parser.add_argument('--iters', type=int, default=400,
                        help='迭代轮次')
    # 学习率
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # 一共20个客户端
    parser.add_argument('--n_clients', type=int,
                        default=20, help='number of clients20')
    # 迪利克雷系数(平衡：0.1、0.01   不平衡：1、0.1)
    parser.add_argument('--non_iid_alpha', type=float,
                        default=0.1, help='data split for label shift')
    # 数据划分方法(迪利克雷2种    病态划分)
    parser.add_argument('--partition_data', type=str,
                        default='non_iid_dirichlet', help='partition data way eg:non_iid_dirichlet、noniid-#label2')
    # 预训练的轮数
    parser.add_argument('--warm_up', type=int,
                        default=60, help='模型预热')
    #seed种子
    parser.add_argument('--seed', type=int, default=28, help='random seed')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    # 是否切换优化器
    parser.add_argument('--transfer', action='store_true',
                        help='true or false of updata classfier')
    args = parser.parse_args()
    # print(args.dataset, args.partition_data, args.warm_up)
    set_random_seed(args.seed)

    exp_folder = (f'fed_selectall_{args.dataset}_{args.alg}_{args.non_iid_alpha}_'
                  f'{args.dis_datapercent}_{args.partition_data}_{args.transfer}_{args.warm_up}')
    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, args.alg)

    init_loggers(args,"./log/")

    best_changed = False
    best_acc = [0] * args.n_clients
    best_tacc = [0] * args.n_clients
    train_loaders, val_loaders, test_loaders, train_dataset = get_data(args.dataset)(args)
    public_dataset = get_distill_date(args)
    algclass = algs.get_algorithm_class(args.alg)(args)
    for a_iter in range(args.iters):
        loger("iteration",f"============ Train round {a_iter} ============")
        if args.alg == 'myfed':
            for client_idx in range(args.n_clients):
                algclass.client_train(
                    client_idx, train_loaders[client_idx], a_iter)
            if a_iter%40 == 0:
                algclass.compute_client_logits(public_dataset)
                algclass.compute_client_group(a_iter)
        algclass.server_aggre(a_iter)
        best_tacc, test_acc_list,best_changed = evalandprint(
            args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_tacc, a_iter,
            best_changed)
        loger("tacc",f"第{a_iter}次内最高准确率{sum(best_tacc) / len(best_tacc):.4f},当前准确率{sum(test_acc_list) / len(test_acc_list):.4f}")


    for client_idx in range(args.n_clients):
        algclass.client_train(client_idx, train_loaders[client_idx], a_iter)

    best_tacc, test_loss_list,best_changed = evalandprint_nosave(
            args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH,  best_tacc, a_iter, best_changed)
    loger("tacc",f"当前最好的测试机准确率为：{best_tacc},最好的测试集平均准确率为{sum(best_tacc) / len(best_tacc)},此次平均损失为：{sum(test_loss_list) / len(test_loss_list)}")