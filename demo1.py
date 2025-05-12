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
    parser.add_argument('--dataset', type=str, default='medmnist',
                        help='[cifar | covid | medmnist | None]')
    parser.add_argument('--distillation_dataset', type=str, default='medmnistC',
                        help='[cifar | covid | medmnistC | chest]')
    parser.add_argument('--pre_dataset', type=str, default='medmnist',
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
    loger(args,"tacc","nihao")