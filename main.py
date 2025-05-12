# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
import wandb


if __name__ == '__main__':
    # 创建了一个参数解析器对象。
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='fedavg_nt',
                        help='Algorithm to choose: [base | fedavg | fedbn | fedprox | fedap | metafed | clusterfed | fedcd | fedmd]')
    parser.add_argument('--datapercent', type=float,
                        default=0.1, help='data percent to use')
    #公共数据集
    parser.add_argument('--dis_datapercent', type=float,
                        default=0.01, help='distillation data percent to use')
    # medmnist医学数据集28*28的图像
    parser.add_argument('--dataset', type=str, default='medmnist',
                        help='[vlcs | pacs | officehome | pamap | covid | medmnist | None]')
    parser.add_argument('--distillation_dataset', type=str, default='medmnist',
                        help='[vlcs | pacs | officehome | pamap | covid | medmnistC | chest]')
    parser.add_argument('--pre_dataset', type=str, default='medmnist',
                        help='[vlcs | pacs | officehome | pamap | covid | medmnistC | chest]')
    parser.add_argument('--root_dir', type=str,
                        default='./data/', help='data path')
    parser.add_argument('--save_path', type=str,
                        default='./Layer-wise/', help='path to save the checkpoint')
    parser.add_argument('--device', type=str,
                        default='cuda', help='[cuda | cpu]')
    # 批大小
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    # 通信轮次300默认
    parser.add_argument('--iters', type=int, default=400,
                        help='iterations for communication')
    # 学习率
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # 一共20个客户端
    parser.add_argument('--n_clients', type=int,
                        default=20, help='number of clients20')
    # 迪利克雷系数(平衡：0.1、0.01   不平衡：1、0.1)
    parser.add_argument('--non_iid_alpha', type=float,
                        default=1.0, help='data split for label shift')
    # 数据划分方法(迪利克雷2种    病态划分)
    parser.add_argument('--partition_data', type=str,
                        default='non_iid_dirichlet', help='partition data way eg:non_iid_dirichlet、noniid-#label2')
    # 预训练的轮数
    parser.add_argument('--pretrained_iters', type=int,
                        default=150, help='iterations for pretrained models')
    #seed种子
    parser.add_argument('--seed', type=int, default=4, help='random seed')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')

    # ？？？？？？？？？？？？？
    parser.add_argument('--nosharebn', action='store_true',
                        help='not share bn')
    # ？？？？？？？？？？？？？
    parser.add_argument('--plan', type=int,
                        default=1, help='choose the feature type')

    # algorithm-specific parameters
    # fedprox
    parser.add_argument('--mu', type=float, default=1e-3,
                        help='The hyper parameter for fedprox')
    # ？？？？？？？？？？？？？
    parser.add_argument('--alp', type=float, default=0.6,
                        help='hyperparmeter for fednt')
    # ？？？？？？？？？？？？？
    parser.add_argument('--init_iters', type=int, default=50,
                        help='iterations for init training')
    # ？？？？？？？？？？？？？
    parser.add_argument('--distill_iters', type=int, default=1,
                        help='iterations for init training')
    # ？？？？？？？？？？？？？
    parser.add_argument('--th_iters', type=int, default=300,
                        help='threshold for iters')

    parser.add_argument('--T', type=int, default=10,
                        help='aggregation T')
    parser.add_argument('--alphaK', type=float, default=5e-4,
                        help='hyperparmeter for fedamp')
    parser.add_argument('--beta', type=float, default=1e-1,
                        help='hyperparmeter for Hfedamp')
    # 是否转换优化器
    parser.add_argument('--transfer', action='store_true',
                        help='true or false of updata classfier')

    # parser.add_argument('--use_non_iid',action = 'store_true',
    #                     help='true or false of use FedCS')
    parser.add_argument('--use_non_iid', type=bool, default=True,
                        help='true or false of use FedCS')

    parser.add_argument('--cluster_num', type=int, default=4,
                        help='The hyper parameter for fedicfa')
    # parser.add_argument('--threshold', type=float, default=0.6,
    #                     help='threshold to use copy or distillation, hyperparmeter for metafed')
    # parser.add_argument('--lam', type=float, default=1.0,
    #                     help='init lam, hyperparmeter for metafed')
    parser.add_argument('--thresh', type=float, default=6,
                        help='threshold to use copy or distillation, hyperparmeter for fedpacfl')
    parser.add_argument('--model_momentum', type=float,
                        default=0.5, help='hyperparameter for fedap')
    parser.add_argument('--choose_cluter', type=str,
                        default='pa')
    parser.add_argument('--interval_increase_factor', type=int,
                        default=2)
    parser.add_argument('--base_interval', type=int,
                        default=1)
    parser.add_argument('--tri', type=int, default=0)
    parser.add_argument('--warm_up', type=int, default=60)
    args = parser.parse_args()



    # 告诉系统只使用GPU编号为0的设备进行计算，而隐藏其他的GPU设备。
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'  多个设备

    # 这个方法返回一个RandomState对象，该对象是一个独立的随机数生成器实例。
    # 你可以创建多个这样的实例，并给每个实例分配不同的种子
    args.random_state = np.random.RandomState(2)
    set_random_seed(args.seed)





    if args.dataset in ['vlcs', 'pacs', 'off_home']:
        #util.config py文件的函数
        args = img_param_init(args)
        # args.n_clients = 4


    exp_folder = (f'fed_selectall_{args.dataset}_{args.alg}_{args.non_iid_alpha}_'
                  f'{args.dis_datapercent}_{args.partition_data}_{args.alp}_{args.tri}_{args.transfer}')
    # args.save_path=./Layer-wise/
    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, args.alg)

    # 下面这一大段都是wandb   evalandprint也有四处wandb注释了！！！
    # experiment_name = 'Layer-wise'
    # wandb_log_dir = os.path.join('./', experiment_name)
    # if not os.path.exists('{}'.format(wandb_log_dir)):
    #     os.makedirs('{}'.format(wandb_log_dir))

    # wandb.init(project='FedCS', name=exp_folder, dir=wandb_log_dir)
    # #保存当前实验运行的基本信息。每次调用此方法，都会将实验的当前状态保存到Wandb服务器以及本地日志文件中。
    # wandb.run.save()
    # #将传入的参数字典args中的键值对更新到Wandb实验的配置信息中。这些配置可以包括模型结构、训练参数等
    # #并可在Wandb界面进行查看和分析，方便实验结果的比较和复现。
    # wandb.config.update(args)

    # 记录训练过程
    # 格式化字符串%(message)s表示只输出日志消息部分
    log_format = '%(message)s'
    # 将日志输出到标准输出流sys.stdout，日志级别设置为INFO，并使用log_format作为日志格式。
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    # 创建一个文件处理器fh，通过logging.FileHandler方法指定日志文件的路径为args.save_path下的log.txt文件。
    fh = logging.FileHandler(os.path.join(args.save_path, 'log.txt'))
    # 使用fh.setFormatter方法设置文件处理器的日志格式为log_format。
    fh.setFormatter(logging.Formatter(log_format))
    # 将文件处理器fh添加到全局日志记录器中，使得日志同时输出到标准输出和文件中。
    logging.getLogger().addHandler(fh)
    logging.info(f' dis_datapercent:{args.dis_datapercent}')
    logging.info(f' dataset:{args.dataset}')
    logging.info(f' distillation_dataset:{args.distillation_dataset}')
    logging.info(f' pre_dataset:{args.pre_dataset}')
    logging.info(f' device:{args.device}')
    logging.info(f' n_clients:{args.n_clients}')
    logging.info(f' non_iid_alpha:{args.non_iid_alpha}')
    logging.info(f' partition_data:{args.partition_data}')
    logging.info(f' seed:{args.seed}')
    logging.info(f' communication rounds:{args.iters}')
    logging.info(f' transfer:{args.transfer}')
    logging.info(f' use_non_iid:{args.use_non_iid}')
    logging.info(f' tri:{args.tri}')
    # 上述参数及其对应的值会被格式化输出并记录到日志文件中，方便开发者或研究人员查看和分析当前任务的配置详情。

    train_loaders, val_loaders, test_loaders, train_dataset = get_data(args.dataset)(args)
    public_dataset = get_distill_date(args)
    algclass = algs.get_algorithm_class(args.alg)(args)

    start = time.perf_counter()
    
    if args.alg == 'fedap':
        algclass.set_client_weight(train_loaders)
    elif args.alg in ['fedHPer']:
        if args.dataset == 'cifar10':
            count_layer = 8
            # count_layer = 1
        elif args.dataset == 'medmnist':
            count_layer = 2
            # count_layer = 1
        algclass.get_layer_index()
        layer_iter = args.iters // count_layer
    elif args.alg == 'metafed':
        algclass.init_model_flag(train_loaders, val_loaders)
        args.iters = args.iters - 1
        print('Common knowledge accumulation stage')
    elif args.alg in ['fedicfa', 'fedourcfl']:
        algclass.set_server_cluster(args)

    best_changed = False

    best_acc = [0] * args.n_clients
    best_tacc = [0] * args.n_clients
    start_iter = 0

    for a_iter in range(start_iter, args.iters):
        logging.info(f"============ Train round {a_iter} ============")
        # local client training
        if args.alg in ['fedicfa']:
            algclass.get_cluster(args, test_loaders)
        if args.alg == 'metafed':
            for c_idx in range(args.n_clients):
                algclass.client_train(
                    c_idx, train_loaders[algclass.csort[c_idx]], a_iter)
            algclass.update_flag(val_loaders)
        # fedavg base prox
        else:
            for wi in range(args.wk_iters):
                for client_idx in range(args.n_clients):
                    algclass.client_train(
                        client_idx, train_loaders[client_idx], a_iter)
            # best_acc, best_tacc, best_changed = evalandprint(
            #             args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, a_iter, best_changed)
            if args.alg in ['fedbn_nt', 'fedprox_nt', 'fedavg_nt', 'fedmlb_nt', 'fedpre']:
                if args.use_non_iid:
                    if a_iter % 40 == 0:
                        algclass.distill_wlocals(public_dataset, args, a_iter)
                        algclass.get_per_logits_soft(args, a_iter)
            elif args.alg == 'fedHPer':
                if (a_iter // layer_iter + 1) <= count_layer:
                    layer_name = f"layer{a_iter // layer_iter + 1}"
                    algclass.layer_index = algclass.layer_parameters[layer_name]
                else:
                    algclass.layer_index = algclass.layer_parameters[f"layer{count_layer}"]
                algclass.get_SSIM(a_iter)
            elif args.alg == 'fedavg_proto':
                algclass.get_per_logits_soft(args, a_iter)
            elif args.alg == 'fedamp':
                algclass.get_weight_matrix_heuristic(args, a_iter)
            elif args.alg == 'fedourcfl':
                algclass.distill_wlocals(public_dataset, args, a_iter)
                algclass.get_cluster(args, test_loaders)
                algclass.set_server_cluster(args)
            elif args.alg in ['fedcfl']:
                algclass.get_cluster(args, train_loaders)
                algclass.set_server_cluster(args)
            elif args.alg in ['fedpacfl']:
                if a_iter == 0:
                    algclass.get_cluster(args, train_loaders)
                    algclass.set_server_cluster(args)
            elif args.alg in ['fedmysoft']:
                if  a_iter ==0:
                    algclass.get_cluster(args)
                    algclass.set_server_cluster(args)
            #疑似消融
            elif args.alg in ['fedmysoft_nt']:
                if args.use_non_iid:
                    if a_iter % 40 == 0:
                        algclass.get_per_logits_soft(args, a_iter)

            elif args.alg in ['fedmd']:
                algclass.distill_wlocals(public_dataset, args, a_iter)
                ensemble_logits = torch.zeros_like(algclass.total_logits[0], dtype=torch.float32)
                for k in range(args.n_clients):
                    ensemble_logits += algclass.client_weight[k] * algclass.total_logits[k]
                for c_idx in range(args.n_clients):
                    algclass.client_distill(c_idx, ensemble_logits)

            # server aggregation
            if args.alg == 'fedlama':
                # 获得更新layer
                algclass.compute_layer_index(a_iter)
                algclass.server_aggre()
                algclass.interval_adjustment()
            elif args.alg in ['fedmd']:
                pass
            #avg,base,cfl,icfa
            else:
                algclass.server_aggre()
        # 下面这个best_acc其实是 best_tacc
        best_acc, best_tacc, best_changed = evalandprint(
            args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, a_iter,
            best_changed)

    if args.alg == 'metafed':
        logging.info('Personalization stage')
        for c_idx in range(args.n_clients):
            algclass.personalization(
                c_idx, train_loaders[algclass.csort[c_idx]], val_loaders[algclass.csort[c_idx]])
        best_acc, loss_list, best_changed = evalandprint(
            args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, a_iter,
            best_changed)

    for wi in range(1):
        for client_idx in range(args.n_clients):
            algclass.client_train_finetune(client_idx, train_loaders[client_idx], a_iter)

    best_acc, test_loss_list, best_changed = evalandprint_nosave(
            args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, a_iter, best_changed)

    end = time.perf_counter()
    # logging.info("运行时间为", round(end - start), 'seconds')
    s = 'Personalized test acc for each client: '
    for item in best_acc:
        s += f'{item:.4f},'
    mean_acc_test = np.mean(np.array(best_acc))
    s += f'\nAverage accuracy: {mean_acc_test:.4f}'
    logging.info(s)
