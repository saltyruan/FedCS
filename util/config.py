# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import numpy as np
import torch


#会设置args.dataset、args.domains、args.img_dataset
#args.shuffle_shape = (3, 228, 228)、args.input_shape = (3, 224, 224)
#args.num_classes = 10
def img_param_init(args):
    dataset = args.dataset
        #看这个！！！它包含 4 个领域，每个领域由 65 个类别组成，每个类别平均 70 张图像，合计包含 15,500 张图像
    if dataset == 'off_home':
        domains = ['Art', 'Clipart', 'Product', 'Real_World']
        # 看这个！！！PACS 它由照片、艺术绘画、卡通图片和素描4个领域的图像组成，每个领域包含7个类别
        """ 1,670 张照片
            2,048 张艺术绘画
            2,344 张卡通图片
            3,929 张素描"""
    elif dataset == 'pacs':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
        # 看这个！！！VLCS域名共有5类:鸟，车，椅子，狗和人
    elif dataset == 'vlcs':
        domains = ['Caltech101']
        # 除了Caltech101，还有'LabelMe', 'SUN09', 'VOC2007'个子数据集


    elif dataset == 'office':
        domains = ['amazon', 'dslr', 'webcam']
    elif dataset == 'office-caltech':
        domains = ['amazon', 'dslr', 'webcam', 'caltech']
    elif dataset == 'dg5':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']
    elif dataset == 'vlcs-caltech101':
        domains = ['Caltech101']
    elif dataset == 'dg4':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn']
    elif dataset == 'terra':
        domains = ['location_38', 'location_43', 'location_46', 'location_100']
    elif dataset == 'domain_net':
        domains = ["clipart", "infograph","painting", "quickdraw", "real", "sketch"]
    else:
        print('No such dataset exists!')

    args.domains = domains

    args.img_dataset = {
        'off_home': ['Art', 'Clipart', 'Product', 'Real_World'],
        'pacs': ['art_painting', 'cartoon', 'photo', 'sketch'],
        'vlcs': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007'],

        'office': ['amazon', 'dslr', 'webcam'],
        'office-caltech': ['amazon', 'dslr', 'webcam', 'caltech'],
        'dg5': ['mnist', 'mnist_m', 'svhn', 'syn', 'usps'],
        'dg4': ['mnist', 'mnist_m', 'svhn', 'syn'],
        'terra': ['location_38', 'location_43', 'location_46', 'location_100'],
        'domain_net': ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    }
    if dataset in ['dg5', 'dg4']:
        args.shuffle_shape = (3, 36, 36)
        args.input_shape = (3, 32, 32)
        args.num_classes = 10
    else:
        args.shuffle_shape = (3, 228, 228)
        args.input_shape = (3, 224, 224)
        if args.dataset == 'off_home':
            args.num_classes = 65
        elif args.dataset == 'pacs':
            args.num_classes = 7
        elif args.dataset == 'vlcs':
            args.num_classes = 5


        elif args.dataset == 'office':
            args.num_classes = 31
        elif args.dataset == 'terra':
            args.num_classes = 10
        elif args.dataset == 'domain_net':
            args.num_classes = 345
        else:
            args.num_classes = 4
    return args


def set_random_seed(seed=0):
    #Python内置的random模块
    random.seed(seed)
    #NumPy库中的随机数生成器种子为给定值
    np.random.seed(seed)
    #分别针对CPU和GPU环境设置PyTorch框架下的全局随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #这是PyTorch中CuDNN的一个性能优化选项。当设为False时，将禁用CuDNN对卷积算法的自动搜索和选择，转而固定使用一种确定性算法
    #这有助于提高模型在不同运行之间的可复现性，但可能会牺牲一定的运算速度。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
