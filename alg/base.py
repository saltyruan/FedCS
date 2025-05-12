import logging
import torchvision
import copy
import torch.nn as nn
from alg.fedavg import fedavg
import torch.optim as optim


class base(fedavg):
    def __init__(self, args):
        super(base, self).__init__(args)
        # if args.dataset == 'cifar10':
        #     self.server_model = torchvision.models.resnet18(pretrained=True).to(args.device)
        #     num_ftrs = self.server_model.fc.in_features  # 获取低级特征维度
        #     self.server_model.fc = nn.Linear(num_ftrs, 10).to(args.device) # 替换新的输出层
        # else:
        #     logging.info('没有定义预训练模型')
        # self.client_model = [copy.deepcopy(self.server_model).to(args.device) for _ in range(args.n_clients)]
        # if args.transfer:
        #     self.optimizers =  [optim.SGD(params=[p for name, p in self.client_model[idx].named_parameters() 
        #                         if 'linear' in name or 'fc' in name], lr=args.lr) for idx in range(args.n_clients)]
        # else:
        #     self.optimizers = [optim.SGD(params=self.client_model[idx].parameters(
        # ), lr=args.lr) for idx in range(args.n_clients)]

    def server_aggre(self):
        pass
