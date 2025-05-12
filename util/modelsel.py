from network.models import AlexNet, AlexNet_AP, PamapModel, lenet5v, ResNet8, ResNet18
import copy
import numpy as np
import torch.nn as nn
import torchvision


def modelsel(args, device):
    if args.dataset in ['vlcs', 'pacs', 'off_home', 'off-cal', 'covid']:
        if args.alg in ['fedap']:
            server_model = AlexNet_AP(num_classes=args.num_classes).to(device)
        else:
            server_model = AlexNet(num_classes=args.num_classes).to(device)
    elif args.dataset in ['medmnist', 'medmnistA']:
        server_model = lenet5v().to(device)
    elif 'cifar' in args.dataset:
        if args.dataset == 'cifar10':
            # server_model = ResNet18().to(device)
            server_model = torchvision.models.resnet18(pretrained=False).to(device)
            num_ftrs = server_model.fc.in_features  # 获取低级特征维度
            server_model.fc = nn.Linear(num_ftrs, 10).to(device) # 替换新的输出层
        else:
            server_model = ResNet18(num_classes=100).to(device)
    elif 'pamap' in args.dataset:
        server_model = PamapModel().to(device)
    # print(args.record_class)
    record_class = args.record_class
    count_client = np.array([np.array(record_class[i])[:, 1].sum() for i in range(0, args.n_clients*3, 3)])
    client_weights = count_client / count_client.sum()
    # client_weights = [1/args.n_clients for _ in range(args.n_clients)]
    #服务端模型进行深拷贝，以确保每个客户端的模型都是独立的副本，不会因为修改其中一个客户端的模型而影响到其他客户端。
    models = [copy.deepcopy(server_model).to(device)
              for _ in range(args.n_clients)]
    return server_model, models, client_weights
