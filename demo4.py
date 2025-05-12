import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import defaultdict, deque
import random
import copy

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 配置参数
class Config:
    data_root = "data/medmnist/"
    num_clients = 20
    num_rounds = 150
    local_epochs = 5
    batch_size = 64
    lr = 0.01
    test_size = 0.2
    P = 5  # 跨轮次聚合频率
    K = 3  # 历史模型保留轮数
    public_ratio = 0.1
    T = 1.0  # 温度系数
    num_classes = 11
    img_size = 28
    selection_ratio = 0.3  # 客户端选择比例

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 数据集类
class MedMNIST(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data.astype(np.uint8)
        self.y_data = y_data.ravel().astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        img = self.x_data[idx]
        label = self.y_data[idx]
        if self.transform:
            img = Image.fromarray(img.squeeze(), mode='L')
            img = self.transform(img)
        else:
            # print(img.shape)
            img = torch.from_numpy(img.transpose(0,1,2)).float() / 255.0
            # print("after:",img.shape)
        return img, label


# 客户端类（含历史模型缓存）
class Client:
    def __init__(self, client_id, train_data):
        self.id = client_id
        self.model = ResNet18().to('cuda')
        self.optimizer = optim.SGD(self.model.parameters(), lr=Config.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(
            train_data,
            batch_size=Config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=11
        )
        self.history_models = deque(maxlen=Config.K)  # 历史模型缓存
        self.current_pac = None  # 当前PAC

    def local_train(self, global_model=None):
        if global_model:
            self.model.load_state_dict(global_model)
        self.model.train()
        for _ in range(Config.local_epochs):
            for data, target in self.train_loader:
                # data=data.swapaxes(0,1)
                data, target = data.to('cuda'), target.to('cuda')
                self.optimizer.zero_grad()
                # print(data.shape)
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        # 保存当前模型到历史
        self.history_models.append(copy.deepcopy(self.model.state_dict()))
        return self.model.state_dict()


# 服务器类（整合跨轮次聚合）
class Server:
    def __init__(self, public_data):
        self.clients = []
        self.public_loader = DataLoader(
            public_data,
            batch_size=Config.batch_size,
            shuffle=False,
            pin_memory=True
        )
        self.hist_similarity = defaultdict(float)
        self.current_global = None

    def select_clients(self):
        num_selected = int(Config.selection_ratio * Config.num_clients)
        return random.sample(self.clients, num_selected)

    def calculate_pac(self, model):
        model.eval()
        confidences = []
        with torch.no_grad():
            for data, _ in self.public_loader:
                data = data.to('cuda')
                output = model(data)
                prob = torch.softmax(output, dim=1)
                confidences.append(prob.mean(dim=0).cpu())
        return torch.stack(confidences).mean(dim=0)

    def cross_round_aggregate(self):
        # 阶段1：计算所有客户端的PAC
        pac_dict = {}
        for client in self.clients:
            client.current_pac = self.calculate_pac(client.model)
            pac_dict[client.id] = client.current_pac

        # 阶段2：动态模型选择
        aggregated = defaultdict(list)
        for client in self.clients:
            # 计算历史模型相似度
            similarities = []
            for hist_model in client.history_models:
                temp_model = ResNet18().to('cuda')
                temp_model.load_state_dict(hist_model)
                hist_pac = self.calculate_pac(temp_model)
                sim = F.cosine_similarity(
                    client.current_pac.flatten(),
                    hist_pac.flatten(),
                    dim=0
                ).item()
                similarities.append(sim)

            # 选择最佳历史模型
            if similarities:
                best_idx = np.argmax(similarities)
                selected_model = client.history_models[best_idx]
            else:
                selected_model = client.model.state_dict()

            aggregated[client.id] = selected_model

        # 阶段3：加权聚合
        global_params = {}
        client_weights = [len(c.train_loader.dataset) for c in self.clients]
        total_samples = sum(client_weights)

        for key in aggregated[self.clients[0].id].keys():
            global_params[key] = sum(
                aggregated[c.id][key] * (client_weights[i] / total_samples)
                for i, c in enumerate(self.clients)
            )

        self.current_global = global_params
        return global_params

    def fedavg_aggregate(self, selected_clients):
        client_weights = [len(c.train_loader.dataset) for c in selected_clients]
        total_samples = sum(client_weights)

        global_params = {}
        for key in selected_clients[0].model.state_dict().keys():
            global_params[key] = sum(
                c.model.state_dict()[key] * (client_weights[i] / total_samples)
                for i, c in enumerate(selected_clients)
            )
        self.current_global = global_params
        return global_params


# ResNet18模型（保持原样）
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)


# class ResNet18(nn.Module):
#     def __init__(self, num_classes=Config.num_classes):
#         super().__init__()
#         self.inplanes = 64
#         self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.maxpool = nn.MaxPool2d(3, 2, 1)
#
#         self.layers = nn.Sequential(
#             self._make_layer(64, 2),
#             self._make_layer(128, 2, stride=2),
#             self._make_layer(256, 2, stride=2),
#             self._make_layer(512, 2, stride=2)
#         )
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, num_classes)
#
#     def _make_layer(self, planes, blocks, stride=1):
#         downsample = None
#         if stride!=1 or self.inplanes!=planes:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
#                 nn.BatchNorm2d(planes)
#             )
#         layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
#         self.inplanes = planes
#         for _ in range(1, blocks):
#             layers.append(BasicBlock(self.inplanes, planes))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.maxpool(x)
#         x = self.layers(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)

class ResNet18(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=11):
        super(ResNet18, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 评估函数
def evaluate(model, test_data):
    model.eval()
    loader = DataLoader(test_data, batch_size=256)
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to('cuda'), target.to('cuda')
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(test_data)

def dirichlet_split(dataset, alpha=0.5):
    labels = dataset.y_data
    client_indices = [[] for _ in range(Config.num_clients)]

    # 对每个类别进行分配
    for class_idx in np.unique(labels):
        indices = np.where(labels==class_idx)[0]
        np.random.shuffle(indices)

        # 生成划分比例
        proportions = np.random.dirichlet(np.repeat(alpha, Config.num_clients))
        proportions = (proportions * len(indices)).astype(int)
        proportions[-1] = len(indices) - np.sum(proportions[:-1])

        # 分配样本
        start = 0
        for client_id, num in enumerate(proportions):
            end = start + num
            client_indices[client_id].extend(indices[start:end])
            start = end

    return [torch.utils.data.Subset(dataset, ids) for ids in client_indices]



import datetime
# 主流程
if __name__=="__main__":
    # 初始化数据
    x_data = np.load(f"{Config.data_root}xdata.npy")
    y_data = np.load(f"{Config.data_root}ydata.npy")
    print("数据导入完毕！")
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, stratify=y_data)
    public_data = MedMNIST(x_test[:int(len(x_test) * 0.1)], y_test[:int(len(y_test) * 0.1)], transform=transform)
    test_data = MedMNIST(x_test, y_test, transform=transform)
    x_train=np.expand_dims(x_train,1)
    x_test=np.expand_dims(x_test,1)
    y_train=np.expand_dims(y_train,1)
    y_test=np.expand_dims(y_test,1)
    # print("xtrain:",x_train.shape," type:",type(x_train))
    # 初始化联邦学习基本数据
    server = Server(public_data)
    client_datasets = dirichlet_split(MedMNIST(x_train, y_train), alpha=0.1)
    server.clients = [Client(i, ds) for i, ds in enumerate(client_datasets)]

    best_acc = 0.0
    for round in range(Config.num_rounds):
        print("开始训练",datetime.datetime.now())
        # 客户端选择
        selected = server.select_clients()
        print("客户端选择：",datetime.datetime.now())
        # 本地训练
        for client in selected:
            client.local_train(server.current_global)
        print("本地训练", datetime.datetime.now())
        # 聚合策略
        if round % Config.P==0 and round > 0:
            # 跨轮次聚合
            global_params = server.cross_round_aggregate()
        else:
            # 常规FedAvg聚合
            global_params = server.fedavg_aggregate(selected)
        print("聚合：", datetime.datetime.now())
        # 更新全局模型
        for client in server.clients:
            client.model.load_state_dict(global_params)
        print("更新全局模型：", datetime.datetime.now())
        # 评估
        if round % 5==0:
            acc = sum(evaluate(c.model, test_data) for c in server.clients) / len(server.clients)
            print(f"Round {round}: Avg Acc {acc:.4f}")
            best_acc = acc
            torch.save(global_params, "best_model.pth")
        print("完成此轮", datetime.datetime.now())
    print(f"Best Accuracy: {best_acc:.4f}")