import copy

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


# 配置参数
class Config:
    data_root = "data/medmnist/"
    num_clients = 20
    num_rounds = 200
    local_epochs = 10
    batch_size = 256
    lr = 0.01
    test_size = 0.2
    P = 5  # 分组更新频率
    public_ratio = 0.1 # 公共数据集比例
    T = 1.0  # 温度系数
    num_classes = 11  # 根据实际标签调整
    img_size = 28
    K = 3  # 保留的历史模型数量
    cross_round_interval = 5  # 跨轮次聚合间隔

# 数据预处理流程
transform = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# 自定义数据集类（修复版）
class MedMNIST(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data.astype(np.uint8)  # 保持uint8格式 (N, H, W, C)
        self.y_data = y_data.ravel().astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        img = self.x_data[idx]  # shape: (28, 28, 1)
        label = self.y_data[idx]

        # 转换为PIL Image处理
        if self.transform:
            img = Image.fromarray(img.squeeze(), mode='L')  # 转换为单通道灰度图
            img = self.transform(img)
        else:
            # 默认转换流程
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img, label


# 数据加载与预处理
def load_data(dataset="cifar10"):
    # 加载原始numpy数据
    x_data = np.load(f"{Config.data_root}xdata.npy")  # 假设shape为(N, 28, 28, 1)
    y_data = np.load(f"{Config.data_root}ydata.npy")  # shape (N, 1)

    # 划分训练测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data,
        test_size=Config.test_size,
        stratify=y_data,
        random_state=42
    )

    # 创建公共数据集（从测试集抽取）
    public_size = int(len(x_test) * Config.public_ratio)
    public_data = MedMNIST(x_test[:public_size], y_test[:public_size], transform=transform)

    return (
        MedMNIST(x_train, y_train, transform=transform),
        MedMNIST(x_test[public_size:], y_test[public_size:], transform=transform),
        public_data
    )


# 非IID数据划分（Dirichlet分布）
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


# 客户端类
class Client:
    def __init__(self, client_id, train_data):
        self.id = client_id
        self.model = ResNet18().to('cuda')

        self.model_cache = deque(maxlen=Config.K)  # 存储最近K轮模型
        self.optimizer = optim.SGD(self.model.parameters(), lr=Config.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(
            train_data,
            batch_size=Config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2
        )

    def local_train(self):
        self.model.train()
        for _ in range(Config.local_epochs):
            for data, target in self.train_loader:
                data, target = data.to('cuda'), target.to('cuda')
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        self.model_cache.append(copy.deepcopy(self.model.state_dict()))  # 保存当前模型
        return self.model.state_dict()


# 服务器类
class Server:
    def __init__(self, public_data):
        self.public_loader = DataLoader(
            public_data,
            batch_size=Config.batch_size,
            shuffle=False,
            pin_memory=True
        )
        self.clients = []
        self.pac = {}
        self.groups = defaultdict(list)
        self.weights = defaultdict(dict)
        self.hist_similarity = defaultdict(float)  # 历史相似度记录

    def calculate_pac(self):
        self.pac.clear()
        with torch.no_grad():
            for client in self.clients:
                client.model.eval()
                confidences = []
                for data, _ in self.public_loader:
                    data = data.to('cuda')
                    output = client.model(data)
                    prob = torch.softmax(output, dim=1)
                    confidences.append(prob.mean(dim=0).cpu())
                self.pac[client.id] = torch.stack(confidences).mean(dim=0)

    def dynamic_grouping(self):
        client_ids = [c.id for c in self.clients]
        n = len(client_ids)
        similarity_matrix = torch.zeros((n, n))

        # 计算带历史衰减的相似度
        decay_factor = 0.9
        for i, c1 in enumerate(client_ids):
            for j, c2 in enumerate(client_ids):
                current_sim = torch.cosine_similarity(
                    self.pac[c1].flatten(),
                    self.pac[c2].flatten(),
                    dim=0
                ).item()

                # 组合历史相似度
                hist_key = (c1, c2)
                weighted_sim = decay_factor * self.hist_similarity.get(hist_key, 0.0)
                weighted_sim += (1 - decay_factor) * current_sim

                similarity_matrix[i, j] = max(0, weighted_sim)
                self.hist_similarity[hist_key] = weighted_sim

        # 动态采样分组
        for i, cid in enumerate(client_ids):
            probs = torch.softmax(similarity_matrix[i] / Config.T, dim=0)
            sampled = torch.multinomial(probs, num_samples=3)  # 每个组选择3个客户端
            self.groups[cid] = [client_ids[idx] for idx in sampled]

    def compute_weights(self):
        for leader in self.groups:
            members = self.groups[leader]
            similarities = [self.hist_similarity[(leader, m)] for m in members]
            weights = torch.softmax(torch.tensor(similarities) / Config.T, dim=0)
            self.weights[leader] = {m: w.item() for m, w in zip(members, weights)}

    def aggregate(self):
        new_models = {}
        for leader in self.groups:
            # 获取组内成员的模型和权重
            members = self.groups[leader]
            weights = [self.weights[leader][m] for m in members]

            # 加权聚合
            averaged_params = {}
            for key in self.clients[members[0]].model.state_dict():
                params = [self.clients[m].model.state_dict()[key] * w
                          for m, w in zip(members, weights)]
                averaged_params[key] = torch.stack(params).sum(dim=0)

            new_models[leader] = averaged_params

            # 将聚合后的参数更新回对应客户端
        for client_id in new_models:
            self.clients[client_id].model.load_state_dict(new_models[client_id])

    def cross_round_aggregate(self, round):
        """每P轮执行跨轮次聚合"""
        if round % Config.P != 0:
            return

        # 1. 计算所有客户端的PAC
        pac_dict = {c.id: self.calculate_pac(c) for c in self.clients}

        # 2. 动态选择模型（简化版）
        selected_models = []
        for client in self.clients:
            # 选择历史模型中与当前全局最相似的
            similarities = [
                F.cosine_similarity(pac, pac_dict[client.id], dim=0)
                for pac in self.pac_cache.get(client.id, [])
            ]
            if similarities:
                best_idx = torch.argmax(torch.stack(similarities))
                selected_models.append(client.model_cache[best_idx])
            else:
                selected_models.append(client.model.state_dict())

        # 3. 加权聚合（示例使用平均权重）
        global_params = {}
        for key in selected_models[0].keys():
            global_params[key] = torch.stack(
                [params[key].float() for params in selected_models]
            ).mean(dim=0)

        # 4. 更新所有客户端模型
        for client in self.clients:
            client.model.load_state_dict(global_params)
            self.pac_cache[client.id] = [self.calculate_pac(client)]  # 更新PAC缓存


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

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
    test_loader = DataLoader(test_data, batch_size=Config.batch_size)
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to('cuda'), target.to('cuda')
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(test_data)


if __name__=="__main__":
    # 初始化数据
    train_data, test_data, public_data = load_data("medmnist")

    # 创建服务器和客户端
    server = Server(public_data)
    client_datasets = dirichlet_split(train_data, alpha=0.5)

    for i in range(Config.num_clients):
        server.clients.append(Client(i, client_datasets[i]))
    # 训练循环
    best_acc = 0.0
    for round in range(Config.num_rounds):
        # 1. 本地训练
        for client in selected_clients:
            client.local_train()

        # 2. 常规聚合（FedAvg）
        avg_params = average_parameters(selected_clients)
        for client in all_clients:  # 更新所有客户端
            client.model.load_state_dict(avg_params)

        # 3. 跨轮次聚合
        server.cross_round_aggregate(round)
    # for round in range(Config.num_rounds):
    #     # 客户端本地训练（使用自己的模型）
    #     for client in server.clients:
    #         client.local_train()  # 不再传递global_model
    #
    #     # 定期更新分组和权重
    #     if round % Config.P==0:
    #         server.calculate_pac()
    #         server.dynamic_grouping()
    #         server.compute_weights()
    #
    #     # 模型聚合并更新客户端
    #     server.aggregate()
    #     acc=0.0
    #     # 评估（随机选择一个客户端模型评估）
    #     for i in range(Config.num_clients):
    #         acc =acc+ evaluate(server.clients[i].model, test_data)
    #     acc=acc/Config.num_clients
    #     print(f"Round {round}: Test Acc {acc:.4f}")

    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import transforms
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from collections import defaultdict, deque
    import random
    import copy


    # 配置参数
    class Config:
        data_root = "data/medmnist/"
        num_clients = 20
        num_rounds = 150
        local_epochs = 5
        batch_size = 256
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
                img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
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
                num_workers=2
            )
            self.history_models = deque(maxlen=Config.K)  # 历史模型缓存
            self.current_pac = None  # 当前PAC

        def local_train(self, global_model=None):
            if global_model:
                self.model.load_state_dict(global_model)
            self.model.train()
            for _ in range(Config.local_epochs):
                for data, target in self.train_loader:
                    data, target = data.to('cuda'), target.to('cuda')
                    self.optimizer.zero_grad()
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


    class ResNet18(nn.Module):
        def __init__(self, num_classes=11):
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(3, 2, 1)

            self.layers = nn.Sequential(
                self._make_layer(64, 2),
                self._make_layer(128, 2, stride=2),
                self._make_layer(256, 2, stride=2),
                self._make_layer(512, 2, stride=2)
            )

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)

        def _make_layer(self, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                    nn.BatchNorm2d(planes)
                )
            layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes
            for _ in range(1, blocks):
                layers.append(BasicBlock(self.inplanes, planes))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layers(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)


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


    # 主流程
    if __name__ == "__main__":
        # 初始化数据
        x_data = np.load(f"{Config.data_root}xdata.npy")
        y_data = np.load(f"{Config.data_root}ydata.npy")
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, stratify=y_data)
        public_data = MedMNIST(x_test[:int(len(x_test) * 0.1)], y_test[:int(len(y_test) * 0.1)], transform=transform)
        test_data = MedMNIST(x_test, y_test, transform=transform)

        # 初始化联邦学习组件
        server = Server(public_data)
        client_datasets = dirichlet_split(MedMNIST(x_train, y_train), alpha=0.5)
        server.clients = [Client(i, ds) for i, ds in enumerate(client_datasets)]

        best_acc = 0.0
        for round in range(Config.num_rounds):
            # 客户端选择
            selected = server.select_clients()

            # 本地训练
            for client in selected:
                client.local_train(server.current_global)

            # 聚合策略
            if round % Config.P == 0 and round > 0:
                # 跨轮次聚合
                global_params = server.cross_round_aggregate()
            else:
                # 常规FedAvg聚合
                global_params = server.fedavg_aggregate(selected)

            # 更新全局模型
            for client in server.clients:
                client.model.load_state_dict(global_params)

            # 评估
            if round % 5 == 0:
                acc = sum(evaluate(c.model, test_data) for c in server.clients) / len(server.clients)
                print(f"Round {round}: Avg Acc {acc:.4f}")
                if acc > best_acc:
                    best_acc = acc
                    torch.save(global_params, "best_model.pth")

        print(f"Best Accuracy: {best_acc:.4f}")