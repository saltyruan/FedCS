import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import defaultdict


# 配置参数
class Config:
    data_root = "data/medmnist/"
    num_clients = 10
    num_rounds = 100
    local_epochs = 3
    batch_size = 128
    lr = 0.01
    test_size = 0.2
    P = 5  # 分组更新频率
    public_ratio = 0.1  # 公共数据集比例
    T = 1.0  # 温度系数
    num_classes = 11  # 根据实际标签调整
    img_size = 28


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
def load_data():
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
        self.model = CNN().to('cuda')
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
                print(output.dtype,target.dtype)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()


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
                    # print(data.shape)
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


# CNN模型（优化版）
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, Config.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


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
    train_data, test_data, public_data = load_data()

    # 创建服务器和客户端
    server = Server(public_data)
    client_datasets = dirichlet_split(train_data, alpha=0.5)

    for i in range(Config.num_clients):
        server.clients.append(Client(i, client_datasets[i]))

    # 训练循环
    best_acc = 0.0
    for round in range(Config.num_rounds):
        # 客户端本地训练（使用自己的模型）
        for client in server.clients:
            client.local_train()  # 不再传递global_model

        # 定期更新分组和权重
        if round % Config.P==0:
            server.calculate_pac()
            server.dynamic_grouping()
            server.compute_weights()

        # 模型聚合并更新客户端
        server.aggregate()

        # 评估（随机选择一个客户端模型评估）
        acc = evaluate(server.clients[0].model, test_data)
        print(f"Round {round}: Test Acc {acc:.4f}")