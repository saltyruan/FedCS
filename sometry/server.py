import torch
import torch.utils.data
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import copy

__all__ = ["resnet10", "resnet18", "resnet34"]


class FedAvgNetMNIST(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(FedAvgNetMNIST, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.linear_2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.ndim < 4:
            x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride!=1 or in_planes!=self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride!=1 or in_planes!=self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet10(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)


def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


class Server(object):
    def __init__(self, conf, eval_dataset):  # 构造函数
        self.conf = conf
        self.global_model = resnet18()
        # self.global_model=FedAvgNetMNIST()
        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"],
            shuffle=True)  # 创建测试集加载器用于测试最终的聚合模型
        if torch.cuda.is_available():
            self.global_model = self.global_model.cuda()

    def model_aggregrate(self, weight_accumulator):  # 聚合函数，更新全局模型用
        # 其中weight_accumulator存放客户端上传参数的变化值, 即更新前全局模型和本地更新后模型的参数变化L_t+1-G_t
        for name, data in self.global_model.state_dict().items():  # 获得全局模型的变量名和参数值
            update_per_layer = weight_accumulator[name] * (1 / self.conf["k"])
            # update_per_layer = weight_accumulator[name] * self.conf["lambda"]  # 乘以系数λ（更新步长）
            if data.type()!=update_per_layer.type():  # 如果数据类型不符
                data.add_(update_per_layer.to(torch.int64))  # 进行数据转换后再累加
            else:
                data.add_(update_per_layer)  # 直接进行累加

    def model_aggregrate_new(self, weight_accumulator, num):  # 聚合函数，更新全局模型用
        # 其中weight_accumulator存放客户端上传参数的变化值, 即更新前全局模型和本地更新后模型的参数变化L_t+1-G_t
        for name, data in self.global_model.state_dict().items():  # 获得全局模型的变量名和参数值
            update_per_layer = weight_accumulator[name] * (1 / num)  # 除一个所有的量
            # update_per_layer = weight_accumulator[name] * self.conf["lambda"]  # 乘以系数λ（更新步长）
            if data.type()!=update_per_layer.type():  # 如果数据类型不符
                data.add_(update_per_layer.to(torch.int64))  # 进行数据转换后再累加
            else:
                data.add_(update_per_layer)  # 直接进行累加

    def model_eval(self):  # 训练结束后，对全局模型进行评估的函数
        self.global_model.eval()  # 标记进入测试模式，模型参数不发生变化

        total_loss = 0.0  # 记录损失和，计算平均损失用
        correct = 0  # 记录正确数目
        dataset_size = 0  # 测试数据总数
        for batch_id, batch in enumerate(self.eval_loader):  # 对测试数据进行编号和按batch提取数据
            data, target = batch  # 解包数据和标签
            dataset_size += data.size()[0]  # 获得当前batch的数据量，进行累加
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()  # 如果pytorch支持GPU，则使用cuda计算
            output = self.global_model(data)  # 获得预测结果
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()  # 把计算的损失进行累加
            pred = output.data.max(1)[1]  # 取预测值最大的类索引
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()  # 获得该batch中预测正确的数目
        acc = 100.0 * (float(correct) / float(dataset_size))  # 得到准确率的百分值
        total_l = total_loss / dataset_size  # 计算平均损失值
        return acc, total_l


