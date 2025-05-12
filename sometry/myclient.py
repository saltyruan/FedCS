import torch
import copy
from collections import deque


class NewClient(object):
    def __init__(self, conf, model, train_dataset, train_dataset_idcs, id=1):
        self.conf = conf  # 配置信息
        self.local_model = copy.deepcopy(model)  # 本地模型
        self.client_id = id  # 客户端id
        self.train_dataset = train_dataset  # 训练集

        # 训练集下标
        self.train_dataset_idcs = train_dataset_idcs
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=conf['batch_size'],
            sampler=torch.utils.data.sampler.SubsetRandomSampler(self.train_dataset_idcs)
        )

        # FedCDA: 添加模型历史缓存
        self.model_history = deque(maxlen=conf.get('memory_k', 3))

        if torch.cuda.is_available():
            self.local_model = self.local_model.cuda()

    def local_train(self, model):
        # 从全局模型复制参数到本地模型
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        # 定义优化器
        optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=self.conf['lr'],
            momentum=self.conf['momentum']
        )

        self.local_model.train()  # 标记为训练模式，参数可以改变

        for e in range(self.conf['local_epochs']):  # 本地轮数
            for batch_id, batch in enumerate(self.train_loader):  # 按batch加载训练数据
                data, target = batch  # 获得本batch数据和标签
                if torch.cuda.is_available():  # 如果GPU可用
                    data, target = data.cuda(), target.cuda()  # 放在GPU计算

                optimizer.zero_grad()  # 优化器置零
                output = self.local_model(data)  # 获得预测结果
                loss = torch.nn.functional.cross_entropy(output, target)  # 获得预测损失
                loss.backward()  # 进行反向传播
                optimizer.step()

            print('本地模型{}完成第{}轮训练'.format(self.client_id, e))  # 打印目前训练进度

        # 保存训练后的模型到历史记录
        model_copy = {}
        for name, data in self.local_model.state_dict().items():
            model_copy[name] = data.clone()

        self.model_history.append(model_copy)

        # 返回训练后的模型参数
        return model_copy

    def get_model_history(self):
        """获取模型历史记录"""
        return list(self.model_history)