import torch
import torch.utils.data
import copy
from server import resnet18  # 复用原有的模型定义


class NewServer(object):
    def __init__(self, conf, eval_dataset):
        self.conf = conf
        self.global_model = resnet18()
        self.eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.conf["batch_size"],
            shuffle=True
        )

        # FedCDA相关参数
        self.current_round = 0
        self.warmup_rounds = conf.get('warmup', 50)

        if torch.cuda.is_available():
            self.global_model = self.global_model.cuda()

    def calculate_divergence(self, model_a, model_b):
        """计算两个模型之间的差异（L2范数）"""
        total_divergence = 0.0
        for name in model_a:
            param_a = model_a[name].cpu()
            param_b = model_b[name].cpu()
            total_divergence += torch.norm(param_a - param_b).item()
        return total_divergence

    def greedy_model_selection(self, client_histories, selected_clients):
        """贪心算法选择最优模型版本组合"""
        selected_versions = {}

        for client_id in selected_clients:
            if client_id not in client_histories or not client_histories[client_id]:
                continue

            min_divergence = float('inf')
            best_idx = 0

            # 遍历该客户端所有历史模型
            for idx, model in enumerate(client_histories[client_id]):
                divergence = self.calculate_divergence(model, self.global_model.state_dict())
                if divergence < min_divergence:
                    min_divergence = divergence
                    best_idx = idx

            selected_versions[client_id] = best_idx

        return selected_versions

    def fedcda_aggregate(self, client_histories, selected_clients, active_client_count):
        """FedCDA聚合策略"""
        # 预热阶段使用最新模型
        if self.current_round < self.warmup_rounds:
            selected_versions = {
                client_id: len(client_histories[client_id]) - 1
                for client_id in selected_clients
                if client_id in client_histories and client_histories[client_id]
            }
        else:
            # 优化阶段：贪心选择最优历史模型
            selected_versions = self.greedy_model_selection(client_histories, selected_clients)

        # 初始化权重累加器
        weight_accumulator = {}
        for name, params in self.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        # 聚合选中的模型
        for client_id in selected_clients:
            if client_id not in client_histories or not client_histories[client_id]:
                continue

            if client_id in selected_versions:
                version_idx = selected_versions[client_id]
                if version_idx < len(client_histories[client_id]):
                    model_params = client_histories[client_id][version_idx]

                    # 计算相对于全局模型的更新
                    for name, data in self.global_model.state_dict().items():
                        weight_accumulator[name].add_(model_params[name] - data)

        # 更新全局模型
        self.model_aggregrate_new(weight_accumulator, active_client_count)
        self.current_round += 1

        return selected_versions

    def model_aggregrate_new(self, weight_accumulator, num):
        """聚合函数，更新全局模型"""
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * (1 / num)
            if data.type()!=update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    def model_eval(self):
        """评估全局模型性能"""
        self.global_model.eval()

        total_loss = 0.0
        correct = 0
        dataset_size = 0

        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = self.global_model(data)
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()

            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l