import os
import json
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil

import dataset
from sometry.myserver import NewServer
from sometry.myclient import NewClient

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'


def monitor_memory():
    """监控内存使用情况"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return f"内存使用: {mem_info.rss / (1024 * 1024):.2f} MB"


if __name__=='__main__':
    # 存储容器，用于绘制图像
    accuracies = []  # 存放准确率
    losses = []  # 存放损失

    # 载入配置文件
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到，请确保config.json位于: {config_path}")

    # 获取训练数据和测试数据
    train_datasets, eval_datasets, client_indices = dataset.get_dataset('../data/',
        config['type'], config.get('alpha', 0.1))

    # 创建服务器
    server = NewServer(config, eval_datasets)

    # 客户端参数
    total_clients = config['no_models']
    memory_size = config['memory_k']
    warmup_rounds = config['warmup']

    # 初始化客户端列表
    clients = []
    for client_idx in range(total_clients):
        clients.append(NewClient(
            config, server.global_model, train_datasets, client_indices[client_idx], client_idx))

    # 记录开始时间
    start_time = time.strftime('%Y%m%d_%H%M%S')
    print(f"开始时间: {start_time}")
    print(f"初始化完成，{monitor_memory()}")

    # 开始联邦学习过程
    for global_round in tqdm(range(config['global_rounds'])):
        print(f"\n===== 第 {global_round} 轮训练开始 =====")
        print(monitor_memory())

        # 当前轮次中活跃的客户端数量
        active_client_count = total_clients

        # 随机选择k个客户端参与本轮训练
        all_client_indices = list(range(total_clients))
        sampled_client_indices = random.sample(all_client_indices, config['k'])

        print(f"选择了客户端: {sampled_client_indices}")

        # 本地训练阶段
        for client_idx in sampled_client_indices:
            client = clients[client_idx]
            client.local_train(server.global_model)

        # 收集所有客户端的模型历史
        client_histories = {i: clients[i].get_model_history() for i in range(total_clients)}

        # FedCDA聚合
        selected_versions = server.fedcda_aggregate(
            client_histories,
            sampled_client_indices,
            active_client_count
        )

        if global_round >= warmup_rounds:
            print(f"选择的模型版本: {selected_versions}")

        # 评估全局模型性能
        accuracy, loss = server.model_eval()
        accuracies.append(accuracy)
        losses.append(loss)
        print(f'全局模型：第{global_round}轮完成！准确率：{accuracy:.2f} loss: {loss:.2f}')

    # 记录结束时间并计算持续时间
    end_time = time.strftime('%Y%m%d_%H%M%S')
    print(f"结束时间: {end_time}")
    print(f"最终内存使用: {monitor_memory()}")

    # 创建结果目录
    os.makedirs("./results", exist_ok=True)

    # 绘制并保存结果图
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(accuracies)), accuracies, label='Accuracy')
    plt.legend()
    plt.xlabel('Global Rounds')
    plt.ylabel('Accuracy')
    plt.title("FedCDA: 跨轮次差异感知聚合 (alpha={})".format(config.get('alpha', 0.1)))
    plt.savefig(f"./results/fedcda_{end_time}.jpg")
    plt.close()

    # 保存数据到CSV
    results_df = pd.DataFrame({
        "round": range(len(accuracies)),
        "accuracy": accuracies,
        "loss": losses
    })
    results_df.to_csv(f"./results/fedcda_data_{end_time}.csv", index=False)

    # 输出运行时间
    print(f"持续时间: {end_time} - {start_time}")