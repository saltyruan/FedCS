import logging
import random
from matplotlib import pyplot as plt
import torch
import copy
import wandb
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from alg.fedavg import fedavg

from util.modelsel import modelsel
from util.traineval import train, test
from alg.core.comm import communication
from util.draw import heatmap, annotate_heatmap


class fedmysoft_nt(fedavg):
    def __init__(self, args):
        super(fedmysoft_nt, self).__init__(args)

    def set_client_weight(self):
        preckpt = '/home/featurize/work/PersonalizationFL-new/new_cks_balance/fed_selectall_medmnist_base_0.1_0.1_non_iid_dirichlet_0.6_0/base'
        i = 0
        for model in self.client_model:
            model.load_state_dict(torch.load(preckpt)['client_model_'+str(i)])
            i = i + 1
        del preckpt

    def get_per_logits_soft(self, args, iter):
        # 获得客户端之间的相似度矩阵 [nclient, nclient]
        num_clients = len(self.client_model)
        similarity_matrix = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            similarity_matrix[i, i] = 1
            for j in range(i + 1, num_clients):
                # 获取第i个和第j个客户端模型的fc层权重
                fc_i = self.client_model[i].fc.weight.data.cpu().numpy()
                fc_j = self.client_model[j].fc.weight.data.cpu().numpy()

                # 执行SVD并获取右奇异向量
                # U: 84 * 84
                # Σ: 84 * 11
                # V: 11 * 11
                # 模型参数的比较：模型参数通常表示模型学习到的特征表示或权重分配，这些参数可以看作是模型对输入数据空间的一种特定“视角”或“投影”。对模型参数进行SVD，实际上是将这种复杂的高维表示转化为一组正交基（右奇异向量）及其对应的权重（奇异值）。通过比较这些正交基，可以评估模型在学习数据特征时的相似性。
                # 余弦相似度计算：余弦相似度衡量的是两个非零向量在方向上的接近程度，而非它们的长度。在SVD上下文中，您关心的是模型参数所对应的右奇异向量（V或V ^ T的列向量）的方向差异，而非它们的尺度（由奇异值控制）。因此，计算两个模型参数SVD后得到的右奇异向量之间的余弦相似度，可以有效地量化模型在学习数据特征时的“视角”或“投影”是否相似。
                # 综上所述，当您使用两个模型参数进行SVD，并希望通过余弦相似度判断它们的相似性时，应关注并使用右奇异值矩阵（V或V ^ T）。具体操作时，可以选择对每个模型参数进行SVD，然后取对应的右奇异向量列向量，计算这些向量之间的余弦相似度，以此评估模型间的相似性。
                _, _, V_i = torch.svd(torch.from_numpy(fc_i).cuda())
                _, _, V_j = torch.svd(torch.from_numpy(fc_j).cuda())

                # 取左奇异向量的前全部列
                V_i_5 = V_i[:, :].cpu().numpy()
                V_j_5 = V_j[:, :].cpu().numpy()
                V_i_5 = V_i_5.flatten()
                V_j_5 = V_j_5.flatten()
                # 计算前5个左奇异向量之间的余弦相似性
                cosine_similarity = np.dot(V_i_5, V_j_5) / (np.linalg.norm(V_i_5) * np.linalg.norm(V_j_5))
                similarity_matrix[i, j] = similarity_matrix[j, i] = cosine_similarity


        selected_clusters = {i: [] for i in range(num_clients)}
        for i in range(num_clients):
            for j in range(num_clients):
                # 等概率随机采样rand
                rand = random.random()
                if rand < similarity_matrix[i,j]:
                    selected_clusters[i].append(j)
        # 获得传送给每个客户的logits
        num_cluster = [len(v) for k,v in selected_clusters.items()]
        weightmatrix = []
        for k,v in selected_clusters.items():
            selectN = v
            gap = torch.from_numpy(similarity_matrix[k][v])
            if num_cluster[k] == 1:
                localweight = [1]
            else:
                localweight = torch.nn.functional.softmax(gap, dim=0)
            weight_one = torch.zeros(num_clients)
            n = 0
            for m in selectN:
                weight_one[m] = localweight[n]
                n = n + 1
            weightmatrix.append(weight_one)
        weightmatrix = torch.stack(weightmatrix, dim=0)
        self.client_weight = weightmatrix

