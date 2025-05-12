import random
import torch
import copy
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from alg.fedavg import fedavg
import torch.nn.init as init
from typing import Iterable 
import skfuzzy as fuzz
from util.modelsel import modelsel
from util.traineval import train, test
from alg.core.comm import communication_cfl,communication_soft_cfl
from collections import OrderedDict
class fedmysoft(fedavg):
    def __init__(self, args):
        super(fedmysoft, self).__init__(args)
        self.weight_client_to_cluster = []  # 每一列值为1，代表客户端对每个簇的权重（和为1）
        self.weight_cluster_to_client = []  # 每一行值为1， 代表簇中每个客户端的权重（和为1）

    def set_server_cluster(self, args):
        self.w_glob_per_cluster = []
        for k in range(args.cluster_num):
            self.w_glob_per_cluster.append(copy.deepcopy(self.server_model))

    def get_cluster(self,args):
        num_clients = len(self.client_model)
        similarity_matrix = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            similarity_matrix[i,i] = 1
            for j in range(i+1, num_clients):
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
                V_i_5=V_i_5.flatten()
                V_j_5=V_j_5.flatten()
                # 计算前5个左奇异向量之间的余弦相似性
                cosine_similarity = np.dot(V_i_5, V_j_5) / (np.linalg.norm(V_i_5) * np.linalg.norm(V_j_5))
                similarity_matrix[i, j] = similarity_matrix[j, i] = cosine_similarity

        self.weight_client_to_cluster = fuzz_clustering(similarity_matrix)

        self.selected_clusters = max_values_softmax_cluster(self.weight_client_to_cluster,4)
        #self.weight_cluster_to_client = self.weight_client_to_cluster / np.sum(self.weight_client_to_cluster, axis=1, keepdims=True)
        self.clinet_to_its = max_values_softmax_client(self.weight_client_to_cluster,2)

        args.cluster_num = len(self.selected_clusters)
        args.selected_clusters = self.selected_clusters
        args.clinet_to_its = self.clinet_to_its

    def server_aggre(self):
        self.w_glob_per_cluster, self.client_model = communication_soft_cfl(
            self.args, self.client_model, self.selected_clusters, self.clinet_to_its,self.w_glob_per_cluster)

def max_values_softmax_cluster(tensor, k):
    result_dict = OrderedDict()
    for row_idx in range(tensor.shape[0]):
        row_values = tensor[row_idx, :]
        topk_indices = np.argsort(row_values)[-k:][::-1]  # 获取最大的前 k 个值的索引
        topk_values = row_values[topk_indices]
        softmaxed_topk = F.softmax(torch.tensor(topk_values), dim=0).numpy()  # 对最大的前 k 个值进行 softmax 操作
        # 将结果添加到有序字典中
        result_dict[row_idx] = [(topk_indices[i], softmaxed_topk[i]) for i in range(k)]
    return result_dict


def max_values_softmax_client(tensor, k):
    result_dict = OrderedDict()
    for col_idx in range(tensor.shape[1]):
        col_values = tensor[:, col_idx]
        topk_indices = np.argsort(col_values)[-k:][::-1]  # 获取最大的前 k 个值的索引
        topk_values = col_values[topk_indices]
        softmaxed_topk = F.softmax(torch.tensor(topk_values), dim=0).numpy()  # 对最大的前 k 个值进行 softmax 操作
        # 将结果添加到有序字典中
        result_dict[col_idx] = [(topk_indices[i], softmaxed_topk[i]) for i in range(k)]
    return result_dict


def fuzz_clustering(similarity_matrix, clu_num=5, m=2, error=0.0005, maxiter=1000, init=None):
    # 执行模糊C均值聚类
    centers, weight, _, _, _, _, _ = fuzz.cluster.cmeans(
        similarity_matrix.T,  # 将相似度矩阵进行转置，以便每行表示一个数据点
        c=clu_num,  # 指定簇的数量
        m=m,  # 模糊度参数，通常设置为2
        error=error,  # error 参数表示算法的收敛条件，即迭代过程中停止的误差阈值。当两次迭代之间的误差小于该阈值时，算法将停止迭代。
        maxiter=maxiter,  # 最大迭代次数
        init=init  # 初始簇中心，如果为None，则随机初始化
    )
    return weight  # 返回数据点关于每个簇权重的二维数组，行为簇，列为数据点。













def weight_flatten(model):
    params = []
    for u in model.parameters():
        params.append(u.view(-1))
    params = torch.cat(params)
    return params

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    return 


def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x