import random
import numpy as np
import torch
import os
import math
import functools

#用于分布式计算和通信的模块
import torch.distributed as dist


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices
        self.replaced_targets = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        return (self.data[data_idx][0], self.data[data_idx][1])

    def update_replaced_targets(self, replaced_targets):
        self.replaced_targets = replaced_targets

        # evaluate the the difference between original labels and the simulated labels.
        count = 0
        for index in range(len(replaced_targets)):
            data_idx = self.indices[index]

            if self.replaced_targets[index] == self.data[data_idx][1]:
                count += 1
        return count / len(replaced_targets)

    def set_targets(self, replaced_targets):
        self.replaced_targets = replaced_targets

    def get_targets(self):
        return self.replaced_targets

    def clean_replaced_targets(self):
        self.replaced_targets = None


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(
        self, conf, data, partition_sizes, partition_type, consistent_indices=True
    ):
        # prepare info.
        self.conf = conf                        #args
        self.partition_sizes = partition_sizes  # 分割大小
        self.partition_type = partition_type    # 如何分割
        self.consistent_indices = consistent_indices    # 应该是不同gpu使用同一个数据索引列表
        self.partitions = []


        self.data_size = len(data)
        if type(data) is not Partition:
            self.data = data
            indices = np.array([x for x in range(0, self.data_size)])
        else:
            self.data = data.data
            indices = data.indices

        self.partition_indices(indices)

    def partition_indices(self, indices):
        indices = self._create_indices(indices)
        if self.consistent_indices:
            indices = self._get_consistent_indices(indices)

        if self.partition_type == 'evenly':
            if self.conf.dataset in ['vlcs', 'pacs', 'off_home']:
                classes = np.unique(self.data.dataset.targets)
            else:
                classes = np.unique(self.data.targets)
            lp = len(self.partition_sizes)
            ti = indices[:, 0]
            ttar = indices[:, 1]
            for i in range(lp):
                self.partitions.append(np.array([]))
            for c in classes:
                tindice = np.where(ttar == c)[0]
                lti = len(tindice)
                from_index = 0
                for i in range(lp):
                    partition_size = self.partition_sizes[i]
                    to_index = from_index + int(partition_size * lti)
                    if i == (lp-1):
                        self.partitions[i] = np.hstack(
                            (self.partitions[i], ti[tindice[from_index:]]))
                    else:
                        self.partitions[i] = np.hstack(
                            (self.partitions[i], ti[tindice[from_index:to_index]]))
                    from_index = to_index
            for i in range(lp):
                self.partitions[i] = self.partitions[i].astype(np.int32).tolist()
        elif 'noniid-#label' in self.partition_type:
            if self.conf.dataset in ['vlcs', 'pacs', 'off_home']:
                classes = np.unique(self.data.dataset.targets)
            else:
                classes = np.unique(self.data.targets)
            ti = indices[:, 0]
            ttar = indices[:, 1]
            num = eval(self.partition_type[13:])
            if self.conf.dataset == 'cifar100':
                K = 100
            elif self.conf.dataset == 'medmnist':
                K = 11
            elif self.conf.dataset == 'covid':
                K = 4
            else:
                K = 10
            print(f'K: {K}')

            for i in range(self.conf.n_clients):
                self.partitions.append(np.array([]))

            times=[0 for i in range(K)]
            contain=[]
            for i in range(self.conf.n_clients):
                current=[i%K]
                times[i%K]+=1
                j=1
                while (j < num):
                    ind=random.randint(0,K-1)
                    if (ind not in current):
                        j=j+1
                        current.append(ind)
                        times[ind]+=1
                contain.append(current)
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(self.conf.n_clients)}
            for i in range(K):
                idx_k = np.where(ttar==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,times[i])
                ids=0
                for j in range(self.conf.n_clients):
                    if i in contain[j]:
                        net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                        self.partitions[j] = np.hstack((self.partitions[j], ti[split[ids]]))
                        ids+=1
            for i in range(self.conf.n_clients):
                self.partitions[i] = self.partitions[i].astype(np.int64).tolist()
        elif self.partition_type == 'non_iid_dirichlet':
            for k in range(len(self.partition_sizes)):
                self.partitions.append(indices[k])
        else:
            from_index = 0
            for partition_size in self.partition_sizes:
                to_index = from_index + int(partition_size * self.data_size)
                self.partitions.append(indices[from_index:to_index])
                from_index = to_index


        if self.conf.dataset in ['vlcs', 'pacs', 'off_home']:
            self.record_class = record_class_distribution(
                self.partitions, self.data.dataset.targets
            )
        else:
            #self.record_class记录了每个客户端每个类的数量一个字典{}，键表示客户端id，值是[(第一个类，数量).....(最后一个类，数量)]
            self.record_class = record_class_distribution(
                self.partitions, self.data.targets
            )

    def _create_indices(self, indices):
        if self.partition_type == "origin":
            pass
        elif self.partition_type == "random":
            # it will randomly shuffle the indices.
            self.conf.random_state.shuffle(indices)
        elif self.partition_type == 'evenly' or 'noniid-#label' in self.partition_type:
            if self.conf.dataset in ['vlcs', 'pacs', 'off_home']:
                indices = np.array([
                    (idx, target)
                    for idx, target in enumerate(self.data.dataset.targets)
                    if idx in indices
                ])
            else:
                indices = np.array([
                    (idx, target)
                    for idx, target in enumerate(self.data.targets)
                    if idx in indices
                ])
        elif self.partition_type == "sorted":
            # it will sort the indices based on the data label.
            indices = [
                i[0]
                for i in sorted(
                    [
                        (idx, target)
                        for idx, target in enumerate(self.data.targets)
                        if idx in indices
                    ],
                    key=lambda x: x[1],
                )
            ]
        elif self.partition_type == "non_iid_dirichlet":
            # 平衡的迪利克雷划分
            #类的个数
            num_classes = len(np.unique(self.data.targets))
            #数据长度
            num_indices = len(indices)
            #20个
            n_workers = len(self.partition_sizes)
            # 20个数组，每个数组是每个客户端的索引
            list_of_indices = build_non_iid_by_dirichlet(
                random_state=self.conf.random_state,
                #一个元组
                indices2targets=np.array(
                    [
                        (idx, target)
                        for idx, target in enumerate(self.data.targets)
                        if idx in indices
                    ]
                ),
                non_iid_alpha=self.conf.non_iid_alpha,
                num_classes=num_classes,
                # 数据长度
                num_indices=num_indices,
                n_workers=n_workers,
            )
            indices = list_of_indices
            indices = functools.reduce(lambda a, b: a + b, list_of_indices)

            # #不平衡的迪利克雷划分
            # indices = dirichlet_split_noniid(self.data.targets, self.conf.non_iid_alpha, len(self.partition_sizes))
        else:
            raise NotImplementedError(
                f"The partition scheme={self.partition_type} is not implemented yet"
            )

        return indices

    def _get_consistent_indices(self, indices):
        if dist.is_initialized():
            # sync the indices over clients.
            indices = torch.IntTensor(indices)
            #dist.broadcast(indices, src=0) 进行广播操作。这个操作会将源设备（这里是id为0的设备）上的 indices 张量内容同步到所有参与训练的其他设备
            #确保每个设备都具有相同的索引数据。
            dist.broadcast(indices, src=0)
            return list(indices)
        else:
            return indices

    def use(self, partition_ind):
        return Partition(self.data, self.partitions[partition_ind])

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    unbalanced
    参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
    '''
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

    class_idcs = [np.argwhere(train_labels==y).flatten() 
           for y in range(n_classes)]
    # 记录每个K个类别对应的样本下标
 
    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        # temp = np.cumsum(fracs)[:-1]
        # temp_c = np.cumsum(fracs)[:-1]*len(c)
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
  
    return client_idcs

def build_non_iid_by_dirichlet(
    random_state, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    """
    balanced
    refer to https://github.com/epfml/quasi-global-momentum/blob/3603211501e376d4a25fb2d427c30647065de8c8/code/pcode/datasets/partition_data.py
    """
    n_auxi_workers = 2
    assert n_auxi_workers <= n_workers

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    #10个
    splitted_targets = []
    num_splits = math.ceil(n_workers / n_auxi_workers)
    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]#[2,2....2] 10个2
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    #[0.1 ,,,,,0.1] 10个0.1
    for idx, _ in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index: (num_indices if idx ==
                             num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        #2个客户端的数据标签
        #也是1/10的数据
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        #确保了在每次循环中，生成的小批量数据的最小样本数量不低总数据的1/40的结果。
        #这样做的目的是确保每个工作进程（或客户端）都能够获得足够多的样本数据。
        while min_size < int(0.50 * _targets_size / _n_workers):
            #只要有一个客户端的数量小于1/40（630个）  就重新划分
            _idx_batch = [[] for _ in range(_n_workers)]
            #一个一个类来弄
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                #[0] 表示只取出索引的第一维，以确保返回一个一维数组而不是元组
                #2个idx_class不一样，第一个是数组的下标，第二个是数据的索引
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            #_targets_size / _n_workers一个客户端的数据量，也就是1/20的数据量
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch


#记录每个分区中各类别的样本数量。
def record_class_distribution(partitions, targets):
    targets_of_partitions = {}
    targets_np = np.array(targets)
    for idx, partition in enumerate(partitions):
        unique_elements, counts_elements = np.unique(
            targets_np[partition], return_counts=True
        )
        targets_of_partitions[idx] = list(
            zip(unique_elements, counts_elements))
    
    return targets_of_partitions

#绘制数据分布图
def draw_data(targets_of_partitions, train_samples_num):
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.array([])
    y = np.array([])
    area = np.array([])
    # x = np.arange(1, 21, 1)
    # y = np.arange(0, 11, 1)
    for i in range(60):
        # if i%3 == 2:
        if i%3 == 0:
            # x = np.hstack((x, np.full(len(targets_of_partitions[i]),(i-2) / 3)))
            x = np.hstack((x, np.full(len(targets_of_partitions[i]),i / 3)))
            y = np.hstack((y, np.array(targets_of_partitions[i])[:, 0]))
            area = np.hstack((area, np.array(targets_of_partitions[i])[:, 1]))
    colors = '#1f77b4'
    area = area / 2  # 0 to 15 point radii

    plt.scatter(x, y, s=area, c=colors, alpha=1)
    plt.xticks(range(0,20))
    plt.yticks(range(0,11))
    plt.xlabel('client')
    plt.ylabel('class')
    # 在每一列的最上方标注指定数字k
    for i in range(0, 20):
        plt.text(i, 11, f'{train_samples_num[i]:.0f}', fontsize=6, ha='center', va='center')
    plt.savefig('./data1.png')


#定义验证数据集。
def define_val_dataset(conf, train_dataset):
    # 改为少数据版本就是[0.2, 0.6, 0.2]
    partition_sizes = [
        0.7, 0.1, 0.2
    ]
    data_partitioner = DataPartitioner(
        conf,
        train_dataset,
        partition_sizes,
        partition_type="evenly",
        # consistent_indices=False,
    )
    return data_partitioner

#定义数据加载器。
def define_data_loader(conf, dataset, data_partitioner=None):
    world_size = conf.n_clients
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    if data_partitioner is None:
        # update the data_partitioner.
        data_partitioner = DataPartitioner(
            conf, dataset, partition_sizes, partition_type=conf.partition_data
        )
    return data_partitioner

#定义数据加载器
def define_data_loader_onedomain(conf, dataset, data_partitioner=None):
    world_size = int(conf.n_clients / len(conf.domains))
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    if data_partitioner is None:
        # update the data_partitioner.
        data_partitioner = DataPartitioner(
            conf, dataset, partition_sizes, partition_type="evenly",
        )
    return data_partitioner

#获取数据加载器
def getdataloader(conf, dataall, root_dir='./split/'):
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(root_dir+conf.dataset+str(conf.datapercent), exist_ok=True)
    file = root_dir+conf.dataset+str(conf.datapercent)+'/partion_'+conf.partition_data + \
        '_'+str(conf.non_iid_alpha)+'_'+str(conf.n_clients)+'.npy'
    if not os.path.exists(file):
        # 获得数据分割索引,截止到这里为止，indice索引都是一个一个类别地有序排列
        #define_data_loader会返回一个DataPartitioner类

        #data_part是DataPartitioner类,主要的是partitions记录了每个客户端得到的数据的索引，[[第一个客户端得到的数据索引]....[最后个客户端得到的数据索引]]
        #还有record_class_distribution记录了每个客户端每个类的数量一个字典{}，键表示客户端id，值是[(第一个类，数量).....(最后一个类，数量)]
        data_part = define_data_loader(conf, dataall)
        tmparr = []
        for i in range(conf.n_clients):
            #tmppart也是DataPartitioner类要的是partitions记录了每个客户端得到的数据的索引[[训练集索引]，[验证集],[测试集]]
            #还有record_class_distribution记录了每个客户端训练，验证，测试集每个类的数量
            # 一个字典{}，键表示训练、验证、测试，值是[(第一个类，数量).....(最后一个类，数量)]

            #data_part.use(i)是一个Partition，重要的是data和indices
            tmppart = define_val_dataset(conf, data_part.use(i))
            tmparr.append(tmppart.partitions[0])
            tmparr.append(tmppart.partitions[1])
            tmparr.append(tmppart.partitions[2])
        #tmparr是60个[]，依次是每个客户端的训练，验证，测试
        tmparr = np.array(tmparr, dtype=object)
        np.save(file, tmparr)
    else:
        conf.partition_data_ori = conf.partition_data
        conf.partition_data = 'origin'
        data_part = define_data_loader(conf, dataall)
    data_part.partitions = np.load(file, allow_pickle=True).tolist()

    ######
    if conf.dataset in ['vlcs', 'pacs', 'off_home']:
        record_class = record_class_distribution(
            data_part.partitions, dataall.dataset.targets
        )
    else:
        #一个字典{}，里面有60个[]，每个[]记录每个类的数量
        record_class = record_class_distribution(
            data_part.partitions, dataall.targets
        )

    conf.record_class = record_class
    # 20个[]，每个[]代表一个客户端
    clienttrain_list = []
    clientvalid_list = []
    clienttest_list = []
    for i in range(conf.n_clients):

        clienttrain_list.append(data_part.use(3*i))
        clienttest_list.append(data_part.use(3*i+1))
        clientvalid_list.append(data_part.use(3*i+2))

    train_indices = [clienttrain_list[j].indices for j in range(len(clienttrain_list))]
    train_samples_num = [len(sublist) for sublist in train_indices]


    draw_data(record_class, train_samples_num)

    #记录了每个客户端训练集每个类的数量
    # 一个字典{}，键表示客户端的训练集，值是[(第一个类，数量).....(最后一个类，数量)]
    conf.train_record_class = record_class_distribution(
            train_indices, dataall.targets
    )
    return clienttrain_list, clientvalid_list, clienttest_list


#定义预训练数据集
def define_pretrain_dataset(conf, train_dataset):
    partition_sizes = [
        0.2, 0.8
    ]
    data_partitioner = DataPartitioner(
        conf,
        train_dataset,
        partition_sizes,
        partition_type="evenly",
        # consistent_indices=False,
    )
    return data_partitioner.use(0)