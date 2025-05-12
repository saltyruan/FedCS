import torchvision.datasets as dataset
import torchvision.transforms as transform
import numpy as np
import split_noniid

np.random.seed(42)


# 生成行和列和均为1的二维数组
def random_sum(row, col, row_sum=1.0, col_sum=1.0):
    print(1)


def get_dataset(dir, name,dirchilet_alpha):
    if name == 'fmnist':
        # 获取训练集和测试集
        train_dataset = dataset.FashionMNIST(dir, train=True, download=True, transform=transform.ToTensor())  # 设置下载数据集并转
        # 换为torch识别的tensor数据类型
        eval_dataset = dataset.FashionMNIST(dir, train=False, transform=transform.ToTensor())  # 测试集
        train_labels = np.array(train_dataset.targets)
        client_idcs = split_noniid.dirichlet_split_noniid(train_labels, alpha=0.1,n_clients=20)
    elif name == 'cifar10':
        transform_train = transform.Compose([  # 数据增强操作，训练集的预处理
            transform.RandomCrop(32, padding=4),  # 随机剪裁，大小为32*32，添加4个像素的填充内容
            transform.RandomHorizontalFlip(),  # 随机垂直方向的翻转
            transform.ToTensor(),
            transform.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 归一化操作，数值是抽样得到的，无需考虑太
            # 多，分别是均值和标准差
        ])
        transform_test = transform.Compose([  # 对测试集进行预处理
            transform.ToTensor(),
            transform.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = dataset.CIFAR10(dir, train=True, download=True, transform=transform_train)  # 获得训练集
        eval_dataset = dataset.CIFAR10(dir, train=False, transform=transform_test)  # 获得测试集

        # noniid划分
        train_labels = np.array(train_dataset.targets)
        client_idcs = split_noniid.dirichlet_split_noniid(train_labels, alpha=dirchilet_alpha,n_clients=20)
        #client_idcs = split_noniid.pathological_non_iid_split(train_dataset,10, 20, 8)

        # print(len(client_idcs))
        # for i in range(len(client_idcs)):
        #     print(len(client_idcs[i]))
    elif name == 'cifar100':
        transform_train = transform.Compose([  # 数据增强操作，训练集的预处理
                transform.RandomCrop(32, padding=4),
                transform.RandomHorizontalFlip(),
                transform.ToTensor(),
                transform.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            # 归一化操作，数值是抽样得到的，无需考虑太
            # 多，分别是均值和标准差
        ])
        transform_test = transform.Compose([  # 对测试集进行预处理
            transform.ToTensor(),
            transform.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])
        train_dataset = dataset.CIFAR100(dir, train=True, download=True, transform=transform_train)  # 获得训练集
        eval_dataset = dataset.CIFAR100(dir, train=False, transform=transform_test)  # 获得测试集

        # noniid划分
        train_labels = np.array(train_dataset.targets)
        # client_idcs = split_noniid.dirichlet_split_noniid(train_labels, alpha=0.5,n_clients=20)
        client_idcs = split_noniid.pathological_non_iid_split(train_dataset,100, 20, 8)
    # 确保train_dataset已定义后再返回
    else:
        print(f"目前未配置数据集{name},请检查是否写错")
        raise Exception(f"目前未配置数据集{name},请检查是否写错")

    return train_dataset, eval_dataset, client_idcs

if __name__ == '__main__':
    # NonIID
    train_dataset, eval_dataset, client_idcs = get_dataset('../data/', "cifar10",0.1)