import os
import pickle
import torch
import numpy as np
#用于对图像进行预处理和数据增强操作
import torchvision.transforms as transforms

#用于图像的读取和处理
from PIL import Image
#用于创建图像数据集对象
from torchvision.datasets import ImageFolder, DatasetFolder
#用于图像的默认加载方式
from torchvision.datasets.folder import default_loader

#用于自定义数据集对象
from torch.utils.data import Dataset
#用于加载数据并组织成批次进行训练
from torch.utils.data import DataLoader

#用于数据加载和数据集划分的操作。
from datautil.datasplit import getdataloader
from datautil.datasplit import define_pretrain_dataset, define_data_loader_onedomain



#用于处理图像文件时的异常情况
from PIL import ImageFile
#以允许加载截断的图像文件。
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_data(data_name):
    """Return the algorithm class with the given name."""
    datalist = {'officehome': 'img_union', 'pacs': 'img_union', 'vlcs': 'img_union', 'medmnist': 'medmnist',
                'medmnistA': 'medmnist', 'medmnistC': 'medmnist', 'pamap': 'pamap', 'covid': 'covid', 'cifar10': 'cifar', 'cifar100': 'cifar'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(data_name))
    return globals()[datalist[data_name]]
def gettransforms():
    transform_train = transforms.Compose([
        transforms.Resize([256, 256]), #将图片大小调整为 256x256 像素
        transforms.RandomHorizontalFlip(), #随机水平翻转图片，增加数据的多样性
        transforms.RandomRotation((-30, 30)), #随机旋转图片，在 -30 度到 30 度之间进行随机旋转。
        transforms.ToTensor(), #将图片转换为 Tensor 格式
    ])
    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])
    return transform_train, transform_test
def gettransforms_datatest():
    transform_all = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])
    return transform_all

def gettransforms_datatest_chest():
    transform_all = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    return transform_all


#作用是返回适用于 CIFAR 数据集的训练集和测试集的数据转换操作
def getcifartransforms():
    # train_transformer = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #         ])
    train_transformer = transforms.Compose([
            transforms.ToTensor(),
             #对图片进行标准化处理。这里使用了 CIFAR 数据集的均值和标准差进行标准化，这样可以使得数据的均值接近 0、标准差接近 1，有利于模型的训练。
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))])
        
    test_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))])

    return train_transformer, test_transformer

class mydataset(object):
    def __init__(self, args):
        self.x = None  #数据样本
        self.targets = None  #目标标签
        self.dataset = None
        self.transform = None
        self.target_transform = None
        self.loader = None      #数据加载器
        self.args = args
        self.idx = False   #表示是否返回样本的索引

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        x = self.input_trans(self.loader(self.x[index]))
        ctarget = self.target_trans(self.targets[index])
        if self.idx:
            return (x, ctarget, index)
        else:
            return x, ctarget

    def __len__(self):
        return len(self.targets)


class ImageDataset(mydataset):
    def __init__(self, args, dataset, root_dir, domain_name):
        super(ImageDataset, self).__init__(args)
        #通过 .imgs 属性获取了图像数据集的路径和标签信息
        self.imgs = ImageFolder(root_dir+domain_name).imgs
        self.domain_num = 0
        self.dataset = dataset
        imgs = [item[0] for item in self.imgs]
        labels = [item[1] for item in self.imgs]
        self.targets = np.array(labels)
        transform, _ = gettransforms()
        target_transform = None
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        self.pathx = imgs
        self.x = self.pathx

from torch.utils.data import Dataset
class MedMnistDataset(Dataset):
    def __init__(self, filename='', transform=None, percent=None):
        self.data = np.load(filename+'xdata.npy')
        self.targets = np.load(filename+'ydata.npy')
        self.targets = np.squeeze(self.targets)
        self.transform = transform
        self.is_idx = False
        if percent:
            self.data = self.data[:int(len(self.data)*percent)]
            self.targets = self.targets[:int(len(self.targets)*percent)]
            print(f'Clip with {percent}, to {int(len(self.data))}')
            self.is_idx = True
        self.data = torch.Tensor(self.data)
        #在第二维度加了一个尺寸 x(25221, 28, 28)->(25221,1,28,28)
        self.data = torch.unsqueeze(self.data, dim=1)

    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        if self.is_idx:
            return (self.data[idx], self.targets[idx], idx)
        else:
            return self.data[idx], self.targets[idx]


class PamapDataset(Dataset):
    def __init__(self, filename='../data/pamap/', transform=None):
        self.data = np.load(filename+'x.npy')
        self.targets = np.load(filename+'y.npy')
        self.select_class()
        self.transform = transform
        self.data = torch.unsqueeze(torch.Tensor(self.data), dim=1)
        self.data = torch.einsum('bxyz->bzxy', self.data)  #函数对数据维度进行调整

    def select_class(self):
        xiaochuclass = [0, 5, 12]
        index = []
        for ic in xiaochuclass:
            index.append(np.where(self.targets == ic)[0])
        index = np.hstack(index)
        allindex = np.arange(len(self.targets))
        allindex = np.delete(allindex, index)
        self.targets = self.targets[allindex]
        self.data = self.data[allindex]
        ry = np.unique(self.targets)
        ry2 = {}
        for i in range(len(ry)):
            ry2[ry[i]] = i
        for i in range(len(self.targets)):
            self.targets[i] = ry2[self.targets[i]]

    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class CovidDataset(Dataset):
    def __init__(self, filename='../data/covid19/', transform=None, percent=None):
        self.data = np.load(filename+'xdata.npy')
        self.targets = np.load(filename+'ydata.npy')
        self.targets = np.squeeze(self.targets)
        self.transform = transform
        self.is_idx = False
        if percent:
            self.data = self.data[:int(len(self.data)*percent)]
            self.targets = self.targets[:int(len(self.targets)*percent)]
            self.is_idx = True
        self.data = torch.Tensor(self.data)
        #torch.einsum() 函数来对 PyTorch 的 Tensor 数据进行维度调整
        self.data = torch.einsum('bxyz->bzxy', self.data )

    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        if self.is_idx:
            #返回3*224*224的图片
            return (self.data[idx], self.targets[idx], idx)
        else:
            return self.data[idx], self.targets[idx]

class CovidDistillDataset(Dataset):
    def __init__(self, filename='../data/covid19/', transform=None, percent=None):
        self.data = np.load(filename+'xdata.npy')
        self.targets = np.ones((self.data.shape[0],1))
        self.transform = transform
        self.is_idx = False
        if percent:
            self.data = self.data[:int(len(self.data)*percent)]
            self.targets = self.targets[:int(len(self.targets)*percent)]
            print(f'Clip with {percent}, to {int(len(self.data)*percent)}')
            self.is_idx = True
        self.data = torch.Tensor(self.data)
        #数据集的样本格式是[batchsize, h, w, ch]，我们需要将其转换为[batchsize, ch，h，w]
        self.data = torch.einsum('bxyz->bzxy', self.data)

    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        if self.transform is not None:
            #Image.fromarray(self.data[idx])将获取的样本数据转换为 PIL 图像对象
            #数据转换操作
            data = self.transform(Image.fromarray(self.data[idx]))
        else:
            data = self.data[idx]
        if self.is_idx:
            return (data, self.targets[idx], idx)
        else:
            return data, self.targets[idx]

class Cifar_Dataset:
    def __init__(self, args, local_dir, data_type, with_coarse_label=False, distill=False, public_percent=1):
        self.distill = distill
        self.transform = None
        data, targets = [], []
        if data_type == 'cifar10':
            local_dir = os.path.join(args.root_dir, 'cifar-10-batches-py')
            for i in range(1, 6):
                file_name = None
                file_name = os.path.join(local_dir , 'data_batch_{0}'.format(i))
                X_tmp, y_tmp = (None, None)
                with open(file_name, 'rb') as (fo):
                    #CIFAR-10 数据集以字节编码的形式存储
                    #特征数据通常被存储为一个形状为 (10000, 3072) 的数组，每行表示一张图像的像素值，标签则是一个长度为 10000 的一维数组。
                    datadict = pickle.load(fo, encoding='bytes')
                X_tmp = datadict[b'data']
                y_tmp = datadict[b'labels']
                X_tmp = X_tmp.reshape(10000, 3, 32, 32)
                y_tmp = np.array(y_tmp)
                data.append(X_tmp)
                targets.append(y_tmp)
            file_name = None
            file_name = os.path.join(local_dir , 'test_batch')
            with open(file_name, 'rb') as (fo):
                datadict = pickle.load(fo, encoding='bytes')
                X_tmp = datadict[b'data']
                y_tmp = datadict[b'labels']
                X_tmp = X_tmp.reshape(10000, 3, 32, 32)
                y_tmp = np.array(y_tmp)
                data.append(X_tmp)
                targets.append(y_tmp)
            #每个数据批次的特征数据都是一个形状为 (10000, 3, 32, 32) 的四维数组，代表了一批图像数据
            #到一个形状为 (50000, 3, 32, 32) 的数组
            data = np.vstack(data)
            #所有标签数据水平拼接成一个一维数组
            targets = np.hstack(targets)
        elif data_type == 'cifar100':
            local_dir = os.path.join(args.root_dir, 'cifar-100-python')
            file_name = None
            file_name = os.path.join(local_dir , 'train')
            with open(file_name, 'rb') as (fo):
                datadict = pickle.load(fo, encoding='bytes')
                img = datadict[b'data']
                if with_coarse_label:
                    gt = datadict[b'coarse_labels']
                else:
                    gt = datadict[b'fine_labels']
                X_tmp = img.reshape(50000, 3, 32, 32)
                y_tmp = np.array(gt)
                data.append(X_tmp)
                targets.append(y_tmp)
            file_name = None
            file_name = os.path.join(local_dir , 'test')
            with open(file_name, 'rb') as (fo):
                datadict = pickle.load(fo, encoding='bytes')
                img = datadict[b'data']
                #100 个细标签分为 20 个粗标签，每个粗标签包含 5 个细标签。
                if with_coarse_label:
                    gt = datadict[b'coarse_labels']
                else:
                    gt = datadict[b'fine_labels']
                # import ipdb; ipdb.set_trace()
                X_tmp = img.reshape(10000, 3, 32, 32)
                y_tmp = np.array(gt)
                data.append(X_tmp)
                targets.append(y_tmp)
            data = np.vstack(data)
            targets = np.hstack(targets)
        self.data = np.asarray(data)
        self.data = self.data.transpose((0, 2, 3, 1))
        self.targets = np.asarray(targets)
        total_N_img = data.shape[0]
        # import ipdb; ipdb.set_trace()
        if public_percent<1:
            total_N_img = int(total_N_img*public_percent)
            self.data = self.data[:total_N_img]
            self.targets = self.targets[:total_N_img]
            print(f'Clip with {public_percent}, to {total_N_img}')
        self.fixid = np.arange(total_N_img)
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        fixid = self.fixid[idx]
        transimage = Image.fromarray(data)
        
        if self.transform is not None:
            transformer = self.transform 
        transimage = transformer(transimage)
        if self.distill:
            return (transimage, target, idx)
        else:
            return transimage, target

# https://github.com/QinbinLi/MOON/blob/6c7a4ed1b1a8c0724fa2976292a667a828e3ff5d/datasets.py#L148
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, data_type=None, train=True, transform=None, target_transform=None, public_percent=None, distill=False):
        self.root = root + '/' + data_type + '/train'
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.distill = distill
        #标准的 PyTorch 图像数据集加载器
        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if public_percent:
            #在 PyTorch 中，ImageFolder 类的 samples 属性保存了图像数据集中每个样本的路径和对应的标签，以元组形式存储在列表中。
            self.samples = np.array(imagefolder_obj.samples)
            self.samples = self.samples[:int(len(self.samples)*public_percent)]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.distill:
            return (sample, target, index)
        else:
            return sample, target 

    def __len__(self):
        return int(len(self.samples))



def getfeadataloader(args):
    trd, vad, ted = [], [], []
    partition_sizes = [
        0.7, 0.1, 0.2
    ]
    for item in args.domains:
        trl, val, tel = [], [], []
        trl_temp, val_temp, tel_temp = [], [], []
        data = ImageDataset(args, args.dataset,
                            args.root_dir+args.dataset+'/', item)
        l = len(data)
        index = np.arange(l)
        np.random.seed(args.seed)
        np.random.shuffle(index)
        l1, l2, l3 = int(l*partition_sizes[0]), int(l *
                                                  partition_sizes[1]), int(l*partition_sizes[2])
        trl_temp.append(torch.utils.data.Subset(data, index[:l1]))
        val_temp.append(torch.utils.data.Subset(data, index[l1:l1+l2]))
        tel_temp.append(torch.utils.data.Subset(data, index[l1+l2:l1+l2+l3]))
        _, target_transform = gettransforms()
        val_temp[-1].transform = target_transform
        tel_temp[-1].transform = target_transform
        trl_part = define_data_loader_onedomain(args, trl_temp[-1])
        val_part = define_data_loader_onedomain(args, val_temp[-1])
        tel_part = define_data_loader_onedomain(args, tel_temp[-1])
        for i in range(len(trl_part.partitions)):
            trl.append(trl_part.use(i))
            val.append(val_part.use(i))
            tel.append(tel_part.use(i))
        for j in range(len(trl)):
            trd.append(torch.utils.data.DataLoader(
                trl[j], batch_size=args.batch, shuffle=True))
            vad.append(torch.utils.data.DataLoader(
                val[j], batch_size=args.batch, shuffle=False))
            ted.append(torch.utils.data.DataLoader(
                tel[j], batch_size=args.batch, shuffle=False))
    return trd, vad, ted, trl


def img_union(args):
    return getfeadataloader(args)


def getlabeldataloader(args, data):
    # 每个客户端占有5%的数据量
    # 这5%的数据量再分别按照70%、10%、20%的比例划分为训练集、验证集、测试集，每个客户端大概有504张图片
    trl, val, tel = getdataloader(args, data)

    if args.dataset in ['cifar10', 'cifar100']:
        train_transform , _ = getcifartransforms()
    trd, vad, ted = [], [], []
    if args.dataset in ['cifar10', 'cifar100']:
        trl[0].data.transform = train_transform

    for i in range(len(trl)):
        # drop_last=True
        trd.append(torch.utils.data.DataLoader(
            #shuffle=True这意味着在每个epoch（完整遍历数据集一次）开始时，数据加载器会对数据集中的样本进行随机重排
            trl[i], batch_size=args.batch, shuffle=True))
        # train_list.append(len(trl[i].indices))
        vad.append(torch.utils.data.DataLoader(
            val[i], batch_size=args.batch, shuffle=False))
        ted.append(torch.utils.data.DataLoader(
            tel[i], batch_size=args.batch, shuffle=False))
    return trd, vad, ted, trl


def medmnist(args):
    # 获取数据集
    data = MedMnistDataset(args.root_dir+args.dataset+'/')
    trd, vad, ted, train_num_list= getlabeldataloader(args, data)
    args.num_classes = 11
    return trd, vad, ted, train_num_list


def pamap(args):
    data = PamapDataset(args.root_dir+'pamap/')
    trd, vad, ted = getlabeldataloader(args, data)
    args.num_classes = 10
    return trd, vad, ted


def covid(args):
    #CovidDataset是一个类
    data = CovidDataset(args.root_dir+'covid19/')

    trd, vad, ted, train_num_list = getlabeldataloader(args, data)

    args.num_classes = 4
    return trd, vad, ted, train_num_list

def cifar(args):
    data = Cifar_Dataset(args, args.root_dir, data_type=args.dataset)
    trd, vad, ted, train_num_list = getlabeldataloader(args, data) 
    if args.dataset == 'cifar10':
        args.num_classes = 10
    if args.dataset == 'cifar100':
        args.num_classes = 100
    return trd, vad, ted, train_num_list

class combinedataset(mydataset):
    def __init__(self, datal, args, percent=None):
        super(combinedataset, self).__init__(args)

        self.x = np.hstack([np.array(item.x) for item in datal])
        self.targets = np.hstack([item.targets for item in datal])
        if percent:
            l = self.x.shape[0]
            index = np.arange(l)
            np.random.seed(args.seed)
            np.random.shuffle(index)
            cuted_index = index[:int(l*percent)]
            self.x = self.x[cuted_index]
            self.targets = self.targets[cuted_index]
            self.idx = True
        s = ''
        for item in datal:
            s += item.dataset+'-'
        s = s[:-1]
        self.dataset = s
        self.transform = datal[0].transform
        self.target_transform = datal[0].target_transform
        self.loader = datal[0].loader


def getwholedataset(args, percent=None):
    datal = []
    for item in args.domains:
        datal.append(ImageDataset(args, args.dataset,
                     args.root_dir+args.dataset+'/', item))
    # data=torch.utils.data.ConcatDataset(datal)
    data = combinedataset(datal, args, percent)
    return data


def img_union_w(args, percent=None):
    
    return getwholedataset(args, percent)


def medmnist_w(args):
    data = MedMnistDataset(args.root_dir+args.dataset+'/')
    args.num_classes = 11
    return data


def pamap_w(args):
    data = PamapDataset(args.root_dir+'pamap/')
    args.num_classes = 10
    return data


def covid_w(args):
    data = CovidDataset(args.root_dir+'covid19/')
    args.num_classes = 4
    return data

def cifar_w(args):
    data = Cifar_Dataset(args, args.root_dir, data_type=args.dataset)
    train_transform , _ = getcifartransforms()
    data.transform = train_transform
    args.num_classes = 10
    return data

def get_whole_dataset(data_name):
    datalist = {'officehome': 'img_union_w', 'pacs': 'img_union_w', 'vlcs': 'img_union_w', 'medmnist': 'medmnist_w',
                'medmnistA': 'medmnist_w', 'medmnistC': 'medmnist_w', 'pamap': 'pamap_w', 'covid': 'covid_w', 'cifar10': 'cifar_w'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(data_name))
    return globals()[datalist[data_name]]

def get_distill_date(args):
    if args.distillation_dataset in ['medmnistC', 'medmnist', 'medmnistA']:
        public_dataset = MedMnistDataset(filename=args.root_dir+args.distillation_dataset+'/', percent=args.dis_datapercent)
    elif args.distillation_dataset in ['chest']:
        public_dataset = CovidDistillDataset(filename=args.root_dir+args.distillation_dataset+'/', percent=args.dis_datapercent)
    elif args.distillation_dataset in ['vlcs']:
        temp_dataset = args.dataset
        temp_domains = args.domains
        args.dataset= args.distillation_dataset
        args.domains = ['Caltech101']
        public_dataset = img_union_w(args, percent=0.1)
        # public_dataset = define_pretrain_dataset(args, data)
        distill_loader = DataLoader(
            dataset=public_dataset, batch_size=args.batch, shuffle=False, 
            num_workers=8, pin_memory=True, sampler=None)
        args.dataset= temp_dataset
        args.domains = temp_domains
    elif args.distillation_dataset in ['cifar100','cifar10']:
        public_dataset = Cifar_Dataset(args, args.root_dir, data_type=args.distillation_dataset, distill = True, public_percent=args.dis_datapercent)
        train_transform , _ = getcifartransforms()
        public_dataset.transform = train_transform
        distill_loader = DataLoader(
            dataset=public_dataset, batch_size=args.batch, shuffle=False, 
            num_workers=8, pin_memory=True, sampler=None)
    elif args.distillation_dataset == 'tinyimagenet':
        public_dataset = ImageFolder_custom(args.root_dir, data_type=args.distillation_dataset, distill = True, public_percent=0.01)
        # train_transform , _ = getcifartransforms()
        # public_dataset.transform = train_transform
        transform = gettransforms_datatest()
        public_dataset.transform = transform
        distill_loader = DataLoader(
            dataset=public_dataset, batch_size=args.batch, shuffle=False, 
            num_workers=8, pin_memory=True, sampler=None)
    else:
        public_dataset = None
    return public_dataset
