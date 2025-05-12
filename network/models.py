import torch
import torch.nn as nn
import torch.nn.functional as func
from collections import OrderedDict
import torchvision.models as models

class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        # Load the pretrained ResNet-18 model
        self.resnet18 = models.resnet18(pretrained=True)
        
        # Extract layers as per your requirements
        self.layer1 = nn.Sequential(
            self.resnet18.conv1,
            self.resnet18.bn1,
            self.resnet18.layer1[0]
        )
        self.layer2 = nn.Sequential(
            self.resnet18.layer1[1]
        )
        self.layer3 = nn.Sequential(
            self.resnet18.layer2[0]
        )
        self.layer4 = nn.Sequential(
            self.resnet18.layer2[1]
        )
        self.layer5 = nn.Sequential(
            self.resnet18.layer3[0],
        )
        self.layer6 = nn.Sequential(
            self.resnet18.layer3[1]
        )
        self.layer7 = nn.Sequential(
            self.resnet18.layer4[0]
        )
        self.layer8 = nn.Sequential(
            self.resnet18.layer4[1]
        )
        self.fc = self.resnet18.fc

    def forward(self, x):
        # Forward pass through the defined layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create an instance of CustomResNet18
custom_resnet18 = CustomResNet18()

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features1 = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.features2 = nn.Sequential(
            OrderedDict([
                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.features3 = nn.Sequential(
            OrderedDict([
                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),
            ])
        )
        self.features4 = nn.Sequential(
            OrderedDict([
                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),
            ])
        )
        self.features5 = nn.Sequential(
            OrderedDict([
                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        #输出特征图的目标尺寸为6x6
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.bn6 = nn.BatchNorm1d(4096)
        self.relu6 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(4096, 4096)
        self.bn7 = nn.BatchNorm1d(4096)
        self.relu7 = nn.ReLU(inplace=True)

        self.linear3 = nn.Linear(4096, num_classes)


    def forward(self, x, feat=False, is_mlb=False, level=0):
        if not is_mlb:
            x = self.features1(x)
            x = self.features2(x)
            x = self.features3(x)
            x = self.features4(x)
            x = self.features5(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.relu6(self.bn6(self.fc1(x)))
            x = self.relu7(self.bn7(self.fc2(x)))
            x = self.linear3(x)
            return x
        else:
            if level <= 0:
                out0 = self.features1(x)
            else:
                out0 = x
            if level <= 1:
                out1 = self.features2(out0)
            else:
                out1 = out0
            if level <= 2:
                out2 = self.features3(out1)
            else:
                out2 = out1
            if level <= 3:
                out3 = self.features4(out2)
            else:
                out3 = out2
            if level <= 4:
                out4 = self.features5(out3)
                out4 = self.avgpool(out4)
                out4 = torch.flatten(out4, 1)
            else:
                out4 = out3
            if level <= 5:
                out5 = self.relu6(self.bn6(self.fc1(out4)))
                out5 = self.relu7(self.bn7(self.fc2(out5)))
            else:
                out5 = out4
            logit = self.linear3(out5)

            if feat:
                return out0, out1, out2, out3, out4, out5, logit
            else:
                return logit
        

class AlexNet_AP(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet_AP, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 4096)),
                ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(4096, 4096)),
                ('bn7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),

                ('fc3', nn.Linear(4096, num_classes)),
            ])
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def getallfea(self, x):
        fealist = []
        for i in range(len(self.features)):
            if i in [1, 5, 9, 12, 15]:
                fealist.append(x.clone().detach())
            x = self.features[i](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for i in range(len(self.classifier)):
            if i in [1, 4]:
                fealist.append(x.clone().detach())
            x = self.classifier[i](x)
        return fealist

    def getfinalfea(self, x):
        for i in range(len(self.features)):
            x = self.features[i](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for i in range(len(self.classifier)):
            if i == 6:
                return [x]
            x = self.classifier[i](x)
        return x

    def get_sel_fea(self, x, plan=0):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if plan == 0:
            y = x
        elif plan == 1:
            y = self.classifier[5](self.classifier[4](self.classifier[3](
                self.classifier[2](self.classifier[1](self.classifier[0](x))))))
        else:
            y = []
            y.append(x)
            x = self.classifier[2](self.classifier[1](self.classifier[0](x)))
            y.append(x)
            x = self.classifier[5](self.classifier[4](self.classifier[3](x)))
            y.append(x)
            y = torch.cat(y, dim=1)
        return y



class PamapModel(nn.Module):
    def __init__(self, n_feature=64, out_dim=10):
        super(PamapModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=27, out_channels=16, kernel_size=(1, 9))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(1, 9))
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.fc1 = nn.Linear(in_features=32*44, out_features=n_feature)
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=n_feature, out_features=out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(self.relu1(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool2(self.relu2(self.bn2(x)))
        x = x.reshape(-1, 32 * 44)
        feature = self.fc1_relu(self.fc1(x))
        out = self.fc2(feature)
        return out

    def getallfea(self, x):
        fealist = []
        x = self.conv1(x)
        fealist.append(x.clone().detach())
        x = self.pool1(self.relu1(self.bn1(x)))
        x = self.conv2(x)
        fealist.append(x.clone().detach())
        return fealist

    def getfinalfea(self, x):
        x = self.conv1(x)
        x = self.pool1(self.relu1(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool2(self.relu2(self.bn2(x)))
        x = x.reshape(-1, 32 * 44)
        feature = self.fc1_relu(self.fc1(x))
        return [feature]

    def get_sel_fea(self, x, plan=0):
        if plan == 0:
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.reshape(-1, 32 * 44)
            fealist = x
        elif plan == 1:
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.reshape(-1, 32 * 44)
            feature = self.fc1_relu(self.fc1(x))
            fealist = feature
        else:
            fealist = []
            x = self.conv1(x)
            x = self.pool1(self.relu1(self.bn1(x)))
            fealist.append(x.view(x.shape[0], -1))
            x = self.conv2(x)
            x = self.pool2(self.relu2(self.bn2(x)))
            fealist.append(x.view(x.shape[0], -1))
            x = x.reshape(-1, 32 * 44)
            feature = self.fc1_relu(self.fc1(x))
            fealist.append(feature)
            fealist = torch.cat(fealist, dim=1)
        return fealist


class lenet5v(nn.Module):
    def __init__(self, num_classes=11):
        super(lenet5v, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        # self.fc1 = nn.Linear(400, 120)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x, feat=False, is_mlb=False, level=0, proto=False):
        if not is_mlb:
            y = self.conv1(x)
            y = self.bn1(y)
            y = self.relu1(y)
            out0 = self.pool1(y)
            y = self.conv2(out0)
            y = self.bn2(y)
            y = self.relu2(y)
            y = self.pool2(y)
            out1 = y.view(y.shape[0], -1)
            y = self.fc1(out1)
            y = self.relu3(y)
            y = self.fc2(y)
            y = self.relu4(y)
            y = self.fc(y)
            return y
        else:
            if level <= 0:
                out0 = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            else:
                out0 = x
            if level <= 1:
                out1 = self.pool2(self.relu2(self.bn2(self.conv2(out0))))
                out1 = out1.view(out1.shape[0], -1)
            else:
                out1 = out0
            if level <= 2:
                out2 = self.relu3(self.fc1(out1))
                out2 = self.relu4(self.fc2(out2))
            else:
                out2 = out1
            logit = self.fc(out2)

            if feat:
                return out0, out1, out2, logit
            if proto:
                return out1, logit
            else:
                return logit


    def getallfea(self, x):
        fealist = []
        y = self.conv1(x)
        fealist.append(y.clone().detach())
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        fealist.append(y.clone().detach())
        return fealist

    def getfinalfea(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        return [y]

    def get_sel_fea(self, x, plan=0):
        if plan == 0:
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.view(x.shape[0], -1)
            fealist = x
        elif plan == 1:
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.view(x.shape[0], -1)
            x = self.relu3(self.fc1(x))
            x = self.relu4(self.fc2(x))
            fealist = x
        else:
            fealist = []
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.view(x.shape[0], -1)
            fealist.append(x)
            x = self.relu3(self.fc1(x))
            fealist.append(x)
            x = self.relu4(self.fc2(x))
            fealist.append(x)
            fealist = torch.cat(fealist, dim=1)
        return fealist


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.featlist = []

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    # def forward(self, x):
    #     out = func.relu(self.bn1(self.conv1(x)))
    #     out = self.bn2(self.conv2(out))
    #     out += self.shortcut(x)
    #     out = func.relu(out)
    #     return out
    def forward(self, x):
        self.featlist = []
        out = self.conv1(x)
        self.featlist.append(out.clone().detach())
        out = func.relu(self.bn1(out))
        out = self.conv2(out)
        self.featlist.append(out.clone().detach()) 
        out = self.bn2(out)
        out += self.shortcut(x)
        out = func.relu(out)
        return out, self.featlist

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.in_planes = 128

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.layer1 = self._make_layer(block,  128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2)
        self.linear1 = nn.Linear(2048, num_classes) #512*exp*block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, feat=False, is_mlb=False, level=0):
        if not is_mlb:
            pout = func.relu(self.bn1(self.conv1(x)))
            pout, _ = self.layer1(pout) #b*128*32*32
            pout, _ = self.layer2(pout)#b*256*16*16
            pout, _ = self.layer3(pout) #b*512*8*8
            out = func.avg_pool2d(pout, 4)
            out = out.view(out.size(0), -1)
            out = self.linear1(out)
            if feat:
                return pout, out
            else:
                return out
        else:
            if level <= 0:
                out0 = func.relu(self.bn1(self.conv1(x)))
            else:
                out0 = x
            if level <= 1:
                out1, _ = self.layer1(out0)
            else:
                out1 = out0
            if level <= 2:
                out2, _ = self.layer2(out1)
            else:
                out2 = out1
            if level <= 3:
                out3, _ = self.layer3(out2)
                out3 = func.avg_pool2d(out3, 4)
                out3 = out3.view(out3.size(0), -1)
            else:
                out3 = out2
    
            logit = self.linear1(out3)

            if feat == True:
                return out0, out1, out2, out3, logit
            else:
                return logit
        
    def getallfea(self, x):
        featlist = []
        y = self.conv1(x)
        featlist.append(y.clone().detach())
        y = func.relu(self.bn1(y))
        y, fealist_temp = self.layer1(y) #b*128*32*32
        for i in fealist_temp:
            featlist.append(i)
        y, fealist_temp = self.layer2(y)#b*256*16*16
        for i in fealist_temp:
            featlist.append(i)
        y, fealist_temp = self.layer3(y) #b*512*8*8
        for i in fealist_temp:
            featlist.append(i)
        return featlist
    
def ResNet8(num_classes=10):
    return ResNet(BasicBlock, [1,1,1], num_classes=num_classes) #2048

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes) #2048
