import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datautil.datasplit import define_pretrain_dataset
from datautil.prepare_data import get_whole_dataset
from alg.SAM import SAM


def train(model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    batch_logits = []
    for data, target in data_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        #前向传播
        output = model(data)
        loss = loss_fun(output, target)
        #item获得数值而不是计算图
        loss_all = loss_all + loss.item()
        total = total + target.size(0)
        pred = output.data.max(1)[1]
        correct = correct + pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # # first forward-backward pass
        # loss = loss_fun(output, target)  # use this loss for any training statistics
        # loss.backward(retain_graph=True)
        # optimizer.first_step(zero_grad=True)
        #
        # # second forward-backward pass
        # loss_fun(model(data), target).backward()  # make sure to do a full forward pass
        # optimizer.second_step(zero_grad=True)

    return loss_all / len(data_loader), correct / total


def train_proto(model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    batch_logits = []
    for data, target in data_loader:
        def closure():
            loss = loss_fun(model(data), target)
            loss.backward()
            return loss

        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        batch_logits.append(output.detach())  ####
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step(closure)
    batch_logits = torch.cat(batch_logits).mean(dim=0).cpu()  ####
    return loss_all / len(data_loader), correct / total, batch_logits  ####


def test(model, data_loader, loss_fun, device):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            output = model(data)
            loss = loss_fun(output, target)
            loss_all += loss.item()
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
        return loss_all / len(data_loader), correct / total


def train_prox(args, model, server_model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for step, (data, target) in enumerate(data_loader):
        def closure():
            loss = loss_fun(model(data), target)
            loss.backward()
            return loss

        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        loss = loss_fun(output, target)
        if step > 0:
            #如果这不是第一个批次的数据，那么计算客户端模型和服务器模型之间的差异。
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)

            w_diff = torch.sqrt(w_diff)
            loss += args.mu / 2. * w_diff

        optimizer.zero_grad()
        loss.backward()
        optimizer.step(closure)

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct / total


def train_amp(args, model, server_model, data_loader, optimizer, loss_fun, device):
    alphaK = 0.1
    mu = 0.01
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for step, (data, target) in enumerate(data_loader):
        def closure():
            loss=loss_fun(target,model(data))
            loss.backward()
            return loss
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        loss = loss_fun(output, target)
        #如果这不是第一个批次的数据，那么计算模型和服务器模型之间的差异。
        if step > 0:
            w_diff = torch.tensor(0., device=device)
            sub = weight_flatten(model) - weight_flatten(server_model)
            w_diff = torch.dot(sub, sub)

            loss += mu / alphaK / 2. * w_diff

        optimizer.zero_grad()
        loss.backward()
        optimizer.step(closure)

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct / total


def train_mlb(args, model, server_model, data_loader, optimizer, loss_fun, device):
    lambda1 = 1
    lambda2 = 1
    lambda3 = 1
    model.train()
    server_model.eval()
    loss_all = 0
    total = 0
    correct = 0
    for step, (data, target) in enumerate(data_loader):
        def closure():
            loss=loss_fun(model(data),target)
            loss.backward()
            return loss
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data, feat=True, is_mlb=True)
        local_features = output[:-1]
        log_probs = output[-1]
        log_prob_branch = []
        ce_branch = []
        kl_branch = []
        num_branch = len(local_features)
        ## Compute loss from hybrid branches
        for it in range(num_branch):
            this_log_prob = server_model(local_features[it], is_mlb=True, level=it + 1)
            this_ce = loss_fun(this_log_prob, target)
            this_kl = KD(this_log_prob, log_probs)
            log_prob_branch.append(this_log_prob)
            ce_branch.append(this_ce)
            kl_branch.append(this_kl)

        ce_loss = loss_fun(log_probs, target)
        loss = lambda1 * ce_loss + lambda2 * (
            sum(ce_branch)) / num_branch + lambda3 * (sum(kl_branch)) / num_branch

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step(closure)

        loss_all += loss.item()
        total += target.size(0)
        pred = log_probs.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct / total


def trainwithteacher(model, data_loader, optimizer, loss_fun, device, tmodel, lam, args, flag):
    model.train()
    if tmodel:
        tmodel.eval()
        if not flag:
            with torch.no_grad():
                for key in tmodel.state_dict().keys():
                    if 'num_batches_tracked' in key:
                        pass
                    elif args.nosharebn and 'bn' in key:
                        pass
                    else:
                        model.state_dict()[key].data.copy_(
                            tmodel.state_dict()[key])
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        def closure():
            loss=loss_fun(model(data),target)
            loss.backward()
            return loss

        optimizer.zero_grad()

        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        f1 = model.get_sel_fea(data, args.plan)
        loss = loss_fun(output, target)
        if flag and tmodel:
            f2 = tmodel.get_sel_fea(data, args.plan).detach()
            loss += (lam * F.mse_loss(f1, f2))
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step(loss)

    return loss_all / len(data_loader), correct / total


def pretrain_model(args, model, filename, device='cuda'):
    print('===training pretrained model===')
    data = get_whole_dataset(args.pre_dataset)(args)
    predata = define_pretrain_dataset(args, data)
    traindata = torch.utils.data.DataLoader(
        predata, batch_size=args.batch, num_workers=8, shuffle=True)
    loss_fun = nn.CrossEntropyLoss()

    base_optimizer = optim.SGD
    opt = SAM(model.parameters(), base_optimizer)
    best_acc = 0
    for _ in range(args.pretrained_iters):
        _, acc = train(model, traindata, opt, loss_fun, device)
        if best_acc < acc:
            torch.save({
                'state': model.state_dict(),
                'acc': acc
            }, filename)
            print('acc: {}'.format(acc))
    print('===done!===')


#KL散度该函数通常在机器学习和深度学习中用于计算两个分布之间的差异
#例如在知识蒸馏中用于将教师模型的分布与学生模型的分布进行比较。
def KD(input_p, input_q, T=1):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    p = F.softmax(input_p / T, dim=1)
    q = F.log_softmax(input_q / T, dim=1)
    result = kl_loss(q, p)
    return result


#将模型的所有参数展平为一维向量
def weight_flatten(model):
    params = []
    for u in model.parameters():
        params.append(u.view(-1))
    params = torch.cat(params)
    return params
