import torch
import numpy as np
def search(all_weights,all_choose_k,random_index_k,start_index,now_path,res,server,cal_num):#利用DFS搜索最优的客户端组合
    if(start_index>=len(random_index_k)):
        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)
        #计算client的中心_1 求和
        for i in range(len(all_weights)):
            if(len(all_weights[i]))!=0:#如果被选中过
                temp = all_weights[i][now_path[i]]
                for name, params in temp.items():
                    weight_accumulator[name] += params
        #计算client的中心_2 除数量
        for name, params in weight_accumulator.items():
            weight_accumulator[name]  = weight_accumulator[name]/cal_num
        #计算client中心到各client的距离
        temp_res = 0
        for i in range(len(all_weights)):
            if(len(all_weights[i]))!=0:#如果被选中过
                temp = all_weights[i][now_path[i]]
                for name, params in temp.items():
                    temp_res += np.linalg.norm((weight_accumulator[name]-params).cpu())#tensor转numpy
        if(temp_res<res):
            print(all_choose_k)
            for i in range(len(now_path)):
                all_choose_k[i] = now_path[i]
            #all_choose_k = copy.copy(now_path) #注意这里不能这样写
            res = temp_res
        return
    for every in range(len(all_weights[random_index_k[start_index]])):
        now_path[random_index_k[start_index]] = every
        search(all_weights,all_choose_k,random_index_k,start_index+1,now_path,res,server,cal_num)


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

