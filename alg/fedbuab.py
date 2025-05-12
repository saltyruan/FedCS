import copy
from alg.fedavg import fedavg
import torch.optim as optim
from util.traineval import train

class fedbuab(fedavg):
    def __init__(self,args):
        super(fedbuab, self).__init__(args)
        self.optimizers_fine_tuning = []
        for idx in range(args.n_clients):
            body_params = [p for name, p in self.client_model[idx].named_parameters() if 'linear' not in name]
            head_params = [p for name, p in self.client_model[idx].named_parameters() if 'linear' in name]
            
            self.optimizers[idx] = optim.SGD([{'params': body_params, 'lr': args.lr},
                                        {'params': head_params, 'lr': 0}])
            self.optimizers_fine_tuning.append(optim.SGD([{'params': body_params, 'lr': 0},
                                        {'params': head_params, 'lr': args.lr}]))
    

    def client_train(self, c_idx, dataloader, round):
        train_loss, train_acc = train(
            self.client_model[c_idx], dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        return train_loss, train_acc

    
        
