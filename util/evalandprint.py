import enum
import logging
import numpy as np
import torch

from util.log import loger, init_loggers


# def evalandprint(args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, a_iter, best_changed):
#     # evaluation on training data
#     for client_idx in range(args.n_clients):
#         train_loss, train_acc = algclass.client_eval(
#             client_idx, train_loaders[client_idx])
#         print(
#             f' Site-{client_idx:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')

#     # evaluation on valid data
#     val_acc_list = [None] * args.n_clients
#     for client_idx in range(args.n_clients):
#         val_loss, val_acc = algclass.client_eval(
#             client_idx, val_loaders[client_idx])
#         val_acc_list[client_idx] = val_acc
#         print(
#             f' Site-{client_idx:02d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

#     if np.mean(val_acc_list) > np.mean(best_acc):
#         for client_idx in range(args.n_clients):
#             best_acc[client_idx] = val_acc_list[client_idx]
#             best_epoch = a_iter
#         best_changed = True

#     if best_changed:
#         best_changed = False
#         # test
#         for client_idx in range(args.n_clients):
#             _, test_acc = algclass.client_eval(
#                 client_idx, test_loaders[client_idx])
#             print(
#                 f' Test site-{client_idx:02d} | Epoch:{best_epoch} | Test Acc: {test_acc:.4f}')
#             best_tacc[client_idx] = test_acc
#         print(f' Saving the local and server checkpoint to {SAVE_PATH}')
#         tosave = {'best_epoch': best_epoch, 'best_acc': best_acc, 'best_tacc': np.mean(np.array(best_tacc))}
#         for i,tmodel in enumerate(algclass.client_model):
#             tosave['client_model_'+str(i)]=tmodel.state_dict()
#         tosave['server_model']=algclass.server_model.state_dict()
#         torch.save(tosave, SAVE_PATH)

#     return best_acc, best_tacc, best_changed

def evalandprint(args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc,  a_iter, best_changed):
    init_loggers(args)
    # evaluation on training data
    train_acc_list = [None] * args.n_clients
    train_loss_list = [None] * args.n_clients
    for client_idx in range(args.n_clients):
        train_loss, train_acc = algclass.client_eval(
            client_idx, train_loaders[client_idx])
        train_acc_list[client_idx] = train_acc
        train_loss_list[client_idx] = train_loss
        loger("iteration",
            f' Site-{client_idx:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        # print(
        #     f' Site-{client_idx:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')

    # wandb.log({"train_loss": np.mean(np.array(train_loss_list)), "epoch":a_iter})
    # wandb.log({"train_acc": np.mean(np.array(train_acc_list)), "epoch":a_iter})

    test_acc_list = [None] * args.n_clients
    test_loss_list = [None] * args.n_clients
    for client_idx in range(args.n_clients):
            test_loss, test_acc = algclass.client_eval(
                client_idx, test_loaders[client_idx])
            loger("iteration",
                f' Test site-{client_idx:02d} | Epoch:{a_iter} | Test Acc: {test_acc:.4f}')
            # print(
            #     f' Site-{client_idx:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
            test_acc_list[client_idx] = test_acc
            test_loss_list[client_idx] = test_loss

    # wandb.log({"test_loss": np.mean(np.array(test_loss_list)), "epoch":a_iter})
    # wandb.log({"test_acc": np.mean(np.array(test_acc_list)), "epoch":a_iter})

    #测试集的精准度高于目前
    if np.mean(test_acc_list) > np.mean(best_acc):
        for client_idx in range(args.n_clients):
            best_acc[client_idx] = test_acc_list[client_idx]
            best_epoch = a_iter
        best_changed = True
        
        logging.info(f' Saving the local and server checkpoint to {SAVE_PATH}, best_tacc: {np.mean(np.array(best_acc))}')
        tosave = {'best_epoch': best_epoch, 'best_tacc': np.mean(np.array(best_acc))}
        for i,tmodel in enumerate(algclass.client_model):
            tosave['client_model_'+str(i)]=tmodel.state_dict()
        tosave['server_model']=algclass.server_model.state_dict()
        torch.save(tosave, SAVE_PATH)

    return best_acc, test_acc_list, best_changed


def evalandprint_nosave(args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc,  a_iter, best_changed):
    # evaluation on training data
    train_acc_list = [None] * args.n_clients
    train_loss_list = [None] * args.n_clients
    for client_idx in range(args.n_clients):
        train_loss, train_acc = algclass.client_eval(
            client_idx, train_loaders[client_idx])
        train_acc_list[client_idx] = train_acc
        train_loss_list[client_idx] = train_loss
        logging.info(
            f' Site-{client_idx:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    #
    # wandb.log({"train_loss": np.mean(np.array(train_loss_list)), "epoch":a_iter})
    # wandb.log({"train_acc": np.mean(np.array(train_acc_list)), "epoch":a_iter})

    test_acc_list = [None] * args.n_clients
    test_loss_list = [None] * args.n_clients
    for client_idx in range(args.n_clients):
            test_loss, test_acc = algclass.client_eval(
                client_idx, test_loaders[client_idx])
            logging.info(
                f' Test site-{client_idx:02d} | Epoch:{a_iter} | Test Acc: {test_acc:.4f}')
            test_acc_list[client_idx] = test_acc
            test_loss_list[client_idx] = test_loss

    # wandb.log({"test_loss": np.mean(np.array(test_loss_list)), "epoch":a_iter})
    # wandb.log({"test_acc": np.mean(np.array(test_acc_list)), "epoch":a_iter})

    if np.mean(test_acc_list) > np.mean(best_acc):
        for client_idx in range(args.n_clients):
            best_acc[client_idx] = test_acc_list[client_idx]
            best_epoch = a_iter
        best_changed = True
        
        logging.info(f' Saving the local and server checkpoint to {SAVE_PATH}, best_tacc: {np.mean(np.array(best_acc))}')
    return best_acc, test_loss_list, best_changed