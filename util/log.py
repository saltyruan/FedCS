# import logging
#
# # 基础配置（直接写入文件）
# logging.basicConfig(
#     filename="app.log",  # 日志文件名
#     level=logging.DEBUG,  # 设置最低日志级别（DEBUG及以上都会记录）
#     format="%(asctime)s - %(levelname)s - %(message)s" , # 日志格式
#     encoding = 'utf-8'
# )
#
# # 测试日志
#
# logging.debug("这是一条调试信息")
# logging.info("这是一条普通信息")
# logging.warning("这是一条警告信息")
# logging.error("这是一条错误信息")
# import logging
# import os
# from datetime import date
#
# def loger(args,log_type,message,base_path = "../log/"):
#
#     today = date.today()
#     folder_path = f"{today.year}-{today.month}-{today.day}"
#     exp_folder = (f'fed_selectall_{args.dataset}_{args.alg}_{args.non_iid_alpha}_'
#                   f'{args.dis_datapercent}_{args.partition_data}_{args.transfer}')
#     path = base_path + folder_path + exp_folder
#     if not os.path.exists(path):
#         os.makedirs(path)
#
#     logger_a = logging.getLogger("a")
#     logger_a.setLevel(logging.INFO)
#     handler_a = logging.FileHandler(filename=path+'/'+'acc.log', encoding='utf-8')
#     logger_a.addHandler(handler_a)
#
#     # 配置第二个Logger（模块B）
#     logger_b = logging.getLogger('b')
#     logger_b.setLevel(logging.INFO)
#     handler_b = logging.FileHandler(filename=path+'/'+'iteration.log', encoding='utf-8')
#     logger_b.addHandler(handler_b)
#
#     logger_c = logging.getLogger('c')
#     logger_c.setLevel(logging.INFO)
#     handler_c = logging.FileHandler(filename=path+'/'+'others.log', encoding='utf-8')
#     logger_c.addHandler(handler_c)
#     if log_type == 'tacc':
#         logger_a.info(message)
#     if log_type == 'iteration':
#         logger_b.info(message)
#     if log_type == 'error':
#         logger_b.debug(message)

# if __name__ == '__main__':
#
#     log_type = 'tacc'
#     loger(log_type,"你好")
import logging
import os
from datetime import date


# 全局初始化 Logger（避免重复创建）
def init_loggers(args, base_path="../log/"):
    today = date.today()
    folder_path = f"{today.year}-{today.month}-{today.day}"
    exp_folder = (f'fed_selectall_{args.dataset}_{args.alg}_{args.non_iid_alpha}_'
                  f'{args.dis_datapercent}_{args.partition_data}_{args.transfer}_warm{args.warm_up}')
    path = os.path.join(base_path, folder_path, exp_folder)

    if not os.path.exists(path):
        os.makedirs(path)

    # 初始化 Logger A（acc.log）
    logger_a = logging.getLogger("a")
    if not logger_a.handlers:  # 避免重复添加 Handler
        logger_a.setLevel(logging.INFO)
        handler_a = logging.FileHandler(filename=os.path.join(path, 'acc.log'), encoding='utf-8')
        logger_a.addHandler(handler_a)

    # 初始化 Logger B（iteration.log）
    logger_b = logging.getLogger('b')
    if not logger_b.handlers:
        logger_b.setLevel(logging.INFO)
        handler_b = logging.FileHandler(filename=os.path.join(path, 'iteration.log'), encoding='utf-8')
        logger_b.addHandler(handler_b)

    # 初始化 Logger C（others.log）
    logger_c = logging.getLogger('c')
    if not logger_c.handlers:
        logger_c.setLevel(logging.INFO)
        handler_c = logging.FileHandler(filename=os.path.join(path, 'others.log'), encoding='utf-8')
        logger_c.addHandler(handler_c)


# 日志记录函数（直接使用全局 Logger）
def loger(log_type, message):
    if log_type=='tacc':
        logging.getLogger("a").info(message)
    elif log_type=='iteration':
        logging.getLogger("b").info(message)
    elif log_type=='error':
        logging.getLogger("c").debug(message)