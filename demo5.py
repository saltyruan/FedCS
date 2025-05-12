# # import logging
# #
# # # 基础配置（直接写入文件）
# # logging.basicConfig(
# #     filename="app.log",  # 日志文件名
# #     level=logging.DEBUG,  # 设置最低日志级别（DEBUG及以上都会记录）
# #     format="%(asctime)s - %(levelname)s - %(message)s" , # 日志格式
# #     encoding = 'utf-8'
# # )
# #
# # # 测试日志
# #
# # logging.debug("这是一条调试信息")
# # logging.info("这是一条普通信息")
# # logging.warning("这是一条警告信息")
# # logging.error("这是一条错误信息")
# import logging
#
# # 配置第一个Logger（模块A）
# logger_a = logging.getLogger('module_a')
# logger_a.setLevel(logging.INFO)
# handler_a = logging.FileHandler(filename='module_a.log',encoding='utf-8')
# formatter_a = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
# handler_a.setFormatter(formatter_a)
# logger_a.addHandler(handler_a)
#
# # 配置第二个Logger（模块B）
# logger_b = logging.getLogger('module_b')
# logger_b.setLevel(logging.DEBUG)
# handler_b = logging.FileHandler(filename='module_b.log',encoding='utf-8')
# formatter_b = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler_b.setFormatter(formatter_b)
# logger_b.addHandler(handler_b)
#
# # 测试日志
# logger_a.info('模块A的日志信息')
# logger_b.debug('模块B的调试信息')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# 设置中文字体，确保中文能正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 定义beta分布的参数
alpha = 2
beta_param = 5  # 为避免与scipy.stats.beta冲突，变量名使用beta_param

# 生成x轴数据，范围从0到1
x = np.linspace(0, 1, 1000)

# 计算beta分布的概率密度函数值
y = beta.pdf(x, alpha, beta_param)

# 绘制beta分布曲线
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)

# 填充曲线下方区域
plt.fill_between(x, y, alpha=0.2, color='blue')

# 添加标题和标签
plt.title(f'Beta分布 (α={alpha}, β={beta_param})')
plt.xlabel('x')
plt.ylabel('概率密度函数值')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图形
plt.tight_layout()
plt.show()