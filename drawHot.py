import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
data = np.random.rand(10,10)

print(data)
# 绘制热力图
plt.imshow(data, cmap='Blues', interpolation='nearest')

# 添加颜色条
plt.colorbar()

# 设置坐标轴标签
plt.xlabel('Party ID')
plt.ylabel('Class ID')

# 显示图形
plt.show()


