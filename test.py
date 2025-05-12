# # # from PIL import Image
# # #
# # # def remove_white_pixels(input_path, output_path, threshold=240):
# # #     """
# # #     去除图像中的白色像素点（转换为透明）
# # #     参数：
# # #     input_path: 输入图像路径
# # #     output_path: 输出图像路径（建议保存为PNG格式）
# # #     threshold: 白色判断阈值（0-255），默认240以上视为白色
# # #     """
# # #     # 打开图像并转换为RGBA模式
# # #     img = Image.open(input_path).convert("RGBA")
# # #     datas = img.getdata()
# # #
# # #     new_data = []
# # #     for item in datas:
# # #         # 判断是否为白色像素（RGB通道都大于阈值）
# # #         if item[0] > threshold and item[1] > threshold and item[2] > threshold:
# # #             # 设置为完全透明
# # #             new_data.append((255, 255, 255, 0))
# # #         else:
# # #             # 保留原始像素
# # #             new_data.append(item)
# # #
# # #     # 更新图像数据并保存
# # #     img.putdata(new_data)
# # #     img.save(output_path ,"PNG")
# # #     print(f"处理完成，结果已保存至 ")
# # # if __name__ == "__main__":
# # # # 使用示例
# # #     print(18467/5)
# # #     #remove_white_pixels("0.png", "output0.png")
# #
# #
# # import numpy as np
# #
# # # 加载 .npy 文件
# # data = np.load('./data/medmnist/ydata.npy')
# #
# # # 获取基本数据信息
# # print("数据类型:", data.dtype)
# # print("数组形状:", data.shape)
# # print("总元素数量:", data.size)
# #
# #
# # # 如果是图像数据，获取样本数和单张图像的维度
# # if len(data.shape) >= 3:  # 假设至少有三个维度（样本数、高度、宽度）
# #     num_samples = data.shape[0]
# #     print("样本数量:", num_samples)
# #     if num_samples > 0:
# #         image_shape = data[0].shape
# #         print("单张图像的维度:", image_shape)
# # else:
# #     # 提取所有唯一的标签
# #     labels = np.unique(data)
# #
# #     # 打印结果
# #     print("总共有 {} 种标签，分别是：{}".format(len(labels), labels))
#
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.colors import ListedColormap
#
# # 定义分组数据
# groups = {
#     0: [0, 4, 6, 8, 10, 12, 16],
#     1: [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19],
#     2: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
#     3: [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 18],
#     4: [0, 1, 2, 4, 5, 7, 8, 10, 12, 13, 16],
#     5: [1, 2, 3, 5, 7, 10, 11, 12, 13, 14, 15, 17, 19],
#     6: [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19],
#     7: [1, 2, 3, 4, 5, 6, 7, 10, 11, 13, 14, 15, 16, 18],
#     8: [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 18],
#     9: [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 14, 16],
#     10: [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18],
#     11: [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
#     12: [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18],
#     13: [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18],
#     14: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19],
#     15: [0, 3, 4, 5, 6, 8, 10, 11, 13, 14, 15, 16],
#     16: [1, 3, 11, 13, 16],
#     17: [1, 2, 3, 5, 13, 17, 18],
#     18: [1, 2, 3, 4, 9, 10, 11, 12, 13, 18],
#     19: [1, 3, 12, 17, 19]
# }
#
# # 创建一个20x20的矩阵，初始值为-1（表示未选中）
# matrix = np.full((20, 20), -1)
#
# # 填充矩阵，每个组的成员在对应的行中被标记为组号
# for group, members in groups.items():
#     for member in members:
#         matrix[group, member] = group
#
# # 创建自定义颜色映射
# colors = ['white'] + [plt.cm.tab20(i) for i in range(20)]
# cmap = ListedColormap(colors)
#
# # 绘制热图
# plt.figure(figsize=(12, 10))
# plt.imshow(matrix, cmap=cmap, vmin=-1, vmax=19)
#
# # 添加网格线
# plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)
# plt.xticks(np.arange(-0.5, 20, 1)), plt.yticks(np.arange(-0.5, 20, 1))
# plt.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
#
# # 添加标题和标签
# plt.title('Client Group Assignment Visualization', fontsize=14)
# plt.xlabel('Client ID', fontsize=12)
# plt.ylabel('Group ID', fontsize=12)
#
# # 添加图例
# patches = [plt.Rectangle((0,0),1,1, color=colors[i+1]) for i in range(20)]
# plt.legend(patches, [f'Group {i}' for i in range(20)], bbox_to_anchor=(1.05, 1), loc='upper left')
#
# plt.tight_layout()
# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import beta
#
# # 定义Beta分布的概率密度函数（直接调用scipy.stats.beta）
# def beta_pdf(x, alpha, beta_param):
#     """
#     计算Beta分布在x处的概率密度值
#     参数:
#         x: 输入值（0到1之间的标量或数组）
#         alpha: Beta分布的参数α（必须 > 0）
#         beta_param: Beta分布的参数β（必须 > 0）
#     返回:
#         Beta分布在x处的概率密度值
#     """
#     return beta.pdf(x, alpha, beta_param)
#
# # 生成x轴数据（0到1之间的均匀间隔点）
# x = np.linspace(0, 1, 1000)
#
# # 设置不同参数组合，对比Beta分布形状
# params = [
#     (0.7, 0.7),  # U型分布（中间低，两边高）
#     # (0.5, 0.5),
#     # # (0.4, 0.4),
#     # (0.3, 0.3),
#     # # (0.7, 0.7),
#     # # (0.2, 0.8),
#     # (0.4, 0.6)
#
#     # (2, 5),       # 右偏分布
#     # (5, 2),       # 左偏分布
#     # (1, 1)        # 均匀分布
# ]
#
# # 绘制不同参数的Beta分布曲线
# plt.figure(figsize=(10, 6))
# for a, b in params:
#     y = beta_pdf(x, a, b)
#     plt.plot(x, y, label=f'α={a}, β={b}')
#
# # 设置图表标题和标签
# plt.title('Beta Distribution with Different Parameters')
# plt.xlabel('x')
# plt.ylabel('Probability Density')
# plt.legend()
# plt.grid(True)
# plt.show()


from PIL import Image
import os
import argparse


def remove_white_pixels(image):
    """将白色像素转为透明"""
    img = image.convert("RGBA")
    datas = img.getdata()

    new_data = []
    for item in datas:
        # 判断是否为纯白像素（RGB值均为255）
        if item[0]==255 and item[1]==255 and item[2]==255:
            # 将Alpha通道设为0（完全透明）
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)

    img.putdata(new_data)
    return img


def process_images(input_dir="data", output_dir="output"):
    # 创建输出目录
    print(f"正在创建输出目录：{output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("目录创建成功！")
    else:
        print("目录已存在。")

    # 遍历所有PNG文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(output_path)

            try:
                with Image.open(input_path) as img:
                    processed_img = remove_white_pixels(img)
                    processed_img.save(output_path, "PNG")
                    print(f"保存成功 → {output_path}")
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data1", help="输入目录")
    parser.add_argument("--output", default="data0", help="输出目录")
    args = parser.parse_args()

    process_images(args.input, args.output)