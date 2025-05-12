# chest   x(10000, 224, 224, 3)  #就是covid的公共数据       拿30个
# cifar-10   32*32
# cifar-100    32*32                         拿30个
# covid19      x(9208,3,244,224)    y(9208)
# medmnist    x(25221, 1,28, 28)     y(25221,)
# medmnistC   x(23660, 1,28, 28)     y(23660,)  拿30个

import random
import numpy as np
import matplotlib.pyplot as plt
from util.draw import heatmap, annotate_heatmap
weight_m=np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        if i == j:
            weight_m[i, j] = 1
        else:
            weight_m[i, j] = random.random()
fig, ax = plt.subplots()
vegetables = ["client1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
farmers = ["client1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
im, cbar = heatmap(weight_m, vegetables, farmers, ax=ax,
                    cmap="Blues", cbarlabel="Distribution Similarity")
texts = annotate_heatmap(im, valfmt="{x:.2f}")

fig.tight_layout()
plt.show()
# img = args.distillation_dataset + '_sim.svg'
# plt.savefig(img)
############################################################
