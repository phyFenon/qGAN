import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import MaskedCelebA  # dataset.py 和本文件在同一目录下
import matplotlib
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #避免多个 OpenMP 库冲突
# 设置中文字体（根据操作系统选择字体）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体（SimHei）
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号


# 加载数据集
batch_size = 2
dataset = MaskedCelebA(root='./data/celeba/img_align_celeba')
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 获取一批样本
masked_imgs, masks, imgs = next(iter(loader))

# 创建子图 (batch_size 行 × 3 列)
fig, axs = plt.subplots(batch_size, 3, figsize=(12, 3 * batch_size))

# 如果 batch_size == 1，axs 是 1D 结构，需转为 2D 以统一处理
if batch_size == 1:
    axs = axs.reshape(1, 3)

for i in range(batch_size):
    axs[i, 0].imshow(imgs[i].permute(1, 2, 0))
    axs[i, 0].set_title("原始图像")
    axs[i, 0].axis('off')

    axs[i, 1].imshow(masks[i].permute(1, 2, 0))
    axs[i, 1].set_title("遮挡掩码")
    axs[i, 1].axis('off')

    axs[i, 2].imshow(masked_imgs[i].permute(1, 2, 0))
    axs[i, 2].set_title("遮挡后的图像")
    axs[i, 2].axis('off')

plt.tight_layout()
plt.show()
