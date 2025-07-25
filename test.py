import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from dataset import MaskedCelebA
from GatedUNet_model import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #避免多个 OpenMP 库冲突
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体（以 SimHei 为例，黑体）
matplotlib.rcParams['font.family'] = 'SimHei'

# 设置负号正常显示
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置设备与加载模型
device = torch.device("cpu" if torch.cuda.is_available() else "cuda")
model = GatedUNet().to(device)
model.load_state_dict(torch.load("./result/GatedUnet/epoch_199.pth", map_location=device))
#model.load_state_dict(torch.load("./result/WGan2_epoch_19.pth", map_location=device))
model.eval()

# 加载完整数据集并划分测试集
full_dataset = MaskedCelebA(root='./data/celeba/img_align_celeba')
total_size = len(full_dataset)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size
train_set, _, test_set = random_split(full_dataset, [train_size, val_size, test_size])
test_loader = DataLoader(train_set, batch_size=1, shuffle=True)

# 获取一批测试图像
masked_imgs, masks, gts = next(iter(test_loader))
masked_imgs = masked_imgs.to(device)
masks = masks.to(device)

# 网络推理得到修复图像
with torch.no_grad():
    input = torch.cat([masked_imgs, masks], dim=1)
    preds = model(input)
    # 修复融合：遮挡区域来自预测，非遮挡区域保持原图
    inpainted = preds * masks + masked_imgs * (1 - masks)

# 可视化
masked_imgs = masked_imgs.cpu()
outputs = preds.cpu()
gts = gts.cpu()

batch_size = masked_imgs.shape[0]
fig, axs = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))

if batch_size == 1:
    axs = axs.reshape(1, 3)

for i in range(batch_size):
    axs[i, 0].imshow(masked_imgs[i].permute(1, 2, 0).clamp(0, 1))
    axs[i, 0].set_title("遮挡图像")
    axs[i, 0].axis("off")

    axs[i, 1].imshow(outputs[i].permute(1, 2, 0).clamp(0, 1))
    axs[i, 1].set_title("修复结果")
    axs[i, 1].axis("off")

    axs[i, 2].imshow(gts[i].permute(1, 2, 0).clamp(0, 1))
    axs[i, 2].set_title("原始图像")
    axs[i, 2].axis("off")

plt.tight_layout()
plt.savefig("test_results.png")
plt.show()
