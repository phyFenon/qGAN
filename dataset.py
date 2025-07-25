import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import random

class MaskedCelebA(Dataset):
    def __init__(self, root='./data/celeba/img_align_celeba'):
        """
        自定义 CelebA 图像修复数据集加载类
        参数：
            root: 图像文件夹的路径，默认为 Kaggle 解压后的 img_align_celeba 路径
        """
        # 获取图像路径列表（按文件名排序）
        self.image_paths = sorted([
            os.path.join(root, fname)
            for fname in os.listdir(root)
            if fname.endswith('.jpg')
        ])

        # 图像预处理：统一调整为 128x128 并转为张量格式
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        # 返回图像总数
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        返回一个样本：被遮挡的图像、遮挡掩码、原始图像
        """
        # 加载图像，转为 RGB 模式
        img = Image.open(self.image_paths[idx]).convert('RGB')

        # 执行预处理：调整大小并转为 Tensor
        img = self.transform(img)

        # 初始化遮挡掩码：全部为 1（表示全图像可见）
        mask = torch.ones_like(img)
        mask = mask[0:1]  # or mask = mask.mean(dim=0, keepdim=True)
        #if mask.shape[0]==3:
        #    mask = mask[0:1,:, :]

        # 生成一个随机遮挡区域（正方形）
        h, w = img.shape[1], img.shape[2]
        h_size = random.randint(h // 8, h // 6)  # 遮挡边长在 1/4~1/2 图像尺寸之间
        w_size = random.randint(w // 8, w // 6)  # 遮挡边长在 1/4~1/2 图像尺寸之间
        x = random.randint(0, w - w_size)        # 左上角 x 坐标
        y = random.randint(0, h - h_size)        # 左上角 y 坐标

        # 将遮挡区域的掩码设置为 0（不可见）
        mask[:, y:y+h_size, x:x+w_size] = 0

        # 应用掩码，得到遮挡图像（作为模型输入）
        masked_img = img * mask

        # 返回：遮挡图像、掩码、原图
        return masked_img, mask, img
