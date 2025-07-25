import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from dataset import MaskedCelebA
from GatedUNet_model import GatedUNet,Discriminator
import time 
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #避免多个 OpenMP 库冲突
import matplotlib.pyplot as plt
import matplotlib
import torchvision.utils as vutils

# 设置中文字体（以 SimHei 为例，黑体）
matplotlib.rcParams['font.family'] = 'SimHei'
# 设置负号正常显示
matplotlib.rcParams['axes.unicode_minus'] = False


# 设置超参数
# =====================
# 1. 超参数优化
# =====================
batch_size = 64  # 增大batch size提升稳定性
epochs = 500      # 延长训练周期
lr_G = 2e-4      # 生成器学习率
lr_D = 1e-4      # 判别器学习率更低
lambda_recon = 200  # 增强重建损失权重
gp_weight = 10    # 梯度惩罚系数
D_update_freq = 2 #判别器更新频率


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device为{device}")


# =====================
# 3. WGAN损失函数设计
# =====================
def compute_gradient_penalty(D, real, fake):
    alpha = torch.rand(real.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty



# 加载数据集并划分 train / val / test
full_dataset = MaskedCelebA(root='./data/celeba/img_align_celeba')
print("图像遮掩步骤完成！")
total_size = len(full_dataset)
print(f"图像集的样本数量为:{total_size}")

train_size = int(0.7 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size
train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# 初始化模型
G = GatedUNet().to(device)
D = Discriminator().to(device)

# 使用Wasserstein损失
recon_loss = nn.L1Loss()
opt_G = optim.SGD(G.parameters(), lr=lr_G, momentum=0.1)
opt_D = optim.SGD(D.parameters(), lr=lr_D, momentum= 0.1)
scheduler_G = CosineAnnealingLR(opt_G, T_max=epochs)
scheduler_D = CosineAnnealingLR(opt_D, T_max=epochs)

# 初始化损失记录列表
G_losses = []
D_losses = []
Val_losses = []


# 在超参数设置后创建保存目录
os.makedirs('fig_gen2', exist_ok=True)  # 创建图片保存目录
# =====================
# 4. 改进训练循环
# =====================
for epoch in range(epochs):
    epoch_g_loss = 0
    epoch_d_loss = 0
    start_time =time.time()
    for i, (masked_img, mask, real_img) in enumerate(train_loader):
        real_img = real_img.to(device) * 2 - 1
        masked_img = masked_img.to(device) * 2 - 1
        mask = mask.to(device)

        # 训练判别器
        if (i+1) % D_update_freq == 0:
            opt_D.zero_grad()
            with torch.no_grad():
                gen_img = G(torch.cat([masked_img, mask], 1))
            d_real = D(real_img)
            d_fake = D(gen_img)
            gp = compute_gradient_penalty(D, real_img, gen_img)
            d_loss = -torch.mean(d_real) + torch.mean(d_fake) + gp_weight * gp
            d_loss.backward()
            opt_D.step()
            epoch_d_loss += d_loss.item()

        # 训练生成器
        opt_G.zero_grad()
        gen_img = G(torch.cat([masked_img, mask], 1))
        g_adv = -torch.mean(D(gen_img))
        g_recon = recon_loss(gen_img, real_img)
        g_loss = g_adv + lambda_recon * g_recon
        g_loss.backward()
        opt_G.step()
        epoch_g_loss += g_loss.item()

    # 更新学习率
    scheduler_G.step()
    scheduler_D.step()

    # 记录损失
    G_losses.append(epoch_g_loss / len(train_loader))
    D_losses.append(epoch_d_loss / (len(train_loader) // D_update_freq)) # 修正判别器损失的计算
    print(f"[Epoch {epoch}/{epochs}] "
      f"G Loss: {G_losses[-1]:.4f} | "
      f"D Loss: {D_losses[-1]:.4f} | "
      f"Time: {time.time() - start_time:.2f}s")

    # 保存模型和验证
    if (epoch+1) % 10 == 0:
        torch.save(G.state_dict(), f'./result/GatedUnet/epoch_{epoch}.pth')


         # 生成并保存图片
        G.eval()
        with torch.no_grad():
            # 取验证集第一个batch
            sample_batch = next(iter(val_loader))
            masked_val = (sample_batch[0].to(device) * 2 - 1)
            mask_val = sample_batch[1].to(device)
            real_val = (sample_batch[2].to(device) * 2 - 1)
            
            # 生成图像
            gen_val = G(torch.cat([masked_val, mask_val], 1))
            
            # 准备可视化数据（反归一化到[0,1]）
            display_masked = (masked_val + 1) / 2
            display_real = (real_val + 1) / 2
            display_gen = (gen_val + 1) / 2
            
            # 创建对比图（原始 | 遮掩 | 生成）
            comparison = torch.cat([display_real, display_masked, display_gen], dim=3)
            
            # 保存为网格图
            save_path = f'./fig_gen/epoch_{epoch}_comparison.png'
            vutils.save_image(comparison[:1],  # 只保存前8个样本
                            save_path,
                            nrow=1,  # 每行显示一个样本的三张图
                            padding=10,
                            normalize=False)
        
        G.train()

        with torch.no_grad():
            val_loss = 0
            for masked, mask, real in val_loader:
                masked = masked.to(device)
                mask = mask.to(device)
                real = real.to(device)
                gen = G(torch.cat([masked, mask], 1))
                val_loss += recon_loss(gen, real).item()
        val_loss /= len(val_loader)
        Val_losses.append(val_loss)
        print(f"Epoch {epoch}: Val L1 Loss={val_loss:.4f}")

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(G_losses, label='Generator Loss')
plt.plot(D_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('WGAN-GP Training Loss')
plt.legend()
plt.grid(True)
plt.savefig('./result/GatedUnet/smallpatch_loss_curve.png')
plt.show()