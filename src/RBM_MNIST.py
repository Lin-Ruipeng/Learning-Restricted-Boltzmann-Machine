import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ==========================================
# 1. 定义受限玻尔兹曼机 (RBM) 模型
# ==========================================
class RBM(nn.Module):
    def __init__(self, n_vis=784, n_hid=256):
        super(RBM, self).__init__()
        self.n_vis = n_vis
        self.n_hid = n_hid

        # 使用 nn.Parameter 注册参数，否则 state_dict() 无法保存它们
        self.W = nn.Parameter(torch.randn(n_vis, n_hid) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))

    def sample_h(self, v):
        p_h = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h):
        p_v = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        return p_v, torch.bernoulli(p_v)

    def train_step(self, v0, lr=0.01):
        _, h0 = self.sample_h(v0)
        p_v1, v1 = self.sample_v(h0)
        p_h1, _ = self.sample_h(v1)

        N = v0.size(0)

        dW = (torch.matmul(v0.t(), h0) - torch.matmul(v1.t(), p_h1)) / N
        dv_bias = torch.mean(v0 - v1, dim=0)
        dh_bias = torch.mean(h0 - p_h1, dim=0)

        # 手动更新 Parameter 需要在 no_grad 上下文中进行
        with torch.no_grad():
            self.W += lr * dW
            self.v_bias += lr * dv_bias
            self.h_bias += lr * dh_bias

        loss = F.mse_loss(v0, p_v1)
        return loss

    def reconstruct(self, v):
        p_h, _ = self.sample_h(v)
        p_v, _ = self.sample_v(p_h)
        return p_v


# ==========================================
# 2. 数据准备与超参数设置
# ==========================================
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.01
MODEL_PATH = "rbm_mnist.pth"

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = RBM(n_vis=784, n_hid=256)

# ==========================================
# 3. 训练与模型保存/加载逻辑
# ==========================================
if os.path.exists(MODEL_PATH):
    print(f"检测到已存在的模型文件 [{MODEL_PATH}]，跳过训练，直接加载...")
    # strict=True 确保加载的参数和模型结构严格对应
    model.load_state_dict(torch.load(MODEL_PATH), strict=True)
else:
    print("未检测到模型文件，开始训练...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            v0 = data.view(-1, 784)
            loss = model.train_step(v0, lr=LR)
            total_loss += loss.item()

            if batch_idx % 200 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}"
                )

        print(
            f"=== Epoch {epoch + 1} 平均重构误差: {total_loss / len(train_loader):.4f} ===\n"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"训练完成，模型已保存至 [{MODEL_PATH}]")

# ==========================================
# 4. 推理与可视化
# ==========================================
print("\n开始从测试集中抽取图片并进行推理可视化...")

test_images, test_labels = next(iter(test_loader))
random_indices = np.random.choice(BATCH_SIZE, size=3, replace=False)

fig, axes = plt.subplots(2, 3, figsize=(10, 6))
plt.subplots_adjust(hspace=0.4)

model.eval()
with torch.no_grad():
    for i, idx in enumerate(random_indices):
        original_img = test_images[idx].view(-1, 784)
        reconstructed_img = model.reconstruct(original_img)

        orig_np = original_img.squeeze().numpy().reshape(28, 28)
        recon_np = reconstructed_img.squeeze().numpy().reshape(28, 28)
        true_label = test_labels[idx].item()

        axes[0, i].imshow(orig_np, cmap="gray")
        axes[0, i].set_title(f"Original (Label: {true_label})")
        axes[0, i].axis("off")

        axes[1, i].imshow(recon_np, cmap="gray")
        axes[1, i].set_title("RBM Reconstructed")
        axes[1, i].axis("off")

plt.suptitle("RBM Image Reconstruction Test", fontsize=14)
plt.show()
