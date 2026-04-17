import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. 自动下载并解压 MovieLens 100k 数据集
# ==========================================
DATA_DIR = "./ml-100k"
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = "ml-100k.zip"

if not os.path.exists(DATA_DIR):
    print("📥 正在下载 MovieLens 100k 数据集...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(".")
    os.remove(ZIP_PATH)
    print("✅ 下载并解压完成！")

# 读取评分数据 (user_id, item_id, rating, timestamp)
cols = ["user_id", "item_id", "rating", "timestamp"]
ratings_df = pd.read_csv(f"{DATA_DIR}/u.data", sep="\t", names=cols)

# 读取电影名称数据 (item_id, title)
movie_titles = {}
with open(f"{DATA_DIR}/u.item", encoding="latin-1") as f:
    for line in f:
        parts = line.strip().split("|")
        movie_titles[int(parts[0])] = parts[1]

# ==========================================
# 1. 数据预处理：构建 Softmax 矩阵
# ==========================================
n_users = ratings_df["user_id"].max()
n_movies = ratings_df["item_id"].max()
K = 5  # 1到5星

print(f"数据维度: {n_users} 个用户, {n_movies} 部电影, {K} 个评分等级")

V = torch.zeros(n_users, n_movies, K)
Mask = torch.zeros(n_users, n_movies)

for row in ratings_df.itertuples():
    u = row.user_id - 1
    m = row.item_id - 1
    r = int(row.rating) - 1
    V[u, m, r] = 1.0
    Mask[u, m] = 1.0


# 截取子集演示
V = V[:200, :500, :]  # V 是三维的 [n_users, n_movies, K]，保留第三个维度
Mask = Mask[:200, :500]  # 【修改这里】Mask 是二维的 [n_users, n_movies]，只切片两个维度

n_users, n_movies = 200, 500
print(f"截取用于快速演示的维度: {n_users} 个用户, {n_movies} 部电影")


# ==========================================
# 2. 定义 Softmax RBM 模型
# ==========================================
class SoftmaxRBM(nn.Module):
    def __init__(self, n_movies, K, n_hidden=100):
        super(SoftmaxRBM, self).__init__()
        self.n_movies = n_movies
        self.K = K
        self.n_hidden = n_hidden

        # 【修复 1】使用 nn.Parameter 注册权重
        self.W = nn.Parameter(torch.randn(n_movies * K, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_movies * K))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_h(self, v_flat, mask_flat):
        # 【修复 2】mask_flat.unsqueeze(-1) 是 3维，repeat 必须传入 3 个参数
        mask_expanded = (
            mask_flat.unsqueeze(-1).repeat(1, 1, self.K).reshape(v_flat.shape)
        )
        v_masked = v_flat * mask_expanded

        p_h = torch.sigmoid(torch.matmul(v_masked, self.W) + self.h_bias)
        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h):
        logits = torch.matmul(h, self.W.t()) + self.v_bias
        logits_reshaped = logits.view(-1, self.n_movies, self.K)
        p_v = F.softmax(logits_reshaped, dim=2)
        return p_v

    def train_step(self, V_batch, Mask_batch, lr=0.005):
        batch_size = V_batch.size(0)

        v0_flat = V_batch.view(batch_size, -1)
        p_h0, h0 = self.sample_h(v0_flat, Mask_batch)

        p_v1 = self.sample_v(h0)
        v1_flat = p_v1.view(batch_size, -1)

        p_h1, _ = self.sample_h(v1_flat, torch.ones_like(Mask_batch))

        mask_expanded = Mask_batch.unsqueeze(-1).repeat(1, 1, self.K)
        v0_masked = (v0_flat.view(-1, self.n_movies, self.K) * mask_expanded).view(
            batch_size, -1
        )
        v1_masked = (p_v1 * mask_expanded).view(batch_size, -1)

        dW = (
            torch.matmul(v0_masked.t(), p_h0) - torch.matmul(v1_masked.t(), p_h1)
        ) / batch_size
        dv_bias = torch.mean(v0_masked - v1_masked, dim=0)
        dh_bias = torch.mean(p_h0 - p_h1, dim=0)

        # 【修复 3】手动更新参数需在 no_grad 上下文
        with torch.no_grad():
            self.W += lr * dW
            self.v_bias += lr * dv_bias
            self.h_bias += lr * dh_bias

        # 【优化】使得 Loss 只在有效打分的电影上做平均，数值更准确
        valid_ratings_count = Mask_batch.sum()
        ce_loss = -torch.sum(
            v0_flat.view(-1, self.n_movies, self.K)
            * mask_expanded
            * torch.log(p_v1 + 1e-6)
        )
        loss = ce_loss / (valid_ratings_count + 1e-8)

        return loss


# ==========================================
# 3. 训练模型
# ==========================================
model = SoftmaxRBM(n_movies, K, n_hidden=64)
EPOCHS = 30

print("\n🚀 开始训练 Softmax RBM (处理缺失值)...")
for epoch in range(EPOCHS):
    loss = model.train_step(V, Mask, lr=0.05)  # 稍微调大了学习率以加快收敛
    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}], Mean Active Cross-Entropy Loss: {loss.item():.4f}"
        )

print("✅ 训练完成！")

# ==========================================
# 4. 课堂演示：填补缺失评分与推荐
# ==========================================
print("\n" + "=" * 50)
print("👨‍🏫 【课堂演示】用户评分填补与电影推荐")
print("=" * 50)

demo_user_id = 0
user_v = V[demo_user_id : demo_user_id + 1]
user_mask = Mask[demo_user_id : demo_user_id + 1]

with torch.no_grad():
    p_v = model.sample_v(model.sample_h(user_v.view(1, -1), user_mask)[0])
    p_v = p_v.squeeze(0)

# 注意：为了让数据格式强匹配，给 arange 指定 dtype
expected_ratings = torch.sum(p_v * torch.arange(1, K + 1, dtype=torch.float32), dim=1)

unwatched_indices = (user_mask.squeeze(0) == 0).nonzero(as_tuple=True)[0].numpy()
recommended_movie_indices = unwatched_indices[
    np.argsort(-expected_ratings[unwatched_indices].numpy())
]

print(f"\n👤 演示用户 ID: {demo_user_id + 1}")
print("-" * 50)

print("🎥 用户【已看过】的电影及真实评分:")
watched_indices = (user_mask.squeeze(0) == 1).nonzero(as_tuple=True)[0].numpy()
for idx in watched_indices[:5]:
    movie_name = movie_titles.get(idx + 1, "未知电影")
    true_rating = torch.argmax(user_v.squeeze(0)[idx]).item() + 1
    # 【修复 4】简化字符串生成，避免替换导致的乱码逻辑
    stars = "⭐" * true_rating
    print(f"   - {movie_name}: {stars} ({true_rating}星)")

print("\n🔮 RBM 填补【未看过】的电影预测评分 (Top 5 推荐):")
for idx in recommended_movie_indices[:5]:
    movie_name = movie_titles.get(idx + 1, "未知电影")
    pred_score = expected_ratings[idx].item()
    stars = "⭐" * int(round(pred_score))
    print(f"   - {movie_name}: 预测 {pred_score:.2f} 星  {stars}")

print("\n💡 原理总结：")
print("RBM 通过隐藏层学习到了 '电影特征' 和 '用户偏好' 的联合分布。")
print("当你把一个不完整的评分向量输入时，网络利用能量最小化原则，")
print("自动把缺失的 Softmax 单元 '拉' 到最符合该用户特征向量的概率分布上！")
