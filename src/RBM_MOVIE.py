import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
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

# 初始化张量
# V: [n_users, n_movies, K]，用于存储 One-Hot 后的评分
V = torch.zeros(n_users, n_movies, K)
# Mask: [n_users, n_movies]，标记哪些位置是已知评分(1)，哪些是缺失(0)
Mask = torch.zeros(n_users, n_movies)

# 填充矩阵
for row in ratings_df.itertuples():
    u = row.user_id - 1  # 转为 0 索引
    m = row.item_id - 1
    r = int(row.rating) - 1  # 转为 0 索引 (0代表1星，4代表5星)
    V[u, m, r] = 1.0
    Mask[u, m] = 1.0

# 为了演示速度，我们只取前 200 个用户和前 500 部电影进行训练
V = V[:200, :500, :]
Mask = Mask[:200, :500, :]
n_users, n_movies = 200, 500
print(f"截取用于快速演示的维度: {n_users} 个用户, {n_movies} 部电影")


# ==========================================
# 2. 定义 Softmax RBM 模型
# ==========================================
class SoftmaxRBM(torch.nn.Module):
    def __init__(self, n_movies, K, n_hidden=100):
        super(SoftmaxRBM, self).__init__()
        self.n_movies = n_movies
        self.K = K
        self.n_hidden = n_hidden

        # W 维度: [n_movies * K, n_hidden]
        # 为什么这样设计？为了能用矩阵乘法一次性算出所有电影的隐藏层概率
        self.W = torch.randn(n_movies * K, n_hidden) * 0.01
        # v_bias 维度: [n_movies * K]
        self.v_bias = torch.zeros(n_movies * K)
        # h_bias 维度: [n_hidden]
        self.h_bias = torch.zeros(n_hidden)

    def sample_h(self, v_flat, mask_flat):
        """
        v_flat: [batch, n_movies * K]
        mask_flat: [batch, n_movies] -> 扩展为 [batch, n_movies * K]
        """
        # 将 mask 扩展到 K 维度，乘以 v，这样缺失的评分(全0)不会影响隐藏层
        mask_expanded = mask_flat.unsqueeze(-1).repeat(1, self.K).reshape(v_flat.shape)
        v_masked = v_flat * mask_expanded

        # [batch, n_movies*K] @ [n_movies*K, n_hidden] -> [batch, n_hidden]
        p_h = torch.sigmoid(torch.matmul(v_masked, self.W) + self.h_bias)
        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h):
        """
        h: [batch, n_hidden]
        """
        # [batch, n_hidden] @ [n_hidden, n_movies*K] -> [batch, n_movies*K]
        logits = torch.matmul(h, self.W.t()) + self.v_bias
        # 重塑为 [batch, n_movies, K] 以便在 K 维度上做 Softmax
        logits_reshaped = logits.view(-1, self.n_movies, self.K)

        # 核心：Softmax！将每个电影小组的能量转化为概率分布
        p_v = F.softmax(logits_reshaped, dim=2)
        return p_v

    def train_step(self, V_batch, Mask_batch, lr=0.005):
        batch_size = V_batch.size(0)

        # 1. 展平输入
        v0_flat = V_batch.view(batch_size, -1)  # [batch, M*K]

        # 2. 正相位 v0 -> h0
        p_h0, h0 = self.sample_h(v0_flat, Mask_batch)

        # 3. 负相位 h0 -> v1 (得到的是概率分布，无需再抛硬币)
        p_v1 = self.sample_v(h0)  # [batch, M, K]
        v1_flat = p_v1.view(batch_size, -1)

        # 4. 负相位 v1 -> h1 (注意：这里的v1没有缺失值，所以不需要mask)
        p_h1, _ = self.sample_h(v1_flat, torch.ones_like(Mask_batch))

        # 5. 计算梯度 (必须在减去之前，先用 mask 把缺失数据的贡献清零)
        mask_expanded = Mask_batch.unsqueeze(-1).repeat(1, 1, self.K)
        v0_masked = (v0_flat.view(-1, self.n_movies, self.K) * mask_expanded).view(
            batch_size, -1
        )
        v1_masked = (p_v1 * mask_expanded).view(batch_size, -1)

        # 梯度公式：正相位 - 负相位
        dW = (
            torch.matmul(v0_masked.t(), p_h0) - torch.matmul(v1_masked.t(), p_h1)
        ) / batch_size
        dv_bias = torch.mean(v0_masked - v1_masked, dim=0)
        dh_bias = torch.mean(p_h0 - p_h1, dim=0)

        # 6. 手动更新参数
        self.W += lr * dW
        self.v_bias += lr * dv_bias
        self.h_bias += lr * dh_bias

        # 返回重构误差（仅计算已知评分位置的交叉熵）
        loss = -torch.mean(
            torch.sum(
                v0_flat.view(-1, self.n_movies, self.K)
                * mask_expanded
                * torch.log(p_v1 + 1e-6),
                dim=2,
            )
        )
        return loss


# ==========================================
# 3. 训练模型
# ==========================================
model = SoftmaxRBM(n_movies, K, n_hidden=64)
EPOCHS = 30

print("\n🚀 开始训练 Softmax RBM (处理缺失值)...")
for epoch in range(EPOCHS):
    # 为了简单，这里使用全批量梯度下降 (数据量已经很小了)
    loss = model.train_step(V, Mask, lr=0.01)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Cross-Entropy Loss: {loss.item():.4f}")

print("✅ 训练完成！")

# ==========================================
# 4. 课堂演示：填补缺失评分与推荐
# ==========================================
print("\n" + "=" * 50)
print("👨‍🏫 【课堂演示】用户评分填补与电影推荐")
print("=" * 50)

# 选择演示用户（比如第 0 号用户）
demo_user_id = 0
user_v = V[demo_user_id : demo_user_id + 1]  # [1, M, K]
user_mask = Mask[demo_user_id : demo_user_id + 1]  # [1, M]

# 1. 推理：网络根据已有评分，预测所有电影的评分概率分布
with torch.no_grad():
    p_v = model.sample_v(model.sample_h(user_v.view(1, -1), user_mask)[0])  # [1, M, K]
    p_v = p_v.squeeze(0)  # [M, K]

# 2. 将概率分布转化为期望评分 (例如：[0.1, 0.1, 0.2, 0.4, 0.2] -> 1*0.1+2*0.1+3*0.2+4*0.4+5*0.2 = 3.5星)
expected_ratings = torch.sum(p_v * torch.arange(1, K + 1), dim=1)  # [M]

# 3. 找出该用户没看过的电影
unwatched_indices = (user_mask.squeeze(0) == 0).nonzero(as_tuple=True)[0].numpy()

# 4. 对没看过的电影按预测评分降序排列
recommended_movie_indices = unwatched_indices[
    np.argsort(-expected_ratings[unwatched_indices].numpy())
]

print(f"\n👤 演示用户 ID: {demo_user_id + 1}")
print("-" * 50)

# 打印用户看过的电影（真实评分）
print("🎥 用户【已看过】的电影及真实评分:")
watched_indices = (user_mask.squeeze(0) == 1).nonzero(as_tuple=True)[0].numpy()
for idx in watched_indices[:5]:  # 只展示5部
    movie_name = movie_titles.get(idx + 1, "未知电影")
    true_rating = torch.argmax(user_v.squeeze(0)[idx]).item() + 1
    print(
        f"   - {movie_name}: ⭐⭐⭐⭐⭐".replace(
            "⭐" * true_rating, f"{'⭐' * true_rating} ({true_rating}星)"
        ).replace("⭐" * (5 - true_rating), "")
    )

print("\n🔮 RBM 填补【未看过】的电影预测评分 (Top 5 推荐):")
for idx in recommended_movie_indices[:5]:
    movie_name = movie_titles.get(idx + 1, "未知电影")
    pred_score = expected_ratings[idx].item()
    # 简单格式化星星
    stars = "⭐" * int(round(pred_score))
    print(f"   - {movie_name}: 预测 {pred_score:.2f} 星  {stars}")

print("\n💡 原理总结：")
print("RBM 通过隐藏层学习到了 '电影特征' 和 '用户偏好' 的联合分布。")
print("当你把一个不完整的评分向量输入时，网络利用能量最小化原则，")
print("自动把缺失的 Softmax 单元 '拉' 到最符合该用户特征向量的概率分布上！")
