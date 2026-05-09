import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 0. 自动下载并解压 MovieLens 100k 数据集
# =====================================================================
DATA_DIR = "./ml-100k"
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = "ml-100k.zip"

if not os.path.exists(DATA_DIR):
    print("[下载] 正在下载 MovieLens 100k 数据集...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(".")
    os.remove(ZIP_PATH)
    print("[OK] 下载并解压完成！")

# 读取评分数据 (user_id, item_id, rating, timestamp)
cols = ["user_id", "item_id", "rating", "timestamp"]
ratings_df = pd.read_csv(f"{DATA_DIR}/u.data", sep="\t", names=cols)

# 读取电影名称数据 (item_id, title)
movie_titles = {}
with open(f"{DATA_DIR}/u.item", encoding="latin-1") as f:
    for line in f:
        parts = line.strip().split("|")
        movie_titles[int(parts[0])] = parts[1]

# =====================================================================
# 1. 数据预处理：构建 Softmax 矩阵 V (n_users, n_movies, K) 和 Mask (n_users, n_movies)
# =====================================================================
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

# 截取子集演示: 200 用户 x 500 电影
V = V[:200, :500, :]
Mask = Mask[:200, :500]

n_users, n_movies = 200, 500
print(f"截取用于快速演示的维度: {n_users} 个用户, {n_movies} 部电影")


# =====================================================================
# 2. 定义 Softmax RBM 模型 (Salakhutdinov et al. 2007)
# =====================================================================
class SoftmaxRBM(nn.Module):
    def __init__(self, n_movies, K=5, n_hidden=64):
        super(SoftmaxRBM, self).__init__()
        self.n_movies = n_movies
        self.K = K
        self.n_hidden = n_hidden

        # 权重矩阵: (n_movies * K, n_hidden) — 每部电影有 K 个可见神经元
        self.W = nn.Parameter(torch.randn(n_movies * K, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_movies * K))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    # -----------------------------------------------------------------
    # 能量函数 (Salakhutdinov et al. 2007)
    # E(V,h) = -sum_{i,j,k} W_{i,j,k} h_j v_i^k - sum_{i,k} b_i^k v_i^k - sum_j b_j h_j
    # -----------------------------------------------------------------
    def energy(self, v_flat, h):
        # v_flat: (batch, n_movies * K),  h: (batch, n_hidden)
        # term1: -v^T W h   bilinear interaction
        term1 = -((v_flat @ self.W) * h).sum(dim=1)
        # term2: -v^T a     visible bias
        term2 = -(v_flat @ self.v_bias)
        # term3: -h^T b     hidden bias
        term3 = -(h @ self.h_bias)
        return term1 + term2 + term3

    # -----------------------------------------------------------------
    # P(h_j=1|V) = sigma(b_j + sum_{i,k} v_i^k W_{i,j,k})  <- S07 Eq. 2
    # -----------------------------------------------------------------
    def sample_h(self, v_flat, mask_flat):
        batch_size = v_flat.size(0)
        # 扩展 mask: (batch, n_movies) -> (batch, n_movies, K) -> (batch, n_movies*K)
        mask_expanded = (
            mask_flat.unsqueeze(-1).repeat(1, 1, self.K).reshape(batch_size, self.n_movies * self.K)
        )
        v_masked = v_flat * mask_expanded

        p_h = torch.sigmoid(v_masked @ self.W + self.h_bias)
        h_sample = torch.bernoulli(p_h)
        return p_h, h_sample

    # -----------------------------------------------------------------
    # P(v_i^k=1|h) = softmax(b_i^k + sum_j h_j W_{i,j,k})  <- S07 Eq. 1
    # -----------------------------------------------------------------
    def sample_v(self, h):
        # h: (batch, n_hidden)
        logits = h @ self.W.t() + self.v_bias  # (batch, n_movies * K)
        logits_reshaped = logits.view(-1, self.n_movies, self.K)
        p_v = F.softmax(logits_reshaped, dim=2)  # (batch, n_movies, K)
        return p_v

    # -----------------------------------------------------------------
    # Gibbs 采样: 从 softmax 分布抽取 1-hot 可见单元 (替代 mean-field 近似)
    # -----------------------------------------------------------------
    def gibbs_sample_v(self, h):
        p_v = self.sample_v(h)  # (batch, n_movies, K)
        batch_size = p_v.size(0)

        # torch.multinomial: 从 categorical 分布采样，返回 one-hot 索引
        p_v_2d = p_v.reshape(-1, self.K)  # (batch * n_movies, K)
        sampled_idx = torch.multinomial(p_v_2d, 1).squeeze(-1)  # (batch * n_movies,)

        # 构造 1-hot 张量
        v_onehot = torch.zeros(batch_size, self.n_movies, self.K, device=h.device)
        flat_idx = torch.arange(batch_size * self.n_movies, device=h.device)
        v_onehot_flat = v_onehot.reshape(-1, self.K)
        v_onehot_flat[flat_idx, sampled_idx] = 1.0

        return v_onehot  # (batch, n_movies, K)

    # -----------------------------------------------------------------
    # CD-1 训练步: dW ∝ <v_i^k h_j>_data - <v_i^k h_j>_recon
    # -----------------------------------------------------------------
    def train_step(self, V_batch, Mask_batch, lr=0.05):
        batch_size = V_batch.size(0)
        v0_flat = V_batch.view(batch_size, -1)  # (batch, n_movies * K)

        # 正相位: 真实数据 -> 采样隐藏层 h0
        p_h0, h0 = self.sample_h(v0_flat, Mask_batch)

        # 负相位: h0 -> Gibbs 采样可见层 v1 (1-hot) -> 概率 p_h1
        p_v1 = self.sample_v(h0)                # softmax 概率 (用于 loss 显示)
        v1_sample = self.gibbs_sample_v(h0)      # 1-hot 采样 (用于梯度!)
        v1_flat = v1_sample.reshape(batch_size, -1)

        p_h1, _ = self.sample_h(v1_flat, torch.ones_like(Mask_batch))

        # 扩展 mask: (batch, n_movies) -> (batch, n_movies, K)
        mask_expanded = Mask_batch.unsqueeze(-1).repeat(1, 1, self.K)

        # 仅对有效评分计算梯度
        v0_masked = (
            v0_flat.view(-1, self.n_movies, self.K) * mask_expanded
        ).view(batch_size, -1)
        v1_masked = (
            v1_sample * mask_expanded
        ).view(batch_size, -1)

        # 梯度: 正相位用采样 h0，负相位用 1-hot 可见 + 概率 p_h1
        dW = (v0_masked.T @ h0 - v1_masked.T @ p_h1) / batch_size
        dv_bias = (v0_masked - v1_masked).mean(dim=0)
        dh_bias = (h0 - p_h1).mean(dim=0)

        # 手动更新权重 (无 autograd)
        with torch.no_grad():
            self.W += lr * dW
            self.v_bias += lr * dv_bias
            self.h_bias += lr * dh_bias

        # 交叉熵损失: 仅对有效评分计算
        valid_count = Mask_batch.sum() + 1e-8
        ce = -(
            v0_flat.view(-1, self.n_movies, self.K)
            * mask_expanded
            * torch.log(p_v1 + 1e-6)
        ).sum()
        loss = ce / valid_count
        return loss


# =====================================================================
# 3. 主程序入口
# =====================================================================
if __name__ == "__main__":
    # -----------------------------------------------------------------
    # 配置
    # -----------------------------------------------------------------
    N_HIDDEN = 64
    EPOCHS = 100
    LR = 0.05
    MODEL_PATH = "src_v2/rbm_movie_v2.pth"
    FIGURE_PATH = "src_v2/movie_recommendation.png"

    model = SoftmaxRBM(n_movies, K, n_hidden=N_HIDDEN)

    # -----------------------------------------------------------------
    # 检查点逻辑：已有模型则跳过训练
    # -----------------------------------------------------------------
    if os.path.exists(MODEL_PATH):
        print(f"加载已存在的模型 [{MODEL_PATH}]，跳过训练...")
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    else:
        print("\n[训练] 开始训练 Softmax RBM (处理缺失值)...")
        for epoch in range(EPOCHS):
            loss = model.train_step(V, Mask, lr=LR)
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], "
                      f"Mean Active Cross-Entropy Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"[OK] 训练完成，模型已保存至 [{MODEL_PATH}]")

    # =================================================================
    # 4. 课堂演示：填补缺失评分与推荐 (终端输出)
    # =================================================================
    print("\n" + "=" * 50)
    print("[课堂] 【课堂演示】用户评分填补与电影推荐")
    print("=" * 50)

    DEMO_USER_ID = 0
    user_v = V[DEMO_USER_ID : DEMO_USER_ID + 1]       # (1, n_movies, K)
    user_mask = Mask[DEMO_USER_ID : DEMO_USER_ID + 1]  # (1, n_movies)

    with torch.no_grad():
        _, h_user = model.sample_h(user_v.view(1, -1), user_mask)
        p_v = model.sample_v(h_user)
        p_v = p_v.squeeze(0)  # (n_movies, K)

    # 期望评分: E[rating] = sum_k k * P(v_i^k = 1)
    expected_ratings = torch.sum(
        p_v * torch.arange(1, K + 1, dtype=torch.float32), dim=1
    )

    # 未观看电影索引（按预测评分降序排列）
    unwatched_indices = (user_mask.squeeze(0) == 0).nonzero(as_tuple=True)[0].numpy()
    recommended_movie_indices = unwatched_indices[
        np.argsort(-expected_ratings[unwatched_indices].numpy())
    ]

    print(f"\n[用户] 演示用户 ID: {DEMO_USER_ID + 1}")
    print("-" * 50)

    # 已看电影
    print("[已看] 用户【已看过】的电影及真实评分:")
    watched_indices = (user_mask.squeeze(0) == 1).nonzero(as_tuple=True)[0].numpy()
    watched_data = []  # 收集已看电影数据，供可视化使用
    for idx in watched_indices[:8]:
        movie_name = movie_titles.get(idx + 1, "未知电影")
        true_rating = torch.argmax(user_v.squeeze(0)[idx]).item() + 1
        stars = "*" * true_rating
        print(f"   - {movie_name}: {stars} ({true_rating}星)")
        watched_data.append((movie_name, true_rating))

    # 未看电影推荐
    print("\n[推荐] RBM 填补【未看过】的电影预测评分 (Top 5 推荐):")
    top_n_recs = 8  # 用于可视化的前 8 推荐
    rec_movies = []  # (idx, name, pred_rating)
    for rank, idx in enumerate(recommended_movie_indices[:top_n_recs]):
        movie_name = movie_titles.get(idx + 1, "未知电影")
        pred_score = expected_ratings[idx].item()
        stars = "*" * int(round(pred_score))
        rec_movies.append((idx, movie_name, pred_score))
        if rank < 5:
            print(f"   - {movie_name}: 预测 {pred_score:.2f} 星  {stars}")

    print("\n[注意] 以上预测评分仅供参考，实际评分可能因用户喜好不同而有所差异。")

    # =================================================================
    # 5. Matplotlib 可视化 (新增)
    # =================================================================
    # 中文字体设置 (Windows)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    def truncate_title(title, max_len=22):
        """截断过长电影标题"""
        return title if len(title) <= max_len else title[:max_len - 1] + "..."

    # 评分-颜色映射
    rating_colors = {
        1: "#e74c3c",   # 红色
        2: "#e67e22",   # 橙色
        3: "#f1c40f",   # 黄色
        4: "#2ecc71",   # 浅绿
        5: "#27ae60",   # 深绿
    }

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Softmax RBM 电影推荐可视化 (MovieLens 100k)",
                 fontsize=16, fontweight="bold", y=0.98)

    # ----------------------------------------------------------------
    # 左侧面板: 横向柱状图 — Top 8 推荐电影
    # ----------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    top_recs = rec_movies[:8]  # (idx, name, pred_rating)

    movie_names = [truncate_title(name) for _, name, _ in top_recs]
    pred_ratings = [score for _, _, score in top_recs]
    # 反转顺序使最高评分在顶部
    movie_names_rev = movie_names[::-1]
    pred_ratings_rev = pred_ratings[::-1]

    bar_colors = [
        rating_colors.get(int(round(r)), "#95a5a6") for r in pred_ratings_rev
    ]
    bars = ax1.barh(movie_names_rev, pred_ratings_rev, color=bar_colors, edgecolor="white", height=0.6)

    # 在柱状图末端标注评分
    for bar, rating in zip(bars, pred_ratings_rev):
        star_count = int(round(rating))
        star_str = "*" * star_count if star_count <= 5 else "*" * 5
        ax1.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2,
                 f"{rating:.1f}  {star_str}",
                 va="center", fontsize=9, fontweight="bold")

    ax1.set_xlim(0, 5.5)
    ax1.set_xlabel("预测评分 (1-5 星)", fontsize=11)
    ax1.set_title("Top 8 推荐电影 (预测评分)", fontsize=13, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3, linestyle="--")

    # ----------------------------------------------------------------
    # 右侧面板: 热力图 — 用户-电影预测评分矩阵
    # ----------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)

    # 选取 5 个演示用户
    DEMO_USERS = [0, 1, 2, 4, 5]  # 用户索引
    top_movie_indices = [idx for idx, _, _ in top_recs]

    heatmap_data = np.zeros((len(DEMO_USERS), len(top_recs)))
    with torch.no_grad():
        for ui, uid in enumerate(DEMO_USERS):
            uv = V[uid:uid + 1]
            um = Mask[uid:uid + 1]
            _, hu = model.sample_h(uv.view(1, -1), um)
            pv_u = model.sample_v(hu).squeeze(0)  # (n_movies, K)
            er_u = torch.sum(pv_u * torch.arange(1, K + 1, dtype=torch.float32), dim=1)
            for mi, midx in enumerate(top_movie_indices):
                heatmap_data[ui, mi] = er_u[midx].item()

    im = ax2.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", vmin=1, vmax=5)

    # 标注每个单元格
    for ui in range(len(DEMO_USERS)):
        for mi in range(len(top_movie_indices)):
            val = heatmap_data[ui, mi]
            ax2.text(mi, ui, f"{val:.1f}", ha="center", va="center",
                     fontsize=9, fontweight="bold",
                     color="white" if val > 3.5 else "black")

    ax2.set_xticks(range(len(top_movie_indices)))
    ax2.set_xticklabels(
        [truncate_title(movie_titles.get(idx + 1, "未知"), 12) for idx in top_movie_indices],
        rotation=45, ha="right", fontsize=8
    )
    ax2.set_yticks(range(len(DEMO_USERS)))
    ax2.set_yticklabels([f"用户 {uid + 1}" for uid in DEMO_USERS], fontsize=10)
    ax2.set_title("用户-电影 预测评分热力图", fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax2, shrink=0.85)
    cbar.set_label("预测评分", fontsize=10)
    cbar.set_ticks([1, 2, 3, 4, 5])

    # ----------------------------------------------------------------
    # 下方面板: 演示用户已看电影记录
    # ----------------------------------------------------------------
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.axis("off")

    watched_text = "[记录] 演示用户 (用户 1) 已看电影:\n\n"
    for name, rating in watched_data[:6]:
        stars = "*" * rating
        watched_text += f"    {truncate_title(name, 35)}   {stars} ({rating}星)\n"
    watched_text += f"\n共观看 {len(watched_indices)} 部电影，以上展示前 {min(6, len(watched_data))} 部"

    ax3.text(0.5, 0.5, watched_text, transform=ax3.transAxes,
             fontsize=12, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=1", facecolor="#f8f9fa", edgecolor="#dee2e6"))

    # 保存并显示
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 确保 src_v2 目录存在
    os.makedirs(os.path.dirname(FIGURE_PATH) or "src_v2", exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    print(f"\n[图表] 可视化图表已保存至 [{FIGURE_PATH}]")
    plt.show()
