# 🔥 PyTorch 从零手搓受限玻尔兹曼机 (RBM)
本项目使用纯 PyTorch 从底层矩阵运算实现了受限玻尔兹曼机，并展示了它在两个截然不同领域的应用：**图像生成（二值 RBM）** 与 **隐式协同过滤（Softmax RBM）**。
> 💡 **写给有深度学习基础的同学**：
> 如果你看惯了 CNN/RNN 和 `loss.backward()`，这个项目会给你打开新世界的大门。在这里，**没有反向传播，没有 Autograd**。我们通过统计物理学中的“能量函数”和吉布斯采样，手动计算梯度来训练模型。
---
## 📁 项目结构
```text
.
├── src/
│   ├── RBM_MNIST.py    # 二值 RBM：处理黑白像素，实现图像重构
│   └── RBM_MOVIE.py    # Softmax RBM：处理 1-5 星评分，实现缺失值填补与推荐
├── src_v2/
│   ├── RBM_MNIST.py    # 升级版: 二值 RBM + 能量函数 + 自由能
│   ├── RBM_MOVIE.py    # 升级版: Softmax RBM + 可视化 + 检查点
│   └── ALGORITHM.md    # 算法详解与公式推导文档
├── pyproject.toml      # uv 依赖管理文件
└── README.md
```
---
## ⚙️ 环境配置 (基于 uv)
本项目使用现代化、极速的 Python 包管理器 [uv](https://github.com/astral-sh/uv) 进行环境管理。
```bash
# 1. 如果你还没有安装 uv，可以先安装它 (以下任一方法均可)
# curl -LsSf https://astral.sh/uv/install.sh | sh  (Linux/macOS)
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex" (Windows)
# pip install uv (Windows/macOS/Linux 通用)
# conda install uv (Windows/macOS/Linux 通用)
# 2. 克隆本项目
git clone https://github.com/Lin-Ruipeng/Learning-Restricted-Boltzmann-Machine.git
cd Learning-Restricted-Boltzmann-Machine-main
# 3. 安装依赖
uv sync
```
---
## 🚀 运行
```bash
uv run python src/RBM_MNIST.py          # 原版: 快速演示
uv run python src/RBM_MOVIE.py          # 原版: 终端推荐

uv run python src_v2/RBM_MNIST.py       # 升级版: 能量函数 + 自由能显示
uv run python src_v2/RBM_MOVIE.py       # 升级版: 可视化柱状图 + 热力图
```
---
## 🎬 效果演示与原理剖析
### 1. 图像重构：`RBM_MNIST.py`
**应用场景**：无监督特征提取与生成。
*   **原理**：将 28x28 的 MNIST 图像展平为 784 维的二值向量（0 或 1）。RBM 通过可见层与隐藏层的双向权重，学习像素间的联合概率分布。
*   **核心看点**：前向传播不是算 Logits，而是算**能量**；神经元输出不是确定的值，而是通过 `torch.bernoulli` **抛硬币采样**。
*   **运行结果**：程序会自动下载 MNIST，训练后弹出 Matplotlib 窗口，展示“原图”与“RBM 梦境重构图”的对比。
### 2. 评分预测与推荐：`RBM_MOVIE.py`
**应用场景**：推荐系统（MovieLens 数据集）。
*   **痛点解决**：传统网络遇到用户没看过的电影（缺失值）需要做复杂的插补。RBM 只需要把没看过的电影状态置为全零，并在计算梯度时使用 `Mask` 掩码屏蔽它们，**天然支持缺失值**。
*   **Softmax 可见单元**：1-5 星不是连续数值，而是 5 个类别。代码中将一部电影映射为 5 个神经元，使用 `F.softmax` 实现“多选一”的概率分布。
*   **运行结果**：自动下载 MovieLens 100k 小样本，训练后在终端打印出某个用户看过的电影，以及 RBM 为他填补的“未看过但可能喜欢”的 Top 5 电影及预测星级。
---

## ⬆️ 升级版本 (src_v2/)

**为什么要升级？** 为了让代码更贴合 Hinton (2002/2012) 和 Salakhutdinov (2007) 论文的公式标准，并增加教学演示功能。

### 核心升级点

- **能量函数 E(v,h)** 和 **自由能 F(v)** 显式实现，代码注释标注论文公式出处
- **梯度方案标准化**: 正相位采样（信息瓶颈正则化），负相位概率（方差缩减）— 严格遵循 Hinton (2012) §3.4
- **模型检查点**: 训练后自动保存 `.pth` 文件，再次运行跳过训练直接推理
- **`if __name__ == "__main__"` 保护**: 代码结构更规范，方便作为模块导入
- **[RBM_MOVIE.py]** 新增 **Matplotlib 可视化**: 推荐电影柱状图 + 用户评分热力图

#### MNIST 升级版 (`src_v2/RBM_MNIST.py`)

```bash
uv run python src_v2/RBM_MNIST.py      # 自动训练/加载 → 显示重构结果 + 自由能值
```

#### MovieLens 升级版 (`src_v2/RBM_MOVIE.py`)

```bash
uv run python src_v2/RBM_MOVIE.py      # 自动训练/加载 → 终端推荐 + 可视化图表
```

📖 算法原理与公式详解请参见: [`src_v2/ALGORITHM.md`](src_v2/ALGORITHM.md)

---

## 🧠 核心算法：为什么不用 `loss.backward()`？
在传统的 PyTorch 训练中，我们习惯：
```python
loss = criterion(output, target)
loss.backward()  # Autograd 魔法
optimizer.step()
```
但在 RBM 中，由于中间引入了不可导的随机采样操作 `torch.bernoulli()`，且理论上求解配分函数需要 $2^N$ 次计算（宇宙毁灭也算不完），我们使用了 Hinton 提出的 **对比散度算法**。
代码中的核心训练循环长这样（以 MNIST 为例）：
```python
# 正相位：真实数据激活特征
_, h0 = self.sample_h(v0)      
# 负相位：特征重构梦境数据
_, v1 = self.sample_v(h0)      
p_h1, _ = self.sample_h(v1)     
# 核心玄学：直接用矩阵乘法硬算梯度 (赫布学习律)
positive_grad = torch.matmul(v0.t(), h0)
negative_grad = torch.matmul(v1.t(), p_h1)
# 手动更新权重 (拉近真实能量，推高虚假能量)
self.W += lr * (positive_grad - negative_grad) / batch_size
```
*这行手动更新权重的代码，体现了统计物理与深度学习最完美的结合。*
---
## 📚 技术栈
- **核心框架**：PyTorch (纯张量操作，无高级 API 依赖)
- **数据集**：MNIST, MovieLens 100k
- **环境管理**：uv
- **可视化**：Matplotlib
---
## 📜 License
MIT License. 随意用于学习、课程演示或二次开发。
