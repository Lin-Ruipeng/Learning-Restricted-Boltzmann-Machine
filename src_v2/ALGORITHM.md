# RBM 算法详解与公式推导

> 本文档配合 `src_v2/` 下的代码使用，每节均标注了对应的代码行号。
> 读者应具备基本的概率论和线性代数知识。所有公式均来自经典文献，代码实现遵循 Hinton (2012) 的工程实践建议。

---

## 1. 受限玻尔兹曼机简介

### 1.1 什么是 RBM

受限玻尔兹曼机 (Restricted Boltzmann Machine, RBM) 是一种**基于能量的生成式模型** (energy-based model)。它由两层神经元构成：

- **可见层 (visible layer)** $\mathbf{v}$：接受输入数据，如图像像素或用户评分。
- **隐藏层 (hidden layer)** $\mathbf{h}$：负责提取特征，隐式表达数据的潜在结构。
- **层间全连接，层内无连接**：这种"二分图"结构 (bipartite graph) 使得条件概率计算可以并行化，这也是"受限"二字的含义。

### 1.2 联合概率分布

RBM 通过能量函数定义联合概率分布：

$$P(\mathbf{v},\mathbf{h}) = \frac{1}{Z}\exp(-E(\mathbf{v},\mathbf{h}))$$

其中 $E(\mathbf{v},\mathbf{h})$ 是能量函数 (见第 2 节)，$Z$ 是**配分函数** (partition function)：

$$Z = \sum_{\mathbf{v},\mathbf{h}} \exp(-E(\mathbf{v},\mathbf{h}))$$

对于 $n$ 个可见单元和 $m$ 个隐藏单元，$Z$ 需要对 $2^{n+m}$ 种状态求和。当 $n=784$ (MNIST)、$m=256$ 时，这个数值比宇宙中的原子数还多，直接计算完全不可行。

### 1.3 回避配分函数：对比散度

由于 $Z$ 不可计算，我们不能直接对 $P(\mathbf{v})$ 做最大似然估计。Hinton (2002) 提出的**对比散度** (Contrastive Divergence, CD) 巧妙绕过了这个问题：

- 不是精确计算梯度，而是用**吉布斯采样** (Gibbs sampling) 近似。
- 只需要 $k$ 步 (通常 $k=1$) 即可得到足够好的梯度方向。
- 训练目标是：**拉低观测数据的能量，抬高模型重构数据的能量**。

> 这是一个"对比"的过程：让真实数据更有可能，让模型幻想的数据更不可能。

**代码结构**：`src_v2/RBM_MNIST.py:15-24` 定义了 RBM 类与参数，
`src_v2/RBM_MOVIE.py:69-79` 定义了 SoftmaxRBM 类与参数。

---

## 2. 能量函数

能量函数是 RBM 的核心。它给每一种可见-隐藏状态组合 $(v,h)$ 分配一个标量能量值。能量越低，该状态出现的概率越高。

### 2.1 符号说明

| 符号 | 代码变量 | 维度 | 含义 |
|------|----------|------|------|
| $\mathbf{v}$ | `v` | $(N, n\_vis)$ | 可见层状态 (二值) |
| $\mathbf{h}$ | `h` | $(N, n\_hid)$ | 隐藏层状态 (二值) |
| $a_i$ | `self.v_bias[i]` | 标量 | 第 $i$ 个可见单元偏置 |
| $b_j$ | `self.h_bias[j]` | 标量 | 第 $j$ 个隐藏单元偏置 |
| $W_{ij}$ | `self.W[i,j]` | 标量 | 可见单元 $i$ 与隐藏单元 $j$ 的权重 |
| $v_i^k$ | `V[u,m,k]` | 标量 (0 或 1) | 用户 $u$ 对电影 $m$ 评分等级 $k$ 的 one-hot 编码 |
| $K$ | `self.K` | 标量 | 评分等级数 (MovieLens: 5) |

### 2.2 Bernoulli RBM (二值-二值) 能量函数

二值 RBM 中，可见单元和隐藏单元均取值为 $\{0, 1\}$。能量函数定义为：

$$E(\mathbf{v},\mathbf{h}) = -\mathbf{a}^T\mathbf{v} - \mathbf{b}^T\mathbf{h} - \mathbf{v}^T\mathbf{W}\mathbf{h}$$

展开为标量形式：

$$E(v,h) = -\sum_i a_i v_i - \sum_j b_j h_j - \sum_{i,j} v_i W_{ij} h_j$$

- 第一项 $-\mathbf{a}^T\mathbf{v}$：可见偏置，衡量可见状态的"固有倾向"。
- 第二项 $-\mathbf{b}^T\mathbf{h}$：隐藏偏置，衡量隐藏状态的"固有倾向"。
- 第三项 $-\mathbf{v}^T\mathbf{W}\mathbf{h}$：交互项，捕捉可见-隐藏单元之间的相关性。这是真正的"学习"发生的地方。

**代码**：`src_v2/RBM_MNIST.py:26-32`

```python
def energy(self, v, h):
    return (
        -torch.matmul(v, self.v_bias)
        - torch.matmul(h, self.h_bias)
        - torch.sum(torch.matmul(v, self.W) * h, dim=1)
    )
```

### 2.3 Softmax RBM (类别-二值) 能量函数

在推荐系统中，评分不是二值的 (1-5 星)。Salakhutdinov, Mnih & Hinton (2007) 将每个可见单元替换为 $K$ 个 softmax 神经元，每个神经元对应一个评分等级。一部电影 $m$ 有 $K=5$ 个可见神经元，其中恰好一个为 1 (one-hot 编码)。

能量函数扩展为：

$$E(V,h) = -\sum_{i=1}^{N}\sum_{j=1}^{F}\sum_{k=1}^{K} W_{i,j,k} \,h_j\, v_i^k - \sum_{i=1}^{N}\sum_{k=1}^{K} b_i^k\, v_i^k - \sum_{j=1}^{F} b_j\, h_j$$

其中 $N$ 是电影数，$F$ 是隐藏单元数，$K$ 是评分等级数。$v_i^k = 1$ 表示用户对电影 $i$ 的评分为 $k$。

代码中将三维张量 $W_{i,j,k}$ 展平为二维矩阵 `self.W`，形状为 `(n_movies * K, n_hidden)`：

**代码**：`src_v2/RBM_MOVIE.py:85-93`

```python
def energy(self, v_flat, h):
    term1 = -((v_flat @ self.W) * h).sum(dim=1)
    term2 = -(v_flat @ self.v_bias)
    term3 = -(h @ self.h_bias)
    return term1 + term2 + term3
```

> **为什么 energy 返回的是标量而不是矩阵？**
> 每个样本 $(v,h)$ 都对应一个标量能量值。对一个 batch 计算时，`energy()` 返回形状为 $(N,)$ 的向量，代表每个样本的能量。后续在 free energy 和 CD 中取平均。

---

## 3. 自由能

### 3.1 定义

自由能 (free energy) 是**对隐藏单元求和**后的"约化"能量函数：

$$F(\mathbf{v}) = -\log\sum_{\mathbf{h}} \exp(-E(\mathbf{v},\mathbf{h}))$$

它与模型对数概率的关系为：

$$\log P(\mathbf{v}) = -F(\mathbf{v}) - \log Z$$

由于 $Z$ 是常数，**自由能下降等价于模型对数概率上升**。因此我们可以用自由能作为训练进度指标：训练过程中自由能持续下降，说明模型在有效学习。

### 3.2 封闭形式

对 Bernoulli RBM，自由能有漂亮的封闭形式：

$$F(\mathbf{v}) = -\mathbf{a}^T\mathbf{v} - \sum_j \text{softplus}\big(\mathbf{W}_j^T\mathbf{v} + b_j\big)$$

其中 $\text{softplus}(x) = \log(1 + e^x)$。推导利用了 $h_j \in \{0,1\}$ 的求和可解析计算的性质。

同样的公式也适用于 Softmax RBM，因为 softplus 项只依赖隐藏单元结构 (仍然是二值的)，与可见层是 Bernoulli 还是 softmax 无关。

**代码 (Bernoulli)**：`src_v2/RBM_MNIST.py:34-37`

```python
def free_energy(self, v):
    hidden_term = torch.matmul(v, self.W) + self.h_bias
    return -torch.matmul(v, self.v_bias) - torch.sum(F.softplus(hidden_term), dim=1)
```

**代码 (Softmax)**：`src_v2/RBM_MOVIE.py:100-102`

```python
def free_energy(self, v_flat):
    hidden_term = v_flat @ self.W + self.h_bias
    return -(v_flat @ self.v_bias) - torch.sum(F.softplus(hidden_term), dim=1)
```

### 3.3 为什么 free_energy 比 energy 更实用

- `energy(v, h)` 需要知道隐藏状态 $h$，但推理时我们不一定有 $h$。
- `free_energy(v)` 只需要可见状态 $v$，可以直接计算单个样本的"质量"。
- 在代码中，`free_energy` 被用作训练监控指标 (打印 Free Energy 的均值) 和推理指标 (重构时显示每张图的 FE 值)。

---

## 4. 条件概率 — Bernoulli RBM (二值-二值)

RBM 的核心优势在于**条件独立性**：给定一层，另一层的各单元条件独立。这使得采样可以高度并行化。

### 4.1 隐藏层给定可见层

$$P(h_j = 1 \mid \mathbf{v}) = \sigma\Big(b_j + \sum_i v_i W_{ij}\Big)$$

其中 $\sigma(x) = 1 / (1 + e^{-x})$ 是 sigmoid 函数。
"三部分"直观理解：

- $b_j$：隐藏单元 $j$ 自身的激活倾向。
- $\sum_i v_i W_{ij}$：所有可见单元对隐藏单元 $j$ 的"投票"总和。
- sigmoid 将投票结果映射到 $[0,1]$ 范围内的概率。

**代码**：`src_v2/RBM_MNIST.py:39-42`

```python
def sample_h(self, v):
    p_h = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
    return p_h, torch.bernoulli(p_h)
```

函数返回两个值：
1. `p_h`：概率值，用于梯度计算 (方差更低)。
2. `torch.bernoulli(p_h)`：采样后的二值状态 (0 或 1)，用于后续的 Gibbs 步骤。

### 4.2 可见层给定隐藏层

$$P(v_i = 1 \mid \mathbf{h}) = \sigma\Big(a_i + \sum_j W_{ij} h_j\Big)$$

对称形式，权重矩阵转置：

**代码**：`src_v2/RBM_MNIST.py:44-47`

```python
def sample_v(self, h):
    p_v = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
    return p_v, torch.bernoulli(p_v)
```

### 4.3 torch.bernoulli：抛硬币采样

`torch.bernoulli(p)` 根据概率 $p$ 独立地对每个神经元"抛硬币"：

- 有 $p$ 的概率输出 1 (神经元激活)。
- 有 $1-p$ 的概率输出 0 (神经元沉默)。

这种随机性对 RBM 至关重要：它引入了**探索性噪声**，使模型能够逃离局部能量极小值。如果没有采样 (直接用概率值)，RBM 退化为确定性自编码器，失去了生成能力。

---

## 5. 条件概率 — Softmax RBM (类别-二值)

推荐系统中，评分是 $K=5$ 类离散值，不能用单个二值神经元表示。Softmax RBM 为每个电影 $m$ 配备 $K$ 个可见神经元，构成 one-hot 编码。

### 5.1 隐藏层 (与 Bernoulli RBM 相同)

$$P(h_j = 1 \mid V) = \sigma\Big(b_j + \sum_{i=1}^{N}\sum_{k=1}^{K} v_i^k W_{i,j,k}\Big)$$

与二值 RBM 的唯一区别：可见层是二维的 (电影 $\times$ 评分等级)，求和范围扩展为 $i$ 和 $k$。

**代码**：`src_v2/RBM_MOVIE.py:107-117`

```python
def sample_h(self, v_flat, mask_flat):
    batch_size = v_flat.size(0)
    mask_expanded = (
        mask_flat.unsqueeze(-1).repeat(1, 1, self.K).reshape(batch_size, self.n_movies * self.K)
    )
    v_masked = v_flat * mask_expanded
    p_h = torch.sigmoid(v_masked @ self.W + self.h_bias)
    h_sample = torch.bernoulli(p_h)
    return p_h, h_sample
```

注意 `mask_flat` 的用法：未观看的电影对应的可见神经元被置零，不参与隐藏单元的激活计算。详见第 8 节。

### 5.2 可见层 (Softmax 多分类)

$$P(v_i^k = 1 \mid \mathbf{h}) = \frac{\exp\big(b_i^k + \sum_j h_j W_{i,j,k}\big)}{\sum_{l=1}^{K} \exp\big(b_i^l + \sum_j h_j W_{i,j,l}\big)}$$

对于电影 $i$，其 $K$ 个神经元的 logits 经过 **softmax** 归一化，确保概率和为 1。这保证了**恰好一个评分等级被选中**。

**代码**：`src_v2/RBM_MOVIE.py:122-127`

```python
def sample_v(self, h):
    logits = h @ self.W.t() + self.v_bias
    logits_reshaped = logits.view(-1, self.n_movies, self.K)
    p_v = F.softmax(logits_reshaped, dim=2)
    return p_v
```

`sample_v` 返回的是概率分布 `p_v`。要用它做下一步 Gibbs 采样，需要从 softmax 分布中抽取一个 one-hot 向量：

**代码**：`src_v2/RBM_MOVIE.py:132-146`

```python
def gibbs_sample_v(self, h):
    p_v = self.sample_v(h)
    batch_size = p_v.size(0)
    p_v_2d = p_v.reshape(-1, self.K)
    sampled_idx = torch.multinomial(p_v_2d, 1).squeeze(-1)
    v_onehot = torch.zeros(batch_size, self.n_movies, self.K, device=h.device)
    flat_idx = torch.arange(batch_size * self.n_movies, device=h.device)
    v_onehot_flat = v_onehot.reshape(-1, self.K)
    v_onehot_flat[flat_idx, sampled_idx] = 1.0
    return v_onehot
```

这里使用 `torch.multinomial` 替代 `torch.bernoulli` 做类别采样，因为 softmax 输出是**互斥的** (只能选一个评分等级)，不能用独立伯努利。

---

## 6. 对比散度算法 (CD-1)

### 6.1 为什么要用 CD

精确的对数似然梯度为：

$$\frac{\partial \log P(\mathbf{v})}{\partial \theta} = -\Big\langle \frac{\partial E(\mathbf{v},\mathbf{h})}{\partial \theta} \Big\rangle_{P(\mathbf{h}|\mathbf{v})} + \Big\langle \frac{\partial E(\mathbf{v},\mathbf{h})}{\partial \theta} \Big\rangle_{P(\mathbf{v},\mathbf{h})}$$

- 第一项 (正相位)：在数据条件下计算，容易近似。
- 第二项 (负相位)：在模型分布下计算，需要遍历所有 $(\mathbf{v},\mathbf{h})$ 组合，不可行。

CD 的核心思想：用 $k$ 步吉布斯采样的结果近似负相位。

### 6.2 CD-1 算法流程

**步骤 1 — 正相位 (positive phase)**：将训练样本 $\mathbf{v}^0$ 固定到可见层，采样隐藏层 $\mathbf{h}^0$。

$$\mathbf{h}^0 \sim P(\mathbf{h} \mid \mathbf{v}^0)$$

**步骤 2 — 吉布斯步 (Gibbs step)**：用 $\mathbf{h}^0$ 重构可见层 $\mathbf{v}^1$，再采样隐藏层 $\mathbf{h}^1$。

$$\mathbf{v}^1 \sim P(\mathbf{v} \mid \mathbf{h}^0), \quad \mathbf{h}^1 \sim P(\mathbf{h} \mid \mathbf{v}^1)$$

**步骤 3 — 梯度近似**：

$$\Delta \mathbf{W} \propto \big\langle \mathbf{v}^0 (\mathbf{h}^0)^T \big\rangle_{\text{data}} - \big\langle \mathbf{v}^1 (\mathbf{h}^1)^T \big\rangle_{\text{recon}}$$

### 6.3 Hinton (2012) 的采样策略

代码实现遵循 Hinton (2012) "A Practical Guide to Training RBM" 第 3.4 节的建议：

| 阶段 | 隐藏层 | 可见层 | 理由 |
|------|--------|--------|------|
| 正相位 | **采样** $\mathbf{h}^0$ (伯努利) | 固定为数据 $\mathbf{v}^0$ | 信息瓶颈正则化 |
| 负相位 | **概率** $\mathbf{p}_{\mathbf{h}}^1$ (无伯努利) | 重构 $\mathbf{v}^1$ | 降低梯度方差 |

> "When hidden units are driven by data, always use stochastic binary states. When driven by reconstructions, always use probabilities without sampling."
> — Hinton (2012)

- 正相位用采样：强制隐藏层做"硬决策"，防止过拟合。
- 负相位用概率：避免采样引入额外噪声，使梯度更稳定。

### 6.4 代码中的 CD-1

**Bernoulli RBM**：`src_v2/RBM_MNIST.py:49-72`

```python
def train_step(self, v0, lr=0.01):
    p_h0, h0 = self.sample_h(v0)          # 正相位: 采样 h0
    p_v1, v1 = self.sample_v(h0)           # Gibbs: 重构
    p_h1, _ = self.sample_h(p_v1)          # 负相位: 用概率 (非采样)

    N = v0.size(0)
    dW = (torch.matmul(v0.t(), h0) - torch.matmul(p_v1.t(), p_h1)) / N
    dv_bias = torch.mean(v0 - p_v1, dim=0)
    dh_bias = torch.mean(h0 - p_h1, dim=0)

    with torch.no_grad():
        self.W += lr * dW
        self.v_bias += lr * dv_bias
        self.h_bias += lr * dh_bias

    mse_loss = F.mse_loss(v0, p_v1)
    fe = self.free_energy(v0).mean()
    return mse_loss, fe
```

仔细观察：`self.sample_h(p_v1)` 的输入是 `p_v1` (概率值) 而非 `v1` (采样值)。这是刻意为之的**方差降低策略**。

**Softmax RBM**：`src_v2/RBM_MOVIE.py:151-195`

```python
def train_step(self, V_batch, Mask_batch, lr=0.05):
    # 正相位
    p_h0, h0 = self.sample_h(v0_flat, Mask_batch)
    # 负相位
    p_v1 = self.sample_v(h0)
    v1_sample = self.gibbs_sample_v(h0)       # 1-hot 采样!
    p_h1, _ = self.sample_h(v1_flat, torch.ones_like(Mask_batch))

    # 梯度...
    dW = (v0_masked.T @ h0 - v1_masked.T @ p_h1) / batch_size
    # ...

    with torch.no_grad():
        self.W += lr * dW
        ...
```

区别：Softmax RBM 的负相位可见层使用 **1-hot 采样** (`gibbs_sample_v`)，而非 softmax 概率。这是因为 softmax 的"全概率"向量 (如 $[0.1, 0.2, 0.6, 0.05, 0.05]$) 物理意义不明确 — 实际评分必须是确定的整数。采样保证了重构的可见状态是合法的评分。

---

## 7. 梯度计算详解

### 7.1 从能量到梯度的数学推导

对权重 $W_{ij}$ 的对数似然梯度为：

$$\frac{\partial \log P(\mathbf{v})}{\partial W_{ij}} = \mathbb{E}_{P_{\text{data}}}[v_i h_j] - \mathbb{E}_{P_{\text{model}}}[v_i h_j]$$

其中：
- $\mathbb{E}_{P_{\text{data}}}[v_i h_j]$：在数据分布下的期望 (正相位)。
- $\mathbb{E}_{P_{\text{model}}}[v_i h_j]$：在模型分布下的期望 (负相位，用 CD 近似)。

**这就是赫布学习律 (Hebbian learning) 的形式：一起激活的神经元之间的连接应该加强。** 正相位增强真实相关性，负相位削弱模型幻想出来的虚假相关性。

### 7.2 Bernoulli RBM 梯度

**代码**：`src_v2/RBM_MNIST.py:58-68`

```python
dW = (torch.matmul(v0.t(), h0) - torch.matmul(p_v1.t(), p_h1)) / N
dv_bias = torch.mean(v0 - p_v1, dim=0)
dh_bias = torch.mean(h0 - p_h1, dim=0)
```

矩阵形式 (对应图 1 的直观理解)：

- `v0.t() @ h0`：正相位梯度矩阵，形状 `(n_vis, n_hid)`。
- `p_v1.t() @ p_h1`：负相位梯度矩阵，形状相同。
- 两者相减后除以 batch size 得到平均梯度方向。

**关键细节**：
- `v0` 是真实数据，`h0` 是采样得到的 (含随机噪声)。
- `p_v1` 和 `p_h1` 都是概率值 (不含采样噪声)。

这是因为**数据需要随机性来打破对称性，重构需要确定性来稳定梯度**。详见第 6.3 节。

### 7.3 Softmax RBM 梯度

**代码**：`src_v2/RBM_MOVIE.py:168-185`

```python
dW = (v0_masked.T @ h0 - v1_masked.T @ p_h1) / batch_size
dv_bias = (v0_masked - v1_masked).mean(dim=0)
dh_bias = (h0 - p_h1).mean(dim=0)
```

与 Bernoulli RBM 的区别：

| 方面 | Bernoulli RBM | Softmax RBM |
|------|---------------|-------------|
| 可见层 | 概率 `p_v1` | 1-hot 采样 `v1_sample` |
| 隐藏层 (负) | 概率 `p_h1` | 概率 `p_h1` |
| 梯度来源 | `p_v1.t() @ p_h1` | `v1_masked.T @ p_h1` |

Softmax RBM 用 1-hot 采样后的可见层计算梯度，是因为 softmax 概率向量违背了可见单元的 one-hot 约束。一个电影如果显示概率 $[0.1, 0.2, 0.6, 0.05, 0.05]$，在物理上不是一个合法的评分。

### 7.4 为什么所有更新都在 torch.no_grad 中

**代码**：`src_v2/RBM_MNIST.py:65`

```python
with torch.no_grad():
    self.W += lr * dW
```

`self.W` 是 `nn.Parameter` 类型。PyTorch 默认会追踪 `Parameter` 上的所有操作构建 autograd 图。由于 RBM 的梯度是手动计算的 (不是通过 `loss.backward()`)，我们必须阻止 autograd 追踪更新操作，否则：

1. 计算图会指数级膨胀 (每次更新都挂到前一次的计算图上)。
2. 内存泄漏。
3. autograd 图记录了大量伯努利采样操作，而这些采样操作是不可导的。

**这条规则非常重要：手动更新 Parameter 必须放在 `torch.no_grad()` 上下文中。** 这是项目中最重要的反模式之一。

---

## 8. 缺失值处理 (Masking)

### 8.1 问题背景

在推荐系统场景中，每个用户只对极少数电影有过评分。例如 MovieLens 100k 数据集中，200 个用户对 500 部电影的评分矩阵稀疏度超过 99%。

传统方法需要做矩阵分解或插补。RBM 的处理方式更优雅：**将未观测的可见单元置为零，并用 Mask 张量屏蔽其梯度贡献。**

### 8.2 Mask 的构造

**代码**：`src_v2/RBM_MOVIE.py:48-56`

```python
V = torch.zeros(n_users, n_movies, K)
Mask = torch.zeros(n_users, n_movies)

for row in ratings_df.itertuples():
    u = row.user_id - 1
    m = row.item_id - 1
    r = int(row.rating) - 1
    V[u, m, r] = 1.0
    Mask[u, m] = 1.0
```

- `V[u, m, k] = 1`：用户 $u$ 对电影 $m$ 的评分为 $k+1$ (one-hot 编码)。
- `Mask[u, m] = 1`：该评分已观测，应参与训练。
- `Mask[u, m] = 0`：未观测，不贡献梯度。

### 8.3 Mask 在梯度计算中的应用

**第一步：扩展 Mask 维度**

`Mask` 形状为 `(batch, n_movies)`，需要扩展到与 `V` 相同的 `(batch, n_movies, K)`：

```python
mask_expanded = Mask_batch.unsqueeze(-1).repeat(1, 1, self.K)
```

**第二步：在 sample_h 中屏蔽未观测数据**

**代码**：`src_v2/RBM_MOVIE.py:107-113`

```python
v_masked = v_flat * mask_expanded
p_h = torch.sigmoid(v_masked @ self.W + self.h_bias)
```

未观测电影的可见神经元全部为零，不参与隐藏单元的激活计算。

**第三步：在梯度计算中只考虑观测位置**

**代码**：`src_v2/RBM_MOVIE.py:165-174`

```python
v0_masked = (v0_flat.view(-1, self.n_movies, self.K) * mask_expanded).view(batch_size, -1)
v1_masked = (v1_sample * mask_expanded).view(batch_size, -1)

dW = (v0_masked.T @ h0 - v1_masked.T @ p_h1) / batch_size
```

### 8.4 Mask 的物理含义

Mask 的本质是**不对未观测数据做任何假设**：

- 正相位：未观测位置贡献为零，隐藏单元完全由观测到的评分决定。
- 负相位：重构时仅约束观测位置的评分，未观测位置自由变化。
- 梯度更新仅发生在观测位置上，不会强行"填零"。

这与矩阵分解中的"忽略未知项"思想一致，但 RBM 是用概率模型天然实现的，不需要手动修改损失函数。

### 8.5 对比：Bernoulli RBM 不需要 Mask

MNIST 数据没有缺失值 — 每张图片的所有 784 个像素都是已知的。因此 `src_v2/RBM_MNIST.py` 中的代码没有使用 Mask。

---

## 9. 模型检查点与自动推理

### 9.1 保存与加载机制

训练结束后，模型参数被保存为 `.pth` 文件。下次运行时，如果检测到文件存在，直接跳过训练进入推理。

**保存**：

```python
torch.save(model.state_dict(), MODEL_PATH)
```

`state_dict()` 是一个 Python 字典，包含所有 `nn.Parameter` 参数的名称和数值张量。

**加载**：

```python
model.load_state_dict(torch.load(MODEL_PATH), strict=True)
```

参数 `strict=True` 确保保存的参数和模型结构完全匹配 — 如果模型结构发生变化 (如改变了隐藏层大小)，加载会报错，避免静默错误。

### 9.2 检查点逻辑

**Bernoulli RBM**：`src_v2/RBM_MNIST.py:114-140`

```python
if os.path.exists(MODEL_PATH):
    print("加载已存在的模型，跳过训练...")
    model.load_state_dict(torch.load(MODEL_PATH), strict=True)
else:
    # 训练循环...
    torch.save(model.state_dict(), MODEL_PATH)
```

**Softmax RBM**：`src_v2/RBM_MOVIE.py:216-227`

```python
if os.path.exists(MODEL_PATH):
    print("加载已存在的模型，跳过训练...")
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
else:
    # 训练循环...
    torch.save(model.state_dict(), MODEL_PATH)
```

两个脚本遵循同样的模式：**检查 → 加载/训练 → 保存**。

### 9.3 推理 (Inference)

加载模型后，推理过程不需要采样 (使用概率值)：

**Bernoulli RBM 重构**：`src_v2/RBM_MNIST.py:74-77`

```python
def reconstruct(self, v):
    p_h, _ = self.sample_h(v)
    p_v, _ = self.sample_v(p_h)
    return p_v
```

**Softmax RBM 评分预测**：`src_v2/RBM_MOVIE.py:240-248`

```python
_, h_user = model.sample_h(user_v.view(1, -1), user_mask)
p_v = model.sample_v(h_user)
expected_ratings = torch.sum(p_v * torch.arange(1, K + 1, dtype=torch.float32), dim=1)
```

对于 Softmax RBM，推理时计算**期望评分** (expected rating) 而非直接输出概率：

$$\mathbb{E}[\text{rating}_m] = \sum_{k=1}^{K} k \cdot P(v_m^k = 1 \mid \mathbf{h})$$

这个期望值可以是非整数 (如 3.72 星)，更准确地反映模型的不确定性。

---

## 10. 参考文献

1. **Hinton, G. E. (2002).** "Training Products of Experts by Minimizing Contrastive Divergence." *Neural Computation*, 14(8), 1771-1800.
   - CD-1 算法的原始论文。提出了用短链吉布斯采样近似负相位梯度的方法，奠定了 RBM 高效训练的理论基础。

2. **Hinton, G. E., Osindero, S., & Teh, Y. W. (2006).** "A Fast Learning Algorithm for Deep Belief Nets." *Neural Computation*, 18(7), 1527-1554.
   - 提出通过逐层预训练 RBM 构建深度信念网络 (DBN)。展示了 RBM 作为"构建块"在深度学习中的价值。

3. **Hinton, G. E. (2012).** "A Practical Guide to Training Restricted Boltzmann Machines." In *Neural Networks: Tricks of the Trade* (2nd ed.), Springer.
   - 工程实践圣经。第 3.4 节详细讨论了正相位用采样、负相位用概率的策略。本文代码的很多设计决策 (如学习率、参数初始化、采样策略) 源于此。

4. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press.
   - 第 20.2 节给出了自由能的推导及其与配分函数的关系。将 RBM 放在"基于能量的模型"框架下讨论，帮助读者理解更广泛的能量模型家族。

5. **Salakhutdinov, R., Mnih, A., & Hinton, G. (2007).** "Restricted Boltzmann Machines for Collaborative Filtering." *Proceedings of ICML 2007*.
   - Softmax RBM 的原创论文。将评分预测建模为多分类问题，在 Netflix 数据集上取得了当时最先进的推荐效果。本文 `src_v2/RBM_MOVIE.py` 的实现直接基于该论文。

---

> **附录：快速查找表**
>
> | 功能 | Bernoulli RBM (MNIST) | Softmax RBM (MovieLens) |
> |------|----------------------|------------------------|
> | 能量函数 | `RBM_MNIST.py:26-32` | `RBM_MOVIE.py:85-93` |
> | 自由能 | `RBM_MNIST.py:34-37` | `RBM_MOVIE.py:100-102` |
> | 隐藏层采样 | `RBM_MNIST.py:39-42` | `RBM_MOVIE.py:107-117` |
> | 可见层采样 | `RBM_MNIST.py:44-47` | `RBM_MOVIE.py:122-127` |
> | Gibbs 采样 | (内置在 train_step) | `RBM_MOVIE.py:132-146` |
> | CD-1 训练步 | `RBM_MNIST.py:49-72` | `RBM_MOVIE.py:151-195` |
> | 梯度计算 | `RBM_MNIST.py:58-68` | `RBM_MOVIE.py:168-185` |
> | 模型持久化 | `RBM_MNIST.py:114-140` | `RBM_MOVIE.py:216-227` |
> | 数据预处理 | (ToTensor 内置) | `RBM_MOVIE.py:48-56` |
