# RBM Documentation Update Plan

## TL;DR

> **Quick Summary**: Update `README.md` to reflect the `src_v2/` code upgrade, and create `src_v2/ALGORITHM.md` with comprehensive algorithm and formula explanations for classroom use. Also update `AGENTS.md` to reflect that `src_v2/` is now populated.
>
> **Deliverables**:
> - `README.md` — Updated with `src_v2/` structure, run commands, and upgrade narrative
> - `src_v2/ALGORITHM.md` — New document explaining RBM formulas mapped to code
> - `AGENTS.md` — Updated structure section to reflect populated `src_v2/`
>
> **Estimated Effort**: Quick
> **Parallel Execution**: YES — 2 parallel + verification

---

## Context

### Original Request
更新项目的README文档，再在src_v2文件夹下写一个文档专门用来解释算法以及公式相关的代码。

### Current State
- `README.md` (79 lines) — Only describes `src/`, no mention of `src_v2/` upgraded code
- `src_v2/` — Contains `RBM_MNIST.py` (173 lines), `RBM_MOVIE.py` (404 lines), checkpoints, PNG
- `AGENTS.md` (80 lines) — Describes `src_v2/` as "Empty - reserved for future refactor"

---

## Work Objectives

### Core Objective
Document the upgraded `src_v2/` code with proper algorithm explanations and paper formula references.

### Concrete Deliverables
- `README.md` — Updated to include `src_v2/` info
- `src_v2/ALGORITHM.md` — Algorithm + formula document
- `AGENTS.md` — Updated structure section

### Must Have
- ALGORITHM.md covers: energy function, CD-1, Bernoulli RBM, Softmax RBM, free energy, masking, gradient scheme
- Each formula has corresponding code reference (file:line)
- README.md includes both `src/` and `src_v2/` run commands
- Academic references: Hinton (2002), Hinton (2012), Goodfellow (2016), Salakhutdinov (2007)

### Must NOT Have
- Do NOT modify `src/` or `src_v2/` Python files
- Do NOT add code examples not in existing codebase
- Do NOT add external links to copyrighted material

---

## Execution Strategy

```
Wave 1 (Start Immediately — fully parallel):
├── Task 1: Create src_v2/ALGORITHM.md [writing]
└── Task 2: Update README.md (add src_v2/ info) [quick]

Wave FINAL (sequential):
├── F1: Verify correctness + format + no broken links
   → Present results → Get explicit user okay
```

---

## TODOs

- [x] 1. Create `src_v2/ALGORITHM.md` — Algorithm and Formula Document

  **What to do**:
  Create a new file `src_v2/ALGORITHM.md` with the following structure:

  **Title**: `RBM 算法详解与公式推导 (Algorithm and Mathematical Derivation)`

  **Sections**:

  1. **受限玻尔兹曼机简介 (RBM Overview)**
     - What is RBM: energy-based model, bipartite graph (visible+hidden layers)
     - Boltzmann distribution: `P(v,h) = exp(-E(v,h)) / Z`
     - The partition function Z is intractable → CD-k bypasses it

  2. **能量函数 (Energy Function)**
     - Bernoulli RBM: `E(v,h) = -aᵀv - bᵀh - vᵀWh`
     - Code reference: `src_v2/RBM_MNIST.py` lines 26-32 (`energy()` method)
     - Softmax RBM (Salakhutdinov 2007): `E(V,h) = -ΣᵢΣⱼΣₖ Wᵢⱼᵏ hⱼ vᵢᵏ - ΣᵢΣₖ bᵢᵏ vᵢᵏ - Σⱼ bⱼ hⱼ`
     - Code reference: `src_v2/RBM_MOVIE.py` lines 85-93 (`energy()` method)

  3. **自由能 (Free Energy)**
     - Definition: `F(v) = -log Σₕ exp(-E(v,h))`
     - Closed form for Bernoulli RBM: `F(v) = -aᵀv - Σⱼ softplus(Wⱼᵀv + bⱼ)`
     - Code reference: `src_v2/RBM_MNIST.py` lines 34-37 (`free_energy()`)
     - Same form for Softmax RBM since visible units are binary (1-hot)
     - Code reference: `src_v2/RBM_MOVIE.py` lines 100-102 (`free_energy()`)
     - Why free energy matters: monitors convergence, `log P(v) = -F(v) - log Z`

  4. **条件概率 (Conditional Probabilities) — Bernoulli RBM**
     - Hidden: `P(hⱼ=1|v) = σ(bⱼ + Σᵢ vᵢ Wᵢⱼ)`
     - Code: `src_v2/RBM_MNIST.py` lines 39-42 (`sample_h()`)
     - Visible: `P(vᵢ=1|h) = σ(aᵢ + Σⱼ Wᵢⱼ hⱼ)`
     - Code: `src_v2/RBM_MNIST.py` lines 44-47 (`sample_v()`)
     - Both use `torch.sigmoid` + `torch.bernoulli` for stochastic sampling

  5. **条件概率 (Conditional Probabilities) — Softmax RBM**
     - Hidden (same as Bernoulli): `P(hⱼ=1|V) = σ(bⱼ + ΣᵢΣₖ vᵢᵏ Wᵢⱼᵏ)`
     - Code: `src_v2/RBM_MOVIE.py` lines 107-117 (`sample_h()`)
     - Visible (softmax per movie): `P(vᵢᵏ=1|h) = softmax(bᵢᵏ + Σⱼ hⱼ Wᵢⱼᵏ)`
     - Code: `src_v2/RBM_MOVIE.py` lines 122-127 (`sample_v()`)
     - Masking for missing ratings: zero out unseen entries + binary Mask tensor
     - Code: `src_v2/RBM_MOVIE.py` lines 48-56 (data prep), 107-113 (mask in `sample_h`)

  6. **对比散度算法 (Contrastive Divergence, CD-1)**
     - Why CD: intractable partition function, Gibbs sampling approximation
     - CD-1 algorithm steps:
       1. Positive phase: clamp `v⁰` to data → sample `h⁰ ~ P(h|v⁰)`
       2. Gibbs step: sample `v¹ ~ P(v|h⁰)` → sample `h¹ ~ P(h|v¹)`
       3. Gradient: `ΔW ∝ ⟨v⁰h⁰⟩_data - ⟨v¹h¹⟩_recon`
     - Key insight: Hinton (2012) recommends **sampled** `h⁰` for positive phase (info bottleneck regularization) and **probabilities** `p_h¹` for negative phase (variance reduction)
     - Code reference (Bernoulli): `src_v2/RBM_MNIST.py` lines 49-72 (`train_step()`)
     - Code reference (Softmax): `src_v2/RBM_MOVIE.py` lines 151-195 (`train_step()`)

  7. **梯度计算详解 (Gradient Computation)**
     - Weight gradient: `dW = (v⁰ᵀ @ h⁰ - p_v¹ᵀ @ p_h¹) / N` — uses `p_v¹` not `v¹` for lower variance
     - Visible bias: `dv_bias = mean(v⁰ - p_v¹)` — uses probabilities
     - Hidden bias: `dh_bias = mean(h⁰ - p_h¹)` — samples vs probabilities
     - Code (Bernoulli): `src_v2/RBM_MNIST.py` lines 58-68
     - Softmax RBM special: `dW = (v0_maskedᵀ @ h⁰ - v1_maskedᵀ @ p_h¹) / batch_size`
       Positive uses sampled `h⁰` (per Hinton), negative uses 1-hot sampled `v¹` (per Salakhutdinov 2007)
     - Code (Softmax): `src_v2/RBM_MOVIE.py` lines 168-185

  8. **缺失值处理 (Handling Missing Ratings)**
     - MovieLens problem: most user-movie pairs are unobserved
     - RBM approach: set unseen visible units to zero + use Mask to zero-out their gradient contribution
     - Code: `src_v2/RBM_MOVIE.py` lines 46-56 (build Mask), 107-113 (apply in `sample_h`), 165-174 (mask in gradient)

  9. **模型检查点与自动推理 (Model Persistence)**
     - Train → save `model.state_dict()` to `.pth` file
     - Rerun → if `.pth` exists, load checkpoint + run inference (skip training)
     - Code (Bernoulli): `src_v2/RBM_MNIST.py` lines 114-140
     - Code (Softmax): `src_v2/RBM_MOVIE.py` lines 216-227

  10. **参考文献 (References)**
      - Hinton (2002) "Training Products of Experts by Minimizing Contrastive Divergence" — CD-1 algorithm
      - Hinton (2006) "A Fast Learning Algorithm for Deep Belief Nets" — RBM pre-training
      - Hinton (2012) "A Practical Guide to Training Restricted Boltzmann Machines" §3.4 — Sampling strategy
      - Goodfellow, Bengio, Courville (2016) "Deep Learning" §20.2 — Free energy derivation
      - Salakhutdinov, Mnih, Hinton (2007) "Restricted Boltzmann Machines for Collaborative Filtering" — Softmax RBM

  **Format**:
  - Use Chinese (primary) + English (formulas/variable names)
  - LaTeX formulas via `$$ ... $$` or `$ ... $` for inline math
  - Code references as `src_v2/RBM_MNIST.py:N` (file:line) or `src_v2/RBM_MOVIE.py:N`
  - Tables where helpful (e.g., symbol-to-variable mappings)
  - Keep educational tone — written for students seeing RBM for the first time

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Pure documentation task requiring educational writing skills

  **QA Scenarios**:
  ```
  Scenario: File exists and has correct structure
    Tool: Bash
    Preconditions: src_v2/ exists
    Steps:
      1. Check src_v2/ALGORITHM.md exists
      2. Check file is not empty (>1000 chars)
      3. Check file has 10 section headers matching the spec
    Expected Result: File exists with all required sections
    Evidence: .sisyphus/evidence/task-1-algorithm-file.txt

  Scenario: LaTeX formulas render correctly
    Tool: Bash
    Preconditions: File exists
    Steps:
      1. Grep for `$$` — verify matching pairs
      2. Grep for `\$` — verify inline math delimiters balanced
    Expected Result: Math delimiters properly balanced
    Evidence: .sisyphus/evidence/task-1-latex-check.txt
  ```

  **Commit**: YES
  - Message: `docs: add RBM algorithm and formula reference doc to src_v2/`

---

- [x] 2. Update `README.md` — Add `src_v2/` Information

  **What to do**:
  Edit `README.md` to add `src_v2/` content. Do NOT remove any existing content about `src/`.

  **Changes**:

  1. **Project Structure** section: Add `src_v2/` entry:
     ```
     ├── src_v2/
     │   ├── RBM_MNIST.py    # 升级版: 二值 RBM + 能量函数 + 自由能
     │   ├── RBM_MOVIE.py    # 升级版: Softmax RBM + 可视化 + 检查点
     │   └── ALGORITHM.md     # 算法详解与公式推导文档
     ```

  2. **Run** section: Add `src_v2/` run commands:
     ```bash
     uv run python src/RBM_MNIST.py          # 原版: 快速演示
     uv run python src/RBM_MOVIE.py          # 原版: 终端推荐

     uv run python src_v2/RBM_MNIST.py       # 升级版: 能量函数 + 自由能显示
     uv run python src_v2/RBM_MOVIE.py       # 升级版: 可视化柱状图 + 热力图
     ```

  3. **New section: "升级版本 (v2)"** — Insert after existing 🎬 section, before the 🧠 algorithm section:
     ```
     ---
     ## ⬆️ 升级版本 (src_v2/)
     **为什么要升级？** 为了让代码更贴合 Hinton (2002/2012) 和 Salakhutdinov (2007) 论文的公式标准，并增加教学演示功能。

     ### 核心升级点
     - **能量函数 E(v,h)** 和 **自由能 F(v)** 显式实现，代码注释标注论文公式出处
     - **梯度方案标准化**: 正相位采样（信息瓶颈正则化），负相位概率（方差缩减）— 严格遵循 Hinton (2012) §3.4
     - **模型检查点**: 训练后自动保存，再次运行跳过训练直接推理
     - **`if __name__ == "__main__"` 保护**: 代码结构更规范
     - **[RBM_MOVIE.py]** 新增 Matplotlib 可视化: 推荐柱状图 + 评分热力图

     #### MNIST 升级版 (`src_v2/RBM_MNIST.py`)
     ```
     uv run python src_v2/RBM_MNIST.py      # 自动训练/加载 → 显示重构结果 + 自由能值
     ```

     #### MovieLens 升级版 (`src_v2/RBM_MOVIE.py`)
     ```
     uv run python src_v2/RBM_MOVIE.py      # 自动训练/加载 → 终端推荐 + 可视化图表
     ```

     📖 算法原理与公式详解请参见: `src_v2/ALGORITHM.md`
     ```

  4. **No other changes** — keep existing content intact

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple edits to existing file, no structural changes needed

  **QA Scenarios**:
  ```
  Scenario: All expected sections present
    Tool: Bash
    Preconditions: README.md exists
    Steps:
      1. Grep for "src_v2" — verify at least 3 occurrences
      2. Grep for "ALGORITHM.md" — verify reference
      3. Grep for "升级版本" or "v2" — verify new section
    Expected Result: All three patterns present
    Evidence: .sisyphus/evidence/task-2-readme-sections.txt

  Scenario: Original content preserved
    Tool: Bash
    Preconditions: README.md exists
    Steps:
      1. Grep for "PyTorch 从零手搓" — verify still present
      2. Grep for "uv run python src/RBM_MNIST.py" — verify original command exists
    Expected Result: Original content intact
    Evidence: .sisyphus/evidence/task-2-original-preserved.txt
  ```

  **Commit**: YES
  - Message: `docs: update README with src_v2/ upgrade info`

---

- [x] 3. Update `AGENTS.md` — Refresh Structure Section

  **What to do**:
  Edit `AGENTS.md` to reflect that `src_v2/` is now populated with upgraded code.

  **Changes**:

  1. Update the structure tree (line 16):
     Change:
     ```diff
     -├── src_v2/             # Empty - reserved for future refactor
     +├── src_v2/             # Upgraded paper-standard RBM implementations
     +│   ├── RBM_MNIST.py    # Bernoulli RBM with energy/free-energy + model persistence
     +│   ├── RBM_MOVIE.py    # Softmax RBM with visualization + model persistence
     +│   └── ALGORITHM.md    # Algorithm and formula reference
     ```

  2. Update `WHERE TO LOOK` table: add row for upgraded code:
     ```
     | Upgraded Bernoulli RBM | `src_v2/RBM_MNIST.py` | Paper-standard with energy + free energy |
     | Upgraded Softmax RBM | `src_v2/RBM_MOVIE.py` | Paper-standard with Matplotlib visualization |
     | Algorithm reference | `src_v2/ALGORITHM.md` | Formula derivations and code mapping |
     ```

  3. Update `COMMANDS` section: add `src_v2/` run commands

  4. Update the note about `src_v2/` (line 53 anti-patterns): Update "Do NOT modify `src/`" to reference the now-populated `src_v2/`

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small targeted edits, existing file

  **QA Scenarios**:
  ```
  Scenario: Structure section updated
    Tool: Bash
    Preconditions: AGENTS.md exists
    Steps:
      1. Grep for "src_v2" — verify "upgraded" or "populated" instead of "Empty"
      2. Verify ALGORITHM.md referenced
    Expected Result: AGENTS.md reflects current state
    Evidence: .sisyphus/evidence/task-3-agents-update.txt
  ```

  **Commit**: YES
  - Message: `docs: update AGENTS.md to reflect populated src_v2/`

---

## Final Verification Wave

- [x] F1. **Cross-File Consistency Check**
  Verify all three files are consistent:
  - README.md references ALGORITHM.md correctly
  - File paths in ALGORITHM.md match actual files
  - Run commands in README.md actually work (can be run)
  - No dead links or broken references between documents

---

## Commit Strategy

- **1**: `docs: add RBM algorithm and formula reference doc to src_v2/`
- **2**: `docs: update README with src_v2/ upgrade info`
- **3**: `docs: update AGENTS.md to reflect populated src_v2/`
