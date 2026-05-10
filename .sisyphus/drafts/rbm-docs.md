# Draft: RBM Documentation Update

## Requirements (confirmed)
1. Update `README.md` to reflect `src_v2/` upgrades
2. Create algorithm/formula explanation document in `src_v2/` (e.g., `ALGORITHM.md`)

## Current State
- AGENTS.md: describes `src_v2/` as "empty - reserved for future refactor" → needs update
- README.md: only has `src/` entries, no mention of `src_v2/`
- src_v2/: now contains RBM_MNIST.py (175 lines), RBM_MOVIE.py (404 lines), checkpoints, visualization PNG

## Proposed Deliverables
1. `src_v2/ALGORITHM.md` — New: comprehensive algorithm explanation with formulas/code mapping (10 sections)
2. `README.md` — Update: add src_v2/ structure, run commands, upgrade section
3. `AGENTS.md` — Update: src_v2/ no longer empty, add WHERE TO LOOK entries

## Content Plan for ALGORITHM.md (10 sections)
1. RBM概述 (overview + Boltzmann distribution)
2. 能量函数 E(v,h) (Bernoulli + Softmax, code refs)
3. 自由能 F(v) (closed-form, code refs)
4. 条件概率 — Bernoulli RBM (sample_h, sample_v)
5. 条件概率 — Softmax RBM (sample_h, sample_v, masking)
6. 对比散度CD-1 (算法步骤, Hinton 2012策略)
7. 梯度计算详解 (dW, dv_bias, dh_bias formulas)
8. 缺失值处理 (Mask构建与应用)
9. 模型检查点与自动推理
10. 参考文献
