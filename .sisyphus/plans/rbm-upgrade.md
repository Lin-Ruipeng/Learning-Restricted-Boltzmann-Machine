# RBM Code Upgrade Plan (src/ → src_v2/)

## TL;DR

> **Quick Summary**: Upgrade two RBM Python scripts (MNIST + MovieLens) to be paper-standard compliant, adding energy functions, proper gradient schemes per Hinton 2012 / Salakhutdinov 2007, model persistence with auto-inference, and Matplotlib visualization for the movie recommendation demo. Save upgraded versions to `src_v2/`.
>
> **Deliverables**:
> - `src_v2/RBM_MNIST.py` — Paper-standard Bernoulli RBM with energy/free-energy + CD-1 training + model save/load + auto-inference
> - `src_v2/RBM_MOVIE.py` — Paper-standard Softmax RBM with energy + CD-1 training + save/load + auto-inference + Matplotlib visualization (bar chart + heatmap)
>
> **Estimated Effort**: Medium (~300 lines each)
> **Parallel Execution**: YES — 2 waves
> **Critical Path**: Task 1 ↔ Task 2 (fully parallel) → Final verification

---

## Context

### Original Request
升级 `src/` 下的两个 RBM 代码，保存到 `src_v2/`，使其更规范（符合 Science 论文公式），添加模型保存/加载/自动推理，并为电影评分增加可视化展示。

### Interview Summary
**Key Discussions**:
- Current code already has bias terms — upgrade focuses on paper-compliance structural improvements
- CD-k: CD-1 (keep 1-step Gibbs) ✅
- MOVIE visualization: Bar chart + Heatmap (both) ✅
- MNIST Loss monitoring: Free Energy + MSE (both) ✅

**Research Findings**:
- Gradient asymmetry (sampled `h0` for positive, probs `p_h1` for negative) is **NOT a bug** — this is standard per Hinton 2012 "Practical Guide to Training RBMs" Section 3.4
- Legitimate fixes: use `p_v1` (probabilities) instead of `v1` (sampled) for visible bias and negative-phase `dW` term
- MOVIE: positive phase should use sampled `h0` (not `p_h0`), negative phase should sample 1-hot from softmax (not use mean-field probs)

### Metis Review
**Identified Gaps** (addressed):
- ❌ "Gradient asymmetry is a bug" → Corrected: it's standard practice, document in comments
- ✅ Visible bias uses `v1` → Fix: use `p_v1` for lower variance
- ✅ MOVIE uses probabilities for dW positive → Fix: use sampled `h0`
- ✅ MOVIE negative phase mean-field → Fix: sample 1-hot from softmax

---

## Work Objectives

### Core Objective
Upgrade both RBM implementations to paper-standard quality in `src_v2/` with model persistence and visualization.

### Concrete Deliverables
- `src_v2/RBM_MNIST.py` — Upgraded Bernoulli RBM
- `src_v2/RBM_MOVIE.py` — Upgraded Softmax RBM with visualization

### Definition of Done
- [ ] `uv run python src_v2/RBM_MNIST.py` runs without errors
- [ ] `uv run python src_v2/RBM_MOVIE.py` runs without errors
- [ ] Both save/load model checkpoints correctly
- [ ] MOVIE shows Matplotlib visualization (bar chart + heatmap)

### Must Have
- Paper-standard energy function E(v,h) and free energy F(v)
- Consistent gradient scheme per Hinton 2012 / Salakhutdinov 2007
- Model persistence: save after train, auto-load + inference when checkpoint exists
- Proper `if __name__ == "__main__"` guard
- Explicit paper formula references in comments

### Must NOT Have (Guardrails)
- Do NOT change the fundamental approach — still manual CD-1 gradients (no autograd)
- Do NOT modify `src/` files — only `src_v2/`
- Do NOT add external dependencies beyond current stack (torch, numpy, matplotlib, pandas)
- Do NOT refactor into shared modules — each file stays self-contained
- Do NOT change input/output shapes or break compatibility with the original checkpoint `rbm_mnist.pth`

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: None (project has no test infra)
- **Agent-Executed QA**: ALWAYS — run the scripts and verify outputs

### QA Policy
Every task includes agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **MNIST**: Run via `uv run python src_v2/RBM_MNIST.py` — verify training output, model save, inference display
- **MOVIE**: Run via `uv run python src_v2/RBM_MOVIE.py` — verify training output, model save, matplotlib window with bar chart + heatmap

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — fully parallel):
├── Task 1: Upgrade RBM_MNIST.py → src_v2/ [medium]
└── Task 2: Upgrade RBM_MOVIE.py → src_v2/ [medium]

Wave FINAL (sequential after both tasks):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review + run test
├── Task F3: Real manual QA — execute both scripts
└── Task F4: Scope fidelity check
   → Present results → Get explicit user okay

Critical Path: Task 1 → F1-F4 → user okay
                Task 2 ↗
Parallel Speedup: ~50% faster than sequential
Max Concurrent: 2 (Wave 1)
```

### Dependency Matrix
- **1**: - (none) - F1-F4, F2-F3
- **2**: - (none) - F1-F4, F2-F3
- **F1**: 1, 2 - -
- **F2**: 1, 2 - -
- **F3**: 1, 2 - -
- **F4**: 1, 2 - -

---

## TODOs

- [x] 1. Upgrade `src/RBM_MNIST.py` → `src_v2/RBM_MNIST.py`

  **What to do**:
  Create a new file `src_v2/RBM_MNIST.py` with the following structure and improvements over the original:

  **Model class `RBM`** (nn.Module):
  - `__init__(n_vis=784, n_hid=256)` — same as original, `W`, `v_bias`, `h_bias` as `nn.Parameter`
  - **`energy(v, h)`** — new: compute `E(v,h) = -vᵀWh - vᵀa - hᵀb` per Hinton 2002 Eq. 2.1
  - **`free_energy(v)`** — new: compute `F(v) = -aᵀv - Σⱼlog(1 + exp(Wⱼᵀv + bⱼ))` per Goodfellow 2016 Eq. 20.10
  - `sample_h(v)` → `(p_h, h_sample)` — same sigmoid + Bernoulli
  - `sample_v(h)` → `(p_v, v_sample)` — same sigmoid + Bernoulli
  - `train_step(v0, lr=0.01)` → CD-1, returns `(loss_mse, free_energy)`:
    - Positive phase: `p_h0, h0 = sample_h(v0)` — sampled `h0` (Hinton: keep sampling for positive!)
    - Gibbs step: `p_v1, v1 = sample_v(h0)`
    - Negative phase: `p_h1, _ = sample_h(p_v1)` — **use `p_v1` not `v1`** for lower variance
    - Gradients:
      - `dW = (v0.T @ h0 - p_v1.T @ p_h1) / N` — **use `p_v1` instead of `v1`**
      - `dv_bias = mean(v0 - p_v1)` — **use `p_v1` instead of `v1`**
      - `dh_bias = mean(h0 - p_h1)`
    - `with torch.no_grad(): self.W += ...`
    - Returns `(mse_loss, free_energy_value)`
  - **`reconstruct(v)`** — same as original: `p_h, _ = sample_h(v); p_v, _ = sample_v(p_h); return p_v`

  **Main script** (`if __name__ == "__main__"` guard — NEW):
  - Config constants: `BATCH_SIZE=64`, `EPOCHS=10`, `LR=0.01`, `N_HID=256`, `MODEL_PATH="src_v2/rbm_mnist_v2.pth"`
  - Data loading: same MNIST transforms + DataLoader
  - Model init: `model = RBM(n_vis=784, n_hid=N_HID)`
  - **If checkpoint exists**: load and skip to inference
  - **If not**: train with CD-1, print epoch loss + free energy, save checkpoint
  - **Inference**: sample 3 random test images, show original vs reconstruction (Matplotlib, same layout as original)

  **Comments**: Add explicit formula references:
  - `# E(v,h) = -vᵀWh - aᵀv - bᵀh  ← Hinton (2002) Eq. 2.1`
  - `# ΔW ∝ ⟨vᵢhⱼ⟩_data - ⟨vᵢhⱼ⟩_recon  ← Hinton (2002) Eq. 3.4`
  - `# Positive hidden: sampled (info bottleneck) ← Hinton (2012) §3.4`
  - `# Negative hidden: probabilities (variance reduction) ← Hinton (2012) §3.4`

  **Must NOT do**:
  - Do NOT change the Bernoulli sampling approach
  - Do NOT change input shape (784 → 256 → 784)
  - Do NOT add autograd/optimizer

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Multiple interdependent concerns (model logic, gradient math, training pipeline, inference visualization) require deep understanding of RBM theory and PyTorch mechanics.
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: F1, F2, F3, F4
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References** (existing code to follow):
  - `src/RBM_MNIST.py` — Original code to upgrade. Follow the same model structure, data loading, and inference visualization pattern. Maintain the same class/method signatures where applicable.

  **API/Type References**:
  - `src/RBM_MNIST.py:RBM` — Class structure to preserve (sample_h, sample_v, train_step, reconstruct)
  - `src/RBM_MNIST.py:36-49` — Current gradient computation. Update `v1` → `p_v1` in `dW` and `dv_bias`

  **External References**:
  - Hinton (2002) "Training Products of Experts" — CD-1 gradient formula
  - Hinton (2012) "A Practical Guide to Training RBMs" §3.4 — Sampling vs probability convention
  - Goodfellow (2016) "Deep Learning" §20.2 — Free energy formula

  **WHY Each Reference Matters**:
  - The original code is the baseline API contract — keep method signatures and data flow
  - The paper references guide the doc-comment formulas and gradient corrections

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Training from scratch + model save
    Tool: Bash (uv run)
    Preconditions: No checkpoint at src_v2/rbm_mnist_v2.pth
    Steps:
      1. Run: `uv run python src_v2/RBM_MNIST.py`
      2. Wait for training to complete (check output for "训练完成" or similar)
      3. Check exit code = 0
      4. Check file exists: src_v2/rbm_mnist_v2.pth
    Expected Result: Training completes successfully, model file saved
    Failure Indicators: Script errors, exit code != 0, no .pth file created
    Evidence: .sisyphus/evidence/task-1-training.txt (capture stdout+stderr)

  Scenario: Inference with existing checkpoint (auto-skip train)
    Tool: Bash (uv run)
    Preconditions: src_v2/rbm_mnist_v2.pth exists from previous run
    Steps:
      1. Run: `uv run python src_v2/RBM_MNIST.py`
      2. Verify output shows "加载已存在的模型" or skips training
      3. Verify shows MSE + Free Energy for inference samples
      4. Verify exit code = 0
    Expected Result: Skips training, loads checkpoint, runs inference, shows Matplotlib window
    Failure Indicators: Tries to train, no model load message, crashes
    Evidence: .sisyphus/evidence/task-1-inference.txt

  Scenario: Free Energy + MSE both printed during training
    Tool: Bash (uv run)
    Preconditions: No checkpoint
    Steps:
      1. Run: `uv run python src_v2/RBM_MNIST.py`
      2. Grep output for "Free Energy" and "MSE" (or Chinese equivalents)
      3. Verify both metrics appear in epoch summary
    Expected Result: Both metrics reported each epoch
    Failure Indicators: Only one metric shown
    Evidence: .sisyphus/evidence/task-1-metrics.txt
  ```

  **Commit**: YES
  - Message: `feat: upgrade Bernoulli RBM to paper-standard with energy/free-energy + model persistence`
  - Files: `src_v2/RBM_MNIST.py`

---

- [x] 2. Upgrade `src/RBM_MOVIE.py` → `src_v2/RBM_MOVIE.py`

  **What to do**:
  Create a new file `src_v2/RBM_MOVIE.py` with the following improvements:

  **Model class `SoftmaxRBM`** (nn.Module):
  - `__init__(n_movies, K=5, n_hidden=64)` — `W: nn.Parameter(n_movies*K, n_hidden)`, `v_bias`, `h_bias`
  - **`energy(v_flat, h)`** — new: compute `E(v,h) = -vᵀWh - vᵀa - hᵀb`
  - **`free_energy(v_flat)`** — new: closed-form free energy for Softmax RBM
  - `sample_h(v_flat, mask_flat)` → `(p_h, h_sample)` — masked input → sigmoid → Bernoulli
    - **Fix**: return sampled `h_sample` for `dW` positive phase (per Hinton convention)
  - `sample_v(h)` → return `p_v` (softmax probs per movie)
  - `gibbs_sample_v(h)` — new: sample 1-hot from softmax distribution (instead of using probs as v1)
  - `train_step(V_batch, Mask_batch, lr=0.05)` → CD-1, returns loss:
    - Positive phase: `p_h0, h0 = sample_h(v0_flat, Mask_batch)`
      - **Fix**: `dW_pos = v0_masked.T @ h0` — use `h0` (sampled) not `p_h0`
    - Gibbs step: `p_v1 = sample_v(h0)` → `v1_sample = gibbs_sample_v(h0)` (1-hot)
    - Negative phase: `p_h1, _ = sample_h(v1_flat, ones_like(Mask))`
    - Gradients:
      - `dW = (v0_masked.T @ h0 - v1_masked.T @ p_h1) / N`
      - `dv_bias = mean(v0_masked - v1_masked)`
      - `dh_bias = mean(h0 - p_h1)`
    - `with torch.no_grad(): self.W += ...`
    - Cross-entropy loss on masked ratings (same as original)

  **Data pipeline** (restructured on `if __name__` guard):
  - Same MovieLens download + parse logic
  - Subset selection: same `V[:200, :500]`, `Mask[:200, :500]`
  - Constants: `N_HIDDEN=64`, `EPOCHS=100`, `LR=0.05`, `MODEL_PATH="src_v2/rbm_movie_v2.pth"`

  **Training**:
  - **If checkpoint exists**: load and skip to inference/visualization
  - **If not**: train, print epoch loss every 5 epochs, save checkpoint

  **Inference + Visualization** (Matplotlib):
  - Load model, pick demo user (ID 0 as before)
  - Compute predicted ratings (same expected value method)
  - **Matplotlib figure with 2+ panels**:
    - **Left panel — Bar chart**: Show user's Top-8 recommended movies (unwatched) with predicted star ratings as colored bars. Bars colored by rating (1-5 color gradient). X-axis = movie title (truncated), Y-axis = predicted rating.
    - **Right panel — Heatmap**: Show a small user-movie rating matrix heatmap. Rows = a few demo users (3-5), Columns = top unwatched movies. Color intensity = predicted rating. Annotate with rating values.
    - Extra: Show user's watched movies with true ratings as a reference table below the charts
  - Save figure to `src_v2/movie_recommendation.png` AND `plt.show()` (blocking display)

  **Comments**: Add explicit formula references:
  - `# E(V,h) = -ΣᵢΣⱼΣₖ Wᵢⱼᵏ hⱼ vᵢᵏ  ← Salakhutdinov et al. (2007)`
  - `# P(vᵢᵏ=1|h) = softmax(bᵢᵏ + Σⱼ hⱼ Wᵢⱼᵏ)  ← S07 Eq. 1`
  - `# P(hⱼ=1|V) = σ(bⱼ + ΣᵢΣₖ vᵢᵏ Wᵢⱼᵏ)  ← S07 Eq. 2`
  - `# Positive hidden: sampled (info bottleneck) ← Hinton (2012) §3.4`

  **Must NOT do**:
  - Do NOT change the masking approach for missing ratings
  - Do NOT change the subset size (200 users, 500 movies)
  - Do NOT add autograd/optimizer

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: This task is ~40% model logic + ~30% data pipeline + ~30% Matplotlib visualization. The visualization is the most novel and requires design judgment (layout, colors, annotations, readability for classroom display).
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: F1, F2, F3, F4
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References** (existing code to follow):
  - `src/RBM_MOVIE.py` — Original code. Keep data loading, parsing, subset selection, expected_ratings computation, demo user selection.

  **API/Type References**:
  - `src/RBM_MOVIE.py:SoftmaxRBM` — Class to upgrade. Maintain sample_h, sample_v, train_step. Add energy, free_energy, gibbs_sample_v.
  - `src/RBM_MOVIE.py:96-134` — Current train_step. Change dW to use h0 (sampled), add gibbs 1-hot sampling for v1.
  - `src/RBM_MOVIE.py:153-193` — Current terminal output. Replace with Matplotlib visualization.

  **External References**:
  - Salakhutdinov, Mnih, Hinton (2007) ICML — Softmax RBM formulas for collaborative filtering
  - Matplotlib docs: `plt.subplots`, `plt.barh`, `plt.imshow` for heatmap, `plt.colorbar`
  - MovieLens dataset format: `u.data` (tab-separated user-item-rating), `u.item` (pipe-separated movie metadata)

  **WHY Each Reference Matters**:
  - The original code provides the proven data pipeline — reuse it
  - The Salakhutdinov paper defines the canonical Softmax RBM energy and gradient
  - Matplotlib docs are needed for the new visualization code

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Training from scratch + model save
    Tool: Bash (uv run)
    Preconditions: No checkpoint at src_v2/rbm_movie_v2.pth, ml-100k/ exists
    Steps:
      1. Run: `uv run python src_v2/RBM_MOVIE.py`
      2. Wait for training to complete (check for "训练完成" or epoch 100/100)
      3. Check exit code = 0
      4. Check file exists: src_v2/rbm_movie_v2.pth
    Expected Result: Training completes, model saved
    Failure Indicators: Errors during training, exit code != 0, no .pth file
    Evidence: .sisyphus/evidence/task-2-training.txt

  Scenario: Inference with visualization (checkpoint exists)
    Tool: Skill (Playwright for screenshot of matplotlib window)
    Preconditions: src_v2/rbm_movie_v2.pth exists
    Steps:
      1. Run: `uv run python src_v2/RBM_MOVIE.py`
      2. Verify output shows model is loaded (skip training)
      3. Matplotlib window should appear with bar chart + heatmap
      4. Take screenshot of the figure
      5. Check src_v2/movie_recommendation.png was saved
    Expected Result: Bar chart shows top-8 recommended movies, heatmap shows user-movie matrix
    Failure Indicators: No matplotlib window, crashes, missing saved figure
    Evidence: .sisyphus/evidence/task-2-viz.png

  Scenario: Output matches original terminal output structure
    Tool: Bash (uv run)
    Preconditions: src_v2/rbm_movie_v2.pth exists
    Steps:
      1. Run: `uv run python src_v2/RBM_MOVIE.py`
      2. Grep output for "已看过" and "未看过" (or their Chinese/English equivalents)
      3. Verify watched movies and recommended movies are listed
    Expected Result: Both watched and recommended movie sections are printed
    Failure Indicators: Missing movie sections, empty recommendations
    Evidence: .sisyphus/evidence/task-2-output.txt

  Scenario: Model loads from checkpoint and skips training
    Tool: Bash (uv run)
    Preconditions: src_v2/rbm_movie_v2.pth exists
    Steps:
      1. Run: `uv run python src_v2/RBM_MOVIE.py`
      2. Check output does NOT contain training epoch messages
    Expected Result: Skips training, goes directly to inference/viz
    Failure Indicators: Training loop starts (epoch messages visible)
    Evidence: .sisyphus/evidence/task-2-skip-train.txt
  ```

  **Commit**: YES
  - Message: `feat: upgrade Softmax RBM to paper-standard with visualization + model persistence`
  - Files: `src_v2/RBM_MOVIE.py`

---

## Final Verification Wave

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run scripts). For each "Must NOT Have": search source for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Check both files for: `as any`/`@ts-ignore` (not applicable for Python), bare `except:`, `console.log` in prod, commented-out code, unused imports. Check specific: gradient computation matches paper formulas, no autograd used, `no_grad()` context for manual updates, proper `if __name__ == "__main__"` guard.
  Output: `Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Run BOTH scripts from clean state. Execute EVERY QA scenario from both tasks, follow exact steps, capture evidence. Test cross-task: verify save/load works end-to-end (train → save → delete .pth → re-train → compare outputs).
  Output: `Scenarios [N/N pass] | Integration [N/N] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual file diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

- **1**: `feat: upgrade Bernoulli RBM to paper-standard with energy/free-energy + model persistence` — `src_v2/RBM_MNIST.py`
- **2**: `feat: upgrade Softmax RBM to paper-standard with visualization + model persistence` — `src_v2/RBM_MOVIE.py`

---

## Success Criteria

### Verification Commands
```bash
uv run python src_v2/RBM_MNIST.py    # Expected: train/inference without errors
uv run python src_v2/RBM_MOVIE.py    # Expected: train/inference + matplotlib visualization
```

### Final Checklist
- [ ] Both files created in `src_v2/`
- [ ] Both scripts run without errors
- [ ] Model checkpoints saved and loadable
- [ ] Energy + free energy methods implemented
- [ ] Gradient computation follows paper conventions
- [ ] MOVIE visualization shows bar chart + heatmap
- [ ] No modifications to `src/` files
- [ ] No autograd/optimizer used
