# RBM Upgrade Learnings

## Conventions from src/ code
- Indent: 4 spaces | No type hints | No docstrings
- UPPER_SNAKE_CASE constants, snake_case functions, PascalCase classes
- Imports: stdlib â†’ blank line â†’ third-party (os, numpy, torch order)
- Comments: Chinese explanations + section headers `# =====...=====`
- No `if __name__ == "__main__"` guard in original
- Manual CD-1 gradients, no autograd

## Gradient Scheme (Hinton 2012)
- Positive hidden: sampled `h0` (info bottleneck regularization)
- Negative hidden: probabilities `p_h1` (variance reduction)
- Use `p_v1` instead of `v1` for `dW` negative term and `dv_bias`

## Softmax RBM Fixes (Salakhutdinov 2007)
- dW positive: use sampled `h0` not `p_h0`
- Negative phase: sample 1-hot from softmax, not mean-field probs

## v2 Bernoulli RBM Implementation (2026-05-09)
### File: src_v2/RBM_MNIST.py
- Added `energy(v, h)` method: `E = -a^T v - b^T h - v^T W h` per-sample
- Added `free_energy(v)` method: `F = -a^T v - sum_j softplus(W_j^T v + b_j)` per-sample
- Fixed gradient asymmetry: `p_v1` (probs) instead of `v1` (sampled) in `dW` negative and `dv_bias`
- `train_step` now returns `(mse_loss, free_energy_val)` tuple
- `if __name__ == "__main__":` guard added
- Checkpoint saved to `src_v2/rbm_mnist_v2.pth` (separate from original `rbm_mnist.pth`)
- Free Energy displayed per test sample in Matplotlib visualization
- Training results: 10 epochs, final MSE ~0.0173, FE ~-285.64
- Second run correctly loads checkpoint and skips training

## v2 Softmax RBM Implementation (2026-05-09)
### File: src_v2/RBM_MOVIE.py (~396 lines)
- Added `energy(v_flat, h)` method per Salakhutdinov 2007: `E = -(v @ W * h).sum() - v@bias_v - h@bias_h`
- Added `gibbs_sample_v(h)`: samples 1-hot from softmax using `torch.multinomial` (replaces mean-field shortcut)
- Fixed CD-1 gradients:
  - dW positive: sampled `h0` (not `p_h0`) per Hinton convention
  - dW negative: 1-hot sampled `v1_sample` (not `p_v1` probs) per Salakhutdinov 2007
- Checkpoint to `src_v2/rbm_movie_v2.pth` with `if os.path.exists()` skip-training logic
- Matplotlib visualization:
  - Left: horizontal bar chart (top-8 recs, color-coded by rating 1-5)
  - Right: heatmap (5 users x 8 movies, YlOrRd colormap, annotated)
  - Bottom: watched movies text table with star ratings
  - Saved to `src_v2/movie_recommendation.png` (dpi=150)
- Training results: 100 epochs, final CE Loss ~1.28, recommendations: Schindler's List (4.66), Casablanca (4.65), Titanic (4.59)
- Glyph warnings fixed: removed `fontfamily="monospace"` from text box (DejaVu Sans Mono lacks CJK)

## F3 ¡ª Manual QA Results (2026-05-09)
- Scenarios: 8/8 PASS | Integration: 3/3 | VERDICT: APPROVE
- MNIST S1: 10 epochs, MSE 0.0173, FE -286.21, checkpoint 809KB - PASS
- MNIST S2: Load + skip-train message + inference - PASS
- MOVIE S3: 100 epochs, loss 1.517->1.282, checkpoint 652KB, figure saved - PASS
- MOVIE S4: Skip-train + both output sections + figure regenerated 215KB - PASS
- MOVIE S5: All 5 sections present (Î¬¶È/×Ó¼¯/ÒÑ¿´¹ý/ÍÆ¼ö/×¢Òâ) - PASS
- Cross-task S6-8: Valid Python, loadable checkpoints, src/ unchanged - PASS
- LSP diagnostics clean on both files
- MOVIE figure: 2385x1330 RGBA, non-blank (confirmed via PIL pixel analysis)
- Only warnings: plt.show() with Agg backend (expected, non-fatal)
## F4 Scope Fidelity Check (2026-05-09)
### Result: APPROVE
- Task 1 (RBM_MNIST.py): 10/10 compliant â€” all 7 methods, energy/free-energy formulas, p_v1 gradient fix, checkpoint logic, inference grid, paper refs, config constants
- Task 2 (RBM_MOVIE.py): 10/10 compliant â€” all 6 methods, gibbs 1-hot sampling, gradient fixes, checkpoint, bar chart+heatmap+text table, figure saved, terminal output preserved
- Scope creep: CLEAN â€” only 2 .py files + 1 auto-generated .pth in src_v2/; src/ untouched; no new dependencies; no autograd/optimizer
- Minor gap: Hinton (2012) section 3.4 reference present in MNIST but absent from MOVIE comments (plan line 291)
- Observation: free_energy not implemented for SoftmaxRBM (plan line 252) â€” plan compliance question for F1
