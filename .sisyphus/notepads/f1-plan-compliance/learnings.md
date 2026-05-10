# Learnings

## 2026-05-09: Added `free_energy` to SoftmaxRBM

- Added `free_energy(self, v_flat)` method to `SoftmaxRBM` in `src_v2/RBM_MOVIE.py` (line 100)
- Formula mirrors Bernoulli RBM: `F(v) = -a^T v - sum_j softplus(W_j^T v + b_j)`
- Same structure works because Softmax RBM visible units are binary (1-of-K encoded per movie)
- Inserted after `energy()` method (line 93) and before `sample_h()` (line 107)
- LSP diagnostics: clean | Syntax: OK
- Already had `torch` and `F` imported — no new imports needed
