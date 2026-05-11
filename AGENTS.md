# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-09
**Commit:** aee567c
**Branch:** main

## OVERVIEW
Educational PyTorch project implementing Restricted Boltzmann Machines (RBM) from scratch — no Autograd, manual Contrastive Divergence. Two standalone demo scripts: binary RBM for MNIST image reconstruction, Softmax RBM for MovieLens collaborative filtering.

## STRUCTURE
```
Project/
├── src/
│   ├── RBM_MNIST.py    # Binary RBM: MNIST image reconstruction
│   └── RBM_MOVIE.py    # Softmax RBM: MovieLens rating prediction
├── src_v2/             # Upgraded paper-standard RBM implementations
│   ├── RBM_MNIST.py    # Bernoulli RBM with energy/free-energy + model persistence
│   ├── RBM_MOVIE.py    # Softmax RBM with Matplotlib visualization + model persistence
│   └── ALGORITHM.md    # Algorithm and formula reference document
├── pyproject.toml      # uv dependencies (Python >=3.12)
├── README.md           # Full Chinese documentation
├── rbm_mnist.pth       # Pre-trained checkpoint (gitignored)
├── data/               # Downloaded MNIST (gitignored)
└── ml-100k/            # Downloaded MovieLens 100k (gitignored)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Binary RBM (images) | `src/RBM_MNIST.py` | Self-contained: model + train + inference |
| Softmax RBM (ratings) | `src/RBM_MOVIE.py` | Self-contained: model + train + recommendation demo |
| Dependencies | `pyproject.toml` | torch/torchvision declared in pyproject.toml — uv sync installs automatically |
| Run commands | `README.md` | `uv run python src/RBM_MNIST.py` / `RBM_MOVIE.py` |
| Upgraded Bernoulli RBM | `src_v2/RBM_MNIST.py` | Paper-standard with energy + free energy |
| Upgraded Softmax RBM | `src_v2/RBM_MOVIE.py` | Paper-standard with Matplotlib visualization |
| Algorithm reference | `src_v2/ALGORITHM.md` | Formula derivations and code mapping |

## CODE MAP
| Symbol | Type | File | Role |
|--------|------|------|------|
| `RBM` | class | `src/RBM_MNIST.py` | Bernoulli RBM: `sample_h`, `sample_v`, `train_step`, `reconstruct` |
| `SoftmaxRBM` | class | `src/RBM_MOVIE.py` | Softmax RBM: `sample_h`, `sample_v` (softmax), `train_step` |

## CONVENTIONS
- **Indent**: 4 spaces | **No type hints** | **No docstrings**
- **Naming**: `UPPER_SNAKE_CASE` constants, `snake_case` functions/vars, `PascalCase` classes
- **Imports**: stdlib → blank line → third-party (os, numpy, torch order)
- **Comments**: Chinese (explanations) + English (code), section headers `# =====...=====`
- **Strings**: f-strings everywhere
- **No `if __name__ == "__main__"` guard** — scripts execute top-to-bottom

## ANTI-PATTERNS (THIS PROJECT)
- **NEVER use `loss.backward()` or `optimizer.step()`** — this project uses CD-k with manual matrix-multiplication gradients
- **ALWAYS wrap weights in `nn.Parameter`** — `state_dict()` silently drops plain tensors
- **ALWAYS wrap manual weight updates in `torch.no_grad()`** — `nn.Parameter += x` outside no_grad corrupts autograd graph
- **ALWAYS use `Mask` tensor for missing ratings** — never impute or zero-fill unseen MovieLens entries
- **ALWAYS specify `dtype=` in `torch.arange()`** — avoids type mismatch with softmax probabilities
- **NEVER change `.repeat()` args without checking dimensionality** — `unsqueeze(-1)` on 2D makes 3D; `.repeat(1,1,K)` needs exactly 3 args
- **Do NOT modify `src/`** — use `src_v2/` which now houses the paper-standard upgraded code

## UNIQUE STYLES
- **Manual weight updates** (no autograd): `self.W += lr * (pos_grad - neg_grad) / N` inside `torch.no_grad()`
- **Bernoulli sampling for activations**: `torch.bernoulli(p)` instead of deterministic ReLU/sigmoid outputs
- **Softmax visible units** (Movie): each movie = 5 neurons, `F.softmax(logits, dim=2)` for multi-class rating distribution
- **Energy-based learning**: minimizing reconstruction error (free energy), not classification loss
- **Script-style execution**: `uv run python src/<file>.py`, no package install, no entry points

## COMMANDS
```bash
# Setup
uv sync                                         # Install all dependencies (including torch/torchvision)
# Run
uv run python src/RBM_MNIST.py                  # MNIST image reconstruction demo
uv run python src/RBM_MOVIE.py                  # MovieLens recommendation demo
uv run python src_v2/RBM_MNIST.py              # Upgraded Bernoulli RBM with free energy display
uv run python src_v2/RBM_MOVIE.py              # Upgraded Softmax RBM with visualization
# Missing (add if needed)
# pytest                                        # No test infrastructure
# ruff check                                    # No linter configured
```

## NOTES
- **PyTorch declared in `pyproject.toml`**: `torch` and `torchvision` are now in `[project] dependencies` — `uv sync` installs everything automatically.
- **Python >=3.12 required** — PyTorch 2.11+ fully supports Python 3.12/3.13/3.14.
- **uv with Tsinghua mirror**: `pyproject.toml` points to `pypi.tuna.tsinghua.edu.cn` — change if outside China.
- **Data auto-downloads**: Both scripts download datasets on first run. `data/` and `ml-100k/` are gitignored.
- **`rbm_mnist.pth` at root**: Pre-trained checkpoint. `RBM_MNIST.py` reuses it to skip training. Gitignored.
- **No tests, no CI, no linting**: Pure research/educational code. Add `pytest + ruff` if evolving to production.
