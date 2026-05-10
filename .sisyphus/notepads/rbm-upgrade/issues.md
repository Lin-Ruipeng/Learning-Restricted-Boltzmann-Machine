# RBM Upgrade Issues

## Known Potential Issues
- PyTorch not in pyproject.toml - might need manual install
- Matplotlib might require GUI backend on Windows
- MovieLens download might fail if network restricted
- Training takes time (MNIST ~10 epochs on CPU, MOVIE ~100 epochs on 200x500 subset)
