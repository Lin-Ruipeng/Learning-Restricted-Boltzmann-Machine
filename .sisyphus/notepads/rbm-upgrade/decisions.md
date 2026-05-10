# RBM Upgrade Decisions

## Checkpoint paths
- MNIST: `src_v2/rbm_mnist_v2.pth` (separate from original `rbm_mnist.pth`)
- MOVIE: `src_v2/rbm_movie_v2.pth`

## CD-k
- Both files use CD-1 (1-step Gibbs)

## Visualization
- MOVIE: Bar chart (top-8 recommendations) + Heatmap (user-movie matrix)
- MNIST: Keep original 3-sample layout
