# Fusion with Sliced-Wasserstein Distance 

This code implement an algorithm for fusion between hyperspectral and multispectral images. 

## Inverse Problem framework 

$$\hat{X} = \operatorname*{argmin}_{X \in \mathbb{R}^{l_h\times p_m}} \Big\{ ||Y_m-LX||^2 + ||Y_h - XBS||^2 \Big\}$$