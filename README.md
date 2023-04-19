# Fusion with Sliced-Wasserstein Distance 

This code implement an algorithm for fusion between hyperspectral and multispectral images. The hyperspectral image img_HS.npy is a $128\times 128$ image with $183$ spectral bands.

## Inverse Problem framework 

The goal of fusion is to retrieve image $\hat{X} \in \mathbb{R}^{l_h\times p_m}$ in the hight spatial and spectral dimension space from $Y_m \in \mathbb{R}^{l_m\times p_m}$ the multispectral image and $Y_h \in \mathbb{R}^{l_h\times p_h}$ the hyperspectral image. In order to solve this inverse problem, we compute a Wasserstein barycenter between hyperspectral and multispectral information:

$$\hat{X} = \operatorname*{argmin}_{X \in \mathbb{R}^{l_h\times p_m}} \frac{1}{2} \{ \widehat{SW_2}(Y_m,L X)^2 + \widehat{SW_2}(Y_h,X B S)^2 \}.$$

Spatial and spectral degradation operators $S$ and $L$ are used with Gaussian blur operator $B$. A gradient descent is used to do the minimisation. It is implemented in the function fusion_SW.py
## Image view as a point cloud distribution

An image with $d$ spectral bands can be see as a point cloud distribution in a $d$-dimensional space. A point cloud distribution of $n$ element noted $\mu$ is a uniform distribution with a support $U_{i \in [1,n]} \in \mathbb{R}^{d}$

$$\mu = \frac{1}{n} \sum_{k = 1}^{n} \delta_{U_k}\quad \nu = \frac{1}{n} \sum_{k = 1}^{n} \delta_{V_k}$$

![boat3](/figures/boat3.jpg)

For instance, a RGB image is equivalent to a 3-D points cloud distribution $d=3$. 
![RGB_points](/figures/RGB_points.png) 

## Sliced-Wasserstein distance 

The following Monte-Carlo approximation of the Sliced-Wasserstein distance [@Peyré, 2011] is used as distance:

$$ \widehat{SW_2}(\mu,\nu)^2 = \frac{1}{|\Psi|}\sum_{\theta \in \Psi} W_2(U_{|\theta},V_{|\theta})^2,$$

with $\mu$ and $\nu$ two distribution with support $U$ and $V$ respectively. $\Psi$ is a unit sphere of dimension $d$ and $W_2(\cdot,\cdot)$ the Wasserstein distance. The gradient has the following expression:

$$ \forall i \in [1,n], \qquad \frac{\partial \widehat{SW_2}(\mu,\nu)^{2}}{\partial U_i}=\frac{2}{|\Psi|} \sum_{\theta \in \Psi}\left\langle U_{i}-V_{s_{\theta}^{\star}(i)}, \theta\right\rangle \theta, $$

with $s_{\theta}^{\star}$ the optimal permutation. $s_v, s_u \in \Sigma_{n}$ denote the permutations that order the value of $\langle U_i | \theta \rangle$ and $\langle V_i | \theta \rangle$ respectively. Thoses equations are implemented in the file Sliced_Wasserstein.py

## References
@inproceedings{peyré,
  TITLE = {{Wasserstein Regularization of Imaging Problems}},
  AUTHOR = {Rabin, Julien and Peyr{\'e}, Gabriel},
  URL = {https://hal.archives-ouvertes.fr/hal-00591279},
  BOOKTITLE = {{ICIP 2011 : 2011 IEEE International Conference on Image Processing}},
  ADDRESS = {Bruxelles, Belgium},
  PAGES = {?},
  YEAR = {2011},
  MONTH = Sep,
  KEYWORDS = {Variational model ; Energy minimization ; Image regularization ; Gradient descent ; color and contrast modification},
  PDF = {https://hal.archives-ouvertes.fr/hal-00591279/file/wasserstein_variational_prox.pdf},
  HAL_ID = {hal-00591279},
  HAL_VERSION = {v1},
}

@article{flamary2021pot, author = {R{'e}mi Flamary and Nicolas Courty and Alexandre Gramfort and Mokhtar Z. Alaya and Aur{'e}lie Boisbunon and Stanislas Chambon and Laetitia Chapel and Adrien Corenflos and Kilian Fatras and Nemo Fournier and L{'e}o Gautheron and Nathalie T.H. Gayraud and Hicham Janati and Alain Rakotomamonjy and Ievgen Redko and Antoine Rolet and Antony Schutz and Vivien Seguy and Danica J. Sutherland and Romain Tavenard and Alexander Tong and Titouan Vayer}, title = {POT: Python Optimal Transport}, journal = {Journal of Machine Learning Research}, year = {2021}, volume = {22}, number = {78}, pages = {1-8}, url = {http://jmlr.org/papers/v22/20-451.html} }

@inproceedings{mifdal:hal-01620601,
  TITLE = {{HYPERSPECTRAL AND MULTISPECTRAL WASSERSTEIN BARYCENTER FOR IMAGE FUSION}},
  AUTHOR = {Mifdal, Jamila and Coll, Bartomeu and Courty, Nicolas and Froment, Jacques and Vedel, B{\'e}atrice},
  URL = {https://hal.science/hal-01620601},
  BOOKTITLE = {{IGARSS 2017}},
  ADDRESS = {Houston, United States},
  YEAR = {2017},
  MONTH = Jul,
  KEYWORDS = {Optimal Transport ; Wasser- stein Barycenter ; Image Fusion},
  PDF = {https://hal.science/hal-01620601/file/fusion.pdf},
  HAL_ID = {hal-01620601},
  HAL_VERSION = {v1},
}
