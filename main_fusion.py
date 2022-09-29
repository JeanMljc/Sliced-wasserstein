import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import Sliced_Wasserstein as SW
import image_proscess as Ip
import operators as Op
import fusion_SW

### retrieval of HS image ###

mat = scipy.io.loadmat('/home/mallejac/Documents/Stage 3A/f130803t01p00r15rdn_refl_img_corr_roi2_denoised.mat')
IHS = mat.get("img")
IHS_lamdas = mat.get("lambdas")

# img = IHS
img = IHS[0:-1:4,0:-1:4,:]
img = img[:,:,8:]
img = img / np.max(img)

N = 128

### pre-processing on HS image ###

X_perf = Ip.im2mat(img)

yh = Op.S_spat(Op.B(X_perf,1),2)
ym = Op.L(X_perf,10)
yh_interpo = Ip.interpo(yh,2)

X0 = Ip.add_posvect(yh_interpo)
Yh = Ip.add_posvect(yh)
Ym = Ip.add_posvect(ym)

### Matrix Delta ###

lc = 1e-2
lp = 1
Del_e = SW.mat_delta(lc,lp,12)

### Fusion ###

n_pro = 10
n_steps = 10
tau = 10

X_n,D,GN = fusion_SW.fuse_ope(Yh,Ym,X0,lc,lp,n_pro,n_steps,tau)

img_final = X_n
img_fused = Ip.torgb(img_final)

### Plot RGB image ###

fig = plt.figure(figsize=(10,10))

gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

ax1.imshow(Ip.torgb(X_perf))
ax1.set_title('Image perfect RGB')
ax2.imshow(Ip.torgb(yh_interpo))
ax2.set_title('Image interpo RGB')
ax3.imshow(img_fused)
ax3.set_title('Image fused RGB')

plt.colorbar
plt.show()