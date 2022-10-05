import numpy as np
import ot 
import scipy.io
import matplotlib.pyplot as plt

import Sliced_Wasserstein as SW
import image_proscess as Ip
import operators as Op
import fusion_SW
import metrics

### retrieval of HS image ###

img = np.load("img_HS.npy")
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
tau = 50

X_n,D,GN = fusion_SW.fuse_ope(Yh,Ym,X0,lc,lp,n_pro,n_steps,tau)

img_final = X_n
img_fused = Ip.torgb(img_final)
img_clean = img_final[:-2]

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

plt.plot(D,label='SWD')
plt.legend()
plt.title('SWD')
plt.show()

### Plot RGB histogram ###

colors = ['r','g','b']

fig, ax = plt.subplots(3,figsize=(10,10))
ax[0].hist(Ip.im2mat(Ip.torgb(X_perf)).T,bins=np.arange(0,1.1,0.05), color=colors, stacked=1)
ax[0].set_title("X_perf")
ax[1].hist(Ip.im2mat(Ip.torgb(yh_interpo)).T,bins=np.arange(0,1.1,0.05), color=colors, stacked=1)
ax[1].set_title("yh_interpo")
ax[2].hist(Ip.im2mat(img_fused).T,bins=np.arange(0,1.1,0.05), color=colors, stacked=1)
ax[2].set_title("img_fused")
plt.show()

### Metrics ###

Pro = ot.sliced.get_random_projections(183,100).T

print("Sliced WD(X0,X_perfect)",SW.Sliced_WD(yh_interpo,X_perf,lc,lp,Pro))
print("Sliced WD(X_n,X_perfect)",SW.Sliced_WD(img_clean,X_perf,lc,lp,Pro))

print("Norm2(X0,X_perfect)",Ip.MSE_image(yh_interpo,X_perf))
print("Norm2(X_n,X_perfect)",Ip.MSE_image(img_clean,X_perf))

v_ssim = metrics.SSIM(X_perf,img_clean)

plt.plot(v_ssim)
plt.title("SSIM index")
plt.xlabel("spectral band l")
plt.show()

img_SAM = metrics.SAM(img_clean,X_perf)

plt.imshow(img_SAM)
plt.colorbar()
plt.show()