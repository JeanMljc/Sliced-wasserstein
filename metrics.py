import image_proscess as Ip
import numpy as np
import spectral

from skimage import metrics

def SSIM(IA,IB):
    Ia = Ip.mat2im(IA)
    Ib = Ip.mat2im(IB)

    value_SSIM = np.zeros(Ia.shape[2])

    for i in range(Ia.shape[2]):
        value_SSIM[i] = metrics.structural_similarity(Ia[:,:,i],Ib[:,:,i])
    return value_SSIM

def SAM(Xp,X):
    ne = np.int(np.sqrt(Xp.shape[1]))
    pix = np.zeros(ne**2)
    X_img = Ip.mat2im(X)
    Xpe = Xp.T

    for k in range(ne**2):
        sb = Xpe[k,:].reshape((1,-1))
        tab = Ip.im2mat(spectral.spectral_angles(X_img,sb)).T
        pix[k] = tab[k]
    img_SAM = pix.reshape((ne,ne))
    return img_SAM