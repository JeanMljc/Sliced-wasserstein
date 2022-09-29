import numpy as np
import ot
from tqdm import tqdm

import image_proscess 
import operators as Op
import Sliced_Wasserstein as SW

def fuse_ope(Yh,Ym,I,lc,lp,n_proj,n_step,tau):
    r"""
     
    Parameters
    ----------
    Yh : array-like, shape (lh,ph) 
        Hyperspectral image with lh spectral bands and ph pixel per spectral bands
    Ym : array-like, shape (lm,pm) 
        Multispectral image with lm spectral bands and pm pixel per spectral bands
    I : array-like, shape (lh,pm) 
        Initiale image of gradient descent X^{(0)}
    lc : int 
         scalar \lambda_c
    lp : int
         scalar \lambda_p
    n_proj : int 
             number of projections
    n_step : int 
             number of steps in the gradient descent
    tau : int 
          learning rate

    Returns
    -------
    X_next : array-like, shape (lh,pm) 
             Fused image : Final image of gradient descent X^{(n_step)} 
    D : array-like, shape (1,n_step)
        Sliced-wasserstein distance at each step k
    GN : array-like, shape (1,n_step)           
         Norme of the gradient at each step k

    """
    (l1,ph) = Yh.shape
    (l2,pm) = Ym.shape
    lh = l1-2
    lm = l2-2
    
    nh = round(np.sqrt(ph))
    n = round(np.sqrt(pm))
    d = 2
    
    P = ot.sliced.get_random_projections(l1,n_proj).T
    D = np.zeros((n_step-1,1))
    GN = np.zeros(n_step-1)

    X_next = I

    for k in tqdm(range(1,n_step)):
        
        X_pre = X_next
        
        G0 = np.zeros((lh,pm))
        G1 = np.zeros((lh,pm))
        
        U = image_proscess.add_posvect(Op.L(X_pre[0:lh,:],lm))
        V = image_proscess.add_posvect(Op.S_spat(Op.B(X_pre[0:lh,:]),d))
        D[k-1] = 0.5*(SW.Sliced_WD(U,Ym,lc,lp,P[:,0:l2]) + SW.Sliced_WD(V,Yh,lc,lp,P[:,0:l1]))
        
        G0 = Op.LT(SW.Grad_SW(U,Ym,lc,lp,n_proj),lh)
        G1 = Op.B(Op.ST_spat(SW.Grad_SW(V,Yh,lc,lp,n_proj),d))
        
        G = 0.5*(G0 + G1)
        GN[k-1] = np.linalg.norm(G)
        X_next[0:lh,:] = X_pre[0:lh,:] - tau*G
 
    return X_next,D,GN