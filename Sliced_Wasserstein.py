import numpy as np
import ot

def mat_delta(lc,lp,dim):
    r"""

    Built matrix \Delta from coefficients \lambda_c and \lambda_p and the dimension dim

    Parameters
    ----------
    lc : int 
         scalar \lambda_c
    lp : int
         scalar \lambda_p
    dim : dimension of matrix \Delta

    Returns
    -------
    Delta :  array-like, shape (dim, dim)
    """
    vec = np.zeros(dim)
    vec[:dim-2] = lc
    vec[dim-2:] = lp
    Delta = np.diag(vec)
    return Delta
    
def Sliced_WD(A,B,lc,lp,P): ## add lc and lp
    r"""

    Compute 2-sliced Wasserstein (using l2-norm for cost function) distance between point clouds A and B 
    with projection P
    
    Parameters
    ----------
    A : array-like, shape (d,N) 
        d-Dimensional point clouds of N elements 
    B : array-like, shape (d,N) 
        d-Dimensional point clouds of N elements
    lc : int 
         scalar \lambda_c
    lp : int
         scalar \lambda_p
    P : array-like, shape (n_proj,d) 
        Array containing n_proj Projections  
    

    Returns
    -------
    Sw : scalar 
         Sliced Wasserstein distance between A and B 
    """
    (l,N) = A.shape    
    n_proj = P.shape[0]
    
    Delta = mat_delta(lc,lp,l)
    ad = P @ Delta @ A
    bd = P @ Delta @ B  
    Wd = np.zeros((n_proj,1))
    
    for i in range(n_proj):
        Wd[i] = ot.emd2_1d(ad[i,:],bd[i,:],p=2) # norm 2
    Sw = np.mean(Wd)
    return Sw

def Grad_SW(A,B,lc,lp,n_proj=10):
    r"""

    Compute the gradient of 2-sliced Wasserstein distance between point clouds A and B 
    with respect to A^(c) (spectral component of A) with n_proj projections
    
    Parameters
    ----------
    A : array-like, shape (d,N) 
        d-Dimensional point clouds of N elements 
    B : array-like, shape (d,N) 
        d-Dimensional point clouds of N elements  
    lc : int 
         scalar \lambda_c
    lp : int
         scalar \lambda_p
    n_proj : int 
             number of projections

    Returns
    -------
    Grad : array-like, shape (d,N)
         Gradient of Sliced Wasserstein distance between A and B with respect to A
    """
    
    (l,N) = A.shape
    
    P = ot.sliced.get_random_projections(l,n_proj).T
    Delta = mat_delta(lc,lp,l)

    ad = Delta @ A
    bd = Delta @ B 
    ap = P @ ad
    bp = P @ bd
    Grad = np.zeros((l-2,N))
    sig_star = []
        
    for i in range(n_proj):
        sig_a_inv = np.argsort(np.argsort(ap[i,:]))
        sig_b = np.argsort(bp[i,:])
        sig_star.append(sig_b[sig_a_inv])

    for k in range(N):
        G = np.zeros((l-2,n_proj))
        for i in range(0,n_proj):
            sig_star_i = sig_star[i]
            cout = ad[:,k]-bd[:,sig_star_i[k]]
            G[:,i] = (cout @ P[i,:])*P[i,0:l-2]
        Grad[:,k] = (2/n_proj)*np.sum(G,axis=1)
    return Grad