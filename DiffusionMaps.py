import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import scipy.io as sio
from scipy import sparse
import time

def getSSM(X):
    """
    Compute a Euclidean self-similarity image between a set of points
    :param X: An Nxd matrix holding the d coordinates of N points
    :return: An NxN self-similarity matrix
    """
    D = np.sum(X**2, 1)[:, None]
    D = D + D.T - 2*X.dot(X.T)
    D[D < 0] = 0
    D = 0.5*(D + D.T)
    D = np.sqrt(D)
    return D

def getW(D, K, Mu = 0.5):
    """
    Return affinity matrix
    [1] Wang, Bo, et al. "Similarity network fusion for aggregating data types on a genomic scale." 
        Nature methods 11.3 (2014): 333-337.
    :param D: Self-similarity matrix
    :param K: Number of nearest neighbors
    """
    #W(i, j) = exp(-Dij^2/(mu*epsij))
    DSym = 0.5*(D + D.T)
    np.fill_diagonal(DSym, 0)

    Neighbs = np.partition(DSym, K+1, 1)[:, 0:K+1]
    MeanDist = np.mean(Neighbs, 1)*float(K+1)/float(K) #Need this scaling
    #to exclude diagonal element in mean
    #Equation 1 in SNF paper [1] for estimating local neighborhood radii
    #by looking at k nearest neighbors, not including point itself
    Eps = MeanDist[:, None] + MeanDist[None, :] + DSym
    Eps = Eps/3
    W = np.exp(-DSym**2/(2*(Mu*Eps)**2))
    return W

def getDiffusionMap(SSM, Kappa, t = -1, includeDiag = True, thresh = 5e-4, NEigs = 51):
    """
    :param SSM: Metric between all pairs of points
    :param Kappa: Number in (0, 1) indicating a fraction of nearest neighbors
                used to autotune neighborhood size
    :param t: Diffusion parameter.  If -1, do Autotuning
    :param includeDiag: If true, include recurrence to diagonal in the markov
        chain.  If false, zero out diagonal
    :param thresh: Threshold below which to zero out entries in markov chain in
        the sparse approximation
    :param NEigs: The number of eigenvectors to use in the approximation
    """
    N = SSM.shape[0]
    #Use the letters from the delaPorte paper
    K = getW(SSM, int(Kappa*N))
    if not includeDiag:
        np.fill_diagonal(K, np.zeros(N))
    RowSumSqrt = np.sqrt(np.sum(K, 1))
    DInvSqrt = sparse.diags([1/RowSumSqrt], [0])

    #Symmetric normalized Laplacian
    Pp = (K/RowSumSqrt[None, :])/RowSumSqrt[:, None]
    Pp[Pp < thresh] = 0
    Pp = sparse.csr_matrix(Pp)

    lam, X = sparse.linalg.eigsh(Pp, NEigs, which='LM')
    lam = lam/lam[-1] #In case of numerical instability

    #Check to see if autotuning
    if t > -1:
        lamt = lam**t
    else:
        #Autotuning diffusion time
        lamt = np.array(lam)
        lamt[0:-1] = lam[0:-1]/(1-lam[0:-1])

    #Do eigenvector version
    V = DInvSqrt.dot(X) #Right eigenvectors
    M = V*lamt[None, :]
    return M/RowSumSqrt[:, None] #Put back into orthogonal Euclidean coordinates

def getPinchedCircle(N):
    t = np.linspace(0, 2*np.pi, N+1)[0:N]
    x = np.zeros((N, 2))
    x[:, 0] = (1.5 + np.cos(2*t))*np.cos(t)
    x[:, 1] = (1.5 + np.cos(2*t))*np.sin(t)
    return x

def getTorusKnot(N, p, q):
    t = np.linspace(0, 2*np.pi, N+1)[0:N]
    X = np.zeros((N, 3))
    r = np.cos(q*t) + 2
    X[:, 0] = r*np.cos(p*t)
    X[:, 1] = r*np.sin(p*t)
    X[:, 2] = -np.sin(q*t)
    return X

if __name__ == '__main__':
    zeroReturn = True
    N = 400
    X = getPinchedCircle(N)
    sio.savemat("X.mat", {"X":X})
    tic = time.time()
    SSMOrig = getSSM(X)
    toc = time.time()
    print("Elapsed time SSM: ", toc - tic)
    Kappa = 0.1

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], 40, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
    plt.axis('equal')
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor((0.15, 0.15, 0.15))
    plt.title("Original Pinched Circle")
    plt.subplot(122)
    plt.imshow(SSMOrig, interpolation = 'nearest', cmap = 'afmhot')
    plt.title("Original SSM")
    plt.savefig("Diffusion0.svg", bbox_inches = 'tight')

    ts = [100]
    for t in ts:
        plt.clf()
        M = getDiffusionMap(SSMOrig, Kappa, t)
        SSM = getSSM(M)
        plt.subplot(121)
        X = M[:, [-2, -3]]
        plt.scatter(X[:, 0], X[:, 1], 40, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
        plt.title("2D Diffusion Map, t = %i, $\kappa = %g$"%(t, Kappa))
        plt.axis('equal')
        plt.xlim([np.min(X[:, 0]) - 0.001, np.max(X[:, 0]) + 0.001])
        plt.ylim([np.min(X[:, 1]) - 0.001, np.max(X[:, 1]) + 0.001])
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor((0.15, 0.15, 0.15))
        plt.subplot(122)
        plt.imshow(SSM, interpolation = 'nearest', cmap = 'afmhot')
        plt.title("Diffusion Distance")
        plt.savefig("Diffusion%i.svg"%t, bbox_inches = 'tight')
