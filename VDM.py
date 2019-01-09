"""
An implementation of vector diffusion maps of point clouds in R^d
"""
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

def getdists(X, i, exclude_self=True):
    """
    Get the distances from the ith point in X to the rest of the points
    """
    x = X[i, :]
    x = x[None, :]
    dsqr = np.sum(x**2) + np.sum(X**2, 1)[None, :] - 2*x.dot(X.T)
    dsqr = dsqr.flatten()
    if exclude_self:
        dsqr[i] = np.inf # Exclude the point itself
    return dsqr

def getGreedyPerm(X):
    N = X.shape[0]
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = getdists(X, 0, exclude_self=False)
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, getdists(X, idx, exclude_self=False))
    return (perm, lambdas)

def getLocalPCA(X, eps_pca, gammadim = 0.9, K_usqr = lambda u: np.exp(-5*u)*(u <= 1)):
    """
    Estimate a basis to the tangent plane TxM at every point
    Parameters
    ----------
    X: ndarray(N, p)
        A Euclidean point cloud in p dimensions
    eps_pca: float
        Square of the radius of the neighborhood to consider at each 
        It is assumed that it is such that every point will have at least
        d nearest neighbors, where d is the intrinsic dimension
    gammadim: float
        The explained variance ratio below which to assume all of
        the tangent space is captured.
        Used for estimating the local dimension d
    K_usqr: function float->float
        A C2 positive monotonic decreasing function of a squared argument
        with support on the interval [0, 1]
    
    Returns
    -------
    bases: list of ndarray(d, p)
        All of the orthonormal basis matrices for each point
    """
    N = X.shape[0]
    bases = []
    ds = np.zeros(N) # Local dimension estimates
    for i in range(N):
        dsqr = getdists(X, i)
        Xi = X[dsqr <= eps_pca, :] - X[i, :]
        di = K_usqr(dsqr[dsqr <= eps_pca]/eps_pca)
        if di.size == 0:
            bases.append(np.zeros((Xi.shape[1], 0)))
            continue
        Bi = (Xi*di[:, None]).T # Transpose to be consistent with Singer paper
        U, s, _ = linalg.svd(Bi) # *Variances* lie along diagonal of S in U*S*V^T (versus typo in Singer paper)
        ds[i] = np.argmax(np.cumsum(s) / np.sum(s) > gammadim) + 1
        bases.append(U)
    d = int(np.median(ds)) # Dimension estimate
    if (d == 0):
        d = 1
    print("Dimension %i"%d)
    for i, U in enumerate(bases):
        if U.shape[1] < d:
            print("Warning: Insufficient rank for epsilon at point %i"%i)
            # There weren't enough nearest neighbors to determine a basis
            # up to the estimated intrinsic dimension, so recompute with 
            # 2*d nearest neighbors
            dsqr = getdists(X, i)
            idx = np.argsort(dsqr)[0:min(2*d, X.shape[0])]
            Xi = X[idx, :] - X[i, :]
            U, _, _ = linalg.svd(Xi.T)
            bases[i] = U[:, 0:d]
        else:
            bases[i] = U[:, 0:d]
    return bases

def getTorusKnot(p, q, pt):
    """
    Return a p-q torus knot parameterized on [0, 1]
    Parameters
    ----------
    p: int
        p parameter for torus knot
    q: int
        q parameter for torus knot
    pt: ndarray(N)
        Parameterization on the interval [0, 1]
    """
    N = len(pt)
    t = 2*np.pi*pt
    X = np.zeros((N, 3))
    r = np.cos(q*t) + 2
    X[:, 0] = r*np.cos(p*t)
    X[:, 1] = r*np.sin(p*t)
    X[:, 2] = -np.sin(q*t)
    return X

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def testVDM3D():
    np.random.seed(10)
    X = getTorusKnot(2, 3, np.linspace(0, 1, 200))
    bases = getLocalPCA(X, eps_pca=0.1)

    ax = plt.subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    (perm, _) = getGreedyPerm(X)
    for idx in perm[0:40]:
        x0 = X[idx, :]
        u = bases[idx][:, 0]
        x = np.concatenate((x0[None, :], x0[None, :] + 0.3*u[None, :]), 0)
        a = Arrow3D(x[:, 0], x[:, 1], x[:, 2], mutation_scale=20, \
                lw=2, arrowstyle="-|>", color="r")
        ax.add_artist(a)
    plt.show()

if __name__ == '__main__':
    testVDM3D()