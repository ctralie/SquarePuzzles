"""
An implementation of vector diffusion maps of point clouds in R^d
"""
import numpy as np
import numpy.linalg as linalg
from scipy import sparse 
from scipy.sparse.linalg import lsqr, cg, eigsh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


"""#####################################################
                VECTOR DIFFUSION MAPS
#####################################################"""


def getdists_sqr(X, i, exclude_self=True):
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
    ds = getdists_sqr(X, 0, exclude_self=False)
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, getdists_sqr(X, idx, exclude_self=False))
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
        dsqr = getdists_sqr(X, i, exclude_self=True)
        Xi = X[dsqr <= eps_pca, :] - X[i, :]
        di = K_usqr(dsqr[dsqr <= eps_pca]/eps_pca)
        if di.size == 0:
            bases.append(np.zeros((Xi.shape[1], 0)))
            continue
        Bi = (Xi*di[:, None]).T # Transpose to be consistent with Singer 
        U, s, _ = linalg.svd(Bi)
        s = s**2
        cumvar_ratio = np.cumsum(s) / np.sum(s)
        ds[i] = np.argmax(cumvar_ratio > gammadim) + 1
        bases.append(U)
    d = int(np.median(ds)) # Dimension estimate
    print(np.mean(ds))
    if (d == 0):
        d = 1
    print("Dimension %i"%d)
    for i, U in enumerate(bases):
        if U.shape[1] < d:
            print("Warning: Insufficient rank for epsilon at point %i"%i)
            # There weren't enough nearest neighbors to determine a basis
            # up to the estimated intrinsic dimension, so recompute with 
            # 2*d nearest neighbors
            dsqr = getdists_sqr(X, i)
            idx = np.argsort(dsqr)[0:min(2*d, X.shape[0])]
            Xi = X[idx, :] - X[i, :]
            U, _, _ = linalg.svd(Xi.T)
            bases[i] = U[:, 0:d]
        else:
            bases[i] = U[:, 0:d]
    return bases


def getConnectionLaplacian(ws, Os, N, k):
    """
    Given a set of weights and corresponding orientation matrices,
    return k eigenvectors of the connection Laplacian.
    Parameters
    ----------
    ws: ndarray(M, 3)
        An array of weights for each included edge
    Os: M-element list of (ndarray(d, d))
        The corresponding orthogonal transformation matrices
    N: int
        Number of vertices in the graph
    k: int
        Number of eigenvectors to compute
    
    Returns
    -------
    w: ndarray(k)
        Array of k eigenvalues
    v: ndarray(N*d, k)
        Array of the corresponding eigenvectors
    """
    d = Os[0].shape[0]
    W = sparse.coo_matrix((ws[:, 2], (ws[:, 0], ws[:, 1])), shape=(N, N)).tocsr()
    ## Step 1: Create D^-1 matrix
    deg = np.array(W.sum(1)).flatten()
    deg[deg == 0] = 1
    deginv = 1.0/deg
    I = (np.arange(N)[:, None]*np.ones((1, d))).flatten()
    V = (deginv[:, None]*np.ones((1, d))).flatten()
    DInv = sparse.coo_matrix((V, (I, I)), shape=(N*d, N*d)).tocsr()

    ## Step 2: Create S matrix
    I = []
    J = []
    V = []
    Jsoff, Isoff = np.meshgrid(np.arange(d), np.arange(d))
    Jsoff = Jsoff.flatten()
    Isoff = Isoff.flatten()
    for idx in range(ws.shape[0]):
        [i, j, wij] = ws[idx, :]
        wijOij = wij*Os[idx]
        wijOij = (wijOij.flatten()).tolist()
        I.append((i*d + Isoff).tolist())
        J.append((j*d + Jsoff).tolist())
        V.append(wijOij)
    I, J, V = np.array(I).flatten(), np.array(J).flatten(), np.array(V).flatten()
    S = sparse.coo_matrix((V, (I, J)), shape=(N*d, N*d)).tocsr()
    DInvS = DInv.dot(S)
    return eigsh(DInvS)


def getConnectionLaplacianPC(X, eps_pca, gammadim, eps):
    pass


"""#####################################################
                        EXAMPLES
#####################################################"""

def testConnectionLaplacianSquareGrid(N, seed=0):
    """
    Randomly rotate square tiles on an NxN grid
    Add increasing amounts of noise to some of the Os
    """
    np.random.seed(seed)
    thetas = 2*np.pi*np.random.rand(N, N)
    ws = []
    Os = []
    for i in range(N):
        for j in range(N):
            idxi = i*N+j
            for di in [-1, 1]:
                if i + di < 0 or i + di >= N:
                    continue
                for dj in [-1, 1]:
                    if j + dj < 0 or j + dj >= N:
                        continue
                    idxj = (i+di)*N+j+dj
                    ws.append([idxi, idxj, 1.0])
                    # Oij moves vectors from j to i
                    theta = thetas[i+di, j+dj] - thetas[i, j]
                    c = np.cos(theta)
                    s = np.sin(theta)
                    Oij = np.array([[c, -s], [s, c]])
                    Os.append(Oij)
    ws = np.array(ws)
    w, v = getConnectionLaplacian(ws, Os, N**2, 2)
    print(w)
    ax = plt.gca()
    for idx in range(N*N):
        i, j = np.unravel_index(idx, (N, N))
        vidx = v[idx*2:(idx+1)*2, 0]
        # Bring into world coordinates
        c = np.cos(thetas[i, j])
        s = np.sin(thetas[i, j])
        Oij = np.array([[c, -s], [s, c]])
        vidx = Oij.dot(vidx)
        #vidx = vidx/np.sqrt(np.sum(vidx**2))
        #vidx = Oij[:, 0]*0.5
        plt.scatter([j], [i], 20, 'k')
        ax.arrow(j, i, vidx[1], vidx[0])
    plt.show()



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
    X = getTorusKnot(2, 3, np.random.rand(1000))
    X += 0.05*np.random.randn(X.shape[0], X.shape[1])
    bases = getLocalPCA(X, eps_pca=0.1, gammadim=0.8)

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
    #testVDM3D()
    testConnectionLaplacianSquareGrid(5)