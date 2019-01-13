import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import scipy.misc

def imscatter(X, Rs, P, zoom=1):
    """
    Plot patches in specified locations in R2
    
    Parameters
    ----------
    X : ndarray (N, 2)
        The positions of the center of each patch in R2, 
        with each patch occupying [0, 1] x [0, 1]
    Rs : list of ndarray(2, 2)
        Rotation matrices for each patch
    P : ndarray (N, dim, dim, 3)
        An array of all of the patches
    
    """
    #https://matplotlib.org/examples/api/demo_affine_image.html
    ax = plt.gca()
    for i in range(P.shape[0]):
        p = P[i, :, :, :]
        im = ax.imshow(p, interpolation='none', extent=(-0.5, 0.5, -0.5, 0.5))
        m = np.eye(3)
        m[0:2, 0:2] = Rs[i]
        m[0:2, 2] = X[i, :]
        trans = mtransforms.Affine2D()
        trans.set_matrix(m)
        im.set_transform(trans + ax.transData)
    plt.xlim([np.min(X[:, 0])-1, np.max(X[:, 0])+1])
    plt.ylim([np.min(X[:, 1])-1, np.max(X[:, 1])+1])
    ax.set_xticks([])
    ax.set_yticks([])

def readImage(filename):
    I = scipy.misc.imread(filename)
    I = np.array(I, dtype=np.float32)/255.0
    return I

def writeImage(I, filename):
    IRet = I*255.0
    IRet[IRet > 255] = 255
    IRet[IRet < 0] = 0
    IRet = np.array(IRet, dtype=np.uint8)
    scipy.misc.imsave(filename, IRet)

def getPatchesColor(I, d):
    """
    Given an image I, return all of the dim x dim patches in I
    Parameters
    ----------
    Given an image I, return all of the dim x dim patches in I
    Parameters
    ----------
    I: ndarray(M, N, 3)
        An M x N x3 color image array
    d: int
        The dimension of the square patches
    
    Returns
    -------
    P: ndarray(ceil(M/d), ceil(N/d), d, d, 3)
        Array of all patches
    """
    M = int(np.ceil(float(I.shape[0])/d))
    N = int(np.ceil(float(I.shape[1])/d))
    P = np.zeros((M, N, d, d, 3))
    for i in range(M):
        for j in range(N):
            patch = I[i*d:(i+1)*d, j*d:(j+1)*d, :]
            if patch.shape[0] < d or patch.shape[1] < d:
                p = np.zeros((d, d, 3))
                p[0:patch.shape[0], 0:patch.shape[1], :] = patch
                patch = p
            P[i, j, :, :, :] = patch
    return P

def testPlottingPieces():
    """
    Come up with a bunch of random rotations for each square piece
    and plot the result
    """
    plt.figure(figsize=(9, 9))
    I = readImage("melayla.jpg")
    d = 28
    PColor = getPatchesColor(I, d)
    X, Y = np.meshgrid(np.arange(PColor.shape[1]), np.arange(PColor.shape[0]))
    Y = PColor.shape[0]-Y
    X = np.array([X.flatten(), Y.flatten()])
    X = X.T
    PColor = np.reshape(PColor, (PColor.shape[0]*PColor.shape[1], d, d, 3))
    Rs = []
    for i in range(X.shape[0]):
        R = np.random.randn(2, 2)
        U, _, _ = np.linalg.svd(R)
        Rs.append(U)
    imscatter(X, Rs, PColor)
    plt.show()

if __name__ == '__main__':
    testPlotting()