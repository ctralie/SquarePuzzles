import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy.misc

def imscatter(X, P, zoom=1):
    """
    Plot patches in specified locations in R2
    
    Parameters
    ----------
    X : ndarray (N, 2)
        The positions of each patch in R2
    P : ndarray (N, dim, dim, 3)
        An array of all of the patches
    
    """
    #https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points
    ax = plt.gca()
    for i in range(P.shape[0]):
        patch = np.array(P[i, :, :, :])
        x, y = X[i, :]
        im = OffsetImage(patch, zoom=zoom, cmap = 'gray')
        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
        ax.add_artist(ab)
    ax.update_datalim(X)
    ax.autoscale()
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

if __name__ == '__main__':
    I = readImage("melayla.jpg")
    d = 28
    PColor = getPatchesColor(I, d)
    X, Y = np.meshgrid(np.arange(PColor.shape[1]), np.arange(PColor.shape[0]))
    Y = PColor.shape[0]-Y
    X = np.array([X.flatten(), Y.flatten()])
    X = X.T
    PColor = np.reshape(PColor, (PColor.shape[0]*PColor.shape[1], d, d, 3))
    imscatter(X, PColor)
    plt.show()