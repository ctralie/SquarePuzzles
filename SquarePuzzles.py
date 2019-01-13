import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import scipy.misc
from VDM import *


"""#####################################################
    LOADING/SAVING AND EXTRACTING/PLOTTING PATCHES
#####################################################"""


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

def plotPatches(ax, X, Rs, P, zoom=1):
    """
    Plot patches in specified locations in R2
    with hints from
    https://matplotlib.org/examples/api/demo_affine_image.html
    
    Parameters
    ----------
    ax : matplotlib axis
        The axis on which to plot the collection of patches
    X : ndarray (N, 2)
        The positions of the center of each patch in R2, 
        with each patch occupying [0, 1] x [0, 1]
    Rs : list of ndarray(2, 2)
        Rotation matrices for each patch
    P : ndarray (N, dim, dim, 3)
        An array of all of the patches
    """
    for i in range(P.shape[0]):
        p = P[i, :, :, :]
        im = ax.imshow(p, interpolation='none', extent=(-0.5, 0.5, -0.5, 0.5))
        m = np.eye(3)
        m[0:2, 0:2] = Rs[i]
        m[0:2, 2] = X[i, :]
        trans = mtransforms.Affine2D()
        trans.set_matrix(m)
        im.set_transform(trans + ax.transData)
    ax.set_xlim([np.min(X[:, 0])-1, np.max(X[:, 0])+1])
    ax.set_ylim([np.min(X[:, 1])-1, np.max(X[:, 1])+1])
    ax.set_xticks([])
    ax.set_yticks([])

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
    plotPatches(plt.gca(), X, Rs, PColor)
    plt.show()

"""#####################################################
    MAHALANOBIS GRADIENT COMPATABILITY (MGC)
    "Jigsaw Puzzles with Pieces of Unknown Orientation"
    by Andrew C. Gallagher
#####################################################"""

DUMMY_DIRS = np.array([ [1, 1, 1], [-1, -1, -1], [0, 0, 1], \
                        [0, 1, 0], [1, 0, 0], [-1, 0, 0], \
                        [0, -1, 0], [0, 0, -1], [0, 0, 0]])

def getMGCLR(pL, pR):
    """
    Compute the Mahalanobis Gradient Compatability (MGC) from 
    the left patch to the right patch
    Parameters
    ----------
    pL: ndarray(p, p, 3)
        Left color patch
    pR: ndarray(p, p, 3)
        Right color patch
    Returns
    -------
    mgc: float
        Directional MGC measure between left and right patch
    """
    GiL = np.array(pL[:, -1, :] - pL[:, -2, :], dtype=float)
    GiL = np.concatenate((GiL, DUMMY_DIRS), 0)
    muiL = np.mean(GiL, 0)
    diff = GiL-muiL
    S = (1.0/(GiL.shape[0]-1))*((diff.T).dot(diff))
    SInv = np.linalg.inv(S)
    GijLR = np.array(pR[:, 0, :] - pL[:, -1, :], dtype=float)
    DLR = GijLR - muiL[None, :]
    return np.sum((DLR.dot(SInv))*DLR)

def getMGC(ppL, ppR):
    """
    Compute the symmetric Mahalanobis Gradient Compatability (MGC)
    between two patches by summing the MGC from the left to the
    right and the MGC from the right to the left
    Parameters
    ----------
    ppL: ndarray(p, p, 3)
        Left color patch
    ppR: ndarray(p, p, 3)
        Right color patch
    Returns
    -------
    mgc_symmetric: float
        Directional MGC measure between left and right patch
    """
    pL = np.array(ppL)
    pR = np.array(ppR)
    # First get from left to right patch
    res = getMGCLR(pL, pR)
    # Now switch roles of left and right patches
    res + getMGCLR(np.fliplr(pR), np.fliplr(pL))
    return res

def getRGB(pL, pR):
    """
    Return the summed rgb difference between the boundary of two patches
    Parameters
    ----------
    pL: ndarray(p, p, 3)
        Left color patch
    pR: ndarray(p, p, 3)
        Right color patch
    Returns
    -------
    rgbdiff: float
        Sum of absolute differences between adjacent pixels
        on the boundary of the overlap
    """
    diff = np.array(pL[:, -1, :] - pR[:, 0, :], dtype=float)
    return np.sum(np.abs(diff))

def getOptimalPairRotation(ppL, ppR, evalfn = getMGC):
    """
    Given a patch to the left and a patch to the right, 
    figure out how to optimally rotate both so that
    they align
    Parameters
    ----------
    pL: ndarray(p, p, 3)
        Left color patch
    pR: ndarray(p, p, 3)
        Right color patch
    evalfn: function(ndarray(p, p, 3), ndarray(p, p, 3))
        A function to compare the similarity of a left patch
        to a right patch
    Returns
    -------
    tuple (int, int):
        The number of 90 degree CCW rotations of the left
        and right patch, respectively
    """
    pL = np.array(ppL)
    pR = np.array(ppR)
    scores = []
    for rotl in range(4):
        pL = np.swapaxes(pL, 0, 1)
        pL = np.fliplr(PL)
        for rotr in range(4):
            
    pass

def testMGC(NToPlot = 5):
    """
    Compare the top 5 patches retrieved by MGC and the top
    5 patches retrieved by RGB (like Figure 2 in the Gallagher paper)
    """
    res = 2
    plt.figure(figsize=(res*(2*NToPlot+3), res*2))

    I = readImage("melayla.jpg")
    d = 28
    Ps = getPatchesColor(I, d)
    Ps = np.reshape(Ps, (Ps.shape[0]*Ps.shape[1], d, d, 3))
    N = Ps.shape[0]
    dMGC = np.zeros(N)
    dRGB = np.zeros(N)

    for p0idx in range(N):
        p0 = Ps[p0idx, :, :, :]
        # Compute MGC and RGB similarity
        for i in range(N):
            dMGC[i] = getMGC(p0, Ps[i])
            dRGB[i] = getRGB(p0, Ps[i])
        idxmgc = np.argsort(dMGC)
        idxrgb = np.argsort(dRGB)
        if idxmgc[0] == idxrgb[0]:
            # Only show the results where MGC and RGB are different
            continue

        # Now plot the results
        plt.clf()
        plt.subplot(2, NToPlot+1, 1)
        plt.imshow(p0)
        plt.title("%i"%p0idx)
        for i in range(NToPlot):
            # Most similar MGC Patches
            plt.subplot(2, NToPlot+1, i+2)
            I2 = np.zeros((d, d*2, 3))
            I2[:, 0:d, :] = p0
            I2[:, d::, :] = Ps[idxmgc[i], :, :, :]
            plt.imshow(I2)
            plt.title("MGC %i (%.3g)"%(idxmgc[i], dMGC[idxmgc[i]]))
            plt.axis('off')

            # Most similar RGB Patches
            plt.subplot(2, NToPlot+1, NToPlot+1+i+2)
            I2[:, d::, :] = Ps[idxrgb[i], :, :, :]
            plt.imshow(I2)
            plt.title("RGB %i (%.3g)"%(idxrgb[i], dRGB[idxrgb[i]]))
            plt.axis('off')
        plt.savefig("%i.png"%p0idx, bbox_inches='tight')

"""#####################################################
            Type 3 Puzzles (Rotation Only)
#####################################################"""

def rotateByZMod4(I, g):
    """
    Apply the cyclic group rotation by 90 degree increments
    Parameters
    Parameters
    ----------
    I: ndarray(M, N, 3)
        A color image
    g: int
        Number of CCW increments by which to rotate I
    Returns
    --------
    I: ndarray(M or N, N or M, 3)
        The rotated image
    """
    IRet = np.array(I)
    for i in range(g%4):
        IRet = np.swapaxes(IRet, 0, 1)
        IRet = np.flipud(IRet)
    return IRet

def solveType3Puzzle(Ps):
    """
    Solve a type 3 (rotations only) puzzle
    Parameters
    ----------
    Ps: ndarray(M, N, d, d, 3)
        An MxN grid of dxd patches
    Returns
    -------
    Rs: MxN list of ndarray(2, 2)
        A list of rotation matrices to apply to each patch
        so that they're in the proper orientation
    """
    M, N = Ps.shape[0], Ps.shape[1]
    NP = M*N
    ws = []
    Os = []
    for i in range(NP):
        i1, j1 = np.unravel_index(i, (M, N))
        # Look at neighbor directly to the right and
        # directly below.  The others will be filled in
        # by symmetry
        for k, (di, dj) in enumerate([(0, 1), (1, 0)]):
            i2 = i1+di
            j2 = j1+dj
            if i2 >= M or j2 >= N:
                continue
            j = i2*N+j2
            p1 = np.array(Ps[i1, j1])
            p2 = np.array(Ps[i2, j2])
            if k == 1:
                # Looking at the neighbor below
                pass





if __name__ == '__main__':
    #testPlottingPieces()
    testMGC()