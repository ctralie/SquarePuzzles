import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import scipy.misc
from VDM import *


"""####################################################
    CONSTANTS
#####################################################"""

R90 = np.array([[0, -1], [1, 0]])
RsMod4 = [np.eye(2)]
for i in range(3):
    RsMod4.append(R90.dot(RsMod4[-1]))


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
    GiLH = np.concatenate((GiL, DUMMY_DIRS), 0)
    muiL = np.mean(GiLH, 0)
    diff = GiLH-muiL
    S = (1.0/(GiLH.shape[0]-1))*((diff.T).dot(diff))
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

def getAllPairRotationScores(ppL, ppR, evalfn = getMGC):
    """
    Given a patch to the left and a patch to the right, 
    compute all scores
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
    scores: ndarray(16, 3):
        First column:  The number of 90 degree CCW rotations of the left patch
        Second column: The number of 90 degree CCW rotations of the right patch
        Third column: Score
    """
    pL = np.array(ppL)
    pR = np.array(ppR)
    scores = []
    for rotl in range(4):
        pL = rotateByZMod4(ppL, rotl)
        for rotr in range(4):
            pR = rotateByZMod4(ppR, rotr)
            scores.append([rotl, rotr, evalfn(pL, pR)])
    return np.array(scores)

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

def testRotationPairs(evalfn = getMGC):
    """
    Test the rotation scores for all patches in an image and
    count how many are in each configuration (the correct answer
    should be (0, 0) for most of them ideally)
    """
    I = readImage("melayla.jpg")
    d = 28
    Ps = getPatchesColor(I, d)
    minrots = {}
    getScore = lambda pL, pR: getMGC(pL, pR) + getRGB(pL, pR)
    for i in range(4):
        for j in range(4):
            minrots[(i, j)] = 0
    for i in range(Ps.shape[0]):
        print(i)
        for j in range(Ps.shape[1]-1):
            pL = Ps[i, j, :, :, :]
            pR = Ps[i, j+1, :, :, :]
            scores = getAllPairRotationScores(pL, pR, getScore)
            idx = np.argmin(scores[:, -1])
            minrots[(scores[idx, 0], scores[idx, 1])] += 1
    print(minrots)
    PercentCorrect = 100.0*float(minrots[(0, 0)])/(Ps.shape[0]*(Ps.shape[1]-1))
    print("%.3g %s correct"%(PercentCorrect, "%"))


"""#####################################################
            Type 3 Puzzles (Rotation Only)
#####################################################"""

def solveType3Puzzle(Ps, ratiocutoff = 1.01, avgweight = 0.5, vote_multiple = False, weighted=False, evalfn=getMGC, vratio=0):
    """
    Solve a type 3 (rotations only) puzzle
    Parameters
    ----------
    Ps: ndarray(M, N, d, d, 3)
        An MxN grid of dxd patches
    ratiocutoff: float
        The cutoff below which to consider two rotation
        scores to be the same
    avgweight: float
        The weight to give an orientation when it's the result
        of averaging several votes
    vote_multiple: boolean
        Whether to vote on multiple orientations for a pair if there
        isn't a clear winner
    weighted: boolean
        Whether to use the weighted connection Laplacian
    evalfn: function(patch left, patch right)
        Function for evaluating similarity of patches
    vratio: float
        The ratio of the second eigenvector to the first eigenvector
        in the weighted sum to determine the direction
    Returns
    -------
    Rsidx: ndarray(M, N)
        The element in Z/4 to apply to each patch to bring it
        into the correct orientation
    Rsidxfloat: ndarray(M, N)
        The relaxed solution for each rotation
    """
    M, N = Ps.shape[0], Ps.shape[1]
    NP = M*N
    ## Step 1: Setup the connection Laplacian
    ws = []
    Os = []
    for i in range(NP):
        if i%25 == 0:
            print("%.3g %s"%(100.0*i/NP, "%"))
        i1, j1 = np.unravel_index(i, (M, N))
        # Look at neighbor directly to the right and
        # directly below.  The others will be filled in
        # by symmetry
        for di, dj in [(0, 1), (1, 0)]:
            i2 = i1+di
            j2 = j1+dj
            if i2 >= M or j2 >= N:
                continue
            j = i2*N+j2
            p1 = np.array(Ps[i1, j1])
            p2 = np.array(Ps[i2, j2])
            if di == 1 and dj == 0:
                # Looking at the neighbor below
                p1 = rotateByZMod4(p1, 1)
                p2 = rotateByZMod4(p2, 1)
            scores = getAllPairRotationScores(p1, p2, evalfn=evalfn)
            idx = np.argsort(scores[:, -1])
            scores = scores[idx, :]
            ratios = np.inf*np.ones(scores.shape[0])
            ratios[0] = 1
            if scores[0, -1] > 0:
                ratios = scores[:, -1]/scores[0, -1]
            scores = scores[ratios < ratiocutoff, :]
            #scores = np.array([[Rsidx[i1][j1], Rsidx[i2][j2]]])
            if scores.shape[0] == 1:
                # One rotation is dominating
                thetai, thetaj = scores[0, 0:2]
                ws.append([i, j, 1.0])
                Os.append(RsMod4[int((thetaj-thetai)%4)])
                # Put in symmetric orientation
                ws.append([j, i, 1.0])
                Os.append(Os[-1].T)
            elif vote_multiple:
                # Need to average several orientations, and make the score lower
                print("%i Competing"%scores.shape[0])
                thetai, thetaj = np.mean(scores[:, 0:2], 0)
                ct = np.cos((np.pi/2)*(thetai-thetaj))
                st = np.cos((np.pi/2)*(thetai-thetaj))
                R1 = np.array([[ct, -st], [st, ct]])
                ws.append([i, j, avgweight])
                Os.append(R1)
                ws.append([j, i, avgweight])
                Os.append(R1.T)
    ws = np.array(ws)

    ## Step 2: Get the top eigenvector of the connection Laplacian and
    ## use this to figure out the rotations
    # Only need to compute the top eigenvector since we know
    # this is a rotation matrix
    w, v = getConnectionLaplacian(ws, Os, NP, 2, weighted=weighted)
    print(w)
    Rsidxfloat = np.zeros((M, N), dtype=float)
    for idx in range(NP):
        i, j = np.unravel_index(idx, (M, N))
        R = v[idx*2:(idx+1)*2, 0:2]
        R = R/np.sqrt(np.sum(R**2, 0)[None, :])
        R = R[:, 0] + vratio*R[:, 1]
        theta = (np.arctan2(R[1], R[0])/(np.pi/2))%4
        Rsidxfloat[i, j] = theta
    
    ## Step 3: Figure out which of the possible 4 global rotations
    ## brings the pieces into the best alignment
    ming = 0
    mincost = np.inf
    for g in range(4):
        Rs = np.array(np.mod(np.round(Rsidxfloat + g), 4), dtype=int)
        cost = 0.0
        for i in range(M-1):
            for j in range(N-1):
                # Piece to the right
                p1 = Ps[i, j, :, :, :]
                p2 = Ps[i, j+1, :, :, :]
                ridx = (4-Rs[i, j])%4
                cost += evalfn(rotateByZMod4(p1, ridx), rotateByZMod4(p2, Rs[i, j]))
                # Piece below
                p2 = Ps[i+1, j, :, :, :]
                cost += evalfn(rotateByZMod4(p1, ridx+1), rotateByZMod4(p2, Rs[i, j]+1))
        print("Trying global solution g = %i, cost=%.3g"%(g, cost))
        if cost < mincost:
            mincost = cost
            ming = g
    print("ming = %i"%ming)
    Rsidx = np.array(np.mod(4-np.round(Rsidxfloat + ming), 4), dtype=int)
    Rsidxfloat = np.mod(Rsidxfloat+ming, 4)
    return Rsidx, Rsidxfloat

def flattenColumnwise(arr):
    ret = []
    for row in arr:
        ret += row
    return ret

def testType3Puzzle(seed = 0, d = 50):
    np.random.seed(seed)

    ## Step 1: Setup puzzle
    I = readImage("melayla.jpg")
    Ps = getPatchesColor(I, d)
    M = Ps.shape[0]
    N = Ps.shape[1]
    RsidxGT = np.random.randint(0, 4, (M, N)) #Ground truth rotations
    X, Y = np.meshgrid(np.arange(Ps.shape[1]), np.arange(Ps.shape[0]))
    Y = Ps.shape[0]-Y
    X = np.array([X.flatten(), Y.flatten()])
    X = X.T
    PsFlatten = np.reshape(Ps, (Ps.shape[0]*Ps.shape[1], d, d, 3))
    # Now actually rotate the pieces
    RsEye = []
    for i in range(M):
        for j in range(N):
            Ps[i, j, :, :, :] = rotateByZMod4(Ps[i, j, :, :, :], RsidxGT[i, j])
            RsEye.append(np.eye(2))

    ## Step 2: Solve puzzle and count correct pieces
    vratio = 0.0
    plt.figure(figsize=(22, 9))
    RsidxSol, RsidxSolfloat = solveType3Puzzle(Ps, weighted=True, vote_multiple=True, vratio=vratio, evalfn=getRGB)
    RsSol = []
    NCorrect = 0
    guesses = np.zeros((4, 4))
    for i in range(M):
        RsSol.append([])
        for j in range(N):
            ridx = RsidxSol[i, j]
            RsSol[i].append(RsMod4[ridx])
            guesses[RsidxGT[i, j], (4-ridx)%4] += 1
    print(guesses)
    NCorrect = np.sum(np.diag(guesses))

    ## Step 3: Plot Results
    plt.subplot(131)
    plotPatches(plt.gca(), X, RsEye, PsFlatten)
    plt.title("%i %ix%i Pieces"%(M*N, d, d))
    plt.subplot(133)
    plotPatches(plt.gca(), X, flattenColumnwise(RsSol), PsFlatten)
    plt.title("%i/%i correct"%(NCorrect, M*N))
    plt.subplot(132)
    plt.scatter(X[:, 0], M-X[:, 1])
    ax = plt.gca()
    for i in range(M):
        for j in range(N):
            theta = RsidxSolfloat[i, j] - RsidxGT[i, j]
            v = [0.5*np.cos(np.pi*theta/2), 0.5*np.sin(np.pi*theta/2)]
            c = 'k'
            if not ((RsidxSol[i, j] + RsidxGT[i, j])%4 == 0):
                c = 'r'
            ax.arrow(j, M-i-1, v[0], v[1], head_width=0.1, head_length=0.2, color=c)
    plt.axis('off')
    plt.title("Relaxed Solution vratio=%.3g"%vratio)

    plt.savefig("%i.png"%d, bbox_inches='tight')

if __name__ == '__main__':
    #testPlottingPieces()
    #testMGC()
    #testRotationPairs()
    for d in [20, 30, 40, 50, 100]:
        testType3Puzzle(d=d)