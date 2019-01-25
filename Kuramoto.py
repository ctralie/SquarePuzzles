import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import sklearn.feature_extraction.image as skimage
from sklearn import manifold
import umap
import time
from mpl_toolkits.mplot3d import Axes3D
from ripser import ripser, plot_dgms
import sys
sys.path.append("DREiMac")
from CircularCoordinates import CircularCoords
from DiffusionMaps import *

def makeVideo(Y, I, Xs, Ts, pd, colorvar):
    c = plt.get_cmap('magma_r')
    C = c(np.array(np.round(255.0*colorvar/np.max(colorvar)), dtype=np.int32))
    C = C[:, 0:3]
    fig = plt.figure(figsize=(12, 6))
    idx = 0
    for i in range(0, Y.shape[0], 20):
        plt.clf()
        plt.subplot(121)
        plt.imshow(I, interpolation='none', aspect='auto', cmap='RdGy')
        x1, x2 = Xs[i], Xs[i]+pd[1]
        y1, y2 = Ts[i], Ts[i]+pd[0]
        plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], 'r')
        plt.xlabel("Space")
        plt.ylabel("Time")
        plt.title("[%i, %i] x [%i, %i]"%(x1, x2, y1, y2))
        plt.xlim([0, I.shape[1]])
        plt.ylim([I.shape[0], 0])

        if Y.shape[1] == 3:
            ax = plt.gcf().add_subplot(122, projection='3d')
            ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=np.array([[0, 0, 0, 0]]))
            ax.scatter(Y[0:i+1, 0], Y[0:i+1, 1], Y[0:i+1, 2], c=C[0:i+1, :])
            ax.scatter(Y[i, 0], Y[i, 1], Y[i, 2], 'r')
        else:
            plt.subplot(122)
            plt.scatter(Y[:, 0], Y[:, 1], 100, c=np.array([[0, 0, 0, 0]]))
            plt.scatter(Y[0:i+1, 0], Y[0:i+1, 1], 20, c=C[0:i+1, :])
            plt.scatter(Y[i, 0], Y[i, 1], 40, 'r')
            plt.axis('equal')
            ax = plt.gca()
            ax.set_facecolor((0.15, 0.15, 0.15))
            ax.set_xticks([])
            ax.set_yticks([])
        plt.savefig("%i.png"%idx)
        idx += 1

def do_umap_and_tda(pd, sub, nperm, patches, I, Xs, Ts, ts):
    ## Step 1: Do Umap
    n_components = 2
    n_neighbors = 4
    Y = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors).fit_transform(patches)


    print(patches.shape)
    res1 = CircularCoords(patches, nperm, cocycle_idx = [0])
    #res2 = CircularCoords(patches, nperm, cocycle_idx = [1])
    perm = res1["perm"]
    dgms = ripser(patches[perm, :], maxdim=1, coeff=41)["dgms"]
    dgms1 = dgms[1]


    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    plt.imshow(I, interpolation='none', aspect='auto', cmap='RdGy', extent=(0, I.shape[1], ts[-1], ts[0]))
    x1, x2 = Xs[5], Xs[5]+pd[1]
    y1, y2 = ts[Ts[5]], ts[Ts[5]+pd[0]]
    plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], 'r')
    plt.xlabel("Space")
    plt.ylabel("Time")
    plt.title("Solution")

    plt.subplot(223)
    plt.scatter(Y[:, 0], Y[:, 1], 20, c=ts[Ts], cmap='magma_r')
    plt.colorbar()
    plt.axis('equal')
    plt.title("Umap By Time")


    plt.subplot(224)
    plt.scatter(Y[:, 0], Y[:, 1], 20, c=Xs, cmap='magma_r')
    plt.colorbar()
    plt.axis('equal')
    plt.title("Umap By Space")

    plt.subplot(222)
    plot_dgms(dgms)
    idx = np.argsort(dgms1[:, 0]-dgms1[:, 1])
    plt.text(dgms1[idx[0], 0], dgms1[idx[0], 1], "1")
    plt.text(dgms1[idx[1], 0], dgms1[idx[1], 1], "2")
    plt.title("Persistence Diagrams")

    """
    plt.subplot(235)
    plt.scatter(Xs, ts[Ts], 40, res1["thetas"], cmap="magma_r")
    plt.gca().invert_yaxis()
    plt.xlabel("Space")
    plt.ylabel("Time")
    plt.xlim([0, I.shape[1]])
    plt.ylim([ts[-1], ts[0]])
    plt.title("Cocycle 1")

    plt.subplot(236)
    plt.scatter(Xs, ts[Ts], 40, res2["thetas"], cmap="magma_r")
    plt.gca().invert_yaxis()
    plt.xlabel("Space")
    plt.ylabel("Time")
    plt.xlim([0, I.shape[1]])
    plt.ylim([ts[-1], ts[0]])
    plt.title("Cocycle 2")
    """
    plt.show()
    return Y

def doDiffusionMaps(patches, Xs, Ts, ts):
    """
    Using https://github.com/jmbr/diffusion-maps
    """
    c = plt.get_cmap('magma_r')
    C1 = c(np.array(np.round(255.0*ts[Ts]/np.max(ts[Ts])), dtype=np.int32))
    C1 = C1[:, 0:3]
    C2 = c(np.array(np.round(255.0*Xs/np.max(Xs)), dtype=np.int32))
    C2 = C2[:, 0:3]
    
    D = getSSM(patches)
    t = 0.4*np.max(D**2)*0.001
    print("t = %g"%t)
    print("Doing diffusion maps on %i points in %i dimensions..."%(patches.shape[0], patches.shape[1]))
    tic = time.time()
    Y = getDiffusionMap(patches, t, neigs=4, thresh=1e-10)
    Y = np.fliplr(Y)
    print("Elapsed Time: %.3g"%(time.time()-tic))

    Y = Y[:, 1::]
    plt.imshow(Y, aspect='auto')
    plt.show()

    plt.figure(figsize=(6, 6))
    """
    ax = plt.gcf().add_subplot(121, projection='3d')
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=C1)
    plt.title("Diffusion Maps By Time")
    """

    ax = plt.gcf().add_subplot(111, projection='3d')
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=C2)
    plt.title("Diffusion Maps By Space")
    plt.show()

    return Y

def testKS_NLDM(pd = (200, 1), sub=(2, 1), nperm = 600):
    """
    Test a nonlinear dimension reduction of the Kuramoto Sivashinsky Equation
    torus attractor
    Parameters
    ----------
    pd: tuple(int, int)
        The dimensions of each patch
    sub: int
        The factor by which to subsample the patches
    nperm: int
        Number of points to take in a greedy permutation
    """
    res = sio.loadmat("KS.mat")
    I = res["data"]
    ts = np.linspace(res["tmin"].flatten(), res["tmax"].flatten(), I.shape[0])
    #I = I[0:106, :]
    ts = ts[0:I.shape[0]]
    IOrig = np.array(I)
    I = np.concatenate((I, I[:, 0:pd[1]]), 1) # Do periodic padding
    M, N = I.shape[0], I.shape[1]
    patches = skimage.extract_patches_2d(I, pd)
    patches = np.reshape(patches, (M-pd[0]+1, N-pd[1]+1, pd[0], pd[1]))
    # Index by spatial coordinate and by time
    Xs, Ts = np.meshgrid(np.arange(patches.shape[1]), np.arange(patches.shape[0]))
    # Subsample patches
    patches = patches[0::sub[0], 0::sub[1], :, :]
    Xs = Xs[0::sub[0], 0::sub[1]]
    Ts = Ts[0::sub[0], 0::sub[1]]
    Xs = Xs.flatten()
    Ts = Ts.flatten()
    patches = np.reshape(patches, (patches.shape[0]*patches.shape[1], pd[0]*pd[1]))




    #Y = do_umap_and_tda(pd, sub, nperm, patches, I, Xs, Ts, ts)
    Y = doDiffusionMaps(patches, Xs, Ts, ts)
    #Y = manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=3, eigen_solver='auto', method='standard').fit_transform(patches)
    #Y = manifold.Isomap(n_neighbors=10, n_components=2).fit_transform(patches)
    makeVideo(Y, IOrig, Xs, Ts, pd, Ts)



if __name__ == '__main__':
    testKS_NLDM()