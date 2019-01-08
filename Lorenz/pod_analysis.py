import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patches

from data_gen import Lorenz


def N_y(W,y,xm):
    # nonlinear operator in modal space (unnormalized)
    return np.matmul(Lorenz.NL(xm+np.matmul(y,W.T)),W)

def N_y_std(W,y,xm,yn):
    # nonlinear operator in modal space (normalized)
    # N_y_std(w, y, meanX, y_std)
    X = xm+np.matmul(y*yn,W.T)
    return np.matmul(Lorenz.NL(X),W)/yn


def threshold_plot(ax, x, y, threshv, colors=None):
    """
    Helper function to plot points above a threshold in a different color

    Parameters
    ----------
    ax : Axes
        Axes to plot to
    x, y : array
        The x and y values

    threshv : float
        Plot using overcolor above this value

    """
    r,k = (1.,0,0,1.),(0,0,0,1.)

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be numlines x points per line x 2 (x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if colors is None:
        colors = []
        for yflag in segments[:,0,-1]:
            if yflag < threshv:
                colors.append(r)
            else:
                colors.append(k)

    # Create the line collection object, setting the colormapping parameters.
    # Have to set the actual values used for colormapping separately.
    # lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc = LineCollection(segments, colors=colors)

    ax.add_collection(lc)
    ax.set_ylim(np.floor(10*np.min(y))/10, np.ceil(10*np.max(y))/10)

    return colors


def threshold_passtime(x, thres):
    # going down columns of x and find first idx for which the value is below thres

    passtime = np.zeros(x.shape[2])
    for dim in range(x.shape[1]):
        for i in range(x.shape[0] - 1):
            if x[i,dim] > thres and x[i+1,dim] < thres:
                passtime[dim] = i + 1 - (thres-x[i+1,dim])/(x[i,dim]-x[i+1,dim])
                break

    return passtime


def main():
    # preparing datas for POD
    ## load data file
    npzfile = np.load('./data/lorenz_traj_pt100k_dt0.001.npz')
    X = npzfile["X"]

    ## center data and calculate emprical covariance
    # column mean (time series mean)
    meanX = np.mean(X,axis=0)
    stdX = np.std(X,axis=0)
    X_center = X - meanX # to make centered to zero mean
    covXX = np.matmul(X_center.T,X_center)/X.shape[0]   # covariance matrix definition

    ## eigen-analysis of covariance
    v,w = np.linalg.eig(covXX)

    # swap last two eigenvalue/vector, why? idk
    v[[-2, -1]] = v[[-1, -2]]
    w[:, [-2, -1]] = w[:, [-1, -2]]

    print('eigenvalues:',v)							# eigenvalues
    print('eigenvectors:',w) 							# eigenvectors
    print('energy percentage:',np.cumsum(v)/np.sum(v)) 		# energy percentage

    # new coordinate \xi = v^T * x
    # y = X * v = \xi
    y = np.matmul(X_center,w)
    print('mode variance:',np.var(y,axis=0))

    ## affine operators in POD space
    # linear term for chi
    # L(chi) = y^T L y
    L_y = np.matmul(np.matmul(w.T, Lorenz.L), w)
    # offset term for chi
    # b(chi) = b * y + x * L * chi
    b_y = np.matmul(Lorenz.b, w) + np.matmul(meanX, np.matmul(Lorenz.L,w))

    ## save standardized data and parameters to file
    y_std = np.sqrt(v)
    y = y/y_std
    b_y = b_y/y_std
    # diag(sigma) * L_y * diag(1/sigma)
    L_y = np.matmul(np.matmul(np.diag(y_std), L_y), np.diag(1/y_std))
    # ??
    # dydt1 = np.matmul(y,L_y) + b_y + N_y_std(w, y, meanX, y_std) 		# standardized
    # evolve dynamics
    dydt2 = (np.matmul(Lorenz.dynamics(X), w))/y_std
    # save operators for POD
    np.savez('data/modal_coords_100k_std_lorenz.npz', xm=meanX, y=y, W=w, L_y=L_y, b_y=b_y, dydt=dydt2, y_std=y_std)

    # NL_y = N_y_std(w,y,meanX,y_std)
    # NL_y_nrm = np.linalg.norm(NL_y,axis=-1)
    # NL_x = CdV.NL(X)
    # NL_x_nrm = np.linalg.norm(NL_x,axis=-1)
    # L_x_nrm = np.linalg.norm(CdV.dynamics(X)-NL_x,axis=-1)

    # plt.figure()
    # plt.title('norm of linear and nonlinear dynamics')
    # plt.plot(NL_x_nrm,'b-',linewidth=1,label='nonlinear')
    # plt.plot(L_x_nrm,'r-',linewidth=1,label='linear')
    # plt.xlim([0,3000])
    # plt.legend(frameon=False)

    # ## verify correctness of POD dynamics
    # print('modal dynamics comparison:')
    # print(dydt1[4])
    # print(dydt2[4])

    ## plot system scatter and time series
    plt.rc('axes', linewidth=1)
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)

    # xy
    fig = plt.figure(figsize=(10, 4))
    ax = plt.subplot2grid((4, 8), (0, 0), rowspan=4, colspan=4)
    ax.scatter(X[:, 0], X[:, 1], s=1, marker='o', edgecolor='none', facecolor='k')
    #ax.set_xlim([.7, 1])
    ax.set_xlabel('$x$', size=8)
    #ax.set_ylim([-.8, -.1])
    ax.set_ylabel('$y$', size=8)
    ax2 = plt.subplot2grid((4, 8), (0, 4), rowspan=2, colspan=4)
    ax3 = plt.subplot2grid((4, 8), (2, 4), rowspan=2, colspan=4, sharex=ax2)

    ax2.plot(X[:,0], 'k-', lw=.75)
    #ax2.set_xlim([0, 1000])
    #ax2.set_ylim([.7, 1])
    ax2.set_ylabel('$x$', size=8)
    #ax2.set_yticks(ax2.get_yticks()[1:])
    # ax2.add_patch(patches.Rectangle((81, 0), 209, 2, color='r', alpha=0.15, lw=0))    # shade blocked regions
    # ax2.add_patch(patches.Rectangle((872, 0), 124, 2, color='r', alpha=0.15, lw=0))
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3.plot(y[:,0], 'b-', lw=.75)
    #ax3.set_xlim([0, 1000])
    ax3.set_xlabel('time', size=8)
    #ax3.set_ylim([-2, 8])
    ax3.set_ylabel(r'$xi_1$', size=8)
    #ax3.set_yticks(ax3.get_yticks()[:-1])
    # ax3.add_patch(patches.Rectangle((81, -10), 209, 20, color='r', alpha=0.15, lw=0))
    # ax3.add_patch(patches.Rectangle((872, -10), 124, 20, color='r', alpha=0.15, lw=0))

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.savefig('./system_xy_x_xi1.png', dpi=350)
    # plt.show()

    # yz
    fig = plt.figure(figsize=(10, 4))
    ax = plt.subplot2grid((4, 8), (0, 0), colspan=4, rowspan=4)
    ax.scatter(X[:, 1], X[:, 2], s=1, marker='o', edgecolor='none', facecolor='k')
    #ax.set_xlim([.7, 1])
    ax.set_xlabel('$y$', size=8)
    #ax.set_ylim([-.8, -.1])
    ax.set_ylabel('$z$', size=8)
    ax2 = plt.subplot2grid((4, 8), (0, 4), colspan=4, rowspan=2)
    ax3 = plt.subplot2grid((4, 8), (2, 4), colspan=4, rowspan=2, sharex=ax2)

    ax2.plot(X[:,1], 'k-', lw=.75)
    #ax2.set_xlim([0, 1000])
    #ax2.set_ylim([.7, 1])
    ax2.set_ylabel('$y$', size=8)
    #ax2.set_yticks(ax2.get_yticks()[1:])
    # ax2.add_patch(patches.Rectangle((81, 0), 209, 2, color='r', alpha=0.15, lw=0))    # shade blocked regions
    # ax2.add_patch(patches.Rectangle((872, 0), 124, 2, color='r', alpha=0.15, lw=0))
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3.plot(y[:,1], 'b-', lw=.75)
    #ax3.set_xlim([0, 1000])
    ax3.set_xlabel('time', size=8)
    #ax3.set_ylim([-2, 8])
    ax3.set_ylabel(r'$xi_2$', size=8)
    #ax3.set_yticks(ax3.get_yticks()[:-1])
    # ax3.add_patch(patches.Rectangle((81, -10), 209, 20, color='r', alpha=0.15, lw=0))
    # ax3.add_patch(patches.Rectangle((872, -10), 124, 20, color='r', alpha=0.15, lw=0))

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.savefig('./system_yz_y_xi2.png', dpi=350)

    # xz
    fig = plt.figure(figsize=(10, 4))
    ax = plt.subplot2grid((4, 8), (0, 0), colspan=4, rowspan=4)
    ax.scatter(X[:, 0], X[:, 2], s=1, marker='o', edgecolor='none', facecolor='k')
    #ax.set_xlim([.7, 1])
    ax.set_xlabel('$x$', size=8)
    #ax.set_ylim([-.8, -.1])
    ax.set_ylabel('$z$', size=8)
    ax2 = plt.subplot2grid((4, 8), (0, 4), colspan=4, rowspan=2)
    ax3 = plt.subplot2grid((4, 8), (2, 4), colspan=4, rowspan=2, sharex=ax2)

    ax2.plot(X[:,2], 'k-', lw=.75)
    #ax2.set_xlim([0, 1000])
    #ax2.set_ylim([.7, 1])
    ax2.set_ylabel('$z$', size=8)
    #ax2.set_yticks(ax2.get_yticks()[1:])
    # ax2.add_patch(patches.Rectangle((81, 0), 209, 2, color='r', alpha=0.15, lw=0))    # shade blocked regions
    # ax2.add_patch(patches.Rectangle((872, 0), 124, 2, color='r', alpha=0.15, lw=0))
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3.plot(y[:,2], 'b-', lw=.75)
    #ax3.set_xlim([0, 1000])
    ax3.set_xlabel('time', size=8)
    #ax3.set_ylim([-2, 8])
    ax3.set_ylabel(r'$xi_3$', size=8)
    #ax3.set_yticks(ax3.get_yticks()[:-1])
    # ax3.add_patch(patches.Rectangle((81, -10), 209, 20, color='r', alpha=0.15, lw=0))
    # ax3.add_patch(patches.Rectangle((872, -10), 124, 20, color='r', alpha=0.15, lw=0))

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.savefig('./system_xz_z_xi3.png', dpi=350)


if __name__ == '__main__':
    main()
