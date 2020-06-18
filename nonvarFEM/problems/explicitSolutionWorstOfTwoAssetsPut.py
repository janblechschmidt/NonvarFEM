import ipdb
import numpy as np
from statsmodels.sandbox.distributions.extras import mvnormcdf

F = 40
R = 0.05
Tmax = 1.0


def Mfunc(V, H, F, tau, R=0.05, sigmaH=0.3, sigmaV=0.3, rho_VH=0.5):

    def covmat(rho): return np.array([[1, rho], [rho, 1]])
    u0 = np.array([0, 0])

    sigma = np.sqrt(sigmaV**2 + sigmaH**2 - 2*rho_VH*sigmaV*sigmaH)
    if F > 0:
        if H > 0:
            gamma1 = (np.log(H/F) + (R - .5*sigmaH**2)*tau) / \
                (sigmaH*np.sqrt(tau))
        else:
            gamma1 = (-np.inf + (R - .5*sigmaH**2)*tau) / (sigmaH*np.sqrt(tau))

        if V > 0:
            gamma2 = (np.log(V/F) + (R - .5*sigmaV**2)*tau) / \
                (sigmaV*np.sqrt(tau))
        else:
            gamma2 = (-np.inf + (R - .5*sigmaV**2)*tau) / (sigmaV*np.sqrt(tau))

    else:
        if H > 0:
            gamma1 = (np.inf + (R - .5*sigmaH**2)*tau) / (sigmaH*np.sqrt(tau))
        else:
            gamma1 = (+ (R - .5*sigmaH**2)*tau) / (sigmaH*np.sqrt(tau))
        if V > 0:
            gamma2 = (np.inf + (R - .5*sigmaV**2)*tau) / (sigmaV*np.sqrt(tau))
        else:
            gamma2 = (+ (R - .5*sigmaV**2)*tau) / (sigmaV*np.sqrt(tau))

    alpha1 = gamma1 + sigmaH*np.sqrt(tau)
    if H > 0:
        if V > 0:
            alpha2 = (np.log(V/H) - 0.5*sigma**2*tau) / (sigma * np.sqrt(tau))
        else:
            alpha2 = (-np.inf - 0.5*sigma**2*tau) / (sigma * np.sqrt(tau))

    else:
        if V > 0:
            alpha2 = (np.inf - 0.5*sigma**2*tau) / (sigma * np.sqrt(tau))
        else:
            alpha2 = (- 0.5*sigma**2*tau) / (sigma * np.sqrt(tau))

    beta1 = gamma2 + sigmaV*np.sqrt(tau)
    if V > 0:
        if H > 0:
            beta2 = (np.log(H/V) - 0.5*sigma**2*tau) / (sigma * np.sqrt(tau))
        else:
            beta2 = (-np.inf - 0.5*sigma**2*tau) / (sigma * np.sqrt(tau))

    else:
        if H > 0:
            beta2 = (- 0.5*sigma**2*tau) / (sigma * np.sqrt(tau))
        else:
            beta2 = (- 0.5*sigma**2*tau) / (sigma * np.sqrt(tau))

    l1 = np.array([alpha1, alpha2]).flatten()

    if any(l1 == -np.inf):
        t1 = 0.0
    else:
        t1 = H * mvnormcdf(l1, u0, covmat((rho_VH*sigmaV-sigmaH)/sigma))

    l2 = np.array([beta1, beta2]).flatten()

    if any(l2 == -np.inf):
        t2 = 0.0
    else:
        t2 = V * mvnormcdf(l2, u0, covmat((rho_VH*sigmaH-sigmaV)/sigma))

    l3 = np.array([gamma1, gamma2]).flatten()

    if any(l3 == -np.inf):
        t3 = 0.0
    else:
        t3 = F*np.exp(-R*tau)*mvnormcdf(l3, u0, covmat(rho_VH))
    if np.isnan(t1) or np.isnan(t2) or np.isnan(t3):
        ipdb.set_trace()

    return t1 + t2 - t3


# solfunc = lambda t,x : np.exp(-R*(Tmax-t)) * F \
#        - Mfunc(x[:,0],x[:,1],0,Tmax-t) + Mfunc(x[:,0],x[:,1],F,Tmax-t);
#
# # Compute one explicit value
#
# #solfunc(0.5,np.array([[40,40]]))
#
#
# # Plot the function
# x = np.linspace(0.0000001,100,101)
# X,Y = np.meshgrid(x,x)
# # XY = np.hstack([X.reshape(-1,1), Y.reshape(-1,1)])
# # solfunc(tau,XY)
# V = np.zeros_like(X)
# for i in range(101):
#     for j in range(101):
#         V[i,j] = solfunc(tau,np.array([[X[i,j],Y[i,j]]]))
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# # %matplotlib inline
# fig = plt.figure('Value function')
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X,Y,V,cmap='coolwarm')
# plt.show()
