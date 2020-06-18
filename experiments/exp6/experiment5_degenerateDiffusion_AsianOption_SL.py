""" Semi-Lagrange approach for FE solution of non-variational PDEs

This notebook is based on the `testSemiLagrange.ipynb` notebook and tries to
valuate an European Asian call option using a non-variational solver for the
stochastic component.

We want to solve the degenerate second-order PDE in non-variational form

-v_t - (A : Hessian(v) + inner(b, Grad(v)) + c v - f) = 0

with

A = .5 [[ (σ x)^2, 0], [0, 0]]
b = [x r, (x-y)/t]
c = -r

and final time conditions $v(T) = (K-y)_+$ for a fixed strike price K.
"""

import csv
from time import time
from dolfin import parameters, refine, MeshFunction, as_backend_type
from dolfin import Function, FunctionSpace, TrialFunction, TestFunction, Constant
from dolfin import grad, div, dx, assemble, as_matrix, as_vector, inner
from dolfin import avg, jump, dS
from dolfin import IntervalMesh, SpatialCoordinate, FacetArea, FacetNormal
import numpy as np
import matplotlib.pyplot as plt
import ufl
import scipy.sparse as sp
import scipy.sparse.linalg as la
from experiment5_degenerateDiffusion_SmoothSol_SL import plotVi3d, plotViContour, Yinterp1


def spmat(M):
    mat = as_backend_type(M).mat()
    return sp.csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)


parameters["reorder_dofs_serial"] = False

model = 'AsianCall'
K = 95
name_id = '_ny_is_nx'
plotSol = False
plotErrors = False
plotMode = '3d'
writeErrors = True
storeV = False
Ndofs = []
Hmax = []
Nt = []
L2err = []
Cinferr = []
ExecTime = []
ExecTimeFac = []
steps = 4


def refineMesh(m, l, u):
    cell_markers = MeshFunction('bool', m, 1)
    cell_markers.set_all(False)
    coords = m.coordinates()
    midpoints = 0.5*(coords[:-1] + coords[1:])
    ref_idx = np.where((midpoints > l) & (midpoints < u))[0]
    cell_markers.array()[ref_idx] = True
    m = refine(m, cell_markers)
    return m


if writeErrors:
    filename = './results/experiment5/SL_' + \
        model + '_K{}'.format(K) + name_id + '.csv'
    csvfile = open(filename, 'w', newline='')
    fieldnames = ['Step', 'Ndofs', 'Hmax', 'Nt', 'ExecTime', 'ExecTimeFac', 'V100']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

for step in range(steps):
    exec_time = time()
    if model == 'AsianCall':
        # Set discretization level
        fac = 2**step
        nt = 40*fac
        nx = 20*fac
        ny = 20*fac
        T = [0.0, .25]
        Xmax = 200
        Ymax = 200

        # Mesh for stochastic part
        mx = IntervalMesh(nx, 0, Xmax)
        C = 100
        mx = refineMesh(mx, C-10, C+10)
        mx = refineMesh(mx, C-25, C+25)
        mx = refineMesh(mx, C-50, C+50)

        # Mesh for deterministic part
        my = IntervalMesh(ny, 0, Ymax)
        my = refineMesh(my, C-10, C+10)
        my = refineMesh(my, C-25, C+25)
        my = refineMesh(my, C-50, C+50)

        x = SpatialCoordinate(mx)[0]
        y = SpatialCoordinate(my)[0]

        # Set up variables for t and y
        t_ = Constant(T[1])
        y_ = Constant(0.0)

        # Data
        sigmax = 0.1
        interest = 0.1

        # Set up coefficients as described above

        # Diffusion in x direction
        ax = as_matrix([[.5 * sigmax**2 * x**2]])

        # Convection in x direction
        bx = as_vector([interest * x])

        # Convection in y direction
        def by(t, x, y):
            if t == 0:
                # return (x-y) / (0.001)
                return np.zeros_like(x)
            else:
                return (x-y) / t

        # Potential term
        c = -interest

        # Homogeneous boundary conditions
        DBC = 1
        # sxx, sx, s0
        BC_LB = [1, 1, 0]
        BC_RB = [1, 1, 1]

        def dbc_bound(x, y):
            return (x > Xmax-1e-6) * (y > Ymax - 1e-6)

        def g(t, x, y):
            return y-K*np.exp(-interest * (T[1]-t))

        # Fixed-strike Asian Call
        def u_T(x, y): return ufl.Max(y-K, 0)
        def u_T_numpy(x, y): return np.maximum(y-K, 0)

        f = Constant(0.0)

    # We have to use at least quadratic polynomials here
    Vx = FunctionSpace(mx, 'CG', 2)
    Vy = FunctionSpace(my, 'CG', 1)

    phi = TrialFunction(Vx)
    psi = TestFunction(Vx)

    v = Function(Vx)
    # We avoid the normalization
    gamma = 1.0
    # Jump penalty term
    stab1 = 2.
    nE = FacetNormal(mx)
    hE = FacetArea(mx)
    test = div(grad(psi))
    S_xx = gamma * inner(ax, grad(grad(phi))) * test * dx(mx) \
        + stab1 * avg(hE)**(-1) * inner(jump(grad(phi), nE),
                                        jump(grad(psi), nE)) * dS(mx)
    S_x = gamma * inner(bx, grad(phi)) * test * dx(mx)
    S_0 = gamma * c * phi * test * dx(mx)

    # This matrix also changes since we are testing the whole equation
    # with div(grad(psi)) instead of psi
    M_ = gamma * phi * test * dx(mx)

    M = assemble(M_)

    # Prepare special treatment of deterministic part and time-derivative.

    # Probably, I'll need here the dof coordinates and the dof list
    xdofs = Vx.tabulate_dof_coordinates().flatten()
    ix = np.argsort(xdofs)

    # Since we're using the mesh in the deterministic direction
    # only for temporary purpose, the coordinates are fine here
    ydofs = Vy.tabulate_dof_coordinates().flatten()
    ydofs.sort()
    ydiff = np.diff(ydofs)
    assert np.all(ydiff > 0)

    # Create meshgrid X, Y of shape (ny+1, nx+1)
    X, Y = np.meshgrid(xdofs, ydofs)

    trange = np.linspace(T[1], T[0], nt + 1)
    dt = -np.diff(trange)
    assert np.all(dt > 0)

    # Set up auxiliary functions
    Mvtmp = Function(Vx)
    vsol = Function(Vx)

    # Initialize value at T[1]
    t = T[1]
    t_.assign(t)
    vold = u_T_numpy(X, Y)

    if storeV:
        Vsol = np.ndarray([nt+1, len(ydofs), len(xdofs)])
        Vsol[0, :, :] = vold

    L_ = gamma * f * test * dx(mx)
    mn = len(xdofs) * len(ydofs)

    Mmat = spmat(M)
    idx_xmin = np.where(xdofs == xdofs.min())[0][0]
    idx_xmax = np.where(xdofs == xdofs.max())[0][0]

    solveSmall = True

    for i, t in enumerate(trange[1:]):
        print('Solve problem, time step t = {}'.format(np.round(t, 4)))

        # Assign current time to variable t_
        t_.assign(t)

        # Compute Ytilde, i.e. y-position of process starting in x_i,y_j
        # after application of convection in y-direction
        Ytilde = Y + by(t, X, Y)*dt[i]
        Vtilde = Yinterp1(vold, Ytilde, ydofs, ydiff)

        if not solveSmall:
            Slist = []
            Lcomp = np.zeros((mn,), dtype='float')

        # Loop over all layers in y-direction
        for j in range(len(ydofs)):
            y_.assign(ydofs[j])

            Mvtmp.vector().set_local(M * Vtilde[j, :])

            L = assemble(L_)
            Lfull = Mvtmp.vector() - dt[i] * L

            Sxx = spmat(assemble(S_xx))
            Sx = spmat(assemble(S_x))
            S0 = spmat(assemble(S_0))

            # Include boundary conditions

            # left boundary
            if BC_LB[0]:
                Sxx[idx_xmin, :] = 0
            if BC_LB[1]:
                Sx[idx_xmin, :] = 0
            if BC_LB[2]:
                S0[idx_xmin, :] = 0

            # right boundary
            if BC_RB[0]:
                Sxx[idx_xmax, :] = 0
            if BC_RB[1]:
                Sx[idx_xmax, :] = 0
            if BC_RB[2]:
                S0[idx_xmax, :] = 0

            Sfull = Mmat - dt[i]*(Sxx + Sx + S0)

            # Include the upper-right corner as Dirichlet data
            dbc_bnd = np.where(dbc_bound(xdofs, ydofs[j]))[0]

            if len(dbc_bnd) > 0:
                Sfull[dbc_bnd, :] = 0
                Sfull[dbc_bnd, dbc_bnd] = 1
                Lfull[dbc_bnd] = g(t, xdofs[dbc_bnd], ydofs[j])

            if solveSmall:
                vsol = la.spsolve(Sfull, Lfull.get_local())
                vold[j, :] = vsol[:]
            else:
                Slist.append(Sfull)
                Lcomp[j*len(xdofs):(j+1)*len(xdofs)] = Lfull.get_local()

        if not solveSmall:
            Scomp = sp.block_diag(Slist, format='csr')
            vcomp = la.spsolve(Scomp, Lcomp)
            vold = vcomp.reshape(len(ydofs), len(xdofs))

        if storeV:
            Vsol[i+1, :, :] = vold

    # End of time loop

    if plotSol:
        if storeV:
            Z = Vsol[i, :, :]
        else:
            Z = vold

        if plotMode == 'contour':
            plotViContour(X[:, ix], Y[:, ix], Z[:, ix],)

        if plotMode == '3d':
            plotVi3d(X[:, ix], Y[:, ix], Z[:, ix],)

    x100 = np.where(xdofs == 100)[0]
    y100 = np.where(ydofs == 100)[0]
    z100 = vold[y100, x100][0]

    ndofs = len(ydofs) * len(xdofs)

    hxmax = max(np.diff(np.sort(mx.coordinates().flatten())))
    hymax = ydiff.max()
    # import ipdb
    # ipdb.set_trace()
    hmax = np.maximum(hxmax, hymax)
    print('Step: ', step)
    print('Ndofs: ', ndofs)
    print('Hmax: ', hmax)
    print('Value: ', z100)
    exec_time = time() - exec_time
    ExecTime.append(exec_time)
    exec_time_fac = np.round(exec_time / ExecTime[0],2)
    ExecTimeFac.append(exec_time_fac)

    if writeErrors:
        writer.writerow({'Step': step,
                         'Ndofs': ndofs,
                         'Hmax': hmax,
                         'Nt': nt,
                         'ExecTime': exec_time,
                         'ExecTimeFac': exec_time_fac,
                         'V100': z100})
if writeErrors:
    csvfile.close()

# Plot and write errors
if plotErrors:
    plt.loglog(Hmax, L2err, label='L2 error')
    plt.loglog(Hmax, Cinferr, label='Cinf error')
    plt.show()
