""" Semi-Lagrange approach for FE solution of non-variational PDEs

This notebook is based on the `testSemiLagrange.ipynb` notebook and performs a
convergence study for a degenerate non-variational problem with known solution.

It shows the efficiency of the SL approach in connection with a method discussed
in Neilan, Salgado, Zhang (2017) for the solution of non-variational PDEs.

We want to solve the degenerate second-order PDE in non-variational form

-v_t - (A : Hessian(v) + inner(b, Grad(v)) + c v - f) = 0

with

A = [[1, 0], [0, 0]],
b = [0, -10 x (y - 0.5)],
c = 0.

The explicit solution is

v(t,x,y) = 0.5 (t^2 + 1) sin(2 pi x) * sin(2 pi y)$.

and the final time condition reads

v(T) = sin(2 pi x) * sin(2 pi y).
"""

import csv
from dolfin import parameters, sin, pi
from dolfin import Function, FunctionSpace, TrialFunction, TestFunction, Constant
from dolfin import grad, div, dx, assemble, as_matrix, as_vector, inner, diff
from dolfin import avg, jump, dS, DirichletBC, solve, norm
from dolfin import UnitIntervalMesh, SpatialCoordinate, FacetArea, FacetNormal
import numpy as np
import matplotlib.pyplot as plt


def plotVi3d(X, Y, Z):

    fig = plt.figure(clear=True)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.7)

    ax.set_xlabel('Stochastic')
    ax.set_ylabel('Deterministic')
    ax.view_init(30, -120)
    plt.show()


def plotViContour(X, Y, Z):

    fig = plt.figure(clear=True)
    ax = fig.add_subplot(111)
    ax.contour(X, Y, Z, cmap='coolwarm')
    ax.set_xlabel('Stochastic')
    ax.set_ylabel('Deterministic')
    plt.show()


def Yinterp1(V, Ytilde, ydofs, dy):

    assert Ytilde.max() <= ydofs.max() + 1e-6
    assert Ytilde.min() >= ydofs.min() - 1e-6
    Ytilde = np.maximum(Ytilde, ydofs.min())
    Ytilde = np.minimum(Ytilde, ydofs.max())

    # Determine dimensions of arrays
    ny, nx = np.array(Ytilde.shape)-1

    G = (Ytilde.reshape(ny+1, nx+1, 1) -
         ydofs[:-1].reshape(1, 1, ny))/dy.reshape(1, 1, ny)
    # There are three cases:
    # - G \in (0,1) is best, since it can only be active for exactly one point
    # - G \in {0,1} can either be fulfilled by only one point (on boundary)
    #   or fulfilled by two points (this has to be handled differently)
    # Therefore, we use the argmax to determine the first index lying withinconditions

    ik = np.argmax((G <= 1) * (G >= 0), axis=2).flatten()
    ii = np.repeat(np.arange(ny+1), nx+1)
    ij = np.repeat(np.arange(nx+1).reshape(nx+1, 1),
                   ny+1, axis=1).T.flatten()
    mu = G[ii, ij, ik]
    assert len(ik) == len(ii)
    assert np.all((mu >= 0) & (mu <= 1))
    Vtilde = (1 - mu)*V[ik, ij] + mu*V[ik+1, ij]

    return Vtilde.reshape(ny+1, nx+1)


if __name__ == '__main__':

    parameters["reorder_dofs_serial"] = False

    model = 'SmoothSol'
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
    L2errTrap = []
    Cinferr = []
    steps = 7
    filename = './results/experiment5/SL_' + model + name_id + '.csv'

    if writeErrors:
        csvfile = open(filename, 'w', newline='')
        fieldnames = ['Step', 'Ndofs', 'Hmax',
                      'L2_err', 'L2_errTrap', 'Cinf_err', 'Nt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for step in range(steps):

        if model == 'SmoothSol':

            fac = 2**(step)
            # Set discretization level
            nt = 30 * fac
            nx = 5 * fac
            ny = 5 * fac
            T = [0, 1]

            # Mesh for stochastic part
            mx = UnitIntervalMesh(nx)
            x = SpatialCoordinate(mx)[0]

            # Mesh for deterministic part
            my = UnitIntervalMesh(ny)
            y = SpatialCoordinate(my)[0]

            # Set up coefficients as described above
            # Diffusion in x direction
            ax = as_matrix([[1]])

            # Convection in x direction
            bx = as_vector([0])

            # Convection in y direction
            def by(t, x, y): return -(y-0.5)*x*10

            # Potential term
            c = 0.0

            # Homogeneous boundary conditions
            g = 0

            # Set up variables for t and y
            t_ = Constant(T[1])
            y_ = Constant(0.0)

            # Define solution
            def u(t, x, y): return 0.5 * (t**2+1) * sin(2*x*pi) * sin(2*y*pi)

            # Declare also a numpy version of the solution (handy for plotting and evaluation)
            def u_numpy(t, x, y): return 0.5 * (t**2+1) * \
                np.sin(2*x*pi) * np.sin(2*y*pi)

            def u_T_numpy(x, y): return u_numpy(T[1], x, y)

            # Define right-hand side
            u_x = u(t_, x, y_)
            by_x = by(t_, x, y_)

            f = diff(u_x, t_) + c * u_x + inner(bx, grad(u_x)) + \
                by_x * diff(u_x, y_) + inner(ax, grad(grad(u_x)))

        # We have to use at least quadratic polynomials here
        Vx = FunctionSpace(mx, 'CG', 2)
        Vy = FunctionSpace(my, 'CG', 1)

        phi = TrialFunction(Vx)
        psi = TestFunction(Vx)

        v = Function(Vx)

        gamma = 1.0

        # Jump penalty term
        stab1 = 2.

        nE = FacetNormal(mx)
        hE = FacetArea(mx)

        test = div(grad(psi))
        S_ = gamma * inner(ax, grad(grad(phi))) * test * dx(mx) \
            + stab1 * avg(hE)**(-1) * inner(jump(grad(phi), nE), jump(grad(psi), nE)) * dS(mx) \
            + gamma * inner(bx, grad(phi)) * test * dx(mx) \
            + gamma * c * phi * test * dx(mx)

        # This matrix also changes since we are testing the whole equation
        # with div(grad(psi)) instead of psi
        M_ = gamma * phi * test * dx(mx)

        bc_Vx = DirichletBC(Vx, g, 'on_boundary')
        S = assemble(S_)
        M = assemble(M_)

        # Prepare special treatment of deterministic part and time-derivative.
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

        # Initialize auxiliary functions
        Mvtmp = Function(Vx)
        vsol = Function(Vx)

        # Set final time
        # Initialize value at T[1]
        t = trange[1]
        vold = u_T_numpy(X, Y)

        if storeV:
            Vsol = np.ndarray([nt+1, len(ydofs), len(xdofs)])
            Vsol[0, :, :] = vold

        L_ = gamma * f * test * dx(mx)

        for i, t in enumerate(trange[1:]):
            print('Solve problem, time step t = {}'.format(np.round(t, 4)))

            # Assign current time to variable t_
            t_.assign(t)

            # Compute Ytilde, i.e. y-position of process starting in x_i,y_j
            # after application of convection in y-direction
            Ytilde = Y + by(t, X, Y)*dt[i]
            Vtilde = Yinterp1(vold, Ytilde, ydofs, ydiff)

            # Set up problem
            Sfull = M - dt[i] * S

            # Apply boundary conditions
            bc_Vx.apply(Sfull)

            # Loop over all layers in y-direction
            for j in range(ny+1):

                y_.assign(ydofs[j])
                L = assemble(L_)

                # This is ordinary matrix multiplication
                # np.max(np.abs(M*Vtilde[3,:] - M.array() @ Vtilde[3,:]))

                Mvtmp.vector().set_local(M * Vtilde[j, :])

                Lfull = Mvtmp.vector() - dt[i] * L
                bc_Vx.apply(Lfull)

                # Solve problem in j'th layer
                solve(Sfull, vsol.vector(), Lfull)

                # Store solution for j'th layer
                vold[j, :] = vsol.vector().get_local()

            if storeV:
                Vsol[i+1, :, :] = vold

        if plotSol:
            Z = np.abs(vold - u_numpy(trange[i], X, Y))

            if plotMode == 'contour':
                plotViContour(X[:, ix], Y[:, ix], Z[:, ix],)

            if plotMode == '3d':
                plotVi3d(X[:, ix], Y[:, ix], Z[:, ix],)

        if storeV:
            # Decide for which time slice to compute the errors
            k = -1
            t = trange[k]
            Z = np.abs(Vsol[k, :, :] - u_numpy(trange[k], X, Y))
        else:
            Z = np.abs(vold - u_numpy(t, X, Y))

        # Determine L^2 error for each slice and sum them
        l2err = 0
        for j in range(len(ydofs)):
            vdiff = Function(Vx)
            vdiff.vector().set_local(Z[j, :])
            if j == 0:
                fac = ydiff[0]/2
            elif j == len(ydofs)-1:
                fac = ydiff[-1]/2
            else:
                fac = ydiff[[j-1, j]].mean()

            l2err += np.power(norm(vdiff, 'L2'), 2) * fac
        cinferr = Z.max()
        l2err = np.sqrt(l2err)

        l2errtrap = 0
        for j in range(len(ydofs)-1):
            vdiff = Function(Vx)
            vdiff.vector().set_local(0.5*(Z[j, :] + Z[j+1, :]))
            l2errtrap += np.power(norm(vdiff, 'L2'), 2) * ydiff[j]

        # Compute L2 error by trapezoidal rule
        l2errtrap = np.sqrt(l2errtrap)
        L2errTrap.append(l2errtrap)

        ndofs = len(ydofs) * len(xdofs)
        Ndofs.append(ndofs)
        hxmax = max(np.diff(np.sort(mx.coordinates().flatten())))
        hymax = ydiff.max()
        hmax = np.maximum(hxmax, hymax)
        Hmax.append(hmax)
        L2err.append(l2err)
        Cinferr.append(cinferr)
        Nt.append(nt)

        print('Step: ', step)
        print('Ndofs: ', ndofs)
        print('Hmax: ', hmax)
        print('L2 err: ', l2err)
        print('L2 err (trapezoidal): ', l2errtrap)
        print('Cinf err: ', cinferr)

        if writeErrors:
            # with open(filename, 'wa', newline='') as csvfile:
            writer.writerow({'Step': step,
                             'Ndofs': ndofs,
                             'Hmax': hmax,
                             'L2_err': l2err,
                             'L2_errTrap': l2errtrap,
                             'Cinf_err': cinferr,
                             'Nt': nt})

    if writeErrors:
        csvfile.close()

    # Plot and write errors
    if plotErrors:
        plt.loglog(Hmax, L2err, label='L2 error')
        plt.loglog(Hmax, Cinferr, label='Cinf error')
        plt.show()
