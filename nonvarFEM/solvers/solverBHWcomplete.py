from dolfin import *
# from dolfin import Function, TestFunctions, TrialFunctions
# from dolfin import DirichletBC
# from dolfin import project, assemble_system, solve
# from dolfin import inner, grad, jump, avg, dx, div, dS, ds
from time import time
from nonvarFEM.norms import vj, mj
import sys


def solverBHWcomplete(P, opt):

    gamma = P.normalizeSystem(opt)

    (trial_H, trial_p, trial_u) = TrialFunctions(P.mixedSpace)
    (test_H,  test_p,  test_u) = TestFunctions(P.mixedSpace)

    # Set up bilinear form

    # 1-1 block: mass matrix in tensor space W_H
    a = inner(trial_H, test_H) * dx

    # 1-3 block: 'stiffness' matrices for FE Hessians
    a += inner(grad(trial_u), div(test_H)) * dx \
        - inner(grad(trial_u), test_H * P.nE) * ds
    if opt['HessianSpace'] == 'DG':
        a += - inner(avg(grad(trial_u)), jump(test_H, P.nE)) * dS

    # qdegree = 1

    # 2-1 block: weighted mass matrix
    a += - gamma * inner(P.a, trial_H) * test_p * dx

    # 2-2 block: mass matrix in W_H
    a += inner(trial_p, test_p) * dx

    # 3-2 block: 'stiffness' matrix for FE Laplacian
    a += - inner(grad(test_u), grad(trial_p)) * dx \
        + inner(grad(test_u), trial_p * P.nE) * ds

    if opt['HessianSpace'] == 'DG':
        a += inner(avg(grad(test_u)), jump(trial_p, P.nE)) * dS

    if P.hasDrift:
        a += - gamma * inner(P.b, grad(trial_u)) * test_p * dx

    if P.hasPotential:
        a += - gamma * P.c * trial_u * test_p * dx

    # Add stabilization if necessary
    # 3-3 block: 'stabilization'
    if opt["stabilizationFlag"] > 0:
        if opt["stabilityConstant1"] > 0:
            a += opt["stabilityConstant1"] * avg(P.hE)**(-1) * \
                inner(vj(grad(trial_u), P.nE), vj(grad(test_u), P.nE)) * dS
        if opt["stabilityConstant2"] > 0:
            a += opt["stabilityConstant2"] * avg(P.hE)**(+1) * \
                inner(mj(grad(grad(trial_u)), P.nE),
                      mj(grad(grad(test_u)), P.nE)) * dS

    # Set up the right-hand side
    f = gamma * P.f

    # Adjust system matrix and load vector in case of time-dependent problem
    if P.isTimeDependant:
        a += +gamma / P.dt * trial_u * test_p * dx  # + seems to be correct here
        f += -gamma / P.dt * P.u_np1               # - seems to be correct here

    fh = project(f, P.W_P)

    # print('write f to file')
    # file_f = File('f.pvd')
    # file_f << fh
    # plt.figure('fh', clear = True)
    # c = plot(fh, mode='color')
    # plt.colorbar(c)
    # plt.show()

    # print(str(P.f)) # Shows that replacement changes coefficients
    # ipdb.set_trace()
    # print('mean of abs of fh: ', np.mean(np.abs(fh.vector().get_local()))) # This is also changed

    L = -inner(grad(fh), grad(test_u))*dx + inner(grad(test_u), P.nE)*fh*ds
    # print('mean of abs of L: ', np.mean(np.abs(assemble(L).get_local()))) # This is also changed

    # Set inhomogeneous Dirichlet boundary conditions for V part of mixed space
    print('Setting boundary conditions')
    if isinstance(P.g, list):
        bc_V = []
        for fun, dom in P.g:
            bc_V.append(DirichletBC(P.mixedSpace.sub(2), fun, dom))
    else:
        if isinstance(P.g, Function):
            bc_V = DirichletBC(P.mixedSpace.sub(2), P.g, 'on_boundary')
        else:
            g = project(P.g, P.V)
            bc_V = DirichletBC(P.mixedSpace.sub(2), g, 'on_boundary')

    if opt['time_check']:
        t1 = time()

    # S, rhs = assemble_system(a, L, bc_V)
    # solve(S, P.x.vector(), rhs)
    solve(a == L, P.x, bc_V, solver_parameters={'linear_solver': 'mumps'})
    import ipdb
    ipdb.set_trace()

    TEST_RITZ = False
    if TEST_RITZ:
        ufl_cell = P.mesh.ufl_cell()
        CG2 = FunctionSpace(P.mesh, FiniteElement("CG", ufl_cell, 2))
        uh = TrialFunction(CG2)
        vh = TestFunction(CG2)

        ah = -inner(grad(uh),grad(vh)) * dx
        fh = P.f * vh * dx
        ustar = Function(CG2)
        solve(ah == fh, ustar, DirichletBC(CG2, P.g, 'on_boundary'))
        c = plot(P.u - ustar)
        import matplotlib.pyplot as plt
        plt.colorbar(c)
        plt.show()
        import ipdb
        ipdb.set_trace()



    # rhs = assemble(L)
    # S = assemble(a)
    # bc_V.apply(S,rhs)
    # ipdb.set_trace()
    # solve(S==rhs)
    # solver =
    # solver.solve()

    # solve(a==L, P.x, bc_V, solver_parameters={'linear_solver': 'mumps'})

    if opt['time_check']:
        print("Solve linear equation system ... %.2fs" % (time()-t1))
        sys.stdout.flush()

    return 1
