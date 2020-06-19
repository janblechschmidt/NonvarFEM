from dolfin import Function, TestFunctions, TrialFunctions
from dolfin import DirichletBC
from dolfin import project, assemble_system, solve
from dolfin import inner, grad, avg, dx, div, dS, ds
from time import time
import sys


def solverNeilan(P, opt):
    '''
    This function implements the method presented in
    Neilan:Convergence Analysis of a Finite Element
    Method for Second Order Non-Variational Problems
    Main characteristics:
    - no stabilization
    - method relies on a discontinuous Hessian approximation
    - no action on test function of solution u_h
    - no normalization of system
    '''

    (trial_u, trial_H) = TrialFunctions(P.mixedSpace)
    (test_u, test_H) = TestFunctions(P.mixedSpace)

    # Define bilinear form
    jump_outer = inner(grad(trial_u), test_H*P.nE) * ds
    jump_inner = inner(avg(grad(trial_u)), test_H(
        '+')*P.nE('+') + test_H('-')*P.nE('-')) * dS

    a_h = -inner(P.a, trial_H) * test_u * dx \
        + inner(trial_H, test_H) * dx \
        + inner(grad(trial_u), div(test_H)) * dx \
        - jump_inner - jump_outer

    if P.hasDrift:
        a_h += - inner(P.b, grad(trial_u)) * test_u * dx

    if P.hasPotential:
        a_h += - P.c * trial_u * test_u * dx

    # Define linear form
    f_h = -P.f * test_u * dx

    # Adjust system matrix and load vector in case of time-dependent problem
    if P.isTimeDependant:
        a_h += 1. / P.dt * trial_u * test_u * dx
        f_h += 1. / P.dt * P.u_np1 * test_u * dx

    # Set boundary conditions
    print('Setting boundary conditions')
    if isinstance(P.g, list):
        bc_V = []
        for fun, dom in P.g:
            bc_V.append(DirichletBC(P.mixedSpace.sub(0), fun, dom))
    else:
        if isinstance(P.g, Function):
            bc_V = DirichletBC(P.mixedSpace.sub(0), P.g, 'on_boundary')
        else:
            g = project(P.g, P.V)
            bc_V = DirichletBC(P.mixedSpace.sub(0), g, 'on_boundary')

    if opt['time_check']:
        t1 = time()

    S, rhs = assemble_system(a_h, f_h, bc_V)
    try:
        solve(S, P.x.vector(), rhs)

    except RuntimeError:
        try:
            solve(S, P.x.vector(), rhs, 'gmres', 'hypre_amg')
        except RuntimeError:
            import ipdb
            ipdb.set_trace()

    N_iter = 1

    if opt['time_check']:
        print("Solve linear equation system ... %.2fs" % (time() - t1))
        sys.stdout.flush()

    return N_iter
