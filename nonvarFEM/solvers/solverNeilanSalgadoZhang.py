from dolfin import Function, TestFunction, TrialFunction
from dolfin import DirichletBC
from dolfin import project, assemble_system, solve, assemble
from dolfin import inner, grad, jump, avg, dx, div, dS

from time import time
import sys


def solverNeilanSalgadoZhang(P, opt):
    '''
    This function implements the method presented in
    Neilan, Salgado, Zhang (2017), Chapter 4
    Main characteristics:
    - test with piecewise second derivative of test_u
    - no discrete Hessian
    - first-order stabilization necessary
    '''

    assert opt["HessianSpace"] == "CG", 'opt["HessianSpace"] has to be "CG"'
    assert opt["stabilityConstant1"], 'opt["stabilityConstant1"] has to be positive'

    gamma = P.normalizeSystem(opt)

    if isinstance(P.g, list):
        bc_V = []
        for fun, dom in P.g:
            bc_V.append(DirichletBC(P.V, fun, dom))
    else:
        if isinstance(P.g, Function):
            bc_V = DirichletBC(P.V, P.g, 'on_boundary')
        else:
            g = project(P.g, P.V)
            bc_V = DirichletBC(P.V, g, 'on_boundary')

    trial_u = TrialFunction(P.V)
    test_u = TestFunction(P.V)

    # Assemble right-hand side in each case
    f_h = gamma * P.f * div(grad(test_u)) * dx
    
    # Adjust load vector in case of time-dependent problem
    if P.isTimeDependant:
        f_h = gamma *P.u_np1 * div(grad(test_u)) * dx - P.dt * f_h
    
    rhs = assemble(f_h)
    if isinstance(bc_V, list):
        for bc_v in bc_V:
            bc_v.apply(rhs)
    else:
        bc_V.apply(rhs)
    
    repeat_setup = False
    
    if hasattr(P, 'timeDependentCoefs'):
        if P.timeDependentCoefs:
            repeat_setup = True
    
    if 'HJB' in str(type(P)):
        repeat_setup = True

    if (not P.isTimeDependant) or (P.iter == 0) or repeat_setup:

        print('Setup bilinear form')

        # Define bilinear form
        a_h = gamma * inner(P.a, grad(grad(trial_u))) * div(grad(test_u)) * dx \
            + opt["stabilityConstant1"] * avg(P.hE)**(-1) * inner(
                jump(grad(trial_u), P.nE), jump(grad(test_u), P.nE)) * dS
    
        if P.hasDrift:
            a_h += gamma * inner(P.b, grad(trial_u)) * div(grad(test_u)) * dx
    
        if P.hasPotential:
            a_h += gamma * P.c * trial_u * div(grad(test_u)) * dx

        # Adjust system matrix in case of time-dependent problem
        if P.isTimeDependant:
            a_h = gamma * trial_u * div(grad(test_u)) * dx - P.dt * a_h

        S = assemble(a_h)
        if isinstance(bc_V, list):
            for bc_v in bc_V:
                bc_v.apply(S)
        else:
            bc_V.apply(S)

        P.S = S

    if opt['time_check']:
        t1 = time()

    solve(P.S, P.u.vector(), rhs)

    # tmp = assemble(inner(P.a,grad(grad(trial_u))) * div(grad(test_u)) * dx)
    # A = assemble(a_h)
    # print('Row sum: ', sum(A.array(),1).round(4))
    # print('Col sum: ', sum(A.array(),0).round(4))
    # print('Total sum: ', sum(sum(A.array())).round(4))
    # ipdb.set_trace()
    # solve(a_h == f_h, P.u, bc_V, solver_parameters={'linear_solver': 'mumps'})

    if opt['time_check']:
        print("Solve linear equation system ... %.2fs" % (time()-t1))
        sys.stdout.flush()

    N_iter = 1

    return N_iter
