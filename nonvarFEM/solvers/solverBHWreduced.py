from dolfin import Function, TestFunction, TrialFunction, Constant
from dolfin import DirichletBC, as_backend_type, assemble, interpolate, assign
from dolfin import project, inner, grad, avg, dx, dS, ds

import ufl
from ufl.conditional import Conditional
from ufl.constantvalue import Zero
from ufl.indexed import Indexed

import matplotlib.pyplot as plt

from time import time
import sys
import os

# Basic linear algebra
import numpy as np
from nonvarFEM.norms import vj, mj

# Advanced linear algebra
import scipy.sparse as sp
import scipy.sparse.linalg as la
import itertools
import ipdb

def checkForZero(this_a):
    if isinstance(this_a, ufl.constantvalue.FloatValue):
        if abs(this_a.value()) < 1e-12:
            return True
        else:
            return False

    elif isinstance(this_a, Constant):
        if abs(this_a.values()[0]) < 1e-12:
            return True
        else:
            return False

    elif isinstance(this_a, Conditional):
        return False

    elif isinstance(this_a, Zero):
        return True
    
    elif isinstance(this_a, Indexed):
        return False
    else:
        print('Unknown instance in check for zeros')
        ipdb.set_trace()
        raise ValueError('Unknown instance in check for zeros')


def spy(A):
    ind = A > 0

    try:
        xi, yi = np.where(ind)
    except ValueError:
        xi, yi = np.where(ind.toarray())

    plt.scatter(xi, yi, s=1)

    ax = plt.gca()
    ylim = ax.get_ylim()
    ax.set_ylim([ylim[1], ylim[0]])
    ax.legend()

    plt.show()


class cgMat(object):
    def __init__(self, A):
        self.A = A
        self.shape = A.shape
        n = self.shape[0]
        self.Prec = sp.spdiags(1. / A.diagonal(), np.array([0]), n, n)

    def solve(self, x):
        return la.cg(self.A, x, M=self.Prec, tol=1e-10)[0]


class gmresMat(object):
    def __init__(self, A, tol=1e-10):
        self.A = A
        self.tol = tol

        self.shape = A.shape
        n = self.shape[0]
        self.Prec = sp.spdiags(1. / A.diagonal(), np.array([0]), n, n)

    def solve(self, x):
        return la.lgmres(self.A, x, M=self.Prec, tol=self.tol)[0]


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.t1 = time()

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            self.t2 = time()
            print('iter %3i\trk = %s\ttime = %s' %
                  (self.niter, str(rk), str(self.t2 - self.t1)))
            self.t1 = time()


def spmat(myM):
    M_mat = as_backend_type(myM).mat()
    M_sparray = sp.csr_matrix(M_mat.getValuesCSR()[::-1], shape=M_mat.size)

    return M_sparray.tocsc()


def solverBHWreduced(P, opt):
    ''' Function to solve the second-order pde in
    nonvariational formulation using gmres '''

    if P.hasDrift:
        raise ValueError('Method currently does not support non-zero drift terms.')
    if P.isTimeDependant:
        raise ValueError('Method currently does not support parabolic problems.')
    if P.hasPotential:
        raise ValueError('Method currently does not support non-zero potential terms.')

    gamma = P.normalizeSystem(opt)

    # Extract local space of tensor space W_h
    W_H_loc = P.mixedSpace.sub(1).extract_sub_space(np.array([0])).collapse()

    trial_u = TrialFunction(P.V)
    test_u = TestFunction(P.V)

    trial_p = TrialFunction(W_H_loc)
    test_p = TestFunction(W_H_loc)

    # Get number of dofs in V
    N = P.V.dim()
    NW = W_H_loc.dim()

    # Get indices of inner and boundary nodes
    # Calling DirichletBC with actual P.g calls project which is expensive
    bc_V = DirichletBC(P.V, Constant(1), 'on_boundary')
    idx_bnd = list(bc_V.get_boundary_values().keys())
    N_bnd = len(idx_bnd)
    N_in = N - N_bnd

    # Get indices of inner nodes
    idx_inner = np.setdiff1d(range(N), idx_bnd)

    def _i2a(S):
        return S[:, idx_inner]

    def _i2i(S):
        return S[idx_inner, :][:, idx_inner]

    def _b2a(S):
        return S[:, idx_bnd]

    def _b2i(S):
        return S[idx_inner, :][:, idx_bnd]

    # Assemble mass matrix and store LU decomposition
    M_W = spmat(assemble(trial_p * test_p * dx))
    # spy(M_W)

    if opt['time_check']:
        t1 = time()

    # M_LU = la.splu(M_W)
    M_LU = cgMat(M_W)

    if opt['time_check']:
        print("Compute LU decomposition of M_W ... %.2fs" % (time() - t1))
        sys.stdout.flush()

    # Check for constant zero entries in diffusion matrix

    # The diagonal entries are set up everytime
    nzdiags = [(i, i) for i in range(P.dim())]

    # The off-diagonal entries
    nzs = []
    Asym = True
    for (i, j) in itertools.product(range(P.dim()), range(P.dim())):
        if i == j:
            nzs.append((i, j))
        else:
            is_zero = checkForZero(P.a[i, j])
            if not is_zero:
                Asym = False
                print('Use value ({},{})'.format(i, j))
                nzs.append((i, j))
            else:
                print('Ignore value ({},{})'.format(i, j))

    def emptyMat(d=P.dim()):
        return [[-1e15] * d for i in range(d)]
        # return [[0] * d for i in range(d)]

    # Assemble weighted mass matrices
    B = emptyMat()
    for (i, j) in nzs:
        # ipdb.set_trace()
        this_form = gamma * P.a[i, j] * trial_p * test_p * dx
        B[i][j] = spmat(assemble(this_form))

    # Init array for partial stiffness matrices
    C = emptyMat()

    # Set up the form for partial stiffness matrices
    def C_form(i, j):
        this_form = -trial_u.dx(i) * test_p.dx(j) * \
            dx + trial_u.dx(i) * test_p * P.nE[j] * ds
        if opt["HessianSpace"] == 'DG':
            this_form += avg(trial_u.dx(i)) \
                * (test_p('+') * P.nE[j]('+') + test_p('-') * P.nE[j]('-')) * dS
        return this_form

    # Ensure the diagonal is set up, necessary for the FE Laplacian
    for i in range(P.dim()):
        C[i][i] = spmat(assemble(C_form(i, i)))

    # Assemble the partial stiffness matrices for off-diagonal entries
    # only if a_i,j is non-zero
    for i, j in nzs:
        if i != j:
            C[i][j] = spmat(assemble(C_form(i, j)))

    C_lapl = sum([C[i][j] for i, j in nzdiags])

    # Set up the form for stabilization term
    if opt["stabilizationFlag"] > 0:

        this_form = 0

        # First stabilization term
        if opt["stabilityConstant1"] > 0:
            this_form += opt["stabilityConstant1"] * avg(P.hE)**(-1) * \
                inner(vj(grad(trial_u), P.nE), vj(grad(test_u), P.nE)) * dS

        # Second stabilization term
        if opt["stabilityConstant2"] > 0:
            this_form += opt["stabilityConstant2"] * avg(P.hE)**(+1) * \
                inner(mj(grad(grad(trial_u)), P.nE),
                      mj(grad(grad(test_u)), P.nE)) * dS

        # Assemble stabilization term
        S = spmat(assemble(this_form))

    else:

        # If no stabilization is used, S is zero
        S = sp.csc_matrix((N, N))

    # Set up matrix-vector product for inner nodes
    def S_II_times_u(x):
        Dv = [B[i][j] * M_LU.solve(_i2a(C[i][j]) * x) for i, j in nzs]
        w = M_LU.solve(sum(Dv))
        return (_i2a(C_lapl)).transpose() * w + _i2i(S) * x

    # Set up matrix-vector product for boundary nodes
    def S_IB_times_u(x):
        Dv = [B[i][j] * M_LU.solve(_b2a(C[i][j]) * x) for i, j in nzs]
        w = M_LU.solve(sum(Dv))
        return (_i2a(C_lapl)).transpose() * w + _b2i(S) * x

    # Assemble f_W
    f_W = assemble(gamma * P.f * test_p * dx)

    M_in = la.LinearOperator((N_in, N_in), matvec=lambda x: S_II_times_u(x))
    M_bnd = la.LinearOperator((N_in, N_bnd), matvec=lambda x: S_IB_times_u(x))

    # Set up right-hand side
    try:
        print('Project boundary data with LU')
        G = project(P.g, P.V, solver_type="lu")
    except RuntimeError:
        print('Out of memory error: Switch to projection with CG')
        G = project(P.g, P.V, solver_type="cg")

    # Compute right-hand side
    rhs = (_i2a(C_lapl)).transpose() * M_LU.solve(f_W.get_local())

    Gvec = G.vector()
    g_bc = Gvec.get_local().take(idx_bnd)
    rhs -= M_bnd * g_bc

    M_W_DiagInv = sp.spdiags(1. / M_W.diagonal(), np.array([0]), NW, NW, format='csc')
    D = sum([sp.spdiags(B[i][j].diagonal(), np.array([0]), NW, NW, format='csc')
             * M_W_DiagInv * _i2a(C[i][j]) for i, j in nzs])
    Prec = (_i2a(C_lapl)).transpose().tocsc() * M_W_DiagInv * D + _i2i(S)

    if opt['time_check']:
        t1 = time()

    # Determine approximate size of LU decomposition in GB
    LU_size = NW**2 / (1024.**3)
    MEM_size = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3)

    if LU_size / MEM_size > 0.8:
        if Asym:
            print('Use CG for preconditioning')
            Prec_LU = cgMat(Prec)
        else:
            print('Use GMRES for preconditioning')
            Prec_LU = gmresMat(Prec)
    else:
        try:
            print('Use LU for preconditioning')
            Prec_LU = la.splu(Prec)
        except MemoryError:
            if Asym:
                print('Use CG for preconditioning')
                Prec_LU = cgMat(Prec)
            else:
                print('Use GMRES for preconditioning')
                Prec_LU = gmresMat(Prec)

    PrecLinOp = la.LinearOperator(
        (N_in, N_in), matvec=lambda x: Prec_LU.solve(x))

    # import ipdb
    # ipdb.set_trace()
    # gmres_mode = 1  # LU decomposition of Prec
    # # gmres_mode = 3  # incomplete LU decomposition of Prec
    # # gmres_mode = 4  # aslinearop
    # # Findings during experiments
    # # - LU factorization of prec is fast, high memory demand
    # # - Using only the diag of prec is not suitable
    # # - Solve routine from scipy is slow

    # # 1st variant: determine LU factorization of preconditioner
    # if gmres_mode == 1:
    #     Prec_LU = la.splu(Prec)
    #     PrecLinOp = la.LinearOperator(
    #         (N_in, N_in), matvec=lambda x: Prec_LU.solve(x))

    # # 3rd variant: determine incomplete LU factorization of preconditioner
    # if gmres_mode == 3:
    #     if P.solDofs(opt) < MEM_THRESHOLD:
    #         fill_factor = 20
    #         fill_factor = 30
    #         print('Use incomplete LU with fill factor {} for preconditioning'.format(fill_factor))
    #         Prec_LU = la.spilu(Prec,
    #                            fill_factor=fill_factor,
    #                            drop_tol=1e-4)
    #     else:
    #         print('Use gmres for preconditioning')
    #         Prec_LU = gmresMat(Prec, tol=1e-8)
    #         # print('Use cg for preconditioning')
    #         # Prec_LU = cgMat(Prec)
    #     PrecLinOp = la.LinearOperator(
    #         (N_in, N_in), matvec=lambda x: Prec_LU.solve(x))

    # if gmres_mode == 4:

    #     if P.solDofs(opt) < MEM_THRESHOLD:
    #         Prec_LU = la.splu(Prec)
    #     else:
    #         Prec_LU = gmresMat(Prec)

    #     PrecLinOp = la.LinearOperator(
    #         (N_in, N_in), matvec=lambda x: Prec_LU.solve(x))

    if opt['time_check']:
        t2 = time()
        print("Prepare GMRES (e.g. LU decomp of Prec) ... %.2fs" % (t2 - t1))
        sys.stdout.flush()

    do_savemat = 0
    if do_savemat:
        from scipy.io import savemat
        savemat('M_{}.mat'.format(P.meshLevel),
                mdict={'Prec': Prec,
                       'B': B,
                       'C': C,
                       'S': S,
                       'M': M_W,
                       'f': f_W.get_local(),
                       'g': Gvec.get_local(),
                       'idx_inner': idx_inner,
                       'idx_bnd': idx_bnd,
                       })

    if P.meshLevel == 1 or not opt["gmresWarmStart"]:
        x0 = np.zeros(N_in)
    else:
        tmp = interpolate(P.uold, P.V)
        x0 = tmp.vector().get_local()[idx_inner]

    # Initialize counter for GMRES
    counter = gmres_counter(disp=True)

    # System solve
    (x, gmres_flag) = la.gmres(A=M_in,
                               b=rhs,
                               M=PrecLinOp,
                               x0=x0,
                               maxiter=2000,
                               tol=opt["gmresTolRes"],
                               atol=opt["gmresTolRes"],
                               restart=20,
                               callback=counter)

    if opt['time_check']:
        print("Time for GMRES ... %.2fs" % (time() - t2))
        sys.stdout.flush()

    print('GMRES output flag: {}'.format(gmres_flag))

    N_iter = counter.niter

    u_loc = Function(P.V)
    u_loc.vector()[idx_bnd] = g_bc
    u_loc.vector()[idx_inner] = x

    # Set solution to problem structure
    assign(P.u, u_loc)

    # Compute FE Hessians
    # import ipdb
    # ipdb.set_trace()

    for (i, j) in itertools.product(range(P.dim()), range(P.dim())):
        if (i, j) in nzs:
            Hij = Function(P.W_H.sub(i*P.dim() + j).collapse())
            hij = M_LU.solve(C[i][j] * u_loc.vector())
            Hij.vector()[:] = hij
            assign(P.H.sub(i*P.dim() + j), Hij)

    return N_iter
