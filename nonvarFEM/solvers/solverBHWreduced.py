from dolfin import Function, TestFunction, TrialFunction, Constant
from dolfin import DirichletBC, as_backend_type, assemble, interpolate, assign
from dolfin import project, inner, grad, avg, dx, dS, ds

import ufl
from ufl.conditional import Conditional

import matplotlib.pyplot as plt

from time import time
import sys

# Basic linear algebra
import numpy as np
from nonvarFEM.norms import vj, mj

# Advanced linear algebra
import scipy.sparse as sp
import scipy.sparse.linalg as la
import itertools


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
    else:
        print('Unknown instance in check for zeros')
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
        return la.cg(self.A, x, M=self.Prec, tol=1e-12)[0]


class gmresMat(object):
    def __init__(self, A):
        self.A = A
        self.shape = A.shape
        n = self.shape[0]
        self.Prec = sp.spdiags(1. / A.diagonal(), np.array([0]), n, n)

    def solve(self, x):
        return la.lgmres(self.A, x, M=self.Prec, tol=1e-12)[0]


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


def spmat(myM):
    M_mat = as_backend_type(myM).mat()
    M_sparray = sp.csr_matrix(M_mat.getValuesCSR()[::-1], shape=M_mat.size)

    return M_sparray.tocsc()


def solverBHWreduced(P, opt):
    ''' Function to solve the second-order pde in
    nonvariational formulation using gmres '''

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
    for (i, j) in itertools.product(range(P.dim()), range(P.dim())):
        if i == j:
            nzs.append((i, j))
        else:
            is_zero = checkForZero(P.a[i, j])
            if not is_zero:
                print('Use value ({},{})'.format(i, j))
                nzs.append((i, j))
            else:
                print('Ignore value ({},{})'.format(i, j))

    def emptyMat(d=P.dim()):
        return [[None] * d for i in range(d)]

    # Assemble weighted mass matrices
    B = emptyMat()
    for (i, j) in nzs:
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
        S = sp.csr_matrix((N, N))

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
    if P.solDofs(opt) < 20000:
        G = project(P.g, P.V, solver_type="lu")
    else:
        G = project(P.g, P.V, solver_type="cg")
    # try:
    #     print('Interpolation works for G')
    #     G = interpolate(P.g, P.V)
    # except AttributeError:
    #     print('Has to use projection for G')
    #     G = project(P.g, P.V)

    # Compute right-hand side
    rhs = (_i2a(C_lapl)).transpose() * M_LU.solve(f_W.get_local())

    Gvec = G.vector()
    g_bc = Gvec.get_local().take(idx_bnd)
    rhs -= M_bnd * g_bc

    # ipdb.set_trace()
    # G2 = dolfin.UserExpression(P.g, element=P.uElement)
    # ipdb.set_trace()
    # rhs2 = (_i2a(C_lapl)).transpose() * M_LU.solve(f_W.get_local())

    # Gvec = G2.vector()
    # g_bc = Gvec.get_local().take(idx_bnd)
    # rhs2 -= M_bnd * g_bc

    # qel = dolfin.VectorElement(family='Quadrature',cell=P.mesh.ufl_cell(),degree=2,quad_scheme='default')
    # Q_V = dolfin.FunctionSpace(P.mesh, qel)
    # Q_g = dolfin.UserExpression(P.g, element=qel)

    # Set up preconditioner
    # M_W_orig = assemble(trial_p * test_p * dx)
    # import ipdb
    # ipdb.set_trace()

    M_W_DiagInv = sp.spdiags(1. / M_W.diagonal(), np.array([0]), NW, NW)
    D = sum([B[i][j] * M_W_DiagInv * _i2a(C[i][j]) for i, j in nzs])
    Prec = (_i2a(C_lapl)).transpose() * M_W_DiagInv * D + _i2i(S)

    # Initialize counter for GMRES
    counter = gmres_counter(disp=True)
    # counter = gmres_counter(disp=False)

    if opt['time_check']:
        t1 = time()

    # gmres_mode = 1  # LU decomposition of Prec
    # gmres_mode = 2  # solve routine of scipy
    # gmres_mode = 3  # incomplete LU decomposition of Prec
    gmres_mode = 4  # aslinearop
    # gmres_mode = 5  # Diag of prec

    # 1st variant: determine LU factorization of preconditioner
    if gmres_mode == 1:
        Prec_LU = la.splu(Prec)
        PrecLinOp = la.LinearOperator(
            (N_in, N_in), matvec=lambda x: Prec_LU.solve(x))

    # 2nd variant: without LU factorization of preconditioner
    if gmres_mode == 2:
        PrecLinOp = la.LinearOperator(
            (N_in, N_in), matvec=lambda x: sp.linalg.spsolve(Prec, x))

    # 3rd variant: determine incomplete LU factorization of preconditioner
    if gmres_mode == 3:
        Prec_LU = la.spilu(Prec)
        PrecLinOp = la.LinearOperator(
            (N_in, N_in), matvec=lambda x: Prec_LU.solve(x))

    if gmres_mode == 4:

        if P.solDofs(opt) < 20000:
            Prec_LU = la.splu(Prec)
        else:
            Prec_LU = gmresMat(Prec)

        PrecLinOp = la.LinearOperator(
            (N_in, N_in), matvec=lambda x: Prec_LU.solve(x))

    if gmres_mode == 5:
        PrecDiag = sp.spdiags(1. / Prec.diagonal(), np.array([0]), N_in, N_in)
        PrecLinOp = la.aslinearoperator(PrecDiag)

    if opt['time_check']:
        t2 = time()
        print("Prepare GMRES (e.g. LU decomp of Prec) ... %.2fs" % (t2 - t1))
        sys.stdout.flush()

    # System solve
    (x, gmres_flag) = la.gmres(A=M_in,
                               b=rhs,
                               M=PrecLinOp,
                               x0=np.zeros(N_in),
                               maxiter=1000,
                               tol=opt["gmresTolRes"],
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
    # TODO adapt this for new implementation
    # Hij = Function(W_H_loc)
    # for (i, j) in itertools.product(range(P.dim()), range(P.dim())):
    #     hij = M_LU.solve(C_in[i*P.dim() + j] * x)
    #     Hij.vector()[:] = hij
    #     assign(P.H.sub(i*P.dim() + j), Hij)

    return N_iter
