# Import FEniCS dolfin
from dolfin import Constant, UserExpression

from dolfin import as_vector, as_matrix, inner, grad, Dx
from dolfin import SpatialCoordinate, UnitIntervalMesh, RectangleMesh, Point, UnitCubeMesh

from dolfin import exp, sin, pi

import numpy as np
import ufl

# Import some standard meshes
from .meshes import mesh_UnitSquare

# Import base class of problems
from nonvarFEM.pdes.parabolicNVP import ParabolicNVP

# Import explicit solution for worst of two assets put
from nonvarFEM.problems.explicitSolutionWorstOfTwoAssetsPut import Mfunc

# --------------------------------------------------
# Test example for solution in H^alpha
# --------------------------------------------------


class PNVP_1_2d(ParabolicNVP):
    """
    This problems is meant for testing. The explicit solution is
    u(x,y) = 1/2 * (t**2 + 1) * sin(2*x*pi) * sin(2*y*pi).
    The coefficients are constant in time:
    a = [[1,  0],[0, 2]]
    b = [0.2, -0.1]
    c = 0.1
    and the source term is chosen appropriately.
    """

    def __init__(self):

        # Time horizon
        self.T = [0., 1.0]
        self.t = Constant(self.T[1])

    def updateCoefficients(self):

        x, y = SpatialCoordinate(self.mesh)

        # Init coefficient matrix
        self.a = as_matrix(
            [[Constant(1.), Constant(0.)],
             [Constant(0.), Constant(2.)]])

        self.b = as_vector([Constant(0.2), Constant(-0.1)])
        self.c = Constant(0.1)

        self.u_ = 0.5 * (self.t**2+1) * sin(2*x*pi) * sin(2*y*pi)

        self.u_T = ufl.replace(self.u_, {self.t: self.T[1]})

        # Init right-hand side
        self.f = self.t * sin(2*pi*x) * sin(2*pi*y) \
            + inner(self.a, grad(grad(self.u_))) \
            + inner(self.b, grad(self.u_)) \
            + self.c * self.u_
        self.g = Constant(0.0)

    def initMesh(self, n):

        self.mesh = mesh_UnitSquare(n)


class PNVP_2(ParabolicNVP):
    def __init__(self):

        # Time horizon
        self.T = [0., 1.0]
        self.t = Constant(self.T[1])

    def updateCoefficients(self):

        x, y = SpatialCoordinate(self.mesh)

        mu = 1

        # Init coefficient matrix
        self.a = as_matrix([[1., Constant(0.)], [Constant(0.), 1.]])

        self.u_ = exp(-mu * (self.T[1] - self.t)) * sin(2*pi*x)*sin(1*pi*y)
        self.u_T = ufl.replace(self.u_, {self.t: self.T[1]})

        # Init right-hand side
        self.f = mu * self.u_ + inner(self.a, grad(grad(self.u_)))

        # Set boundary conditions to exact solution
        self.g = Constant(0.0)

    def initMesh(self, n):

        # Set mesh to square on [-1,1]^2
        self.mesh = mesh_UnitSquare(n)


class ExplicitSolution_WorstOfTwoAssetsPut(UserExpression):
    def __init__(self, t, K, R, Tmax, sigma1, sigma2, rho, **kwargs):

        super().__init__(**kwargs)

        self.t = t
        self.F = K
        self.R = R
        self.Tmax = Tmax
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho

    def eval(self, values, x):
        s = self.t.values()[0]
        values[0] = np.exp(-self.R*(self.Tmax-s)) * self.F \
            - Mfunc(x[0], x[1], 0, self.Tmax-s,
                    self.R, self.sigma1, self.sigma2, self.rho) \
            + Mfunc(x[0], x[1], self.F, self.Tmax-s,
                    self.R, self.sigma1, self.sigma2, self.rho)

    def value_shape(self):
        return ()


class PNVP_WorstOfTwoAssetsPut(ParabolicNVP):
    def __init__(self):

        # Time horizon
        self.T = [0.5, 1.0]
        self.t = Constant(self.T[1])

    def updateCoefficients(self):

        x, y = SpatialCoordinate(self.mesh)

        sigma1 = 0.3
        sigma2 = 0.3
        rho = 0.5
        r = 0.05
        K = 40

        self.u_T = ufl.Max(K - ufl.Min(x, y), 0)
        self.u_ = ExplicitSolution_WorstOfTwoAssetsPut(
            self.t, K, r, self.T[1], sigma1, sigma2, rho,
            cell=self.mesh.ufl_cell(), domain=self.mesh)

        # Init coefficient matrix
        self.a = 0.5 * \
            as_matrix([[sigma1**2 * x**2, rho*sigma1*sigma2*x*y],
                       [rho*sigma1*sigma2*x*y, sigma2**2 * y**2]])

        self.b = as_vector([r * x, r*y])
        self.c = Constant(-r)

        # Init right-hand side
        self.f = Constant(0.0)

        # Set boundary conditions to exact solution
        # self.g_t = lambda t : [(Constant(0.0), 'near(max(x[0],x[1]),200)'),
        #           (Constant(K * exp(-r*(self.T[1]-t))), 'near(min(x[0],x[1]),0)')]
        self.g = ExplicitSolution_WorstOfTwoAssetsPut(
            self.t, K, r, self.T[1], sigma1, sigma2, rho,
            cell=self.mesh.ufl_cell(), domain=self.mesh)

    def initMesh(self, n):
        xmax = 200
        ymax = 200
        self.mesh = RectangleMesh(Point(0., 0.), Point(xmax, ymax), n, n)


class PNVP_1_1d(ParabolicNVP):
    """
    This problems is meant for testing. The explicit solution is
    u(x,y) = 1/2 * (t**2 + 1) * sin(2*x*pi)
    The coefficients are constant in time:
    a = [[1]]
    b = [0.2]
    c = 0.1
    and the source term is chosen appropriately.
    """

    def __init__(self):

        # Time horizon
        self.T = [0., 1.0]
        self.t = Constant(self.T[1])

    def updateCoefficients(self):

        x = SpatialCoordinate(self.mesh)[0]

        # Init coefficient matrix
        self.a = as_matrix([[Constant(1.)]])
        self.b = as_vector([Constant(0.2)])
        self.c = Constant(0.1)

        self.u_ = 0.5 * (self.t**2+1) * sin(2*x*pi)
        self.u_T = ufl.replace(self.u_, {self.t: self.T[1]})

        # Init right-hand side
        self.f = self.t * sin(2*pi*x) \
            + inner(self.a, grad(grad(self.u_))) \
            + inner(self.b, grad(self.u_)) \
            + self.c * self.u_
        self.g = Constant(0.0)

    def initMesh(self, n):

        self.mesh = UnitIntervalMesh(n)


class PNVP_1_3d(ParabolicNVP):
    """
    This problems is meant for testing. The explicit solution is
    u(x,y) = 1/2 * (t**2 + 1) * sin(2*x*pi) * sin(2*y*pi) * sin(2*z*pi).
    The coefficients are constant in time:
    a = [[1,  0, 0],[0, 2, 0], [0, 0, 1]]
    b = [0.2, -0.1, 0.1]
    c = 0.1
    and the source term is chosen appropriately.
    """

    def __init__(self):

        # Time horizon
        self.T = [0., 1.0]
        self.t = Constant(self.T[1])

    def updateCoefficients(self):

        x, y, z = SpatialCoordinate(self.mesh)

        # Init coefficient matrix
        self.a = as_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
        self.b = as_vector([0.2, -0.1, 0.1])
        self.c = Constant(0.1)

        self.u_ = 0.5 * \
            (self.t**2+1) * sin(2*x*pi) * sin(2*y*pi) * sin(2*z*pi)
        # self.u_T = ufl.replace(self.u_, {self.t: self.T[1]})

        # Init right-hand side
        self.f = +self.t * sin(2*pi*x) * sin(2*pi*y)*sin(2*pi*z) \
            + inner(self.a, grad(grad(self.u_))) \
            + inner(self.b, grad(self.u_)) \
            + self.c * self.u_
        self.g = Constant(0.0)

    def initMesh(self, n):

        self.mesh = UnitCubeMesh(n, n, n)


class PNVP_Degenerate_1(ParabolicNVP):
    """
    This problems is meant for testing. The explicit solution is
    u(x,y) = 1/2 * (t**2 + 1) * sin(2*x*pi) * sin(2*y*pi)
    The coefficients are constant in time:
    a = [[1, 0],[0, 0]]
    b = [0.0, -10*(y-0.5)*x]
    c = 0.0
    and the source term is chosen appropriately.
    """

    def __init__(self, eps=0.01):

        # Time horizon
        self.T = [0., 1.0]
        self.t = Constant(self.T[1])
        self.eps = eps

    def updateCoefficients(self):

        x, y = SpatialCoordinate(self.mesh)

        # Init coefficient matrix
        self.a = as_matrix([[1., 0], [0, self.eps]])
        self.b = as_vector([0.0, -10*(y-0.5)*x])
        self.c = Constant(0.0)

        # self.u_T = Constant(0)
        self.u_ = 0.5 * (self.t**2+1) * sin(2*x*pi) * sin(2*y*pi)
        self.u_T = ufl.replace(self.u_, {self.t: self.T[1]})

        # Init right-hand side
        self.f = +self.t * sin(2*pi*x) * sin(2*pi*y) \
            + inner(self.a[0][0], Dx(Dx(self.u_, 0), 0)) \
            + inner(self.b, grad(self.u_)) \
            + self.c * self.u_
        # self.f = +self.t * sin(2*pi*x) * sin(2*pi*y) \
        # + inner(self.a, grad(grad(self.u_))) \
        # + inner(self.b, grad(self.u_)) \
        # + self.c * self.u_

        self.g = Constant(0.0)
        # self.g_t = lambda t: [(Constant(0.0), 'near(x[0],0)'),
        # (Constant(0.0), 'near(x[0],1)')]

    def initMesh(self, n):

        self.mesh = mesh_UnitSquare(n)
