# Import FEniCS dolfin
from dolfin import Constant
from dolfin import as_vector, as_matrix
from dolfin import inner, grad
from dolfin import SpatialCoordinate, RectangleMesh, Point, UnitIntervalMesh, BoxMesh
from dolfin import sqrt, sin, pi, atan
from ufl import atan_2
from dolfin import cos, conditional, exp, ln


# Import some standard meshes
from .meshes import mesh_UnitSquare, mesh_Square

# Import base class of problems
from nonvarFEM.pdes import NVP

# --------------------------------------------------
# Test example for solution in H^alpha
# --------------------------------------------------


class Sol_in_H_alpha(NVP):
    """
    Example problem whose solution is in H^s with s < 1+\alpha
    """
    def __init__(self, alpha=1.5):

        # Set paramater
        self.alpha = alpha

    def updateCoefficients(self):

        x, y = SpatialCoordinate(self.mesh)

        # Init coefficient matrix
        self.a = as_matrix([[1., Constant(0.)], [Constant(0.), 1.]])

        # Set up explicit solution
        r = sqrt(x**2 + y**2)
        phi = atan_2(x, y)
        self.u_ = r**self.alpha * sin(2. * phi) * (1-x) * (1-y)

        # Init right-hand side
        self.f = inner(self.a, grad(grad(self.u_)))

        # Set boundary conditions to exact solution
        self.g = self.u_

    def initMesh(self, n):

        # Set mesh to square on [0,1]^2
        self.mesh = mesh_UnitSquare(n)


class Sol_in_H_alpha_3d(NVP):
    """
    Example problem whose solution is in H^s with s < 1.5+\alpha
    """
    def __init__(self, alpha=1.):

        # Set paramater
        self.alpha = alpha

    def updateCoefficients(self):

        x, y, z = SpatialCoordinate(self.mesh)

        # Init coefficient matrix
        self.a = as_matrix([[1., Constant(0.), Constant(0.)],
                            [Constant(0.), 1., Constant(0.)],
                            [Constant(0.), Constant(0.), 1.]])

        # Set up explicit solution
        r = sqrt((x-.5)**2 + (y-.5)**2 + (z-.5)**2)
        self.u_ = r**self.alpha * x * y * z * (1-x) * (1-y) * (1-z)

        # Init right-hand side
        self.f = inner(self.a, grad(grad(self.u_)))

        # Set boundary conditions to exact solution
        self.g = self.u_

    def initMesh(self, n):

        # Set mesh to square on [0,1]^3
        self.mesh = BoxMesh(Point(0., 0., 0.), Point(1., 1., 1.), n, n, n)

# --------------------------------------------------
# Test example for which Cordes condition is
# - fulfilled          for kappa in (-1,1)
# - not fulfilled      otherwise
# --------------------------------------------------


class No_Cordes(NVP):
    def __init__(self, kappa=1.0):

        self.kappa = kappa

    def updateCoefficients(self):

        x, y = SpatialCoordinate(self.mesh)

        # Init coefficient matrix
        self.a = as_matrix([[1.0, self.kappa], [self.kappa, 1.0]])

        # Init right-hand side
        self.f = (1 + x**2) * sin(pi*x) * sin(2*pi*y)

        # Set boundary conditions to exact solution
        self.g = Constant(0.0)

    def initMesh(self, n):
        # Set mesh to square on [0,1]^2
        self.mesh = mesh_UnitSquare(n)

# --------------------------------------------
# Test example: Poisson equation
# --------------------------------------------


class Poisson(NVP):

    def updateCoefficients(self):

        x, y = SpatialCoordinate(self.mesh)

        self.a = as_matrix([[1., 0.], [0., 1.]])
        self.u_ = sin(pi*x) * sin(pi*y)

        self.f = inner(self.a, grad(grad(self.u_)))

        self.g = Constant(0.0)

    def initMesh(self, n):
        self.mesh = mesh_UnitSquare(n)

# --------------------------------------------
# Test example: Poisson equation with inhomogeneous bc's
# --------------------------------------------


class Poisson_inhomoBC(NVP):

    def updateCoefficients(self):

        x, y = SpatialCoordinate(self.mesh)

        self.a = as_matrix([[1., 0.], [0., 1.]])
        self.u_ = x*(1-x) + y*(1-y)
        # self.u_ = sin(x)*exp(cos(y))

        self.f = inner(self.a, grad(grad(self.u_)))

        self.g = self.u_

    def initMesh(self, n):
        self.mesh = mesh_UnitSquare(n)

# --------------------------------------------
# Test example with smooth solution
# --------------------------------------------


class Cinfty(NVP):
    def __init__(self, kappa=.99):

        self.kappa = kappa

    def updateCoefficients(self):

        x, y = SpatialCoordinate(self.mesh)

        self.a = as_matrix([[Constant(1.), Constant(self.kappa)],
            [Constant(self.kappa), Constant(1.)]])
        self.u_ = sin(2*pi*x) * sin(2*pi*y)

        self.f = inner(self.a, grad(grad(self.u_)))

        self.g = Constant(0.0)

    def initMesh(self, n):

        # Set mesh to square on [0,1]^2
        self.mesh = mesh_UnitSquare(n)

# --------------------------------------------------
# Test example for which the diffusion matrix is discontinuous
# --------------------------------------------------


class Discontinuous_A(NVP):

    def updateCoefficients(self):
        # Init coefficient matrix
        x, y = SpatialCoordinate(self.mesh)

        self.a = as_matrix(
            [[.02, .01], [0.01, conditional(x*x*x-y > 0.0, 2.0, 1.0)]])

        # Init right-hand side
        self.f = -1.0

        # Set boundary conditions to exact solution
        self.g = Constant(0.0)

    def initMesh(self, n):
        self.mesh = mesh_Square(n)

# --------------------------------------------------
# Test example for identity as diffusion matrix
# --------------------------------------------------


class Identity_A(NVP):

    def __init__(self, kappa=0.0):

        self.kappa = kappa

    def updateCoefficients(self):

        x, y = SpatialCoordinate(self.mesh)

        # Init coefficient matrix
        self.a = as_matrix([[1.0, self.kappa], [self.kappa, 1.0]])

        # Init right-hand side
        self.f = (1 + x**2) * sin(pi * x) * pi**2

        # Set boundary conditions to exact solution
        self.g = Constant(0.0)

    def initMesh(self, n):
        # Set mesh to square on [0,1]^2
        self.mesh = mesh_UnitSquare(n)


class Boundary_Layer(NVP):

    def updateCoefficients(self):

        # Init coefficient matrix
        self.a = as_matrix([[.5, 0.95], [.95, 2.]])

        # Init right-hand side
        self.f = 1.0

        # Set boundary conditions to exact solution
        self.g = Constant(0.0)

    def initMesh(self, n):
        # Set mesh to square on [0,1]^2
        self.mesh = mesh_UnitSquare(n)


class Discontinuous_2nd_Derivative(NVP):

    def updateCoefficients(self):

        x, y = SpatialCoordinate(self.mesh)

        self.a = as_matrix([[1.0, 0.], [0., 1.]])

        # Init right-hand side
        xi = 0.5+pi/100
        self.u_ = conditional(x <= xi, 0.0, 0.5 * (x - xi)**2)
        # self.f = inner(self.a, grad(grad(self.u_)))
        self.f = conditional(x <= xi, 0.0, 1.0)

        # Set boundary conditions to exact solution
        self.g = self.u_

    def initMesh(self, n):

        self.mesh = mesh_UnitSquare(n)

# --------------------------------------------------
# Test example 1 from Neilan:
# --------------------------------------------------


class Neilan_Test1(NVP):

    def updateCoefficients(self):

        x, y = SpatialCoordinate(self.mesh)

        # Init coefficient matrix
        self.a = as_matrix([[pow(abs(sin(4 * pi * (x - 0.5))), 1.0/5), cos(2*pi*x*y)],
                            [cos(2*pi*x*y), pow(abs(sin(4 * pi * (y - 0.5))), 1.0/5) + 1.]])

        # Init exact solution
        self.u_ = x * y * sin(2 * pi * x) * sin(3 * pi *
                                                y) / (pow(x, 2) + pow(y, 2) + 1)

        # Init right-hand side
        self.f = inner(self.a, grad(grad(self.u_)))

        # Set boundary conditions to exact solution
        self.g = self.u_

    def initMesh(self, n):
        self.mesh = mesh_UnitSquare(n)


class Neilan_5_2(NVP):

    def updateCoefficients(self):

        x, y = SpatialCoordinate(self.mesh)

        self.a = as_matrix([[-5. / ln(sqrt(pow(x, 2) + pow(y, 2))) + 15, 1.0],
                            [1.0, -1. / ln(sqrt(pow(x, 2) + pow(y, 2))) + 3]])

        # Init exact solution
        self.u_ = pow(sqrt(pow(x, 2) + pow(y, 2)), 7./4)

        # Init right-hand side
        self.f = inner(self.a, grad(grad(self.u_)))

        # Set boundary conditions to exact solution
        self.g = self.u_

    def initMesh(self, n):
        self.mesh = RectangleMesh(Point(0., 0.), Point(0.5, 0.5), n, n)


class Neilan_Talk_2(NVP):

    def updateCoefficients(self):
        x, y = SpatialCoordinate(self.mesh)

        # Init coefficient matrix
        self.a = as_matrix([[2.0, sin(pi*(20*x*y+.5)) * conditional(x*y > 0, 1, -1)],
                            [sin(pi*(20*x*y+.5)) * conditional(x*y > 0, 1, -1), 2.0]])

        self.u_ = x * y * sin(2 * pi * x) * \
            sin(3 * pi * y) / (x ** 2 + y ** 2 + 1)

        # Init right-hand side
        self.f = inner(self.a, grad(grad(self.u_)))

        # Set boundary conditions to exact solution
        self.g = self.u_

    def initMesh(self, n):
        # Set mesh to square on [-1,1]^2
        self.mesh = mesh_Square(n)

# --------------------------------------------------
# Test example 1 from Lakkis and Pryer:
# --------------------------------------------------


class LP_4_1_Nondiff_A(NVP):

    def updateCoefficients(self):
        # Init coefficient matrix
        x, y = SpatialCoordinate(self.mesh)
        self.a = as_matrix(
            [[1., 0.], [0., (pow(pow(x, 2) * pow(y, 2), 1.0/3) + 1.0)]])

        # Init exact solution
        self.u_ = exp(-10 * (pow(x, 2) + pow(y, 2)))

        # Init right-hand side
        self.f = inner(self.a, grad(grad(self.u_)))

        # Set boundary conditions to exact solution
        self.g = self.u_

    def initMesh(self, n):
        self.mesh = mesh_Square(n)

# --------------------------------------------------
# Test example 2 from Lakkis and Pryer:
# --------------------------------------------------


class LP_4_2_Conv_Dominated_A(NVP):
    def __init__(self, K=5000):

        # Set parameter
        self.K = K

    def updateCoefficients(self):

        x, y = SpatialCoordinate(self.mesh)

        # Init coefficient matrix
        self.a = as_matrix(
            [[1., 0.], [0., 2. + atan(self.K * (pow(x, 2) + pow(y, 2) - 1.0))]])

        # Init exact solution
        self.u_ = sin(pi * x) * sin(pi * y)

        # Init right-hand side
        self.f = inner(self.a, grad(grad(self.u_)))

        # Set boundary conditions to exact solution
        self.g = self.u_

    def initMesh(self, n):
        self.mesh = mesh_Square(n)

# --------------------------------------------------
# Test example 3 from Lakkis and Pryer:
# --------------------------------------------------


class LP_4_3_Singular_Sol(NVP):

    def updateCoefficients(self):

        x, y = SpatialCoordinate(self.mesh)

        # Init coefficient matrix
        self.a = as_matrix(
            [[1., 0.], [0., 2.0 + sin(1.0 / (abs(x) + abs(y)))]])

        # Init exact solution
        self.u_ = pow(2 - pow(x, 2) - pow(y, 2), 0.5)

        # Init right-hand side
        self.f = inner(self.a, grad(grad(self.u_)))

        # Set boundary conditions to exact solution
        self.g = self.u_

    def initMesh(self, n):
        self.mesh = mesh_Square(n)

# --------------------------------------------------
# Test example 4 from Lakkis and Pryer:
# --------------------------------------------------


class LP_4_4_Nonsymmetric_Hessian(NVP):

    def updateCoefficients(self):

        x, y = SpatialCoordinate(self.mesh)

        # Init coefficient matrix
        self.a = as_matrix([[1.0, pow(pow(x, 2) * pow(y, 2), 1.0/3)],
                            [pow(pow(x, 2) * pow(y, 2), 1.0/3), 2.]])

        # Init exact solution
        self.u_ = x * y * (x**2 - y**2) / (x**2 + y**2)

        # Init right-hand side
        self.f = inner(self.a, grad(grad(self.u_)))

        # Set boundary conditions to exact solution
        self.g = self.u_

    def initMesh(self, n):
        self.mesh = mesh_Square(n)

# --------------------------------------------------
# Test example 1 from Feng, Neilan, Schnake
# --------------------------------------------------


class FNS_5_1_Hoelder_Cont_Coeffs(NVP):

    def updateCoefficients(self):
        x, y = SpatialCoordinate(self.mesh)
        z = pow(sqrt(pow(x, 2) + pow(y, 2)), .5)

        # Init coefficient matrix
        self.a = as_matrix([[z + 1., -z],
                            [-z, 5*z+1.]])

        self.u_ = sin(2*pi*x)*sin(2*pi*y)*exp(x*cos(y))

        # Init right-hand side
        self.f = inner(self.a, grad(grad(self.u_)))

        # Set boundary conditions to exact solution
        self.g = self.u_

    def initMesh(self, n):
        self.mesh = RectangleMesh(Point(-.5, -.5), Point(0.5, 0.5), n, n)

# --------------------------------------------------
# Test example 2 from Feng, Neilan, Schnake
# --------------------------------------------------


class FNS_5_2_Uniform_Cont_Coeffs(NVP):

    def updateCoefficients(self):

        x, y = SpatialCoordinate(self.mesh)
        z = -1./ln(sqrt(pow(x, 2) + pow(y, 2)))

        # Init coefficient matrix
        self.a = as_matrix([[5*z + 15, 1.], [1., 1*z + 3]])
        self.u_ = pow(pow(x, 2) + pow(y, 2), 7./8)
        # Init right-hand side
        self.f = inner(self.a, grad(grad(self.u_)))

        # Set boundary conditions to exact solution
        self.g = self.u_

    def initMesh(self, n):
        self.mesh = RectangleMesh(Point(0., 0.), Point(0.5, 0.5), n, n)

# --------------------------------------------------
# Test example 3 from Feng, Neilan, Schnake
# --------------------------------------------------


class FNS_5_3_Degenerate_Coeffs(NVP):

    def updateCoefficients(self):
        x, y = SpatialCoordinate(self.mesh)

        # Init coefficient matrix
        self.a = as_matrix([[16./9*pow(x, 2./3), -16./9*pow(x*y, 1.3)],
                            [-16./9*pow(x*y, 1.3), 16./9*pow(y, 2./3)]])

        self.u_ = pow(x, 4./3) - pow(y, 4./3)

        # Init right-hand side
        self.f = 0.0

        # Set boundary conditions to exact solution
        self.g = self.u_

    def initMesh(self, n):
        self.mesh = RectangleMesh(Point(0., 0.), Point(1.0, 1.0), n, n)

# --------------------------------------------------
# Test example 4 from Feng, Neilan, Schnake
# --------------------------------------------------


class FNS_5_4_L_inf_Coeffs(NVP):

    def updateCoefficients(self):
        # Init coefficient matrix
        x, y = SpatialCoordinate(self.mesh)

        self.a = conditional(x*y > 0,
                             as_matrix([[2.0, 1.0], [1.0, 2.0]]),
                             as_matrix([[2.0, -1.0], [-1.0, 2.0]]))

        self.u_ = x*y*(1-exp(1-abs(x))) * (1-exp(1-abs(y)))

        # Init right-hand side
        self.f = inner(self.a, grad(grad(self.u_)))

        # Set boundary conditions to exact solution
        self.g = self.u_

    def initMesh(self, n):
        self.mesh = mesh_Square(n)

# --------------------------------------------------
# Test example with convection and potential
# --------------------------------------------------


class FullNVP_1(NVP):

    def updateCoefficients(self):
        # Init coefficient matrix
        x, y = SpatialCoordinate(self.mesh)

        self.a = as_matrix([[1.0, 0.0], [0.0, 1.0]])
        self.b = as_vector([1.0, -1.0])
        self.c = -0.1

        self.u_ = x*y*(1-exp(1-abs(x))) * (1-exp(1-abs(y)))

        # Init right-hand side
        self.f = inner(self.a, grad(grad(self.u_))) \
            + inner(self.b, grad(self.u_)) \
            + self.c * self.u_

        # Set boundary conditions to exact solution
        self.g = self.u_

    def initMesh(self, n):
        self.mesh = mesh_Square(n)


class FullNVP_2(NVP):

    def updateCoefficients(self):
        # Init coefficient matrix
        x, y = SpatialCoordinate(self.mesh)

        self.a = as_matrix([[1.0, 0.0], [0.0, 1.0]])
        self.b = 100*as_vector([1.0, -1.0])
        self.c = -0.1

        self.u_ = x*y*(1-exp(1-abs(x))) * (1-exp(1-abs(y)))

        # Init right-hand side
        self.f = inner(self.a, grad(grad(self.u_))) \
            + inner(self.b, grad(self.u_)) \
            + self.c * self.u_

        # Set boundary conditions to exact solution
        self.g = self.u_

    def initMesh(self, n):
        self.mesh = mesh_Square(n)


class FullNVP_3_1d(NVP):

    def updateCoefficients(self):
        # Init coefficient matrix
        x = SpatialCoordinate(self.mesh)[0]

        self.a = as_matrix([[1.0]])
        self.b = as_vector([0.1])
        self.c = -0.1

        self.u_ = x*(1-exp(1-abs(x)))

        # Init right-hand side
        self.f = inner(self.a, grad(grad(self.u_))) \
            + inner(self.b, grad(self.u_)) \
            + self.c * self.u_

        # Set boundary conditions to exact solution
        self.g = self.u_

    def initMesh(self, n):
        self.mesh = UnitIntervalMesh(n)
