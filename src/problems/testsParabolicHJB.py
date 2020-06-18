# Import FEniCS dolfin
from dolfin import FunctionSpace, Constant, Dx
from dolfin import as_vector, as_matrix
from dolfin import SpatialCoordinate, IntervalMesh, RectangleMesh, Point

from dolfin import conditional, exp, sqrt, sin, pi

import ufl
# Import some standard meshes
from .meshes import mesh_UnitSquare, mesh_Triangle

# Import base class of problems
from .parabolicHJB import ParabolicHJB as pHJB


class Mother(pHJB):

    def __init__(self):
        self.T = [0, 1]
        self.t = Constant(self.T[1])

        self.alpha0 = 0.1
        self.alpha1 = 0.2

    def initControl(self):

        # Initialize control spaces
        self.controlSpace = [FunctionSpace(self.mesh, "DG", 0)]

        # Initialize controls
        u_x = Dx(self.u, 0)
        u_y = Dx(self.u, 1)
        u_lapl = Dx(u_x, 0) + Dx(u_y, 1)
        self.gamma = [conditional(u_lapl >= 0, self.alpha0, self.alpha1)]

    def updateControl(self):
        """ The optimal continuous control in this example depends on the
        Laplacian. """
        print('call updateControl')

        # Illustrate discrete Laplacian
        # DG0 = FunctionSpace(self.mesh, "DG", 0)
        # fig = plt.figure('Broken Laplacian', clear=True)
        # c = plot(project(u_lapl,DG0))
        # plt.colorbar(c)
        # plt.draw()
        # plt.pause(0.01)
        # ipdb.set_trace()

        # Method 1: yields oscillating control
        # g_a = conditional(u_lapl >= 0, a0, a1)
        # self.gamma[0].vector()[:] = a0 *(tmp >= 0) + a1 * (tmp < 0)
        # Method 2: yields oscillating control
        # dg0test = TestFunction(self.controlSpace[0])
        # tmp = assemble(u_lapl * dg0test * dx)
        # self.gamma[0].vector()[:] = a0 *(tmp >= 0) + a1 * (tmp < 0)

        # ipdb.set_trace()
        # self.gamma[0] = g_a

        # self.gamma[0] = project(g_a, self.controlSpace[0])

    def updateCoefficients(self):
        # Init coefficient matrix
        x, y = SpatialCoordinate(self.mesh)

        self.a = self.gamma[0] * \
            as_matrix([[1.0, 0.0], [0.0, 1.0]])
        self.b = as_vector([Constant(1.0), Constant(1.0)])
        self.c = Constant(0.0)

        # Init right-hand side
        self.f = -Constant(1.0)

        self.u_T = Constant(0.0)

        # Set boundary conditions
        self.g = Constant(0.0)

    def initMesh(self, n):
        self.mesh = mesh_UnitSquare(n)


class JensenSmears_1(pHJB):

    def __init__(self):
        self.T = [0, 1]
        self.t = Constant(self.T[1])

    def initControl(self):

        # Initialize control spaces
        self.controlSpace = [FunctionSpace(self.mesh, "DG", 1),
                             FunctionSpace(self.mesh, "DG", 1)]

        # Initialize controls
        u_x = Dx(self.u, 0)
        u_y = Dx(self.u, 1)
        u_norm = sqrt(u_x**2 + u_y**2)

        self.gamma = []
        self.gamma.append(u_x / u_norm)
        self.gamma.append(u_y / u_norm)

    def updateControl(self):
        """ The optimal continuous control in this example depends on the
        gradient. """
        print('call updateControl')

    def updateCoefficients(self):
        # Init coefficient matrix
        x, y = SpatialCoordinate(self.mesh)

        self.a = +0.5 * \
            sqrt((x**2 + y**2) / (self.T[1] - self.t + 1)) \
            * as_matrix([[1.0, 0.0], [0.0, 1.0]])

        self.b = -0.5 * (self.T[1] - self.t + 1)**(-.5) \
            * as_vector([self.gamma[0], self.gamma[1]])

        self.u_ = exp(-sqrt((x**2 + y**2) / (self.T[1] - self.t + 1))) \
            + sqrt((x**2 + y**2) / (self.T[1] - self.t + 1))

        self.u_T = ufl.replace(self.u_, {self.t: self.T[1]})

        # Init right-hand side
        self.f = +0.5 * \
            sqrt(x**2 + y**2) * (self.T[1] - self.t + 1)**(-1.5)

        # Set boundary conditions
        self.g = self.u_

    def initMesh(self, n):
        self.mesh = mesh_Triangle(n)


class JensenSmears_2(pHJB):

    def __init__(self):
        self.T = [0, 0.009]
        self.t = Constant(self.T[1])
        self.alpha0 = 0.045
        self.alpha1 = 0.09

    def initControl(self):

        # Initialize control spaces
        self.controlSpace = []
        self.controlSpace.append(FunctionSpace(self.mesh, "DG", 0))
        self.controlSpace.append(FunctionSpace(self.mesh, "DG", 1))
        self.controlSpace.append(FunctionSpace(self.mesh, "DG", 1))

        # Initialize controls
        u_x = Dx(self.u, 0)
        u_y = Dx(self.u, 1)
        u_lapl = Dx(u_x, 0) + Dx(u_y, 1)

        g_a = conditional(u_lapl >= 0, self.alpha0, self.alpha1)

        u_norm = sqrt(u_x**2 + u_y**2)
        g_x = conditional(u_norm > 0, u_x / u_norm, 0)
        g_y = conditional(u_norm > 0, u_y / u_norm, 0)
        self.gamma = []
        self.gamma.append(g_a)
        self.gamma.append(g_x)
        self.gamma.append(g_y)

    def updateControl(self):
        """ The optimal continuous control in this example depends on the
        gradient. """

        print('call updateControl')
        # a0 = self.alpha0
        # a1 = self.alpha1
        # u_x = Dx(self.u, 0)
        # u_y = Dx(self.u, 1)
        # u_lapl = Dx(u_x, 0) + Dx(u_y, 1)

        # g_a = conditional(u_lapl >= 0, a0, a1)

        # # u_lapl = project(u_lapl, self.controlSpace[0])
        # # e = 0.1
        # # g_a = conditional(u_lapl < -e, a1,\
        # #         conditional(u_lapl > e, a0,\
        # #         a1 + (u_lapl + e) / (2*e) * (a0 - a1)))

        # u_norm = sqrt(u_x**2 + u_y**2)
        # g_x = u_x / u_norm
        # g_y = u_y / u_norm
        # self.gamma[0] = project(g_a, self.controlSpace[0])
        # self.gamma[1] = project(g_x, self.controlSpace[1])
        # self.gamma[2] = project(g_y, self.controlSpace[2])

    def updateCoefficients(self):
        # Init coefficient matrix
        x, y = SpatialCoordinate(self.mesh)

        self.a = self.gamma[0] * as_matrix([[1.0, 0.0], [0.0, 1.0]])
        self.b = as_vector([self.gamma[1], self.gamma[2]])

        # Init right-hand side
        # self.f_t = 529 * (sin(xxx)
        #  + 0.5 * sin(2 * xxx) + 4./10 * sin(8 * xxx))**2
        # self.f_t = (sin(pi**2*(x-0.63)*(y-0.26)/0.07))**2
        # self.f_t = Constant(1.)

        self.f = -10 * (sin(pi**2*(x-0.63)*(y-0.26)/0.07)
                        + 0.5 * sin(2 * pi**2*(x-0.63)
                                    * (y-0.26)/0.07)
                        + 4./10 * sin(8 * pi**2*(x-0.63)
                                      * (y-0.26)/0.07))**2

        # Set boundary conditions
        self.g = Constant(0.0)
        self.u_T = Constant(0.0)

    def initMesh(self, n):
        # self.mesh = mesh_Lshape(n)
        self.mesh = mesh_UnitSquare(n)


class MinimumArrivalTimeParabolic(pHJB):

    def __init__(self, alpha=0.1, beta=0.1, cmin=-1., cmax=1.0):
        self.T = [0, 1]
        self.t = Constant(self.T[1])

        self.alpha = alpha
        self.beta = beta
        self.cmin = cmin
        self.cmax = cmax

    def initControl(self):

        # Initialize control spaces
        self.controlSpace = []
        self.controlSpace.append(FunctionSpace(self.mesh, "DG", 0))
        self.controlSpace.append(FunctionSpace(self.mesh, "DG", 0))

        # Initialize controls
        Dxu = Dx(self.u, 0)
        spx = Dxu + self.beta
        smx = Dxu - self.beta

        Dyu = Dx(self.u, 1)
        spy = Dyu + self.beta
        smy = Dyu - self.beta

        if self.alpha < 1e-15:
            gx = conditional(
                spx < 0, self.cmax, conditional(smx > 0, self.cmin, 0))
            gy = conditional(
                spy < 0, self.cmax, conditional(smy > 0, self.cmin, 0))
        else:
            e1 = ufl.Min(-1.0 * spx / self.alpha, self.cmax)
            e2 = conditional(smx > 0,
                             ufl.Max(-1.0 * smx / self.alpha, self.cmin), 0)
            gx = conditional(spx < 0, e1, e2)

            e3 = ufl.Min(-1.0 * spy / self.alpha, self.cmax)
            e4 = conditional(smy > 0,
                             ufl.Max(-1.0 * smy / self.alpha, self.cmin), 0)
            gy = conditional(spy < 0, e3, e4)

        self.gamma = [gx, gy]

    def updateControl(self):
        """ The optimal continuous control in this example depends on the
        gradient. """

        print('call updateControl')

    def updateCoefficients(self):
        # Init coefficient matrix
        x, y = SpatialCoordinate(self.mesh)

        self.a = as_matrix([[1.0, 0.0], [0.0, 1.0]])
        self.b = as_vector([self.gamma[0], self.gamma[1]])

        # Init right-hand side
        self.f = -1 \
            - 0.5 * self.alpha \
            * sqrt(pow(self.gamma[0], 2) + pow(self.gamma[1], 2)) \
            - self.beta * (abs(self.gamma[0]) + abs(self.gamma[1]))

        # Set boundary conditions
        self.g = Constant(0.0)

        self.u_T = Constant(0.0)

    def initMesh(self, n):
        self.mesh = mesh_UnitSquare(n)
        # self.mesh = mesh_Lshape(n)


class Chen_Forsyth(pHJB):

    def __init__(self):
        self.T = [0, 3]
        self.t = Constant(self.T[1])
        # self.T = [0, 0.1]
        # self.alpha = 0.1
        self.alpha = 2.38
        # self.r = 0.0
        self.r = 0.1
        self.sigma = 0.59
        self.pmin = 0.0
        self.pmax = 12.0
        self.Imin = 0.0
        self.Imax = 2000
        self.K0 = 6
        self.k1 = 2040.41
        self.k2 = 730000
        self.k3 = 500
        self.k4 = 2500
        self.k5 = 1.7*365

    def initControl(self):

        # Initialize control spaces
        self.controlSpace = [FunctionSpace(self.mesh, "DG", 1)]

        # Initialize controls

        P, Inv = SpatialCoordinate(self.mesh)

        cmax = self.k1 * sqrt(Inv)
        cmin = -self.k2 * sqrt(1/(Inv + self.k3) - 1/self.k4)
        ui = Dx(self.u, 1)
        u1 = self.Gamma(cmin, P, ui)
        u2 = self.Gamma(-Constant(self.k5), P, ui)
        u3 = self.Gamma(Constant(0.0), P, ui)
        u4 = self.Gamma(cmax, P, ui)
        umax = ufl.Max(u1, ufl.Max(u2, ufl.Max(u3, u4)))
        # self.gamma[0] = cmin
        g1 = conditional(u1 >= umax, cmin,
                         conditional(u2 >= umax, -self.k5,
                                     conditional(u3 >= umax, 0.0, cmax)))
        self.gamma = [g1]

    def updateControl(self):
        """ The optimal continuous control in this example depends on the
        Laplacian. """
        print('call updateControl')

    def cost(self, c):
        return conditional(c >= 0, 0, self.k5)

    def Gamma(self, c, P, ui):
        return (c - self.cost(c)) * P - (c + self.cost(c)) * ui

    def updateCoefficients(self):

        # Init coefficient matrix
        P, Inv = SpatialCoordinate(self.mesh)

        self.a = as_matrix([[+.5 * (P * self.sigma)**2, 0], [0, 0]])

        # def K(t): return self.K0 + beta_SA * sin(4*pi*(t - t_SA))
        def K(t): return self.K0

        self.b = as_vector(
            [+self.alpha*(K(self.t) - P),
             -(self.gamma[0] + self.cost(self.gamma[0]))])

        self.c = Constant(-self.r)

        # Init right-hand side
        self.f = -(self.gamma[0] - self.cost(self.gamma[0])) * P

        self.u_T = - 2 * P * ufl.Max(1000 - Inv, 0)
        # self.u_T = Constant(0.0)

        # Set boundary conditions
        # self.g_t = lambda t : [(Constant(0.0), "near(x[0],0)")]
        self.g = self.u_T
        # self.g_t = lambda t : self.u_t(t)

        # self.loc = conditional(x > 0.5, conditional(x < 1.5, 1, 0), 0)

    def initMesh(self, n):
        self.mesh = RectangleMesh(
            Point(self.pmin, self.Imin),
            Point(self.pmax, self.Imax), n, n)


class Merton(pHJB):

    def __init__(self):
        self.T = [0, 1]
        self.t = Constant(self.T[1])
        self.mu = 0.1
        self.r = 0.01
        self.sigmax = 1.2
        self.alpha = 0.5

    def initControl(self):

        # Initialize control spaces
        self.controlSpace = [FunctionSpace(self.mesh, "DG", 1)]

        # Initialize controls
        x = SpatialCoordinate(self.mesh)[0]

        u_x = Dx(self.u, 0)
        u_xx = Dx(u_x, 0)
        g1 = conditional(
            x > 0, - (self.mu - self.r) * u_x / (x * self.sigmax**2 * u_xx), 0)
        self.gamma = [g1]

    def updateControl(self):
        """ The optimal continuous control in this example depends on the
        Laplacian. """
        print('call updateControl')

    def updateCoefficients(self):

        # Init coefficient matrix
        x = SpatialCoordinate(self.mesh)[0]

        self.a = as_matrix(
            [[.5 * (x * self.gamma[0] * self.sigmax)**2]])
        self.b = as_vector(
            [x * (self.gamma[0] * (self.mu - self.r) + self.r)])
        self.c = Constant(0.0)

        # Init right-hand side
        self.f = Constant(0.0)
        self.u_ = exp(((self.mu - self.r)**2 / (2 * self.sigmax**2) * self.alpha / (
            1-self.alpha) + self.r * self.alpha)
            * (self.T[1] - self.t)) * (x ** self.alpha) / self.alpha
        self.u_T = (x ** self.alpha) / self.alpha

        # Set boundary conditions
        # self.g_t = lambda t : [(Constant(0.0), "near(x[0],0)")]
        self.g = Constant(0.0)
        # self.g_t = lambda t : self.u_t(t)

        self.gamma_star = [
            Constant((self.mu - self.r) / (self.sigmax**2 * (1 - self.alpha)))]

        self.loc = conditional(x > 0.5, conditional(x < 1.5, 1, 0), 0)

    def initMesh(self, n):
        self.mesh = IntervalMesh(n, 0., 10.)
