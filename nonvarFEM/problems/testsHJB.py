# Import FEniCS dolfin
from dolfin import Constant, FunctionSpace, UserExpression
from dolfin import as_vector, as_matrix
from dolfin import inner, grad, interpolate, project, Dx
from dolfin import SpatialCoordinate
from dolfin import sqrt, sin, pi
from dolfin import cos, conditional, exp

import numpy as np
import ufl

# Import some standard meshes
from .meshes import mesh_UnitSquare, mesh_Square

# Import base class of problems
from nonvarFEM.pdes.hjb import HJB


class MinimumArrivalTime(HJB):

    def __init__(self, alpha=0.4, beta=0.4, cmin=-1., cmax=1.0):
        self.alpha = alpha
        self.beta = beta
        self.cmin = cmin
        self.cmax = cmax

    def initControl(self):
        self.controlSpace = [FunctionSpace(self.mesh, "DG", 0),
                             FunctionSpace(self.mesh, "DG", 0)]

        self.gamma = []
        # Dxu = project(Dx(self.u,0),FunctionSpace(self.mesh, "DG", 0))
        Dxu = Dx(self.u, 0)
        spx = Dxu + self.beta
        smx = Dxu - self.beta

        # Dyu = project(Dx(self.u,1),FunctionSpace(self.mesh, "DG", 0))
        Dyu = Dx(self.u, 1)
        spy = Dyu + self.beta
        smy = Dyu - self.beta

        if self.alpha < 1e-15:
            self.gamma.append(conditional(
                spx < 0, self.cmax, conditional(smx > 0, self.cmin, 0)))
            self.gamma.append(conditional(
                spy < 0, self.cmax, conditional(smy > 0, self.cmin, 0)))
        else:

            self.gamma.append(conditional(
                spx < 0,
                ufl.Min(-1.0 * spx / self.alpha, self.cmax),
                conditional(smx > 0, ufl.Max(-1.0 * smx / self.alpha, self.cmin), 0)))

            self.gamma.append(conditional(
                spy < 0,
                ufl.Min(-1.0 * spy / self.alpha, self.cmax),
                conditional(smy > 0, ufl.Max(-1.0 * smy / self.alpha, self.cmin), 0)))

        # self.gamma = [interpolate(Constant("0.0"), self.controlSpace[0]),
        # interpolate(Constant("0.0"), self.controlSpace[1])]

    def updateControl(self):
        """ The optimal continuous control in this example depends on the
        gradient. """
        print('call updateControl')

        # # Dxu = project(Dx(self.u,0),FunctionSpace(self.mesh, "DG", 0))
        # Dxu = Dx(self.u, 0)
        # spx = Dxu + self.beta
        # smx = Dxu - self.beta

        # # Dyu = project(Dx(self.u,1),FunctionSpace(self.mesh, "DG", 0))
        # Dyu = Dx(self.u, 1)
        # spy = Dyu + self.beta
        # smy = Dyu - self.beta

        # if self.alpha < 1e-15:
        #     self.gamma[0] = conditional(
        #         spx < 0, self.cmax, conditional(smx > 0, self.cmin, 0))
        #     self.gamma[1] = conditional(
        #         spy < 0, self.cmax, conditional(smy > 0, self.cmin, 0))
        # else:

        #     import ipdb
        #     ipdb.set_trace()
        #     self.gamma[0] = conditional(
        #         spx < 0,
        #         ufl.Min(-1.0 * spx / self.alpha, self.cmax),
        #         conditional(smx > 0, ufl.Max(-1.0 * smx / self.alpha, self.cmin), 0))

        #     self.gamma[1] = conditional(
        #         spy < 0,
        #         ufl.Min(-1.0 * spy / self.alpha, self.cmax),
        #         conditional(smy > 0, ufl.Max(-1.0 * smy / self.alpha, self.cmin), 0))

    def updateCoefficients(self):
        # Init coefficient matrix
        x, y = SpatialCoordinate(self.mesh)

        self.a = as_matrix([[1.0, 0.0], [0.0, 1.0]])
        self.b = as_vector([self.gamma[0], self.gamma[1]])

        # self.u_ = x*y*(1-exp(1-abs(x))) * (1-exp(1-abs(y)))

        # Init right-hand side
        self.f = -1 \
            - 0.5 * self.alpha * sqrt(pow(self.gamma[0], 2) + pow(self.gamma[1], 2)) \
            - self.beta * (abs(self.gamma[0]) + abs(self.gamma[1]))

        # Set boundary conditions
        self.g = Constant(0.0)

    def initMesh(self, n):
        self.mesh = mesh_Square(n)


class MinimumArrivalTimeRadial(HJB):

    def __init__(self, alpha=0.4):

        assert alpha > 0, 'P.alpha has to be greater than zero.'

        self.alpha = alpha

    def initControl(self):
        self.controlSpace = [FunctionSpace(self.mesh, "DG", 1),
                             FunctionSpace(self.mesh, "DG", 1)]

        u_x = Dx(self.u, 0)
        u_y = Dx(self.u, 1)
        # phi = atan(u_y/u_x) <==> sin(phi) / cos(phi) = u_y / u_x

        self.gamma = []
        phi = ufl.atan_2(u_y, u_x)
        self.gamma.append(1./self.alpha * (cos(phi) * u_x + sin(phi) * u_y))
        self.gamma.append(phi)

    def updateControl(self):
        """ The optimal continuous control in this example depends on the
        gradient. """
        print('call updateControl')
        # u_x = Dx(self.u, 0)
        # u_y = Dx(self.u, 1)
        # # phi = atan(u_y/u_x) <==> sin(phi) / cos(phi) = u_y / u_x
        # self.gamma[1] = ufl.atan_2(u_y, u_x)
        # self.gamma[0] = 1./self.alpha * \
        #    # (cos(self.gamma[1]) * u_x + sin(self.gamma[1]) * u_y)

        # self.gamma[1] = project(self.gamma[1], self.controlSpace)
        # self.gamma[0] = project(self.gamma[0], self.controlSpace)

    def updateCoefficients(self):
        # Init coefficient matrix
        x, y = SpatialCoordinate(self.mesh)

        self.a = as_matrix([[1.0, 0.0], [0.0, 1.0]])
        # self.a = as_matrix([[1.0 , 0.0], [0.0, 1.0]])
        self.b = -self.gamma[0] * \
            as_vector([cos(self.gamma[1]), sin(self.gamma[1])])

        # self.u_ = x*y*(1-exp(1-abs(x))) * (1-exp(1-abs(y)))

        # Init right-hand side
        self.f = -1 \
            - 0.5 * self.alpha * sqrt(pow(self.gamma[0], 2))

        # Set boundary conditions
        self.g = Constant(0.0)

    def initMesh(self, n):
        self.mesh = mesh_Square(n)
        # self.mesh = mesh_Lshape(n)


class Gallistl_Sueli_1_optControl(UserExpression):
    def __init__(self, dxx, dxy, dyx, dyy, alphamin=0, alphamax=pi, **kwargs):

        super().__init__(**kwargs)

        self.dxx = dxx
        self.dxy = dxy
        self.dyx = dyx
        self.dyy = dyy
        self.amax = alphamax
        self.amin = alphamin

    def eval(self, values, x):

        # Discretize control set
        n_alpha = 200
        alpha_space = np.linspace(self.amin, self.amax, n_alpha+1)
        alpha_space = alpha_space[:-1]

        vxx = np.ndarray((1,))
        vxy = np.ndarray((1,))
        vyx = np.ndarray((1,))
        vyy = np.ndarray((1,))

        self.dxx.eval(vxx, x)
        self.dxy.eval(vxy, x)
        self.dyx.eval(vyx, x)
        self.dyy.eval(vyy, x)
        maxres = -1e10
        # print(x)
        for alpha in alpha_space:
            R = np.array([[cos(alpha), - sin(alpha)],
                          [sin(alpha), cos(alpha)]])
            A = 0.5 * R.T @ np.array([[20, 1], [1, 0.1]]) @ R

            res = A[0, 0] * vxx \
                + A[0, 1] * vxy \
                + A[1, 0] * vyx \
                + A[1, 1] * vyy

            if res > maxres:
                maxres = res
                values[:] = alpha

    def value_shape(self):
        return ()


class Gallistl_Sueli_1(HJB):

    def __init__(self):

        self.alphamin = 0
        self.alphamax = pi
        self.delta = 0.1

    def initControl(self):
        self.controlSpace = [FunctionSpace(self.mesh, "DG", 0)]

        x, y = SpatialCoordinate(self.mesh)
        u_ = (2*x-1.) \
            * (exp(1 - abs(2*x-1.)) - 1) \
            * (y + (1 - exp(y/self.delta))/(exp(1/self.delta)-1))
        cs = self.controlSpace[0]
        du = self.u - u_

        du_xx = Dx(Dx(du, 0), 0)
        du_xy = Dx(Dx(du, 0), 1)
        du_yx = Dx(Dx(du, 1), 0)
        du_yy = Dx(Dx(du, 1), 1)

        du_xx_proj = project(du_xx, cs)
        du_xy_proj = project(du_xy, cs)
        du_yx_proj = project(du_yx, cs)
        du_yy_proj = project(du_yy, cs)

        # Use the UserExpression
        gamma_star = Gallistl_Sueli_1_optControl(
            du_xx_proj, du_xy_proj, du_yx_proj, du_yy_proj, self.alphamin, self.alphamax)
        # Interpolate evaluates expression in centers of mass
        # Project evaluates expression in vertices
        self.gamma = [gamma_star]
        # self.gamma = [interpolate(Constant("0.0"), self.controlSpace[0])]

    def updateControl(self):
        """ The optimal continuous control in this example depends on the
        gradient. """
        print('call updateControl')

        # cs = self.controlSpace[0]

        # du = self.u - self.u_

        # du_xx = Dx(Dx(du, 0), 0)
        # du_xy = Dx(Dx(du, 0), 1)
        # du_yx = Dx(Dx(du, 1), 0)
        # du_yy = Dx(Dx(du, 1), 1)

        # du_xx_proj = project(du_xx, cs)
        # du_xy_proj = project(du_xy, cs)
        # du_yx_proj = project(du_yx, cs)
        # du_yy_proj = project(du_yy, cs)

        # # Use the UserExpression
        # gamma_star = Gallistl_Sueli_1_optControl(
        #     du_xx_proj, du_xy_proj, du_yx_proj, du_yy_proj, self.alphamin, self.alphamax)
        # # Interpolate evaluates expression in centers of mass
        # # Project evaluates expression in vertices
        # self.gamma[0] = interpolate(gamma_star, cs)

        # dofs = cs.tabulate_dof_coordinates()
        # n_control = dofs.shape[0]
        # du_xx_val = np.ndarray(shape=(1,1))
        # du_xy_val = np.ndarray(shape=(1,1))
        # du_yy_val = np.ndarray(shape=(1,1))
        #
        # n_alpha = 100
        # alpha_space = np.linspace(self.alphamin, self.alphamax, n_alpha+1)
        # alpha_space = alpha_space[:-1]
        # Res = np.zeros(shape=(n_control, n_alpha))
        #
        # for i, alpha in enumerate(alpha_space):
        #     R = np.array([[cos(alpha) , - sin(alpha)], [ sin(alpha), cos(alpha)]])
        #     A = 0.5 * R.T @ np.array([[20, 1], [1, 0.1]]) @ R
        #     for j in range(n_control):
        #         du_xx_proj.eval(du_xx_val, dofs[j])
        #         du_xy_proj.eval(du_xy_val, dofs[j])
        #         du_yy_proj.eval(du_yy_val, dofs[j])
        #         res = A[0,0] * du_xx_val \
        #                 + (A[0,1]+ A[1,0]) * du_xy_val \
        #                 + A[1,1] * du_yy_val
        #         # print('i:', i, 'j:', j, 'res:', res)
        #         Res[j,i] = res.flatten()
        #
        #     # a_alpha = 0.5 * as_matrix([[cos(alpha) , sin(alpha)],
        #     #          [ - sin(alpha), cos(alpha)]]) \
        #     #     * as_matrix([[20, 1], [1, 0.1]]) \
        #     #     * as_matrix([[cos(alpha) , - sin(alpha)], [ sin(alpha), cos(alpha)]])
        #     # ipdb.set_trace()
        #     # res = inner(a_alpha, grad(grad(self.u))) - inner(a_alpha, grad(grad(self.u_)))
        #     # res_proj = project(res, cs)
        #     # res_vec = np.ndarray(shape=(n_control,1))
        #     # for j in range(n_control):
        #     #     res_proj.eval(res_vec[j], dofs[j])
        #     # Res[:,i] = res_vec.flatten()
        #
        # self.gamma[0].vector()[:] = alpha_space[Res.argmax(axis=1)]
        # # ipdb.set_trace()

    def updateCoefficients(self):

        # Init coefficient matrix
        x, y = SpatialCoordinate(self.mesh)

        self.a = 0.5 * as_matrix(
            [[cos(self.gamma[0]), sin(self.gamma[0])],
             [- sin(self.gamma[0]), cos(self.gamma[0])]]) \
            * as_matrix([[20, 1], [1, 0.1]]) \
            * as_matrix(
            [[cos(self.gamma[0]), - sin(self.gamma[0])],
             [sin(self.gamma[0]), cos(self.gamma[0])]])

        self.b = as_vector([Constant(0.0), Constant(1.0)])

        self.c = Constant(-10.0)

        self.u_ = (2*x-1.) \
            * (exp(1 - abs(2*x-1.)) - 1) \
            * (y + (1 - exp(y/self.delta))/(exp(1/self.delta)-1))

        # Init right-hand side
        self.f = inner(self.a, grad(grad(self.u_))) \
            + inner(self.b, grad(self.u_)) \
            + self.c * self.u_

        # Set boundary conditions
        self.g = Constant(0.0)

    def initMesh(self, n):
        self.mesh = mesh_UnitSquare(n)


class Smears_Sueli_1(HJB):

    def __init__(self):

        self.thetamin = 0
        self.thetamax = pi/3
        self.alphamin = 0
        self.alphamax = pi

    def initControl(self):
        self.controlSpace = []
        self.controlSpace.append(FunctionSpace(self.mesh, "CG", 2))
        self.controlSpace.append(FunctionSpace(self.mesh, "CG", 2))
        self.gamma = []
        self.gamma.append(interpolate(Constant("0.0"), self.controlSpace))
        self.gamma.append(interpolate(Constant("0.0"), self.controlSpace))

    def updateControl(self):
        """ The optimal continuous control in this example depends on the
        gradient. """

        # u_x = Dx(self.u,0)
        # u_y = Dx(self.u,1)
        # # phi = atan(u_y/u_x) <==> sin(phi) / cos(phi) = u_y / u_x
        # self.gamma[1] = ufl.atan_2(u_y, u_x)
        # self.gamma[0] = 1./self.alpha * ( cos(self.gamma[1]) * u_x + sin(self.gamma[1]) * u_y)
        #
        # self.gamma[1] = project(self.gamma[1], self.controlSpace)
        # self.gamma[0] = project(self.gamma[0], self.controlSpace)

    def updateCoefficients(self):

        # Init coefficient matrix
        x, y = SpatialCoordinate(self.mesh)

        self.a = 0.5 * as_matrix(
            [[cos(self.gamma[0]), sin(self.gamma[0])],
             [- sin(self.gamma[0]), cos(self.gamma[0])]]) \
            * as_matrix([[1, sin(self.gamma[1])], [0, cos(self.gamma[1])]]) \
            * as_matrix([[1, 0], [sin(self.gamma[1]), cos(self.gamma[1])]]) \
            * as_matrix(
            [[cos(self.gamma[0]), - sin(self.gamma[0])],
             [sin(self.gamma[0]), cos(self.gamma[0])]])
        self.b = as_vector([Constant(0.0), Constant(0.0)])
        self.c = -pi**2

        self.u_ = exp(x*y) * sin(pi*x) * sin(pi * y)

        # Init right-hand side
        self.f = - sqrt(3) * (sin(self.gamma[1])/pi)**2 \
            + 111111
        # TODO work here

        # Set boundary conditions
        self.g = 0.0

    def initMesh(self, n):
        self.mesh = mesh_UnitSquare(n)


class Mother(HJB):

    def __init__(self):
        self.alpha0 = 0.1
        self.alpha1 = 0.2

    def initControl(self):

        # Initialize control spaces
        self.controlSpace = [FunctionSpace(self.mesh, "DG", 0)]

        # Initialize controls
        u_lapl = Dx(Dx(self.u, 0), 0) + Dx(Dx(self.u, 1), 1)

        # Method 1:
        self.gamma = [conditional(u_lapl >= 0, self.alpha0, self.alpha1)]

    def updateControl(self):
        """ The optimal continuous control in this example depends on the
        gradient. """

        print('call updateControl')
        # a0 = self.alpha0
        # a1 = self.alpha1
        # u_x = Dx(self.u, 0)
        # u_y = Dx(self.u, 1)
        # u_lapl = Dx(u_x, 0) + Dx(u_y, 1)

        # # Method 1:
        # self.gamma[0] = conditional(u_lapl >= 0, a0, a1)

        # u_lapl = tr(self.H)

        # Method 0: Evaluating optimal control in each dof
        # cs = self.controlSpace[0]
        # u_lapl_proj = project(u_lapl, cs)
        # dofs = cs.tabulate_dof_coordinates()
        # n_control = dofs.shape[0]
        # u_lapl_vec = np.ndarray(shape=(n_control,1))
        # for i in range(n_control):
        #     u_lapl_proj.eval(u_lapl_vec[i], dofs[i])
        # # print(dofs[abs(u_lapl_vec.flatten())<1e-1])
        # pos_lapl = (u_lapl_vec.flatten() >= 0)
        # n_pos = sum(pos_lapl)
        # tmp = self.gamma[0].vector()[:]
        # tmp[pos_lapl] = a0
        # tmp[np.logical_not(pos_lapl)] = a1
        # self.gamma[0].vector()[:] = tmp

        # Illustrate discrete Laplacian
        # DG0 = FunctionSpace(self.mesh, "DG", 0)
        # fig = plt.figure('Broken Laplacian', clear=True)
        # c = plot(project(u_lapl,DG0))
        # plt.colorbar(c)
        # plt.draw()
        # plt.pause(0.01)

        # Method 1: yields oscillating control
        # g_a = conditional(u_lapl > 0, a0, a1)
        # self.gamma[0].vector()[:] = a0 *(tmp >= 0) + a1 * (tmp < 0)
        # Method 2: yields oscillating control
        # dg0test = TestFunction(self.controlSpace[0])
        # tmp = assemble(u_lapl * dg0test * dx)
        # self.gamma[0].vector()[:] = a0 *(tmp >= 0) + a1 * (tmp < 0)

        # ipdb.set_trace()

        # tmp = project(g_a, self.controlSpace[0])
        # tmp.eval(uval,np.array([0.5,0.9])); print(uval)
        #
        #
        # self.gamma[0] = project(g_a, self.controlSpace[0])

    def updateCoefficients(self):
        # Init coefficient matrix
        x, y = SpatialCoordinate(self.mesh)

        self.a = self.gamma[0] * as_matrix([[1.0, 0.0], [0.0, 1.0]])
        self.b = as_vector([Constant(0.0), Constant(0.0)])
        self.c = Constant(0.0)

        # Init right-hand side
        # self.f_t = 529 * (sin(xxx) + 0.5 * sin(2 * xxx) + 4./10 * sin(8 * xxx))**2
        # self.f_t = (sin(pi**2*(x-0.63)*(y-0.26)/0.07))**2
        # self.f_t = Constant(1.)
        # self.u_ = sin(pi*x) * sin(pi * y)
        # self.f = -self.alpha1 * 2 * pi**2 * sin(pi*x) * sin(pi*y)

        self.u_ = sin(2*pi*x) * sin(2*pi * y)
        self.f = -conditional((x-0.5)*(y-0.5) > 0,
                              self.alpha1, self.alpha0) * 8*pi**2 * self.u_

        # Set boundary conditions
        self.g = Constant(0.0)

    def initMesh(self, n):
        # self.mesh = mesh_Lshape(n)
        self.mesh = mesh_UnitSquare(n)
        # self.mesh = mesh_UnitSquare(n, 'mshr')
