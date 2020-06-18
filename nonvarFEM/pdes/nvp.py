import itertools
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from nonvarFEM.helpers.plotErrorEstimates import plotErrorEstimates
from dolfin import FunctionSpace, TestFunction, MeshFunction, Function
from dolfin import inner, grad, assemble, dx, dS, avg, tr, plot
from dolfin import project, interpolate, refine
from dolfin import File, error, warning
from dolfin import FacetNormal, FacetArea
from dolfin import FiniteElement, TensorElement, MixedElement
from dolfin import FunctionAssigner

# Import solvers
from nonvarFEM.solvers import solverFEHessianDirect, solverFEHessianGMRES
from nonvarFEM.solvers import solverNeilanSalgadoZhang, solverNeilan

from nonvarFEM.norms import vj


# --------------------------------------------------
# Base class for nonvariational problems:
# --------------------------------------------------


class NVP:

    def solve(self, opt):

        if opt["solutionMethod"] == 'FEHessianDirect':
            N_iter = solverFEHessianDirect(self, opt)

        elif opt["solutionMethod"] == 'FEHessianGmres':
            N_iter = solverFEHessianGMRES(self, opt)

        elif opt["solutionMethod"] == 'NeilanSalgadoZhang':
            N_iter = solverNeilanSalgadoZhang(self, opt)

        elif opt["solutionMethod"] == 'Neilan':
            opt["HessianSpace"] = "DG"
            N_iter = solverNeilan(self, opt)

        return N_iter

    def determineErrorEstimates(self, opt):
        """ This method determines the error estimates
            eta^2 = sum(eta_T^2)
        with 
            eta_T^2 = || γ f - γ A : Hess(u) ||_{L^2(T)}^2
                      + sum_{e \in E_T^I} h_e^{-1} * || jump(grad(u)) ||_{L^2(e)}^2
        according to (52) in the paper. """

        DG0 = FunctionSpace(self.mesh, "DG", 0)
        dg = TestFunction(DG0)

        # Cell error
        cell_residual = self.f - inner(self.a, grad(grad(self.u)))
        if self.hasDrift:
            cell_residual -= inner(self.b, grad(self.u))
        if self.hasPotential:
            cell_residual -= self.c * self.u

        gamma = self.normalizeSystem(opt)

        eta_cell = assemble((gamma * cell_residual) ** 2 * dg * dx)

        # Edge error

        edge_residual = avg(
            self.hE)**(-1) * inner(vj(grad(self.u), self.nE), vj(grad(self.u), self.nE))
        # edge_residual = avg(self.hE)**(-1) * inner( jump(grad(self.u)), jump(grad(self.u)))

        eta_edge = assemble(edge_residual * avg(dg) * dS)

        eta = eta_cell.get_local() + eta_edge.get_local()

        if opt["plotErrorEstimates"]:
            plotErrorEstimates(eta_cell, eta_edge, eta,
                               self.mesh, dim=self.dim())

        return eta

    def normalizeSystem(self, opt):
        if opt["normalizeA"]:

            gamma_nom = tr(self.a)
            gamma_denom = inner(self.a, self.a)

            if self.hasDrift:
                gamma_denom += inner(self.b, self.b) / (2*opt["lambda"])

            if self.hasPotential:
                gamma_nom += self.c / opt["lambda"]
                gamma_denom += (self.c / opt["lambda"])**2

            return gamma_nom / gamma_denom

        else:
            return 1.0

    def markCells(self, opt, eta):

        # Sort descending
        idx = np.argsort(-eta)

        # Sum of eta
        eta_sum = sum(eta)

        eta_cumsum = np.cumsum(eta[idx])

        ref_idx = idx[eta_cumsum < opt["refinementThreshold"] * eta_sum]

        n_ref = len(ref_idx)
        min_ref_number = 10

        if n_ref < min_ref_number:
            ref_idx = idx[:min_ref_number]
            n_ref = len(ref_idx)

        print("Marked %d cells for refinement" % n_ref)

        # cell_markers = MeshFunctionBool(self.mesh, self.dim())
        cell_markers = MeshFunction('bool', self.mesh, self.dim())
        cell_markers.set_all(False)
        cell_markers.array()[ref_idx] = True

        return cell_markers

    def doPlots(self, opt):

        if opt["plotMesh"]:
            self.plotMesh()

            if opt["saveMesh"]:
                plt.axis("off")
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig('./pdf/' + opt["id"] + '_mesh_dimV_%05d.pdf' %
                            self.solDofs(opt), bbox_inches='tight', pad_inches=0)

        if opt["plotSolution"]:

            self.plotSolution()

            if hasattr(self, 'H'):
                self.plotFEHessian()

        if opt["saveSolution"]:
            file = File("./pvd/" + opt["id"] + '_' +
                        '_sol_dimV_%05d.pvd' % self.solDofs(opt))
            file << self.u

    def plotMesh(self, fig=None):

        if not fig:
            fig = plt.figure('mesh', clear=True)

        xy = self.mesh.coordinates()
        plt.triplot(tri.Triangulation(
            xy[:, 0], xy[:, 1], self.mesh.cells()), linewidth=0.3, color='k')
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.draw()
        plt.pause(0.01)

    def plotFEHessian(self, fig=None):

        if not fig:
            fig = plt.figure('FE Hessian', clear=True)
        D = self.dim()
        for (i, j) in itertools.product(range(D), range(D)):
            fig.add_subplot(D, D, i*D+j+1)
            if D == 1:

                # plot(self.H)
                #
                if self.W_H.ufl_element().degree() == 1:
                    error('Not implemented')
                    # x = V.tabulate_dof_coordinates()
                    # y1 = self.H.vector().get_local()
                else:
                    x = self.mesh.coordinates().flatten()
                    xidx = np.argsort(x)
                    x = x[xidx]
                    y1 = self.H.compute_vertex_values()[xidx]

                plt.plot(x, y1)
            elif D == 2:
                plot(self.H.sub(i*D+j))
            else:
                warning('No plots in dimension %i possible' % self.dim())
                return
            ax = plt.gca()
            ax.set_aspect('auto')

        # plt.axis('tight')
        plt.draw()
        plt.pause(0.01)

    def plotSolution(self, fig=None, mode=1):
        if not fig:
            fig = plt.figure('u', clear=True)
        V = self.V
        # V = FunctionSpace(self.mesh,'CG',1)

        if mode == 1:
            if not self.hasSolution:  # No exact solution is available, we plot only the function u
                if self.dim() == 1:
                    if V.ufl_element().degree() == 1:
                        x = V.tabulate_dof_coordinates()
                        y1 = self.u.vector().get_local()
                    else:
                        x = self.mesh.coordinates().flatten()
                        xidx = np.argsort(x)
                        x = x[xidx]
                        y1 = self.u.compute_vertex_values()[xidx]

                    plt.plot(x, y1)
                    plt.title('Numerical solution')
                    plt.axis('tight')
                else:
                    # ax = fig.add_subplot(111, projection='3d')
                    ax = fig.add_subplot(111)
                    c = plot(self.u)
                    plt.colorbar(c)
                    ax.set_aspect('auto')

            else:  # Exact solution is available, we plot them besides with the difference

                if isinstance(self.u_, Function):
                    u_exact = Function(V)
                    u_exact.interpolate(self.u_)
                else:
                    u_exact = project(self.u_, self.V)
                u_diff = self.u - u_exact

                if self.dim() == 1:

                    ax = fig.add_subplot(3, 1, 1)
                    self.plot_1d_fun(self.u, self.V)
                    plt.title('Numerical solution')
                    plt.axis('tight')

                    fig.add_subplot(3, 1, 2)
                    self.plot_1d_fun(self.u_, self.V)
                    plt.title('Analytical solution')
                    # plt.plot(x, y2)
                    plt.axis('tight')

                    fig.add_subplot(3, 1, 3)
                    self.plot_1d_fun(u_diff, self.V)
                    plt.title('Pointwise difference')
                    # plt.plot(x, y3)
                    plt.axis('tight')

                # if self.dim() == 1:
                #     if V.ufl_element().degree() == 1:
                #         x = V.tabulate_dof_coordinates()
                #         y1 = self.u.vector().get_local()
                #         y2 = u_exact.vector().get_local()
                #         y3 = u_diff.vector().get_local()

                #     else:
                #         x = self.mesh.coordinates().flatten()
                #         xidx = np.argsort(x)
                #         x = x[xidx]
                #         y1 = self.u.compute_vertex_values()[xidx]
                #         y2 = u_exact.compute_vertex_values()[xidx]
                #         y3 = y1 - y2
                #         #  y3 = u_diff.compute_vertex_values()[xidx]

                #         # Approximate y for higher polynomials
                #         # V_dofs = V.tabulate_dof_coordinates()
                #         # xmin = min(V_dofs)
                #         # xmax = max(V_dofs)
                #         # Nint = 10
                #         # y1=np.empty((Nint,))
                #         # y2=np.empty((Nint,))
                #         # y3=np.empty((Nint,))
                #         # x = np.linspace(xmin,xmax,Nint)
                #         # u.eval(y1, x)
                #         # u_exact.eval(y2, x)
                #         # u_diff.eval(y3, x)
                #         # sys.exit()

                #     ax = fig.add_subplot(3,1,1)
                #     plt.plot(x, y1)
                #     plt.title('Numerical solution')
                #     plt.axis('tight')
                #
                #     fig.add_subplot(3,1,2)
                #     plt.title('Analytical solution')
                #     plt.plot(x, y2)
                #     plt.axis('tight')

                #     fig.add_subplot(3,1,3)
                #     plt.title('Pointwise difference')
                #     plt.plot(x, y3)
                #     plt.axis('tight')

                elif self.dim() == 2:
                    ax = fig.add_subplot(3, 1, 1)
                    c1 = plot(self.u)
                    plt.colorbar(c1)
                    plt.title('Numerical solution')
                    ax.set_aspect('auto')

                    ax = fig.add_subplot(3, 1, 2)
                    c2 = plot(u_exact)
                    plt.colorbar(c2)
                    plt.title('Analytical solution')
                    ax.set_aspect('auto')

                    ax = fig.add_subplot(3, 1, 3)
                    c3 = plot(u_diff)
                    plt.colorbar(c3)
                    plt.title('Pointwise difference')
                    ax.set_aspect('auto')

                else:
                    warning('No plots in dimension %i possible' % self.dim())
                    return

            plt.draw()
            plt.pause(0.01)
            return ax

        if mode == 2:
            # ax = fig.add_subplot(1,1,1, projection='3d')
            ax = fig.add_subplot(1, 1, 1)
            tp = plot(self.u)
            tp.set_cmap("viridis")
            # plt.title('Numerical solution $u_h$')

            # ax = fig.add_subplot(1,1,1, projection='3d')
            # plot(self.u)
            # plot(self.u.root_node(), mode='warp')
            ax.set_aspect('auto')
            return ax

        if mode == 3:
            if self.hasSolution:

                # ax = fig.add_subplot(1,1,1, projection='3d')
                ax = fig.add_subplot(1, 1, 1)

                # sub2Func = FunctionAssigner(V, self.mixedSpace.sub(0))
                # u_onV = Function(V)
                # sub2Func.assign(u_onV, self.x.sub(0))
                # u_exact = interpolate(self.u_, V)
                # u_diff = Function(V)
                # u_diff.vector()[:] = u_onV.vector() - u_exact.vector()
                # u_diff.vector().abs()
                #
                # if not fig:
                #     fig = plt.figure()
                #     fig.clf()
                # plot(u_diff)

                u_exact = interpolate(self.u_, V)
                tp = plot(u_exact - self.u)
                tp.set_cmap("viridis")
                # plt.title('Difference $|u - u_h|$')
                # plot(u_diff.root_node(), mode='warp')
                return ax

    def plot_1d_fun(self, fun, space, *args, **kwargs):

        if not isinstance(fun, Function):
            fun = project(fun, space)

        element_degree = fun.ufl_element().degree()

        if element_degree == 0:  # DG 0
            x = self.mesh.coordinates().flatten()
            y1 = fun.vector().get_local()
            X = np.vstack([x[:-1], x[1:]])
            Y = np.vstack([y1, y1])

        elif element_degree == 1:
            # ipdb.set_trace()
            X = space.tabulate_dof_coordinates()
            Y = fun.vector().get_local()
            if fun.ufl_element().family() == 'Discontinuous Lagrange':
                X = X.reshape((-1, 2)).T
                Y = Y.reshape((-1, 2)).T
        else:
            x = self.mesh.coordinates().flatten()
            xidx = np.argsort(x)
            X = x[xidx]
            Y = fun.compute_vertex_values()[xidx]
        plt.plot(X, Y, *args, **kwargs)

    def printCordesInfo(self):

        CordesCond = inner(self.a, self.a) / pow(tr(self.a), 2)

        TempSpace = FunctionSpace(self.mesh, "DG", 0)
        cc = project(CordesCond, TempSpace)

        max_lhs = max(cc.vector().get_local())
        cordes_eps = 1./max_lhs - self.dim() + 1

        print("Cordes epsilon %6.4f should be in (0,1]" % (cordes_eps))

        if cordes_eps > 0 and cordes_eps <= 1:
            print('Cordes condition is met')
        else:
            print('WARNING: Cordes condition is NOT met')

    def refineMesh(self, cell_markers=None, N=None):

        # Update mesh
        if cell_markers:
            print("Adaptive refinement")
            self.mesh = refine(self.mesh, cell_markers)
        elif N:
            print("Init mesh with N = ", N)
            self.initMesh(N)
        else:
            print("Uniform refinement")
            self.mesh = refine(self.mesh)

    def solDofs(self, opt):
        return self.V.dim()

    def totalDofs(self, opt):
        if opt["solutionMethod"] == 'NeilanSalgadoZhang':
            return self.V.dim()
        else:
            return self.mixedSpace.dim()

    def updateFunctionSpaces(self, opt):

        m_ufl_cell = self.mesh.ufl_cell()
        if opt["solutionMethod"] == 'FEHessianDirect':
            self.HElement = TensorElement(
                opt['HessianSpace'], m_ufl_cell, opt["q"])
            self.pElement = FiniteElement(
                opt['HessianSpace'], m_ufl_cell, opt["q"])
            self.uElement = FiniteElement("CG", m_ufl_cell, opt["p"])

            mixedElement = MixedElement(
                [self.HElement, self.pElement, self.uElement])

            self.mixedSpace = FunctionSpace(self.mesh, mixedElement)
            self.W_H = self.mixedSpace.sub(0).collapse()
            self.W_P = self.mixedSpace.sub(1).collapse()
            self.V = self.mixedSpace.sub(2).collapse()
            self.uAssigner = FunctionAssigner(self.V, self.mixedSpace.sub(2))

            self.x = Function(self.mixedSpace)
            (self.H, self.p, self.u) = self.x.split()

        elif opt["solutionMethod"] == 'NeilanSalgadoZhang':
            self.uElement = FiniteElement("CG", m_ufl_cell, opt["p"])

            self.V = FunctionSpace(self.mesh, self.uElement)
            self.x = Function(self.V)
            self.u = self.x
            self.uAssigner = FunctionAssigner(self.V, self.V)

        else:

            self.uElement = FiniteElement("CG", m_ufl_cell, opt["p"])
            self.HElement = TensorElement(
                opt['HessianSpace'], m_ufl_cell, opt["q"])
            print("Function spaces: Solution %s%d, Hessian %s%d"
                  % ('CG', opt['p'], opt['HessianSpace'], opt['q']))

            self.mixedElement = MixedElement([self.uElement, self.HElement])

            # Initialize solution space and function
            self.mixedSpace = FunctionSpace(self.mesh, self.mixedElement)
            self.x = Function(self.mixedSpace)
            (self.u, self.H) = self.x.split()

            # if self.hasDrift:
            #     (self.u, self.H, self.G) = self.x.split()
            # else:
            #     (self.u, self.H) = self.x.split()

            self.W_H = self.mixedSpace.sub(1).collapse()
            self.V = self.mixedSpace.sub(0).collapse()
            self.uAssigner = FunctionAssigner(self.V, self.mixedSpace.sub(0))

        self.nE = FacetNormal(self.mesh)
        self.hE = FacetArea(self.mesh)**(1.0/(self.dim()-1))

    def getType(self):
        return 'NVP'

    def checkVariables(self):
        self.hasSolution = 1 if hasattr(
            self, 'u_') or hasattr(self, 'u_t') else 0
        self.hasDrift = 1 if hasattr(self, 'b') or hasattr(self, 'b_t') else 0
        self.hasPotential = 1 if hasattr(
            self, 'c') or hasattr(self, 'c_t') else 0
        self.isTimeDependant = 1 if hasattr(self, 'T') else 0

    def dim(self):
        return self.mesh.topology().dim()

    def updateCoefficients(self):
        raise NameError("updateCoefficients for this problem not implemented")

    # TODO: In case we need this, we should update this, otherwise delete it!
    # def computeEstimateOfConstants(self):
    #
    #     # Check discrete Miranda Talenti estimate
    #     h2h = H2h_norm(self.mesh, self.u)
    #     hess_L2 = L2_norm(self.mesh, self.H)
    #     lapl_L2 = L2_norm(self.mesh, self.H.sub(0) + self.H.sub(3))
    #
    #     C_S = h2h / hess_L2
    #     C_MT = h2h / lapl_L2
    #     C_H = lapl_L2 / hess_L2
    #
    #     print("Stability constant:       C_S  = ", C_S)
    #     print("Miranda-Talenti constant: C_MT = ", C_MT)
    #     print("||H_lapl(u)||_L^2 \ ||H(u)||_L^2 = ", C_H)
    #
    #     return (C_S, C_MT, C_H)
