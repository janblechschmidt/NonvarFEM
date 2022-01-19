# from dolfin import *
from .nvp import NVP
import numpy as np
from nonvarFEM.norms import L2_norm, H10_norm, H20_norm
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from dolfin import Function, project, plot


class HJB(NVP):

    def solve(self, opt):
        """ This routine implements Howards algorithm
        aka Policy iteration.
        INIT: Solve problem with current control gives u_prev
        Repeat until done:
          - Update control
          - Solve problem and determine u_curr
          - if u_curr and u_prev are close, you're done
        """

        # Initialize Howard iteration
        Howard_done = 0
        Howard_iter = 1
        Howard_max_iter = 500
        Howard_threshold = 1e-10

        uold = Function(self.V)
        unew = Function(self.V)

        cycleDetection = True

        # Solve non-variational pde with initial control
        super().solve(opt)
        self.uAssigner.assign(uold, self.u)
        if cycleDetection:
            u_mean_array = np.zeros(shape=(Howard_max_iter+1,), dtype='float')
            u_mean_array[0] = uold.vector().get_local().mean()
        print('-------------------------------------')
        print('        Howard iteration             ')
        print('---------------------------------------------')
        print('     |        || u^k - u^{k-1} ||     ')
        print('Iter |  L2_norm |  H1_semi |  H2_semi ')
        while not Howard_done:

            self.updateControl()

            # This incorporates the current control
            self.updateCoefficients()

            if self.isTimeDependant:
                self.updateTime(self.current_time)

            # Solve non-variational pde with current control
            super().solve(opt)
            self.uAssigner.assign(unew, self.u)

            udiff = unew - uold
            Howard_error = L2_norm(udiff)

            print('  {:2d} | {:.2e} | {:.2e} | {:.2e} '.format(
                Howard_iter,
                Howard_error,
                H10_norm(udiff),
                H20_norm(udiff)))
            if Howard_error < Howard_threshold:
                print('Howard iteration converged in {} iterations'.format(Howard_iter))
                Howard_done = True
            elif Howard_iter > Howard_max_iter:
                print('Howard iteration did not converge in {} iterations'.format(
                    Howard_iter))
                Howard_done = True
            else:
                if cycleDetection:
                    u_mean_new = unew.vector().get_local().mean()
                    if np.min(np.abs(u_mean_new-u_mean_array)) < 1e-14:
                        print('Cycle detected (Dist {})'.format(
                            np.min(np.abs(u_mean_new-u_mean_array))))
                        Howard_done = True
                        break
                    else:
                        u_mean_array[Howard_iter] = u_mean_new
                self.uAssigner.assign(uold, self.u)

            # self.plotSolution()
            # self.plotControl()
            # import ipdb
            # ipdb.set_trace()
            Howard_iter = Howard_iter + 1

        return Howard_iter

    def plotSolution(self, fig=None):
        super().plotSolution()
        self.plotControl()

    def plotControl(self, fig=None, mode=2):
        fig = plt.figure('control', clear=True)

        nc = len(self.gamma)

        if mode == 1:
            xy = self.mesh.coordinates()
            T = tri.Triangulation(xy[:, 0], xy[:, 1], self.mesh.cells())
            for i in range(nc):
                # fig.add_subplot(nc,1,i+1,projection='3d')
                fig.add_subplot(nc, 1, i+1)
                tmp = project(self.gamma[i], self.controlSpace[i])
                plt.tripcolor(T, tmp.vector().get_local())
        else:
            for i in range(nc):

                ax = fig.add_subplot(nc, 1, i+1)

                if self.dim() == 1:
                    self.plot_1d_fun(self.gamma[i], self.controlSpace[i], 'k-')

                    if hasattr(self, 'gamma_star'):
                        self.plot_1d_fun(
                            self.gamma_star[i], self.controlSpace[i], 'r-')

                else:
                    if isinstance(self.gamma[i], Function):
                        cbar = plot(self.gamma[i])
                    else:
                        cbar = plot(project(self.gamma[i], self.controlSpace[i]))

                    try:
                        plt.colorbar(cbar)
                    except AttributeError:
                        pass
                ax.set_aspect('auto')

        # fig = plt.figure('FEHessian', clear=True)
        # fig.add_subplot(1, 1, 1)
        # cbar2 = plot(project(self.H[0,0], self.controlSpace[0]))
        # plt.colorbar(cbar2)
        plt.draw()
        plt.pause(0.01)
        # import ipdb
        # ipdb.set_trace()

    def initControl(self, opt):
        raise NameError("initControl for this problem not implemented")

    def updateControl(self):
        raise NameError("updateControl for this problem not implemented")

    def getType(self):
        return 'HJB'
