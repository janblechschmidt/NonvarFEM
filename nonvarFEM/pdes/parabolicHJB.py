# Dolfin imports
from dolfin import Function, File
from dolfin import assign, project

# Numpy
import numpy as np

# Definition of NVP class
from .hjb import HJB


class ParabolicHJB(HJB):

    def solve(self, opt):
        """ This procedure implements a first-order
        semi-Lagrangian time-stepping scheme to solve a parabolic
        second-order HJB equation in non-variational form
        - du/dt - sup_gamma{a^gamma : D^2 u + b^gamma * D u + c^gamma * u - f^gamma}= 0
        """

        if hasattr(self, 'dt'):
            opt["timeSteps"] *= opt["timeStepFactor"]

        nt = opt["timeSteps"]
        nc = len(self.gamma)

        Tspace = np.linspace(self.T[1], self.T[0], nt+1)
        self.dt = (self.T[1] - self.T[0]) / nt
        self.u_np1 = Function(self.V)
        
        print('Setting final time conditions')
        assign(self.u, project(self.u_T, self.V))

        if opt["saveSolution"]:
            file_u = File('./pvd/u.pvd')
            file_gamma = []
            for i in range(nc):
                file_gamma.append(File('./pvd/gamma_{}.pvd'.format(i)))

        for i, s in enumerate(Tspace[1:]):
            self.current_time = s
            print('Iteration {}/{}:\t t = {}'.format(i+1, nt, s))
            self.iter = i

            # Update time in coefficient functions
            self.updateTime(s)

            assign(self.u_np1, self.u)
            # Solve problem for current time step
            super().solve(opt)
            # self.plotControl()
            # self.plotSolution()

            if opt["saveSolution"]:
                file_u << self.u
                for i in range(nc):
                    try:
                        file_gamma[i] << self.gamma[i]
                    except AttributeError:
                        file_gamma[i] << project(
                            self.gamma[i], self.controlSpace[i])

    def getType(self):
        return 'ParabolicHJB'

    def updateTime(self, t):
        """ This is a copy of the method within parabolicNVP.
        """

        print('Update time: t = ', t)
        self.t.assign(t)
