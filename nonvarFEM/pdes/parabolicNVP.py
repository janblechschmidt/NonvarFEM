# Dolfin imports
from dolfin import Function, File
from dolfin import assign, project

import numpy as np

# Definition of NVP class
from .nvp import NVP


class ParabolicNVP(NVP):

    def solve(self, opt):
        """ This procedure implements a first-order
        semi-Lagrangian time-stepping scheme to solve a parabolic
        second-order PDE in non-variational form
                - du/dt - (a : D^2 u + b * D u + c * u )  =  - f
        <==>    - du/dt - (a : D^2 u + b * D u + c * u - f ) = 0
        """

        if hasattr(self, 'dt'):
            opt["timeSteps"] *= opt["timeStepFactor"]
        nt = opt["timeSteps"]

        Tspace = np.linspace(self.T[1], self.T[0], nt+1)
        self.dt = (self.T[1] - self.T[0]) / nt
        self.u_np1 = Function(self.V)

        if opt["saveSolution"]:
            file_u = File('./pvd/u.pvd')

        print('Setting final time conditions')
        assign(self.u, project(self.u_T, self.V))
        if opt["saveSolution"]:
            file_u << self.u

        for i, s in enumerate(Tspace[1:]):

            print('Iteration {}/{}:\t t = {}'.format(i+1, nt, s))
            self.iter = i

            # Update time in coefficient functions
            self.updateTime(s)

            assign(self.u_np1, self.u)

            # Solve problem for current time step
            super().solve(opt)

            if opt["saveSolution"]:
                file_u << self.u


    def getType(self):
        return 'ParabolicNVP'

    def updateTime(self, t):

        print('Update time: t = ', t)
        self.t.assign(t)
        # self.a = self.a_t(t)
        # if self.hasDrift:
        # self.b = self.b_t(t)
        # if self.hasPotential:
        # self.c = self.c_t(t)
        # if self.hasSolution:
        # self.u_ = self.u_t(t)
        # self.f = self.f_t(t)
        # self.g = self.g_t(t)
