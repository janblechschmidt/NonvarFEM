# -*- coding: utf-8 -*-

# Auxiliary stuff
import nonvarFEM.helpers as hlp

# Elliptic non-variational pdes
import nonvarFEM.problems.testsNVP as NVP

# Parabolic non-variational pdes
import nonvarFEM.problems.testsParabolicNVP as PNVP

# Elliptic HJB equations
import nonvarFEM.problems.testsHJB as HJB

import nonvarFEM.problems.testsParabolicHJB as PHJB

from nonvarFEM import solveProblem


if __name__ == "__main__":

    opt = hlp.standardOptions()

    opt["initialMeshResolution"] = 2
    opt["timeSteps"] = 10
    opt["timeStepFactor"] = 2
    opt["printCordesInfo"] = 0

    opt["plotSolution"] = 1
    opt["plotErrorEstimates"] = 0
    opt["plotConvergenceRates"] = 0
    opt["plotMesh"] = 0
    opt["saveMesh"] = 0
    opt["holdOn"] = 0
    opt["normalizeSystem"] = 0
    opt["meshRefinement"] = 1

    opt["refinementThreshold"] = .80
    opt["p"] = 2
    opt["q"] = 2
    opt["HessianSpace"] = "CG"
    # opt["NdofsThreshold"] = 50000
    opt["NdofsThreshold"] = 4000
    opt["errorEstimationMethod"] = 1
    opt["time_check"] = 1
    opt["stabilizationFlag"] = 0
    opt["stabilityConstant1"] = 2  # Stability constant for first-order term
    opt["stabilityConstant2"] = 0  # Stability constant for second-order term
    # opt["solutionMethod"] = 'BHWcomplete'
    opt["solutionMethod"] = 'BHWreduced'
    # opt["solutionMethod"] = 'NeilanSalgadoZhang'
    # opt["solutionMethod"] = 'Neilan'

    opt["dolfinLogLevel"] = 21

    # P = NVP.Cinfty(0.99)
    # alpha = 1.5
    # P = NVP.Sol_in_H_alpha(alpha)
    # alpha = 0.25
    # alpha = 0.5
    alpha = 0.75
    # alpha = 1.0
    # alpha = 1.25
    P = NVP.Sol_in_H_alpha_3d(alpha)
    # P = NVP.No_Cordes()
    # P = NVP.Poisson()
    # P = NVP.Poisson_inhomoBC()
    # kappa = 0.5
    # P = NVP.Cinfty(kappa)
    # P = NVP.Discontinuous_A()
    # P = NVP.Identity_A()
    # P = NVP.Boundary_Layer()

    # The following problem doesn't show nice convergence rates
    # P = NVP.Discontinuous_2nd_Derivative()

    # P = NVP.Neilan_Test1()
    # P = NVP.Neilan_5_2()
    # P = NVP.Neilan_Talk_2()

    # Problems from Lakkis, Pryer
    # P = NVP.LP_4_1_Nondiff_A()
    # P = NVP.LP_4_2_Conv_Dominated_A()
    # P = NVP.LP_4_3_Singular_Sol()
    # P = NVP.LP_4_4_Nonsymmetric_Hessian()

    # Problems from Feng, Neilan, Schnake
    # P = NVP.FNS_5_1_Hoelder_Cont_Coeffs()
    # P = NVP.FNS_5_2_Uniform_Cont_Coeffs()
    # P = NVP.FNS_5_3_Degenerate_Coeffs()
    # P = NVP.FNS_5_4_L_inf_Coeffs()

    # Further elliptic nonvariational pdes
    # P = NVP.FullNVP_1()
    # P = NVP.FullNVP_2()
    # P = NVP.FullNVP_3_1d()

    # Parabolic PDEs
    # P = PNVP.PNVP_1_2d()
    # P = PNVP.PNVP_2()
    # P = PNVP.PNVP_WorstOfTwoAssetsPut()
    # P = PNVP.PNVP_1_1d()
    # P = PNVP.PNVP_1_3d()
    # P = PNVP.PNVP_Degenerate_1()

    # Elliptic HJB equations
    # P = HJB.MinimumArrivalTime(alpha=0.1, beta=0.1)
    # P = HJB.MinimumArrivalTimeRadial(alpha=0.1)
    # P = HJB.Mother()
    # opt["lambda"] = 1.
    # P = HJB.Gallistl_Sueli_1()

    # Parabolic HJB equations
    # P = PHJB.JensenSmears_1()
    # P = PHJB.JensenSmears_2()
    # P = PHJB.Mother()
    # P = PHJB.MinimumArrivalTimeParabolic()
    # P = PHJB.Merton()
    # P = PHJB.Chen_Forsyth()
    from time import time
    t1 = time()
    df = solveProblem(P, opt)
    print('Time for solution: {}'.format(time() - t1))
