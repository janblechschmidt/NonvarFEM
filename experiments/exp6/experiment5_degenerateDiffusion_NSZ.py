import sys
sys.path.insert(0, '../..')

# Solve routine
from nonvarFEM import solveProblem

import nonvarFEM.problems.testsParabolicNVP as PNVP
import dolfin

import nonvarFEM.helpers as hlp

WRITE_CSV = True

def experiment(P, opt, expname):
    # First run with regular refinement
    opt["meshRefinement"] = 1
    df_uniform = solveProblem(P, opt)
    if WRITE_CSV:
        hlp.writeOutputToCsv(df_uniform, opt, opt['id'] + expname + '_uniform', df_uniform)


if __name__ == "__main__":

    dolfin.set_log_level(21)

    """
    Global settings
    """
    opt = hlp.standardOptions()
    # Works well
    eps_range = [0.1, 0.01, 0.001]
    # Errors explode for this case
    # eps_range = [0.0001]
    for eps in eps_range:
        P = PNVP.PNVP_Degenerate_1(eps)

        opt["id"] = "Degenerate_eps_{}".format(eps)

        # Threshold for dofs
        opt["NdofsThreshold"] = 50000
        opt["initialMeshResolution"] = 10
        opt["timeSteps"] = 10
        opt["timeStepFactor"] = 2

        # Fix polynomial degree
        opt["p"] = 2
        opt["q"] = 2

        opt["plotSolution"] = 0
        opt["plotErrorEstimates"] = 0
        opt["plotConvergenceRates"] = 0
        opt["plotMesh"] = 0
        opt["meshRefinement"] = 1  # Uniform refinement
        opt["stabilizationFlag"] = 1
        # Stability constant for first-order term
        opt["stabilityConstant1"] = 2
        # Stability constant for second-order term
        opt["stabilityConstant2"] = 0
        opt["solutionMethod"] = 'NeilanSalgadoZhang'

        df = solveProblem(P, opt)

        # Determine method to estimate error norms
        opt["errorEstimationMethod"] = 1

        if WRITE_CSV:
            hlp.writeOutputToCsv(df, opt, opt['id'])
