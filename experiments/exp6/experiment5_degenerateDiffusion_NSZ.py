from solve import solveProblem
import problems.testsParabolicNVP as PNVP
import dolfin

from stats.writeOutputToCsv import writeOutputToCsv
from standardOptions import standardOptions

CSV_DIR = './results/experiment5/'
WRITE_CSV = True


def experiment(P, opt, expname):
    # First run with regular refinement
    opt["meshRefinement"] = 1
    df_uniform = solveProblem(P, opt)
    if WRITE_CSV:
        writeOutputToCsv(CSV_DIR + opt['id'] +
                         expname + '_uniform.csv', df_uniform)


if __name__ == "__main__":

    dolfin.set_log_level(21)

    """
    Global settings
    """
    opt = standardOptions()
    eps_range = [0.1, 0.01, 0.001]
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
        opt["plotErrorRates"] = 0
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
            writeOutputToCsv(CSV_DIR + opt['id'] + '.csv', df)
