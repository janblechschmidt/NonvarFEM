import dolfin

from solve import solveProblem
import problems.testsNVP as NVP

from stats.writeOutputToCsv import writeOutputToCsv
from standardOptions import standardOptions

CSV_DIR = './results/Sol_in_H_alpha_3d/new_'
WRITE_CSV = True

if __name__ == "__main__":

    dolfin.set_log_level(21)

    """
    Global settings
    """
    opt = standardOptions()
    opt["initialMeshResolution"] = 2

    alpha = 0.75
    P = NVP.Sol_in_H_alpha_3d(alpha)

    opt["id"] = '3d_alpha_deg2_0_75_'

    # Fix polynomial degree
    opt["p"] = 2
    opt["q"] = 2

    # Determine method to estimate error norms
    opt["errorEstimationMethod"] = 1

    # Specify experiments method
    """
    Designing the different experiments.
    """
    # Own uniform
    opt["NdofsThreshold"] = 32000
    opt["solutionMethod"] = 'FEHessianGmres'
    opt["meshRefinement"] = 1
    opt["stabilizationFlag"] = 0
    df = solveProblem(P, opt)
    if WRITE_CSV:
        writeOutputToCsv(CSV_DIR + opt['id'] +
                         'CG_0_stab' + '_uniform.csv', df)

    # Own adaptive
    # opt["NdofsThreshold"] = 25000
    # opt["solutionMethod"] = 'FEHessianGmres'
    # opt["meshRefinement"] = 2
    # opt["refinementThreshold"] = 0.9
    # opt["stabilizationFlag"] = 0
    # df = solveProblem(P, opt)
    # if WRITE_CSV:
    #     writeOutputToCsv(CSV_DIR + opt['id'] +
    #                      'CG_0_stab' + '_adaptive.csv', df)

    # # Neilan, Salgado, Zhang approach
    # opt["NdofsThreshold"] = 32000
    # opt["solutionMethod"] = 'NeilanSalgadoZhang'
    # opt["meshRefinement"] = 1
    # opt["stabilizationFlag"] = 1
    # opt["stabilityConstant1"] = 1.0  # Stability constant for first-order term
    # opt["stabilityConstant2"] = 0  # Stability constant for second-order term
    # df = solveProblem(P, opt)
    # if WRITE_CSV:
    #     writeOutputToCsv(CSV_DIR + opt['id'] +
    #                      'NeilanSalgadoZhang' + '_uniform.csv', df)
