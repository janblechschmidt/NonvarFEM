from NVFEM import solveProblem
import dolfin

from stats.writeOutputToCsv import writeOutputToCsv
from NonvariationalProblems import Prob_Sol_in_H_alpha
from Problems_Feng_Neilan_Schnake import Prob_FNS_5_4_L_inf_Coeffs
from standardOptions import *

CSV_DIR = './results/Sol_in_H_alpha/'
WRITE_CSV = True

def experiment(P, opt, expname):
    # First run with regular refinement
    opt["meshRefinement"]    = 1
    df_uniform = solveProblem(P, opt)
    if WRITE_CSV:
        writeOutputToCsv(CSV_DIR + opt['id'] + expname + '_uniform.csv', df_uniform)

    # Second run with adaptive refinement
    opt["meshRefinement"]      = 2
    opt["refinementThreshold"] = 0.9
    df_adaptive = solveProblem(P, opt)
    if WRITE_CSV:
        writeOutputToCsv(CSV_DIR + opt['id'] + expname + '_adaptive.csv', df_adaptive)

if __name__ == "__main__":


    dolfin.set_log_level(21)

    """
    Global settings
    """
    global_opt = standardOptions()
    
    P = Prob_FNS_5_4_L_inf_Coeffs()
    global_opt["id"] = "FNS_5_4_"

    # Threshold for dofs
    global_opt["NdofsThreshold"]   = 100000

    # Fix polynomial degree
    global_opt["p"] = 2
    global_opt["q"] = 2

    # Determine method to estimate error norms
    global_opt["errorEstimationMethod"] = 1
    
    # Specify experiments method

    """
    Designing the different experiments.
    """

    experiment(P,
            opt_Neilan(global_opt),
            'Neilan')

    experiment(P,
            opt_NeilanSalgadoZhang(global_opt),
            'NeilanSalgadoZhang')

    experiment(P,
            opt_Own_CG_0_stab(global_opt),
            'CG_0_stab')

    experiment(P,
            opt_Own_CG_1_stab(global_opt),
            'CG_1_stab')

    experiment(P,
            opt_Own_CG_2_stab(global_opt),
            'CG_2_stab')
    
    experiment(P,
            opt_Own_DG_0_stab(global_opt),
            'DG_0_stab')

    experiment(P,
            opt_Own_DG_1_stab(global_opt),
            'DG_1_stab')
    
