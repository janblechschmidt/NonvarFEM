import sys
sys.path.insert(0, '../..')

# Solve routine
from nonvarFEM import solveProblem

# Auxiliary stuff
import nonvarFEM.helpers as hlp

import nonvarFEM.problems.testsParabolicHJB as PHJB
import nonvarFEM.problems.testsHJB as HJB

import dolfin

WRITE_CSV = True

def experiment(P, opt, expname):
    # First run with regular refinement
    opt["meshRefinement"]    = 1
    opt["plotSolution"] = 0
    df_uniform = solveProblem(P, opt)
    if WRITE_CSV:
        hlp.writeOutputToCsv(df_uniform, opt, opt["id"] + expname + '_uniform')

    # Second run with adaptive refinement
    # opt["meshRefinement"]      = 2
    # opt["refinementThreshold"] = 0.95
    # opt["plotMesh"] = 0
    # opt["saveMesh"] = 0
    # opt["plotSolution"] = 0
    # opt["timeSteps"] = 320
    # opt["timeStepFactor"] = 1
    # df_adaptive = solveProblem(P, opt)
    # if WRITE_CSV:
    #     hlp.writeOutputToCsv(df_adaptive, opt, opt["id"] + expname + '_adaptive')

if __name__ == "__main__":


    dolfin.set_log_level(21)

    """
    Global settings
    """
    global_opt = hlp.standardOptions()

    # Polynomial degree
    p = 2
    
    # P = HJB.MinimumArrivalTime(alpha=0.5,
    #         beta=0.5,
    #         sigmax=0.5,
    #         sigmay=0.5,
    #         cmin = -100.0,
    #         cmax = 100.0,
    #         corrxy=0.9)
    # global_opt["id"] = "HJB_MAT_stationary_corr_0_9_deg_{}".format(p)

    # P = HJB.MinimumArrivalTime(alpha=0.5,
    #         beta=0.5,
    #         sigmax=0.5,
    #         sigmay=0.5,
    #         cmin = -100.0,
    #         cmax = 100.0,
    #         corrxy='pw')
    # global_opt["id"] = "HJB_MAT_stationary_corr_pw_deg_{}".format(p)

    P = PHJB.MinimumArrivalTimeParabolic(alpha=0.5,
            beta=0.5,
            sigmax=0.5,
            sigmay=0.5,
            cmin = -1.,
            cmax = 1.,
            corrxy=0.0)
    global_opt["id"] = "HJB_MAT_bc_no_normalization_corr_0_0_deg_{}".format(p)

    P = PHJB.MinimumArrivalTimeParabolic(alpha=0.5,
            beta=0.5,
            sigmax=0.5,
            sigmay=0.5,
            cmin = -1.,
            cmax = 1.,
            corrxy=0.9)
    global_opt["id"] = "HJB_MAT_bc_no_normalization_corr_0_9_deg_{}".format(p)

    P = PHJB.MinimumArrivalTimeParabolic(alpha=0.5,
            beta=0.5,
            sigmax=0.5,
            sigmay=0.5,
            cmin = -1.,
            cmax = 1.,
            corrxy='pw')
    global_opt["id"] = "HJB_MAT_bc_no_normalization_corr_pw_deg_{}".format(p)

    global_opt["normalizeSystem"] = 0  # Normalize A

    # Threshold for dofs
    global_opt["NdofsThreshold"]   = 17000
    # global_opt["NdofsThreshold"]   = 3000
    global_opt["initialMeshResolution"]  = 2
    global_opt["gmresTolRes"] = 1e-7
    global_opt["timeSteps"] = 640
    global_opt["timeStepFactor"] = 1
    
    # Fix polynomial degree
    global_opt["p"] = p
    global_opt["q"] = p
    global_opt["plotSolution"] = 0
    
    # Determine method to estimate error norms
    global_opt["errorEstimationMethod"] = 1
    
    experiment(P,
            hlp.opt_NeilanSalgadoZhang(global_opt),
            'NSZ')
