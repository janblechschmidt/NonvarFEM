import sys
sys.path.insert(0, '../..')

# Solve routine
from nonvarFEM import solveProblem

# Auxiliary stuff
import nonvarFEM.helpers as hlp

import nonvarFEM.problems.testsHJB as HJB

import dolfin

WRITE_CSV = True

def experiment(P, opt, expname):
    # First run with regular refinement
    opt["meshRefinement"]    = 1
    df_uniform = solveProblem(P, opt)
    if WRITE_CSV:
        hlp.writeOutputToCsv(df_uniform, opt, opt['id'] + expname + '_uniform')

    # Second run with adaptive refinement
    opt["meshRefinement"]      = 2
    opt["refinementThreshold"] = 0.95
    opt["plotMesh"] = 0
    opt["saveMesh"] = 1
    opt["plotSolution"] = 0
    df_adaptive = solveProblem(P, opt)
    if WRITE_CSV:
        hlp.writeOutputToCsv(df_adaptive, opt, opt['id'] + expname + '_adaptive')

if __name__ == "__main__":


    dolfin.set_log_level(21)

    """
    Global settings
    """
    global_opt = hlp.standardOptions()
    P = HJB.HJB_H_alpha(alpha=1.3)
    p = 2
    global_opt["id"] = "HJB_HalphaSol_deg_{}".format(p)


    # Threshold for dofs
    global_opt["NdofsThreshold"]   = 30000
    # global_opt["NdofsThreshold"]   = 3000
    global_opt["initialMeshResolution"]  = 2
    global_opt["gmresTolRes"] = 1e-7
    
    # Fix polynomial degree
    global_opt["p"] = p
    global_opt["q"] = p
    
    # Determine method to estimate error norms
    global_opt["errorEstimationMethod"] = 1

    # Specify experiments method
    experiment(P,
            hlp.opt_Own_CG_0_stab(global_opt),
            'CG_0_stab')

    # experiment(P,
    #         hlp.opt_NeilanSalgadoZhang(global_opt),
    #         'NSZ_stab2')
    # experiment(P,
    #         opt_Own_CG_1_stab(global_opt),
    #             'CG_0_stab')
