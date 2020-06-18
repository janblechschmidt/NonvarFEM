from nvfem import solveProblem
import problems.testsNVP as NVP
import dolfin

from stats.writeOutputToCsv import writeOutputToCsv
from standardOptions import *

CSV_DIR = './results/experiment4/'
WRITE_CSV = True

def experiment(P, opt, expname):
    # First run with regular refinement
    opt["meshRefinement"]    = 1
    df_uniform = solveProblem(P, opt)
    if WRITE_CSV:
        writeOutputToCsv(CSV_DIR + opt['id'] + expname + '_uniform.csv', df_uniform)

    # Second run with adaptive refinement
    opt["meshRefinement"]      = 2
    opt["refinementThreshold"] = 0.95
    df_adaptive = solveProblem(P, opt)
    if WRITE_CSV:
        writeOutputToCsv(CSV_DIR + opt['id'] + expname + '_adaptive.csv', df_adaptive)

if __name__ == "__main__":


    dolfin.set_log_level(21)

    """
    Global settings
    """
    global_opt = standardOptions()
    P = NVP.Discontinuous_A()

    for p in range(1,5):
        global_opt["id"] = "Disc_A_deg_{}_".format(p)


        # Threshold for dofs
        global_opt["NdofsThreshold"]   = 50000
    
        # Fix polynomial degree
        global_opt["p"] = p
        global_opt["q"] = p
    
        # Determine method to estimate error norms
        global_opt["errorEstimationMethod"] = 1
        
        # Specify experiments method
    
        """
        Designing the different experiments.
        """
        # 
        # experiment(P,
        #         opt_Neilan(global_opt),
        #         'Neilan')
        # 
        # experiment(P,
        #         opt_NeilanSalgadoZhang(global_opt),
        #         'NeilanSalgadoZhang')
        # 
        # experiment(P,
        #         opt_Own_CG_0_stab(global_opt),
        #         'CG_0_stab')
    
        experiment(P,
                opt_Own_CG_1_stab(global_opt),
                'CG_1_stab')
    
        # experiment(P,
        #         opt_Own_CG_2_stab(global_opt),
        #         'CG_2_stab')
        # 
        # experiment(P,
        #         opt_Own_DG_0_stab(global_opt),
        #         'DG_0_stab')
        # 
        # experiment(P,
        #         opt_Own_DG_1_stab(global_opt),
        #         'DG_1_stab')
        # 
