import sys
sys.path.insert(0, '../..')

# Problem to solve
from nonvarFEM.problems.testsParabolicNVP import PNVP_WorstOfTwoAssetsPut

# Solve routine
from nonvarFEM import solveProblem

# Auxiliary stuff
import nonvarFEM.helpers as hlp

WRITE_CSV = True


def experiment(P, opt, expname):
    # First run with regular refinement
    opt["meshRefinement"] = 1
    df_uniform = solveProblem(P, opt)
    if WRITE_CSV:
        fname = '{}_{}_uniform'.format(opt['id'], expname)
        hlp.writeOutputToCsv(df_uniform, opt, fname)

    # Second run with adaptive refinement
    # opt["meshRefinement"] = 2
    # opt["refinementThreshold"] = 0.95
    # df_adaptive = solveProblem(P, opt)
    # if WRITE_CSV:
    #     fname = '{}_{}_adaptive'.format(opt['id'], expname)
    #     hlp.writeOutputToCsv(df_adaptive, opt, fname)


if __name__ == "__main__":

    """
    Global settings
    """
    global_opt = hlp.standardOptions()
    P = PNVP_WorstOfTwoAssetsPut()

    for p in range(1, 2):
    # for p in range(2, 3):
    # for p in range(3, 4):
    # for p in range(4, 5):
        global_opt["id"] = "WorstAssetPut_true2_bc_deg_{}".format(p)

        # Threshold for dofs
        global_opt["NdofsThreshold"] = 20000
        global_opt["plotMesh"] = 0   # Plot mesh flag
        global_opt["plotSolution"] = 0   # Plot solution flag
        global_opt["initialMeshResolution"] = 7

        # Fix polynomial degree
        global_opt["p"] = p
        global_opt["q"] = p
        global_opt["holdOn"] = 0
        global_opt["timeSteps"] = 1000
        global_opt["timeStepFactor"] = 1

        # Determine method to estimate error norms
        global_opt["errorEstimationMethod"] = 1
        global_opt["writeToCsv"] = 0
        """
        Designing the different experiments.
        """
        experiment(P,
                hlp.opt_NeilanSalgadoZhang(global_opt),
                'NeilanSalgadoZhang')
        
        # Reduced version does currently not support a non-zero drift and potential term
        # experiment(P,
        #            hlp.opt_Own_CG_0_stab(global_opt),
        #            'CG_0_stab')
        
        # The following is rather slow
        # experiment(P,
        #            hlp.opt_Own_CG_0_stab_complete(global_opt),
        #            'CG_0_stab')
