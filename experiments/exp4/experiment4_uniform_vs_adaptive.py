import sys
sys.path.insert(0, '../..')

# Problem to solve
from nonvarFEM.problems.testsNVP import Discontinuous_A

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
    opt["meshRefinement"] = 2
    opt["refinementThreshold"] = 0.95
    df_adaptive = solveProblem(P, opt)
    if WRITE_CSV:
        fname = '{}_{}_adaptive'.format(opt['id'], expname)
        hlp.writeOutputToCsv(df_adaptive, opt, fname)


if __name__ == "__main__":

    """
    Global settings
    """
    global_opt = hlp.standardOptions()
    P = Discontinuous_A()

    for p in range(1, 5):
        global_opt["id"] = "Discontinuous_A_deg_{}".format(p)

        # Threshold for dofs
        global_opt["NdofsThreshold"] = 50000

        # Fix polynomial degree
        global_opt["p"] = p
        global_opt["q"] = p

        # Determine method to estimate error norms
        global_opt["errorEstimationMethod"] = 1
        global_opt["writeToCsv"] = 0

        # Specify experiments method

        """
        Designing the different experiments.
        """
        #
        # experiment(P,
        #         hlp.opt_Neilan(global_opt),
        #         'Neilan')
        #
        # experiment(P,
        #         hlp.opt_NeilanSalgadoZhang(global_opt),
        #         'NeilanSalgadoZhang')
        #
        experiment(P,
                   hlp.opt_Own_CG_0_stab(global_opt),
                   'CG_0_stab')

        experiment(P,
                   hlp.opt_Own_CG_1_stab(global_opt),
                   'CG_1_stab')

        # experiment(P,
        #         hlp.opt_Own_CG_2_stab(global_opt),
        #         'CG_2_stab')
        #
        # experiment(P,
        #         hlp.opt_Own_DG_0_stab(global_opt),
        #         'DG_0_stab')
        #
        # experiment(P,
        #         hlp.opt_Own_DG_1_stab(global_opt),
        #         'DG_1_stab')
        #
