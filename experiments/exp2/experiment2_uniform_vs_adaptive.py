import sys
sys.path.insert(0, '../..')

# Problem to solve
from nonvarFEM.problems.testsNVP import Sol_in_H_alpha

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
    opt["refinementThreshold"] = 0.9
    df_adaptive = solveProblem(P, opt)
    if WRITE_CSV:
        fname = '{}_{}_adaptive'.format(opt['id'], expname)
        hlp.writeOutputToCsv(df_adaptive, opt, fname)


if __name__ == "__main__":

    """
    Global settings
    """
    global_opt = hlp.standardOptions()

    alpha = 1.5
    P = Sol_in_H_alpha(alpha)
    global_opt["id"] = 'Sol_in_H_2.5'

    # Threshold for dofs
    global_opt["NdofsThreshold"] = 100000

    # Fix polynomial degree
    global_opt["p"] = 2
    global_opt["q"] = 2

    # Deactivate write in solveProblem
    global_opt["writeToCsv"] = 0

    # Determine method to estimate error norms
    global_opt["errorEstimationMethod"] = 1

    # Specify experiments method

    """
    Designing the different experiments.
    """

    experiment(P,
               hlp.opt_Neilan(global_opt),
               'Neilan')

    experiment(P,
               hlp.opt_NeilanSalgadoZhang(global_opt),
               'NeilanSalgadoZhang')

    experiment(P,
               hlp.opt_Own_CG_0_stab(global_opt),
               'CG_0_stab')

    experiment(P,
               hlp.opt_Own_CG_1_stab(global_opt),
               'CG_1_stab')

    experiment(P,
               hlp.opt_Own_CG_2_stab(global_opt),
               'CG_2_stab')

    experiment(P,
               hlp.opt_Own_DG_0_stab(global_opt),
               'DG_0_stab')

    experiment(P,
               hlp.opt_Own_DG_1_stab(global_opt),
               'DG_1_stab')
