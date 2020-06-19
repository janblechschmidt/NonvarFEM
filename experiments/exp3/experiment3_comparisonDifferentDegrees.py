import sys
sys.path.insert(0, '../..')

# Problem to solve
from nonvarFEM.problems.testsNVP import FNS_5_4_L_inf_Coeffs

# Solve routine
from nonvarFEM import solveProblem

# Auxiliary stuff
import nonvarFEM.helpers as hlp

# Nice organization of results
import pandas as pd

WRITE_CSV = True


def degreeStudy(P, opt, minDegree=1, maxDegree=4, filename=None):

    out = []

    for deg in range(minDegree, maxDegree+1):
        opt["p"] = deg
        opt["q"] = deg
        df = solveProblem(P, opt)
        out.append(df)
        if WRITE_CSV:
            fname = '{}_degree_study_{}_deg_{}'.format(opt['id'], filename, deg)
            hlp.writeOutputToCsv(df, opt, fname)

    return out


if __name__ == "__main__":

    global_opt = hlp.standardOptions()

    """
    Set up structure of experiment
    """

    P = FNS_5_4_L_inf_Coeffs()
    global_opt["id"] = "FNS_5_4"

    # Threshold for dofs
    global_opt["NdofsThreshold"] = 1000

    # Determine method to estimate error norms
    global_opt["errorEstimationMethod"] = 1
    # Determine initial mesh
    global_opt["initialMeshResolution"] = 6

    # Regular refinement
    global_opt["meshRefinement"] = 1
    global_opt["writeToCsv"] = 0

    minDegree = 1

    # Choose maximal degree of polynomials
    maxDegree = 4

    Neilan = degreeStudy(P,
                         hlp.opt_Neilan(global_opt),
                         minDegree,
                         maxDegree,
                         'Neilan')

    NSZ = degreeStudy(P,
                      hlp.opt_NeilanSalgadoZhang(global_opt),
                      minDegree,
                      maxDegree,
                      'NeilanSalgadoZhang')

    cg0stab = degreeStudy(P,
                          hlp.opt_Own_CG_0_stab(global_opt),
                          minDegree,
                          maxDegree,
                          'cg0stab')

    cg1stab = degreeStudy(P,
                          hlp.opt_Own_CG_1_stab(global_opt),
                          minDegree,
                          maxDegree,
                          'cg1stab')

    cg2stab = degreeStudy(P,
                          hlp.opt_Own_CG_2_stab(global_opt),
                          minDegree,
                          maxDegree,
                          'cg2stab')

    dg0stab = degreeStudy(P,
                          hlp.opt_Own_DG_0_stab(global_opt),
                          minDegree,
                          maxDegree,
                          'dg0stab')

    dg1stab = degreeStudy(P,
                          hlp.opt_Own_DG_1_stab(global_opt),
                          minDegree,
                          maxDegree,
                          'dg1stab')

    # Set up one large dataframe to store relevant output
    dfs = []
    keys = ['L2_error', 'H1_error', 'H2_error', 'H2h_error',
            'L2_eoc', 'H1_eoc', 'H2_eoc', 'H2h_eoc']

    for deg in range(maxDegree):
        df = pd.DataFrame(columns=['Ndofs',
                                   'hmax'], dtype=float)

        df['Ndofs'] = Neilan[deg]['Ndofs']
        df['hmax'] = Neilan[deg]['hmax']
        for key in keys:
            df['neilan_' + key] = Neilan[deg][key]
            df['nsz_' + key] = NSZ[deg][key]
            df['cg0stab_' + key] = cg0stab[deg][key]
            df['cg1stab_' + key] = cg1stab[deg][key]
            df['cg2stab_' + key] = cg2stab[deg][key]
            df['dg0stab_' + key] = dg0stab[deg][key]
            df['dg1stab_' + key] = dg1stab[deg][key]

        dfs.append(df)

        if WRITE_CSV:
            fname = global_opt['id'] + '_deg_{}'.format(deg + 1)
            hlp.writeOutputToCsv(df, global_opt, fname)
