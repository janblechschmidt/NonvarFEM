from nvfem import solveProblem
import problems.testsNVP as NVP
import dolfin

from writeOutputToCsv import writeOutputToCsv
from standardOptions import *

import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = './results/Degree_Study/'
PREFIX = 'degree_study_qd100_'
WRITE_CSV = True

def degreeStudy(P, opt, minDegree=1, maxDegree=4, filename = None):

    out = []

    for deg in range(minDegree, maxDegree+1):
        opt["p"]    = deg
        opt["q"]    = deg
        df = solveProblem(P, opt)
        out.append(df)
        if WRITE_CSV:
            writeOutputToCsv(CSV_DIR + PREFIX + '_' + opt["id"] + '_{}'.format(filename) + '_deg_{}.csv'.format(deg), df)

    return out

if __name__ == "__main__":


    dolfin.set_log_level(21)

    global_opt = standardOptions()

    """
    Set up structure of experiment
    """

    P = NVP.FNS_5_4_L_inf_Coeffs()
    global_opt["id"] = "FNS_5_4"
    
    # Threshold for dofs
    global_opt["NdofsThreshold"]   = 50000

    # Determine method to estimate error norms
    global_opt["errorEstimationMethod"] = 1
    # Determine initial mesh
    global_opt["initialMeshResolution"]  = 6

    # Regular refinement
    global_opt["meshRefinement"]    = 1

    minDegree = 1
    
    # Choose maximal degree of polynomials
    maxDegree = 4

    Neilan = degreeStudy(P,
            opt_Neilan(global_opt),
            minDegree,
            maxDegree,
            'Neilan')

    NSZ = degreeStudy(P,
            opt_NeilanSalgadoZhang(global_opt),
            minDegree,
            maxDegree,
            'NeilanSalgadoZhang')

    cg0stab = degreeStudy(P,
            opt_Own_CG_0_stab(global_opt),
            minDegree,
            maxDegree,
            'cg0stab')

    cg1stab = degreeStudy(P,
            opt_Own_CG_1_stab(global_opt),
            minDegree,
            maxDegree,
            'cg1stab')

    cg2stab = degreeStudy(P,
            opt_Own_CG_2_stab(global_opt),
            minDegree,
            maxDegree,
            'cg2stab')

    dg0stab = degreeStudy(P,
            opt_Own_DG_0_stab(global_opt),
            minDegree,
            maxDegree,
            'dg0stab')

    dg1stab = degreeStudy(P,
            opt_Own_DG_1_stab(global_opt),
            minDegree,
            maxDegree,
            'dg1stab')

    # Set up one large dataframe to store relevant output
    dfs = []
    keys = ['L2_error', 'H1_error', 'H2_error', 'H2h_error', 'L2_eoc', 'H1_eoc', 'H2_eoc', 'H2h_eoc']
    
    for deg in range(maxDegree):
        df = pd.DataFrame(columns = ['Ndofs',
        'hmax'], dtype=float)

        df['Ndofs'] = Neilan[deg]['Ndofs']
        df['hmax'] = Neilan[deg]['hmax']
        for key in keys:
            df['neilan_'+key] = Neilan[deg][key]
            df['nsz_'+key] = NSZ[deg][key]
            df['cg0stab_'+key] = cg0stab[deg][key]
            df['cg1stab_'+key] = cg1stab[deg][key]
            df['cg2stab_'+key] = cg2stab[deg][key]
            df['dg0stab_'+key] = dg0stab[deg][key]
            df['dg1stab_'+key] = dg1stab[deg][key]

        dfs.append(df)
        
        if WRITE_CSV:
            df.to_csv(CSV_DIR + PREFIX + global_opt['id'] + '_deg_{}.csv'.format(deg+1))
