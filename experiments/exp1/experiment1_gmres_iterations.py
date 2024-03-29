import sys
sys.path.insert(0, '../..')

# Problem to solve
from nonvarFEM.problems.testsNVP import Cinfty

# Solve routine
from nonvarFEM import solveProblem

# Auxiliary stuff
from nonvarFEM.helpers import standardOptions, writeOutputToCsv

# Nice organization of results
import pandas as pd

WRITE_CSV = True

if __name__ == "__main__":
    kappa_list = [0.9, 0.99, 0.999]
    eta_list = [5, 1, 0]
    opt = standardOptions()

    # Fix polynomial degree
    opt["HessianSpace"] = 'CG'
    opt["p"] = 2
    opt["q"] = 2

    # We want to compare gmres iteration numbers
    opt["solutionMethod"] = 'BHWreduced'
    # Regular refinement
    opt["meshRefinement"] = 1
    # Initial mesh resolution h = 2^-3
    opt["initialMeshResolution"] = 8
    # Maximum number of dofs
    opt["NdofsThreshold"] = 70000
    # Don't write all the information
    opt["writeToCsv"] = 0
    # Set tolerance to 10^-8
    opt["gmresTolRes"] = 1e-8
    opt["gmresWarmStart"] = False
    # Prepare dataframe for results
    out = pd.DataFrame()

    # First experiment: Without stabilization
    for i, kappa in enumerate(kappa_list):
        for j, eta in enumerate(eta_list):
            opt["stabilizationFlag"] = 0 if eta < 1e-15 else 1
            opt["stabilizationConstant1"] = eta
            opt["stabilizationConstant2"] = 0
            opt["id"] = "Cinfty_eta_{}_kappa_{}_".format(eta, kappa)
            P = Cinfty(kappa)
            df = solveProblem(P, opt)
            if i == 0 and j == 0:
                out['h'] = ['2^{{-{}}}'.format(j + 3)
                            for j in range(df.shape[0])]
            out['eta_{}_kappa_{}'.format(
                eta, kappa)] = df.loc[:, 'N_iter'].astype('int')

    # Write csv file
    if WRITE_CSV:
        opt['id'] = 'gmres_iter'
        writeOutputToCsv(out, opt)
    print(out)
