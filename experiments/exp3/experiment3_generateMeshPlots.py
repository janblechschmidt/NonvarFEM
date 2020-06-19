import sys
sys.path.insert(0, '../..')

# Problem to solve
from nonvarFEM.problems.testsNVP import FNS_5_4_L_inf_Coeffs

# Solve routine
from nonvarFEM import solveProblem

# Auxiliary stuff
import nonvarFEM.helpers as hlp


if __name__ == "__main__":

    # Load standard options
    global_opt = hlp.standardOptions()

    P = FNS_5_4_L_inf_Coeffs()
    global_opt["id"] = "FNS_5_4"

    global_opt["NdofsThreshold"] = 30000
    # Fix polynomial degree
    global_opt["p"] = 2
    global_opt["q"] = 2
    global_opt["plotMesh"] = 1
    global_opt["saveMesh"] = 1
    global_opt["plotSolution"] = 1
    global_opt["saveSolution"] = 1
    global_opt["meshRefinement"] = 2
    global_opt["refinementThreshold"] = .90
    global_opt["writeToCsv"] = 0

    # Determine method to estimate error norms
    global_opt["errorEstimationMethod"] = 1
    df = solveProblem(P, hlp.opt_Own_CG_0_stab(global_opt))
