import sys
sys.path.insert(0, '../..')

# Problem to solve
from nonvarFEM.problems.testsNVP import Sol_in_H_alpha

# Solve routine
from nonvarFEM import solveProblem

# Auxiliary stuff
from nonvarFEM.helpers import standardOptions, opt_Own_CG_0_stab


if __name__ == "__main__":

    # Load standard options
    global_opt = standardOptions()

    # Discontinuous A
    P = Sol_in_H_alpha(1.5)
    global_opt["id"] = 'Sol_in_H_2.5'

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
    df = solveProblem(P, opt_Own_CG_0_stab(global_opt))
