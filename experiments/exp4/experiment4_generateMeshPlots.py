import sys
sys.path.insert(0, '../..')

# Problem to solve
from nonvarFEM.problems.testsNVP import Discontinuous_A

# Solve routine
from nonvarFEM import solveProblem

# Auxiliary stuff
import nonvarFEM.helpers as hlp

if __name__ == "__main__":

    # Load problem
    P = Discontinuous_A()

    # Load standard options
    global_opt = hlp.standardOptions()

    global_opt["id"] = 'Discontinuous_A'

    global_opt["NdofsThreshold"] = 30000
    # Fix polynomial degree
    global_opt["p"] = 2
    global_opt["q"] = 2
    global_opt["plotMesh"] = 1
    global_opt["saveMesh"] = 1
    global_opt["plotSolution"] = 1
    global_opt["saveSolution"] = 1
    global_opt["meshRefinement"] = 2
    global_opt["refinementThreshold"] = .95
    global_opt["writeToCsv"] = 0

    # Determine method to estimate error norms
    global_opt["errorEstimationMethod"] = 1
    df = solveProblem(P, hlp.opt_Own_CG_1_stab(global_opt))
