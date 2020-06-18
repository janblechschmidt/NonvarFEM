from nvfem import solveProblem
import problems.testsNVP as NVP
import dolfin

from stats.writeOutputToCsv import writeOutputToCsv
from standardOptions import *

if __name__ == "__main__":

    dolfin.set_log_level(21)
    
    # Load standard options
    global_opt = standardOptions()

    # Discontinuous A
    P = NVP.Discontinuous_A()
    global_opt["id"] = 'experiment4'

    global_opt["NdofsThreshold"]   = 30000
    # Fix polynomial degree
    global_opt["p"] = 2
    global_opt["q"] = 2
    global_opt["plotMesh"] = 1
    global_opt["saveMesh"] = 1
    global_opt["plotSolution"] = 1
    global_opt["saveSolution"] = 1
    global_opt["meshRefinement"] = 2
    global_opt["refinementThreshold"] = .95

    # Determine method to estimate error norms
    global_opt["errorEstimationMethod"] = 1
    df = solveProblem(P, opt_Own_CG_1_stab(global_opt))
