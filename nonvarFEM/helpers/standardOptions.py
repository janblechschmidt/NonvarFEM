def standardOptions():
    opt = dict()

    opt["id"] = 'standard'  # Prefix for filenames
    opt["holdOn"] = 0      # Flag to hold on between different mesh levels

    opt["printCordesInfo"] = 0

    # Choose dimension of function spaces
    opt["p"] = 2  # Dimension of space of trial functions v_h
    opt["q"] = 2  # Dimension of space of Hessians
    opt["r"] = 2  # Dimension of space of gradients

    opt["normalizeA"] = 1  # Normalize A
    # Normalization parameter (should be optimized, is problem specific)
    opt["lambda"] = 1
    opt["HessianSpace"] = 'CG'
    opt["GradientSpace"] = 'CG'
    # opt["solutionMethod"] = 'BHWcomplete'
    # opt["solutionMethod"] = 'BHWreduced'
    opt["solutionMethod"] = 'NeilanSalgadoZhang'
    # opt["stabilizationFlag"] = 0 # no stabilization
    opt["stabilizationFlag"] = 1  # stabilization
    opt["stabilityConstant1"] = 1.  # Stability constant for first-order term
    opt["stabilityConstant2"] = 1.  # Stability constant for second-order term
    opt["gmresTolRes"] = 1e-10  # GMRES tolerance (abs and rel)

    opt["errorEstimationMethod"] = 1
    opt["dolfinLogLevel"] = 21

    # Choose options for output during computation
    opt["plotMesh"] = 0   # Plot mesh flag
    opt["plotSolution"] = 0   # Plot solution flag
    opt["plotErrorEstimates"] = 0   # Plot error estimates flag
    opt["plotConvergenceRates"] = 0   # Plot of error rates at the end
    opt["zAxisMin"] = 0.0   # Option to fix range of z axis
    opt["zAxisMax"] = 0.0   # Option to fix range of z axis
    opt["time_check"] = 1  # Print computation times

    # Select parts that have to be stored
    opt["writeToCsv"] = 1
    opt["saveSolution"] = 0
    opt["saveMesh"] = 0

    # Select mesh mode
    # meshRefinement = 0 # No refinement - use only initial mesh resolution
    # meshRefinement = 1 # Use regular refinement
    # meshRefinement = 2 # Use adaptive refinement
    opt["meshRefinement"] = 1
    opt["initialMeshResolution"] = 6
    opt["numberMeshLevels"] = 100
    opt["NdofsThreshold"] = 10000

    # Number of equidistant time steps in parabolic problems
    opt["timeSteps"] = 20
    # Number of equidistant time steps in parabolic problems
    opt["timeStepFactor"] = 2

    opt["refinementThreshold"] = 0.9  # only relevant for mesh mode 2

    # Options for error estimation
    opt["cell_residual_with_FEGradient"] = 0

    opt["methodName"] = 'standard'
    opt["outputDir"] = 'results'

    return opt


def opt_NeilanSalgadoZhang(opt=standardOptions()):
    opt["solutionMethod"] = 'NeilanSalgadoZhang'
    opt["HessianSpace"] = "CG"
    opt["stabilizationFlag"] = 1  # first-order stabilization
    opt["stabilityConstant1"] = 1.0  # Stability constant for first-order term
    opt["stabilityConstant2"] = 0  # Stability constant for second-order term
    return opt


def opt_Neilan(opt=standardOptions()):
    opt["solutionMethod"] = 'Neilan'
    opt["HessianSpace"] = "DG"
    opt["normalizeA"] = 0  # No normalization
    return opt


def opt_Own_CG_0_stab(opt=standardOptions()):
    opt["solutionMethod"] = 'BHWreduced'
    opt["HessianSpace"] = 'CG'
    opt["stabilizationFlag"] = 0  # no stabilization
    return opt


def opt_Own_CG_1_stab(opt=standardOptions()):
    opt["solutionMethod"] = 'BHWreduced'
    opt["HessianSpace"] = 'CG'
    opt["stabilizationFlag"] = 1  # first-order stabilization
    opt["stabilityConstant1"] = 1.0  # Stability constant for first-order term
    opt["stabilityConstant2"] = 0  # Stability constant for first-order term
    return opt


def opt_Own_CG_2_stab(opt=standardOptions()):
    opt["solutionMethod"] = 'BHWreduced'
    opt["HessianSpace"] = 'CG'
    opt["stabilizationFlag"] = 1  # first-order stabilization
    opt["stabilityConstant1"] = 1.0  # Stability constant for first-order term
    opt["stabilityConstant2"] = 1.0  # Stability constant for second-order term
    return opt


def opt_Own_DG_0_stab(opt=standardOptions()):
    opt["solutionMethod"] = 'BHWreduced'
    opt["HessianSpace"] = 'DG'
    opt["stabilizationFlag"] = 0  # first-order stabilization
    return opt


def opt_Own_DG_1_stab(opt=standardOptions()):
    opt["solutionMethod"] = 'BHWreduced'
    opt["HessianSpace"] = 'DG'
    opt["stabilizationFlag"] = 1  # first-order stabilization
    opt["stabilityConstant1"] = 1.0  # Stability constant for first-order term
    opt["stabilityConstant2"] = 0  # Stability constant for second-order term
    return opt
