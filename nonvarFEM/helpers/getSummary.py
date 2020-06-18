def getSummary(P, opt):

    s = "-"*50 + '\n'
    s += 'SUMMARY:\n'
    s += "-"*50 + '\n'
    s += '\tProblem: {}\n'.format(P.__class__.__name__)
    s += '\tSolution method: {}\n'.format(opt['solutionMethod'])
    s += '\tSpace of u: {}{}\n'.format('CG', opt['p'])
    if opt['solutionMethod'] == 'NeilanSalgadoZhang':
        s += '\tSpace of H: {}\n'.format('No discrete Hessian')
    else:
        s += '\tSpace of H: {}{}\n'.format(opt['HessianSpace'], opt['p'])
    s += '\tNormalization: {}\n'.format(opt['normalizeA'])
    s += '\tInitial mesh size: {}\n'.format(opt['initialMeshResolution'])
    if opt['meshRefinement'] == 1:
        s += '\tMesh refinement: Quasi-uniform\n'
    elif opt['meshRefinement'] == 2:
        s += '\tMesh refinement: Adaptive\n'
    s += '\tError estimation method: {}\n'.format(opt['errorEstimationMethod'])
    # Now comes the stuff that is available after execution of P.checkVariables()
    s += '\tAnalytical solution available: {}\n'.format(P.hasSolution)
    s += "-"*50 + '\n'
    return s
