# FEniCS stuff
from dolfin import parameters, set_log_level
from dolfin import Function, assign, interpolate

# Basic linear algebra
import numpy as np

# Auxiliary stuff
import nonvarFEM.helpers as hlp

# Norms
from nonvarFEM.norms import EdgeJump_norm, H20_norm, H10_norm, L2_norm

import nonvarFEM.problems.testsParabolicNVP as PNVP

from dolfin import XDMFFile

import dolfin

def solveProblem(P, opt):

    set_log_level(opt['dolfinLogLevel'])

    # parameters["reorder_dofs_serial"] = False
    # parameters['form_compiler']['quadrature_rule'] = 'vertex'
    parameters["form_compiler"]["quadrature_degree"] = 15
    # parameters['ghost_mode'] = 'shared_facet'
    parameters['krylov_solver']['absolute_tolerance'] = 1e-10
    parameters['krylov_solver']['relative_tolerance'] = 1e-10
    parameters['krylov_solver']['maximum_iterations'] = 1000
    parameters['krylov_solver']['monitor_convergence'] = True

    df = hlp.initDataframe()

    if opt["meshRefinement"]:
        n_range = range(opt["numberMeshLevels"])
    else:
        n_range = [opt["initialMeshResolution"]]

    print('-' * 50 + '\n')

    n = opt["initialMeshResolution"]

    # Initialize mesh
    P.initMesh(n)
    P.meshLevel = 1

    for (k, n) in enumerate(n_range):

        # updateFunctionSpaces also, since it depends on the mesh
        P.updateFunctionSpaces(opt)

        if P.getType().find('HJB') > -1:
            # initControl has to be run on each mesh once
            P.initControl()

        # update Coefficients needs control, therefore after initControl
        P.updateCoefficients()
        # checkVariables sets things like hasSolution, hasDrift etc.
        P.checkVariables()

        h = P.mesh.hmax()
        ndofs = P.solDofs(opt)
        ndofsMixed = P.totalDofs(opt)

        if k == 0:
            print(hlp.getSummary(P, opt))

        print("Solve problem with %d (%d) dofs." % (ndofs, ndofsMixed))

        if opt["printCordesInfo"]:
            P.printCordesInfo()

        N_iter = P.solve(opt)

        # Save solution as uold
        P.uold = P.u.copy()

        P.doPlots(opt)

        df.loc[k, ['hmax', 'Ndofs', 'NdofsMixed', 'N_iter']] = (
            h, ndofs, ndofsMixed, N_iter)

        # Check error estimation option during the first iteration
        if k == 0 and not P.hasSolution and opt["errorEstimationMethod"] == 1:
            print("""WARNING: Switch to errorEstimationMethod 2;
            no explicit solution is available.""")

            opt["errorEstimationMethod"] = 2

        if k == 0 and opt["errorEstimationMethod"] == 2:
            # Initialize list to store all solutions
            U = []

        if isinstance(P, PNVP.PNVP_WorstOfTwoAssetsPut):
            if k == 0:
                qoi_val_list = []
            qoi_val = P.u([40,40])
            qoi_val_list.append(qoi_val)

            print('--------------------------------------------------\n')
            print('Value at qoi: %10.8f\n' % qoi_val)
            print('--------------------------------------------------\n')

        # Method 1: Determine norms of residual on current mesh
        if opt["errorEstimationMethod"] == 1:

            # Expression for difference between discrete
            # and analytical solution
            if hasattr(P, 'loc'):
                u_diff = (P.u - P.u_) * P.loc
            else:
                u_diff = P.u - P.u_
            # Determine error norms
            l2err = L2_norm(u_diff)
            h1semierr = H10_norm(u_diff)
            h2semierr = H20_norm(u_diff)
            jumperr = EdgeJump_norm(u_diff, P.hE, P.nE)

            # Write errors to dataframe
            df.loc[k, 'L2_error'] = l2err
            df.loc[k, 'H1_error'] = h1semierr
            df.loc[k, 'H2_error'] = h2semierr
            df.loc[k, 'EdgeJump_error'] = jumperr
            df.loc[k, 'H2h_error'] = np.sqrt(
                pow(h2semierr, 2) + pow(jumperr, 2))

            # Method 2: Collect the current solution,
            # compute the norms of residual on finest mesh
        if opt["errorEstimationMethod"] == 2:
            uerr = Function(P.V)
            assign(uerr, P.u)
            U.append(uerr)

        if opt["plotSolution"]:
            P.plotSolution()

        # Determine error estimates
        print('WARNING: Introduce flag to estimate errors')
        print('Estimate errors')
        eta = P.determineErrorEstimates(opt)
        eta_est = np.sum(np.power(eta, 2))
        df.loc[k, 'Eta_global'] = eta_est

        # Mesh refinement
        if opt["meshRefinement"] > 0:
            if ndofs < opt["NdofsThreshold"]:
                P.meshLevel += 1
                # Refine mesh depending on the strategy
                if opt["meshRefinement"] == 1:
                    print('Refine mesh uniformly')
                    P.refineMesh()

                elif opt["meshRefinement"] == 2:
                    print("Refine mesh adaptively")
                    cell_markers = P.markCells(opt, eta)
                    P.refineMesh(cell_markers)

            else:
                print("Reached maximum number of dofs")
                break

        if opt["holdOn"]:
            inp = input('Press enter to continue, "x" to abort: ')

            if inp == 'x':
                break

    if opt["errorEstimationMethod"] == 2:
        """ This method computes the error norms all on the finest mesh.
        Since the errornorm command is not available for the H^2_h norm,
        we compute the respective norms directly.
        """

        u_exact = P.u_ if P.hasSolution else interpolate(P.u, P.V)

        for (k, u) in enumerate(U):

            print("Compute errors on mesh level %i / %i" % (k + 1, len(U)))

            # Interpolate u_k on the finest mesh
            u = interpolate(u, P.V)
            if hasattr(P, 'loc'):
                u_diff = (u - u_exact) * P.loc
            else:
                u_diff = u - u_exact

            # Determine norms of residual
            EdgeJumperr = EdgeJump_norm(u_diff, P.hE, P.nE)
            L2err = L2_norm(u_diff)
            H1err = H10_norm(u_diff)
            H2err = H20_norm(u_diff)
            H2herr = np.sqrt(pow(H2err, 2) + pow(EdgeJumperr, 2))

            # Write errors to dataframe
            df.loc[k, 'L2_error'] = L2err
            df.loc[k, 'H1_error'] = H1err
            df.loc[k, 'H2_error'] = H2err
            df.loc[k, 'H2h_error'] = H2herr
            df.loc[k, 'EdgeJump_error'] = EdgeJumperr

    if len(n_range) > 1:
        df = hlp.determineEOCforDataframe(P.dim(), df)

        if opt["plotConvergenceRates"]:
            hlp.plotConvergenceRates(df)

    if isinstance(P, PNVP.PNVP_WorstOfTwoAssetsPut):
        df['qoi_val'] = qoi_val_list
    
    if opt["writeToCsv"]:
        hlp.writeOutputToCsv(df, opt)

    # xf = XDMFFile("mesh.xdmf")
    # xf.write(P.u)
    return df
