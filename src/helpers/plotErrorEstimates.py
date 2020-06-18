from dolfin import *
import matplotlib.pyplot as plt

def plotErrorEstimates(eta_cell, eta_edge, eta, mesh, fig = None, dim = 2):

    if not fig:
        fig = plt.figure('errorEstimates', clear=True)

    DG0 = FunctionSpace(mesh, "DG", 0)

    if dim == 1:
        DG0_dofs = DG0.tabulate_dof_coordinates()
        I = np.argsort(DG0_dofs)
    
        fig.add_subplot(3,1,1)
        plt.step(DG0_dofs[I], eta_cell.get_local()[I], where='mid')
        plt.title('Cell error estimate')
        plt.axis('tight')
    
        fig.add_subplot(3,1,2)
        plt.title('Edge error estimate')
        plt.step(DG0_dofs[I], eta_edge.get_local()[I], where='mid')
        plt.axis('tight')
    
        fig.add_subplot(3,1,3)
        plt.title('Total error estimate')
        plt.step(DG0_dofs[I], eta.get_local()[I], where='mid')
        plt.axis('tight')
    
    elif dim == 2:
        r_c_func = Function(DG0)
        r_c_func.vector()[:] = eta_cell

        r_e_func = Function(DG0)
        r_e_func.vector()[:] = eta_edge

        eta_func = Function(DG0)
        eta_func.vector()[:] = eta
    
        fig.add_subplot(3,1,1)
        tp = plot(r_c_func)
        tp.set_cmap("viridis")
        plt.title('Cell error estimate')
        plt.colorbar(tp)
    
        fig.add_subplot(3,1,2)
        tp = plot(r_e_func)
        tp.set_cmap("viridis")
        plt.title('Edge error estimate')
        plt.colorbar(tp)
    
        fig.add_subplot(3,1,3)
        tp = plot(eta_func)
        tp.set_cmap("viridis")
        plt.title('Total error estimate')
        plt.colorbar(tp)

    elif dim == 3:
        print('TODO: Implement plot function for 3d')
    else:
        error('No plots in dimension %i possible' % dim)

    plt.draw()
    plt.pause(0.01)
