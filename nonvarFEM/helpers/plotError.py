import matplotlib.pyplot as plt
from nonvarFEM.helpers import estimateOrderOfConvergence
#import estimateOrderOfConvergence

def plotError(df, fig=None, dim = 2, title = "Absolute errors", dofs = "Ndofs"):
    if not fig:
        fig = plt.figure('errorRates', clear=True)
    modes = ["L2", "H1", "H2h"]
    clrs = ['firebrick', 'darkblue',  'limegreen']

    for (i,mode) in enumerate(modes):
        eoc = estimateOrderOfConvergence(df, dim, mode)
        x = df[dofs]
        y = df[mode + "_error"]
        plt.loglog(x, y, label='%s (%4.2f)' % (mode, eoc), color = clrs[i])
        plt.title(title)
        plt.xlabel(dofs)
        plt.ylabel('Errors')
    ax = plt.gca()
    ax.legend(loc='lower left', shadow=True)
    plt.show()
