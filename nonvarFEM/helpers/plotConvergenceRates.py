import numpy as np
import matplotlib.pyplot as plt


def estimateOrderOfConvergence(df, dim, mode, n = None):
    """
    Function to determine the experimental order
    of convergence through a (linear) regression line.
    """

    max_elem = 4

    ydata = df[mode + "_error"].values
    xdata = df["Ndofs"].values
    if ydata[-1] < 1e-15:
        ydata = ydata[:-1]
        xdata = xdata[:-1]

    y = np.log(ydata)
    x = - (1.0 / dim) * np.log(xdata)
    if not n:
        n = min(len(x), max_elem)
    x = x[-n:]
    y = y[-n:]

    A = np.vstack([np.ones(len(x)), x]).T
    con, rate = np.linalg.lstsq(A, y)[0]
    return rate


def plotConvergenceRates(df, fig=None, dim=2, title="Absolute errors", dofs="Ndofs"):
    if not fig:
        fig = plt.figure('errorRates', clear=True)
    modes = ["L2", "H1", "H2h"]
    clrs = ['firebrick', 'darkblue', 'limegreen']

    for (i, mode) in enumerate(modes):
        eoc = estimateOrderOfConvergence(df, dim, mode)
        x = df[dofs]
        y = df[mode + "_error"]
        plt.loglog(x, y, label='%s (%4.2f)' % (mode, eoc), color=clrs[i])
        plt.title(title)
        plt.xlabel(dofs)
        plt.ylabel('Errors')
    ax = plt.gca()
    ax.legend(loc='lower left', shadow=True)
    plt.show()
