import numpy as np
def estimateOrderOfConvergence(df, dim, mode):
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
    x = - (1.0/dim) * np.log(xdata)
    n = min(len(x),max_elem)
    x = x[-n:]
    y = y[-n:]
    
    A=np.vstack([np.ones(len(x)), x]).T
    con, rate = np.linalg.lstsq(A,y)[0]
    return rate
