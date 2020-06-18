import numpy as np

def determineEOCforDataframe(dim, df, verbose=True):
    """
    Function to determine the estimated order of convergence for
    various error (semi-)norms.
    """

    modes = ['L2', 'H1', 'H2', 'H2h', 'EdgeJump']
    n1 = df['Ndofs'].values[1:]
    n0 = df['Ndofs'].values[0:-1]
    scaling = -dim / \
        (np.log(n1 / n0))
    nvals = df.shape[0]

    for mode in modes:
        x1 = df[mode + '_error'].values[1:]
        x0 = df[mode + '_error'].values[0:-1]
        if nvals > 1:
            if x1[-1] < 1e-15:
                eoc = scaling[:-1] * np.log(x1[:-1]/x0[:-1])
                df.loc[:, mode+'_eoc'] = np.r_[0, eoc, np.inf]
            else:
                eoc = scaling * np.log(x1 / x0)
                df.loc[:, mode+'_eoc'] = np.r_[0, eoc]
        else:
            df.loc[0, mode+'_eoc'] = 0

    if verbose:
        print('\nError norms\n')
        print(df[['Ndofs', 'L2_error', 'H1_error',
                  'H2_error', 'H2h_error', 'EdgeJump_error']])
        print('\nConvergence rates\n')
        print(df[['Ndofs', 'L2_eoc', 'H1_eoc', 'H2_eoc', 'H2h_eoc', 'EdgeJump_eoc']])

    return df
