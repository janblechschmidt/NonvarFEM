import sys
sys.path.insert(0, '../..')

import pandas as pd

from nonvarFEM.helpers import determineEOCforDataframe, estimateOrderOfConvergence

if __name__ == '__main__':
    lu = pd.read_csv("./results/Sol_in_H_2.25_3d_uniform.csv")
    la = pd.read_csv("./results/Sol_in_H_2.25_3d_adaptive.csv")
    ru = pd.read_csv("./results/Sol_in_H_2.50_3d_uniform.csv")
    ra = pd.read_csv("./results/Sol_in_H_2.50_3d_adaptive.csv")
    dim = 3
    #determineEOCforDataframe(3, lu)
    N = 3
    lul = []
    lul.append(estimateOrderOfConvergence(lu, 3, 'L2', n=N))
    lul.append(estimateOrderOfConvergence(lu, 3, 'H1', n=N))
    lul.append(estimateOrderOfConvergence(lu, 3, 'H2h', n=N))
    lal = []
    lal.append(estimateOrderOfConvergence(la, 3, 'L2', n=N))
    lal.append(estimateOrderOfConvergence(la, 3, 'H1', n=N))
    lal.append(estimateOrderOfConvergence(la, 3, 'H2h', n=N))
    rul = []
    rul.append(estimateOrderOfConvergence(ru, 3, 'L2', n=N))
    rul.append(estimateOrderOfConvergence(ru, 3, 'H1', n=N))
    rul.append(estimateOrderOfConvergence(ru, 3, 'H2h', n=N))
    ral = []
    ral.append(estimateOrderOfConvergence(ra, 3, 'L2', n=N))
    ral.append(estimateOrderOfConvergence(ra, 3, 'H1', n=N))
    ral.append(estimateOrderOfConvergence(ra, 3, 'H2h', n=N))
    print('Left uniform: {}'.format(lul))
    print('Left adaptive: {}'.format(lal))
    print('Right uniform: {}'.format(rul))
    print('Right adaptive: {}'.format(ral))
