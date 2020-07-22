from scipy.io import savemat

from varname import nameof

def exportAsMat(M, s='M.mat'):
    """ Exports all elements in the list M into a Matlab mat-object.
    """

    for m in M:
        print(nameof(m))

        # if do_savemat:
        #     if meshtype == 'acute':
        #         s = './mat-files/acute.mat'
        #     else:
        #         s = './mat-files/mat_%s_%03d.mat' % (prob, n)
        #     savemat( s ,
        #         mdict={'S': S,
        #             'F': F,
        #             'vf': vf,
        #             'vz': vz,
        #             'N': N,
        #             'N_inner': N_inner,
        #             'A': mA[idx_inner, :],
        #             'B': mB,
        #             'C': mC[:,idx_inner],
        #             'mA': [x[idx_inner,:] for x in tmpA],
        #             'mAfull': tmpA,
        #             'mB': tmpB,
        #             'mC': [x[:,idx_inner] for x in tmpC],
        #             'mK': mK,
        #             'mM': mM,
        #             'eps': eps,
        #             'xy': mesh.coordinates(),
        #             'tri': mesh.cells() + 1,
        #             'idx_inner': idx_inner + 1,
        #             'vert2dof': vertex_to_dof_map(V)+1})
