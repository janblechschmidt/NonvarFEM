import pandas as pd


def initDataframe():
    return pd.DataFrame(
        columns=['Ndofs', 'NdofsMixed', 'hmax',
                 'L2_error', 'H1_error', 'H2_error', 'H2h_error', 'EdgeJump_error',
                 'N_iter', 'L2_eoc', 'H1_eoc', 'H2_eoc', 'H2h_eoc',
                 'EdgeJump_eoc', 'Eta_global'],
        dtype=float)
