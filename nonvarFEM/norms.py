from dolfin import *

"""
This file collects different norms and scalar products used throughout the
numerical study of finite element methods for non-variational 2nd-order pdes
"""

def L2_scpr(u, v):
    """ L^2 scalar product """
    return assemble( inner(u, v) * dx)

def L2_norm(u):
    """ L^2 norm"""
    return sqrt(L2_scpr(u, u))

def H10_scpr(u, v):
    """ H^1_0 inner product """
    return assemble( inner(grad(u), grad(v)) * dx)

def H10_norm(u):
    """ H^1_0 semi norm """
    return sqrt(H10_scpr(u, u))

def H20_scpr(u, v):
    """ H^2_0 inner product """
    return assemble( inner(grad(grad(u)), grad(grad(v))) * dx)

def H20_norm(u):
    """ H^2_0 semi norm """
    return sqrt(H20_scpr(u, u))

def vj(u, n):
    """ Jump of a vector. """
    return outer(u('+'),n('+'))+outer(u('-'),n('-'))

def mj(u, n):
    """ Jump of a matrix. """
    return u('+')*n('+')+u('-')*n('-')

def EdgeJump_scpr(u, v, hE, nE):
    """ Edge jump inner product in H^2_h norm """
    return assemble( avg(hE)**(-1) * inner( vj(grad(u),nE), vj(grad(u),nE)) * dS )

def EdgeJump_norm(u, hE, nE):
    """ Edge jump part in H^2_h norm """
    return sqrt(EdgeJump_scpr(u, u, hE, nE))

def H2h_norm(u, hE, nE):
    """ H^2_h norm """
    return sqrt( pow(H20_norm(u), 2) + pow(EdgeJump_norm(u, hE, nE), 2) )
