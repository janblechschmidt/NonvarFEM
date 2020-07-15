from dolfin import *
import mshr

dom = mshr.Box(Point(0., 0., 0.), Point(1., 1., 1.))
m_orig = mshr.generate_mesh(dom,0.01)

xf = File('mesh_orig.pvd')
xf << m_orig

m = refine(m_orig)

V = FunctionSpace(m, 'Lagrange', 2)
u = TrialFunction(V)
v = TestFunction(V)

M = assemble(u * v * dx)

M_diag = as_backend_type(M).mat().getDiagonal().array
print('Min of diag of M: {}'.format(M_diag.min()))

xf = File('mesh.pvd')
xf << m
