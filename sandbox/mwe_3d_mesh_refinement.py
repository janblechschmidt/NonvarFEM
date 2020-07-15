from dolfin import *
import mshr
n = 2
p = 3
dom = mshr.Box(Point(0., 0., 0.), Point(1., 1., 1.))
m = mshr.generate_mesh(dom, n)
# , "cgal")

# m = BoxMesh(Point(0., 0., 0.), Point(1., 1., 1.), n, n, n)

V = FunctionSpace(m, 'Lagrange', p)
u = TrialFunction(V)
v = TestFunction(V)

M = assemble(u * v * dx)
M_diag = as_backend_type(M).mat().getDiagonal().array
print('Min of diag of M: {}'.format(M_diag.min()))

m = refine(m)
V = FunctionSpace(m, 'Lagrange', p)
u = TrialFunction(V)
v = TestFunction(V)
M = assemble(u * v * dx)
M_diag = as_backend_type(M).mat().getDiagonal().array
print('Min of diag of M: {}'.format(M_diag.min()))
