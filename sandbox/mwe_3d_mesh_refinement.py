from dolfin import *
import mshr
n = 2
p = 3
# dom = mshr.Box(Point(.0,.0,.25), Point(.75, .75, .75))
# m = mshr.generate_mesh(dom, 5)
m = BoxMesh(Point(0., 0., 0.), Point(1., 1., 1.), n, n, n)
K = 4
for k in range(K):
    V = FunctionSpace(m, 'Lagrange', p)
    u = TrialFunction(V)
    v = TestFunction(V)

    M = assemble(u * v * dx)
    M_diag = as_backend_type(M).mat().getDiagonal().array
    print('Min of diag of M: {}'.format(M_diag.min()))
    print('Dim V: {}'.format(V.dim()))
    if k < K:
        m = refine(m)
