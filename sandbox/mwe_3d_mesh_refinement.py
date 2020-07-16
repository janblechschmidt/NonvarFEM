from dolfin import *
import mshr
# n = 2
# p = 3
# # dom = mshr.Box(Point(.0,.0,.25), Point(.75, .75, .75))
# # m = mshr.generate_mesh(dom, 5)
# m = BoxMesh(Point(0., 0., 0.), Point(1., 1., 1.), n, n, n)
# K = 4
# for k in range(K):
#     V = FunctionSpace(m, 'Lagrange', p)
#     u = TrialFunction(V)
#     v = TestFunction(V)

#     M = assemble(u * v * dx)
#     M_diag = as_backend_type(M).mat().getDiagonal().array
#     print('Min of diag of M: {}'.format(M_diag.min()))
#     print('Dim V: {}'.format(V.dim()))
#     if k < K:
#         m = refine(m)

dom = mshr.Box(Point(0., 0., 0.), Point(1., 1., 1.))
# m_orig = mshr.generate_mesh(dom,0.01)

gen = mshr.CSGCGALMeshGenerator3D()

#gen.parameters["facet_angle"] = 30.0
#gen.parameters["facet_size"] = 0.5
#gen.parameters["edge_size"] = 0.5
gen.parameters["mesh_resolution"] = 0.01
gen.parameters["exude_optimize"] = True

m_orig = gen.generate(mshr.CSGCGALDomain3D(dom))

xf = File('mesh_orig.pvd')
xf << m_orig

V = FunctionSpace(m_orig, 'DG', 0)
w = TestFunction(V)
s = assemble(w*dx)
print(s.get_local())

# m = refine(m_orig)
# 
# V = FunctionSpace(m, 'Lagrange', 2)
# u = TrialFunction(V)
# v = TestFunction(V)
# 
# M = assemble(u * v * dx)
# 
# M_diag = as_backend_type(M).mat().getDiagonal().array
# print('Min of diag of M: {}'.format(M_diag.min()))
# 
# xf = File('mesh.pvd')
# xf << m
# 
