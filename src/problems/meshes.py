import mshr
from dolfin import Point, Mesh, RectangleMesh, sqrt


# Define various meshes
def mesh_AcuteSquare(n=0):
    mbasename = 'AcuteSquareMesh2'
    mesh = Mesh(mbasename + '_%d.xml' % n)
    return mesh


def mesh_Lshape(n=10):
    dom = mshr.Rectangle(Point(0., 0.), Point(1., .5)) + \
        mshr.Rectangle(Point(.5, .5), Point(1., 1.))
    mesh = mshr.generate_mesh(dom, n, "cgal")

    return mesh


def mesh_UnitSquare(n=10, method='regular'):
    if method == 'mshr':
        dom = mshr.Rectangle(Point(0., 0.), Point(1., 1.))
        mesh = mshr.generate_mesh(dom, n, "cgal")
    elif method == 'regular':
        mesh = RectangleMesh(Point(0., 0.), Point(1., 1.), n, n)
    return mesh


def mesh_Square(n=10, method='regular'):
    if method == 'mshr':
        dom = mshr.Rectangle(Point(-1., -1.), Point(1., 1.))
        mesh = mshr.generate_mesh(dom, n, "cgal")
    elif method == 'regular':
        mesh = RectangleMesh(Point(-1., -1.), Point(1., 1.), n, n)
    return mesh


def mesh_Triangle(n=10):
    dom = mshr.Polygon(
        [Point(0., -1.), Point(sqrt(3)/2, 1./2), Point(-sqrt(3)/2, 1./2)])
    mesh = mshr.generate_mesh(dom, n, "cgal")
    return mesh
