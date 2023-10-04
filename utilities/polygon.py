import numpy as np


# https://stackoverflow.com/questions/39660851/deciding-if-a-point-is-inside-a-polygon
def point_inside_polygon(point, poly, include_edges=True):
    '''
    Test if point (x, y) is inside polygon poly.

    poly is N-vertices polygon defined as
    [(x1, y1),...,(xN, yN)] or [(x1, y1),...,(xN, yN),(x1, y1)]
    (function works fine in both cases)

    Geometrical idea: point is inside polygon if horisontal beam
    to the right from point crosses polygon even number of times.
    Works fine for non-convex polygons.
    '''
    x, y = point
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if p1y == p2y:
            if y == p1y:
                if min(p1x, p2x) <= x <= max(p1x, p2x):
                    # point is on horisontal edge
                    inside = include_edges
                    break
                elif x < min(p1x, p2x):  # point is to the left from current edge
                    inside = not inside
        else:  # p1y!= p2y
            if min(p1y, p2y) <= y <= max(p1y, p2y):
                xinters = (y - p1y) * (p2x - p1x) / float(p2y - p1y) + p1x

                if x == xinters:  # point is right on the edge
                    inside = include_edges
                    break

                if x < xinters:  # point is to the left from current edge
                    inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def polygonRes(x, poly, TH=1.):
    inpoly = point_inside_polygon(x, poly)
    min_dist = 10000.
    min_res = None
    for i in range(poly.shape[0]):
        j = (i+1) % poly.shape[0]
        p1 = poly[i]
        p2 = poly[j]
        r = (p2 - p1).dot(x - p1)
        r /= np.linalg.norm(p2 - p1)**2
        if r < 0:
            dist = np.linalg.norm(x - p1)
            res = -p1 + x
        elif r > 1:
            dist = np.linalg.norm(p2 - x)
            res = -p2 + x
        else:
            dist = np.sqrt(np.linalg.norm(x - p1)**2 - (r * np.linalg.norm(p2-p1))**2)
            res = -r*(p2-p1) + (x - p1)
        if (min_dist > dist):
            min_dist = dist
            min_res = res
    if inpoly:
        return -min_res * (TH) / np.linalg.norm(min_res)
    else:
        if min_dist > TH:
            return np.array([0., 0.])
        else:
            return min_res * (TH - min_dist)/np.linalg.norm(min_res)

if __name__ == '__main__':
    from matplotlib.patches import Polygon
    import matplotlib.pyplot as plt
    # test
    ploy = np.array([[0, 0], [1, 1.], [0, 2.], [-1., 1]])
    vec = polygonRes(np.array([-1, 2.]), ploy)
    fig = plt.figure('after loop-closing')
    axes = fig.gca()
    axes.add_patch(Polygon(ploy, alpha=0.5))
    for i in np.arange(-2, 2, 0.15):
        for j in np.arange(-1, 3, 0.15):
            vec = polygonRes(np.array([i, j]), ploy)
            if (np.linalg.norm(vec)):
                vec *= 0.2
                axes.arrow(i, j, vec[0], vec[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
            else:
                axes.scatter(i, j, color='black', s=3)
    plt.show()
