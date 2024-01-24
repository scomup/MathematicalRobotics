from load_ba_datasets import BALLoader
from projection import *
from utilities.math_tools import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.linalg import lu_factor, lu_solve
from scipy.linalg import lapack, solve
from scipy.sparse.linalg import spsolve_triangular
from threading import Thread
from gui import *
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)


def undistort_point(u, K, dist_coeffs):
    # Convert to homogeneous coordinates
    u_homo = np.array([u[0], u[1], 1])
    k1 = dist_coeffs[0]
    k2 = dist_coeffs[1]
    # Normalize coordinates
    p = np.linalg.inv(K) @ u_homo

    # Apply distortion correction
    r_squared = np.sum(p[:2]**2)
    """
    The camera model is described in:
    https://grail.cs.washington.edu/projects/bal/
    """
    radial_correction = 1 + k1 * r_squared + k2 * r_squared**2

    u_corrected = np.array([
        p[0] * radial_correction,
        p[1] * radial_correction
    ])

    # Convert back to homogeneous coordinates
    u_undist_homo = np.array([u_corrected[0], u_corrected[1], 1])

    # Project back to image coordinates
    u_undist = (K @ u_undist_homo)[:2]

    return u_undist


def solveBA(graph, viewer):
    min_score_change = 1.
    last_score = np.inf
    itr = 0
    step = 0
    while(True):
        viewer.setVertices(graph.vertices)
        draw_reproj(graph)

        start = time.time()
        dx, score = graph.solve_once()
        end = time.time()
        itr += 1
        time_diff = end - start

        info_text = 'iter %d: solve time: %f error: %f' % (itr, time_diff, score)
        print(info_text)
        viewer.setText(info_text)
        if (last_score - score < min_score_change and itr > 5):
            print('Solved!')
            break
        if (step > 0 and np.max(dx) > step):
            dx = dx/np.max(dx) * step
        graph.update(dx)
        last_score = score
        if (itr >= 10):
            break
    graph.report()


def runQt(app):
    while(True):
        app.processEvents()


def draw_reproj(graph):
    draw_frame_num = 6
    col = 2
    row = int(draw_frame_num/col)
    draw_frame_num = col * row
    plt.close('all')
    fig, axes = plt.subplots(col, row, num='reprojection')

    img_pts = [[] for i in range(draw_frame_num)]
    reproj_pts = [[] for i in range(draw_frame_num)]
    for e in graph.edges:
        if (type(e).__name__ != 'ReprojEdge'):
            continue
        cam_id = e.link[0]
        points_id = e.link[1]
        if(cam_id >= draw_frame_num):
            continue
        u, K = e.z
        Twc = graph.vertices[cam_id].x
        pw = graph.vertices[points_id].x
        u_reproj = reproject(transform_inv(Twc, pw), K)
        img_pts[cam_id].append(u)
        reproj_pts[cam_id].append(u_reproj)
    img_pts = [np.array(i) for i in img_pts]
    reproj_pts = [np.array(i) for i in reproj_pts]

    for i in range(draw_frame_num):
        ax = axes[int(i / row), int(i % row)]
        ax.set_xlim(np.min(img_pts[i][:, 0]), np.max(img_pts[i][:, 0]))
        ax.set_ylim(np.min(img_pts[i][:, 1]), np.max(img_pts[i][:, 1]))
        ax.invert_yaxis()
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        ax.scatter(img_pts[i][:, 0], img_pts[i][:, 1], color='r', s=1, alpha=0.5)
        ax.scatter(reproj_pts[i][:, 0], reproj_pts[i][:, 1], color='c', s=1, alpha=0.5)
        ax.set_title("camera%d" % i)
    plt.pause(0.1)


if __name__ == '__main__':
    print("Load dataset...")

    loader = BALLoader()
    path = os.path.dirname(os.path.abspath(__file__)) + '/../data/ba/'
    filename = "problem-49-7776-pre.txt"  # Ladybug
    # filename = "problem-170-49267-pre.txt"  # Trafalgar
    # filename = "problem-427-310384-pre.txt"  # Venice

    if loader.load_file(path + filename):
        pass
    else:
        print("Error loading file.")

    print("Undistort points...")
    for obs in loader.observations:
        cam = loader.cameras[obs.camera_id]
        obs.u_undist = undistort_point(obs.u, cam.K, cam.dist_coeffs)

    # Add noise
    loader.points += np.random.normal(0, 0.05, loader.points.shape)
    graph = GraphSolver(use_sparse=True)
    app = QApplication([])
    viewer = BAViewer()
    viewer.show()

    print("Add camera vertex... (piror pose for camera)")
    for i, cam in enumerate(loader.cameras):
        Twc = makeT(cam.R, cam.t)
        if(i == 0):
            graph.add_vertex(CameraVertex(Twc), is_constant=True)
        else:
            cid = graph.add_vertex(CameraVertex(Twc))  # add vertex to graph
            graph.add_edge(CameraEdge([cid], Twc, np.eye(6) * 1e5))  # add piror pose

    print("Add point vertex...")
    for pw in loader.points:
        graph.add_vertex(PointVertex(pw))

    viewer.setVertices(graph.vertices)
    # app.exec_()
    print("Add edges...")
    camera_size = len(loader.cameras)
    kernel = HuberKernel(np.sqrt(5))
    for obs in loader.observations:
        cam = loader.cameras[obs.camera_id]
        graph_pid = camera_size + obs.point_id
        graph.add_edge(ReprojEdge(
            [obs.camera_id, graph_pid],
            [obs.u_undist, cam.K],
            np.eye(2), kernel))
    print("start BA...")
    # t1 = Thread(target=solveBA, args=[graph, viewer])
    # t1.start()
    t2 = Thread(target=runQt, args=[app])
    t2.start()
    solveBA(graph, viewer)

    app.exec_()
