import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.special
import pickle
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
import scipy.sparse.linalg
import copy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def rel_err(a, b):
    return np.abs((a - b) / a)


def pickle_save(data, file_name):
    file = open(f"{file_name}.pkl", "wb")
    pickle.dump(data, file)
    file.close()


def pickle_load(file_name):
    file = open(f"{file_name}.pkl", "rb")
    data = pickle.load(file)
    file.close()

    return data


class Plot:

    def __init__(self):
        self.fig = plt.figure(figsize=self.figsize)
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


class Plot_Loglog(Plot):

    def __init__(self, figsize=(10, 8)):
        self.figsize = figsize
        Plot.__init__(self)

        self.ax = self.fig.add_subplot(111)

    def __call__(self, x, y, linestyle='o-', c=None, linewidth=2, markersize=8, label=None):
        self.ax.loglog(x, y, linestyle,
                       linewidth=linewidth, markersize=markersize, c=c, label=label)

    def show(self):
        return self.fig

    def finish(self, ylims=None, title="", fontsize=17, legend_ncol=1):
        self.ax.set_title(title, fontsize=fontsize)

        self.ax.set_ylim(ylims)

        self.ax.legend(fontsize=fontsize, ncol=legend_ncol)

        self.ax.tick_params(axis='both', which='major', labelsize=fontsize)
        self.ax.tick_params(axis='both', which='minor', labelsize=fontsize)


class Plot_Eigenfunction(Plot):

    def __init__(self, figsize=(10, 8)):
        self.figsize = figsize
        Plot.__init__(self)

        self.ax = self.fig.add_subplot(111, projection='3d')

    def __call__(self, g_domain, f, markersize=40):
        self.ax.scatter3D(g_domain[:, 0], g_domain[:, 1], g_domain[:, 2], c=f, cmap="plasma")

    def finish(self, lim=0.7, view=[-58.4, 0]):
        self.ax.set_xlim([-lim, lim])
        self.ax.set_ylim([-lim, lim])
        self.ax.set_zlim([-lim, lim])

        self.ax.set_box_aspect([1, 1, 1])
        self.ax.axis('off')
        self.ax.view_init(*view)
        self.fig.tight_layout()


def tidy_3dplot(ax, lim=0.7, view=[-58.4, 0]):
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')
    ax.view_init(*view)


def normalize_array(x, return_norms=False, ndim=3):
    shape = x.shape

    if shape[-1] == ndim:

        norms = np.linalg.norm(x, axis=len(shape) - 1, keepdims=True)

        if return_norms:
            return x / norms, norms

        return x / norms

    raise ValueError("Make sure x.shape == (number of points, ndim)")


def convert_to_column_vector(v):
    try:

        shape = v.shape
        dims = len(shape)

        if (dims == 2 and np.any(shape == 1)) or dims == 1:
            return v.reshape(max(shape), 1)

    except:

        v = np.array([v])

        shape = v.shape
        dims = len(shape)

        if dims == 1 or min(shape) == 1:
            return v.reshape(max(shape), 1)

        raise ValueError("Cannot be converted to column vector: v has bad type or shape")


def tablelegend(ax, handles=[],
                col_labels=['$n=1$', '$n=2$', '$n=3$'],
                row_labels=['$m=1$', '$m=2$', '$m=3$'],
                title_label="", *args, **kwargs):
    if not handles:
        handles, labels, extra_args, kwargs = mlegend._parse_legend_args([ax], *args, **kwargs)
    else:
        _, labels, extra_args, kwargs = mlegend._parse_legend_args([ax], *args, **kwargs)

    if len(extra_args):
        raise TypeError('legend only accepts two non-keyword arguments')

    if col_labels is None and row_labels is None:
        ax.legend_ = mlegend.Legend(ax, handles, labels, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_

    else:
        ncol = kwargs.pop('ncol')
        handletextpad = kwargs.pop('handletextpad', 0 if col_labels is None else -2)
        title_label = [title_label]

        extra = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)]

        empty = [""]

        nrow = len(handles) // ncol

        if col_labels is None:
            assert nrow == len(
                row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (
            nrow, len(row_labels))
            leg_handles = extra * nrow
            leg_labels = row_labels
        elif row_labels is None:
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (
            ncol, len(col_labels))
            leg_handles = []
            leg_labels = []
        else:
            assert nrow == len(
                row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (
            nrow, len(row_labels))
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (
            ncol, len(col_labels))
            leg_handles = extra + extra * nrow
            leg_labels = title_label + row_labels
        for col in range(ncol):
            if col_labels is not None:
                leg_handles += extra
                leg_labels += [col_labels[col]]
            leg_handles += handles[col * nrow:(col + 1) * nrow]
            leg_labels += empty * nrow

        ax.legend_ = mlegend.Legend(ax, leg_handles, leg_labels, ncol=ncol + int(row_labels is not None),
                                    handletextpad=handletextpad, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_


class Graph:
    def calculate_csc(self, k, l):

        return 1 / np.sin(k * l)

    def calculate_dcsc(self, k, l):

        return l * -1 / np.sin(k * l) * 1 / np.tan(k * l)

    def calculate_cot(self, k, l):

        return 1 / np.tan(k * l)

    def calculate_dcot(self, k, l):

        return l * -1 / np.sin(k * l) ** 2

    def calculate_sec(self, k, l):

        return 1 / np.cos(k * l)

    def calculate_dsec(self, k, l):

        return l * 1 / np.cos(k * l) * np.tan(k * l)

    def calculate_e_length_flat(self, u_coord, v_coord):

        return np.linalg.norm(u_coord - v_coord)

    def calculate_e_length_spherical(self, u_coord, v_coord):

        dot = np.dot(u_coord, v_coord)
        cross = np.linalg.norm(np.cross(u_coord, v_coord))

        return np.arctan2(cross, dot)

    def calculate_r_vw_flat(self, u_coord, v_coord):

        r_vw = u_coord - v_coord

        return r_vw, r_vw

    def calculate_r_vw_spherical(self, u_coord, v_coord):

        dot = np.dot(u_coord, v_coord)

        tangent_u = normalize_array(dot * u_coord - v_coord)
        tangent_v = normalize_array(dot * v_coord - u_coord)

        return tangent_u, tangent_v

    def update_R_flat(self):

        return lambda r, l: np.tensordot(r, r) / l

    def update_R_spherical(self):

        return lambda r, l: np.tensordot(r, r) * l

    def construct_cart_to_spher_transformation_matrix(self, theta, phi, r):

        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)

        return np.array(([cos_theta * cos_phi, -sin_phi, sin_theta * cos_phi],
                         [cos_theta * sin_phi, cos_phi, sin_theta * sin_phi],
                         [-sin_theta, 0, cos_theta]))

    def construct_E_lengths_and_wadjacency_matrix(self):

        E_lengths = []
        wadjacency_matrix = scipy.sparse.lil_matrix((self.num_Vs, self.num_Vs), dtype=np.float64)

        if self.geometry == "flat":

            e_length_calculator = self.calculate_e_length_flat

        elif self.geometry == "spherical":

            e_length_calculator = self.calculate_e_length_spherical

        for v0_num, v1_num in self.E_by_v_num:
            e_length = e_length_calculator(self.V_coords[v0_num],
                                           self.V_coords[v1_num])

            E_lengths.append(e_length)

            wadjacency_matrix[v0_num, v1_num] = e_length
            wadjacency_matrix[v1_num, v0_num] = e_length

        return np.array(E_lengths), wadjacency_matrix.tocsr()

    def construct_g_domain(self, points=10, fixed_num_points=True):

        if fixed_num_points:
            calculate_points_per_edge = lambda l: points
        else:
            calculate_points_per_edge = lambda l: max(2, int(l * points))

        g_domain = []

        for e_num, (v_num, w_num) in enumerate(self.E_by_v_num):

            v_coord = self.V_coords[v_num]
            w_coord = self.V_coords[w_num]

            l_vw = self.E_lengths[e_num]
            points_per_edge = calculate_points_per_edge(l_vw)
            x = np.linspace(0, 1, points_per_edge)
            l = (x * w_coord[:, np.newaxis]
                 + (1 - x) * v_coord[:, np.newaxis])
            if self.geometry == "spherical":
                l /= np.linalg.norm(l, axis=0, keepdims=True)

            g_domain.append(l)

        return g_domain

    def construct_L(self, k, deriv=False):

        if not deriv:
            calculate_csc = self.calculate_csc
            calculate_cot = self.calculate_cot
        else:
            calculate_csc = self.calculate_dcsc
            calculate_cot = self.calculate_dcot

        matrix_csc = self.wadjacency_matrix.copy()
        matrix_csc.data = calculate_csc(k, matrix_csc.data)

        matrix_cot = self.wadjacency_matrix.copy()
        matrix_cot.data = calculate_cot(k, matrix_cot.data)
        matrix_cot = scipy.sparse.diags(matrix_cot.sum(axis=0).flat)

        return (matrix_cot - matrix_csc).tocsc()[self.interior_V_num[:, None], self.interior_V_num]

    def construct_R(self):

        R = [0 for _ in range(self.num_Vs)]

        if self.geometry == "flat":

            calculate_r_vw = self.calculate_r_vw_flat
            update_R = self.update_R_flat

        elif self.geometry == "spherical":

            calculate_r_vw = self.calculate_r_vw_spherical
            update_R = self.update_R_spherical

        for e_num, v_num, w_num in enumerate(self.E_by_v_num):
            r_vw, r_wv = calculate_r_vw(self.V_coords[v_num], self.V_coords[w_num])

            R[v_num] += update_R(r_vw, self.E_lengths[e_num])
            R[w_num] += update_R(r_wv, self.E_lengths[e_num])

        if self.geometry == "spherical":

            for r_num, (theta, phi, r) in enumerate(self.spher_V_coords):
                transf = self.construct_cart_to_spher_transformation_matrix(theta, phi, r)
                R[r_num] = (transf.T @ R[r_num] @ transf)[:-1, :-1]

        return R


class SpiderWeb(Graph):

    def __init__(self, num_radial_Vs, num_angular_Vs, j=None, rtype=0):

        self.num_radial_Vs = num_radial_Vs
        self.num_angular_Vs = num_angular_Vs
        self.num_Vs = self.num_angular_Vs * (self.num_radial_Vs - 1) + 1

        self.j = j
        self.rtype = rtype

        self.dtheta = 2 * np.pi / self.num_angular_Vs

        self.interior_V_num = np.arange(self.num_radial_Vs)

        self.geometry = "flat"

        self.r = self.construct_radial_distribution()
        self.theta = np.linspace(0, 2 * np.pi, self.num_angular_Vs, endpoint=False)

        self.V_coords = np.array([[0, 0]] + [[i * np.cos(j), i * np.sin(j)]
                                             for i in self.r[1:] for j in self.theta])

        self.E_by_v_num = self.construct_E_by_v_num()

        self.interior_v_num = np.arange(self.num_Vs - self.num_angular_Vs, self.num_Vs)

        self.radial_lengths = np.array([np.linalg.norm(self.r[i] - self.r[i - 1])
                                        for i in range(1, self.num_radial_Vs)])

    def construct_radial_distribution(self, rtype=0):  # [..., 1(1-dtheta)(1-dtheta), 1(1-dtheta), 1]

        if self.rtype == 0:

            dtheta = 2 * np.pi / self.num_angular_Vs

            radial_distribution = [1]

            for i in range(self.num_radial_Vs - 2):
                radial_distribution = [radial_distribution[0] * (1 - dtheta)] + radial_distribution

            return np.array([0] + radial_distribution)

        elif self.rtype == 1:

            dtheta = 2 * np.pi / (self.num_angular_Vs + 1)

            radial_distribution = [1]

            for i in range(self.num_radial_Vs - 1):
                radial_distribution = [radial_distribution[0] * (1 - dtheta)] + radial_distribution

            radial_distribution = np.array(radial_distribution)
            radial_distribution -= radial_distribution[0]
            radial_distribution /= radial_distribution[-1]

            return radial_distribution

    def construct_E_by_v_num(self):

        E_by_v_num = [[0, w_num] for w_num in np.arange(1, self.num_angular_Vs + 1)]

        for i in range(1, self.num_angular_Vs + 1):
            neighbours = [np.mod(i, self.num_angular_Vs) + 1,
                          np.mod(i + self.num_angular_Vs - 2, self.num_angular_Vs) + 1,
                          i + self.num_angular_Vs]

            E_by_v_num += [[i, neighbour] for neighbour in neighbours]

            a = self.num_Vs - self.num_angular_Vs + i - 1
            b = self.num_angular_Vs * (self.num_radial_Vs - 2)
            neighbours = [b + np.mod(a, self.num_angular_Vs) + 1,
                          b + np.mod(a + self.num_angular_Vs - 2, self.num_angular_Vs) + 1,
                          a - self.num_angular_Vs]

            E_by_v_num += [[a, neighbour] for neighbour in neighbours]

        for i in range(1, self.num_radial_Vs - 2):

            a = i * self.num_angular_Vs + 1

            for j in range(self.num_angular_Vs):
                b = a + j
                c = self.num_angular_Vs * i
                neighbours = [c + np.mod(b, self.num_angular_Vs) + 1,
                              c + np.mod(b + self.num_angular_Vs - 2, self.num_angular_Vs) + 1,
                              b - self.num_angular_Vs,
                              b + self.num_angular_Vs]

                E_by_v_num += [[b, neighbour] for neighbour in neighbours]

        return np.unique(np.sort(E_by_v_num, axis=1), axis=0)

    def construct_L(self, k, deriv=False):

        matrix = scipy.sparse.lil_matrix((self.num_radial_Vs, self.num_radial_Vs), dtype=np.float64)

        if not deriv:

            calculate_csc = self.calculate_csc
            calculate_cot = self.calculate_cot
            calculate_sec = self.calculate_sec
            matrix[0, 0] = 1
            matrix[-1, -1] = 1

        else:

            calculate_csc = self.calculate_dcsc
            calculate_cot = self.calculate_dcot
            calculate_sec = self.calculate_dsec

        if self.j == 0:
            matrix[0, 1] = -calculate_sec(k, self.radial_lengths[0])  # / self.num_angular_Vs

        for i in range(1, self.num_radial_Vs - 1):
            back = self.radial_lengths[i - 1]
            forward = self.radial_lengths[i]

            rho = 2 * np.sum(self.radial_lengths[:i]) * np.sin(self.dtheta / 2)

            matrix[i, i - 1] = -calculate_csc(k, back)
            matrix[i, i] = (calculate_cot(k, back) + calculate_cot(k, forward)
                            + 2 * (calculate_cot(k, rho) - np.cos(self.j * self.dtheta) * calculate_csc(k, rho)))
            matrix[i, i + 1] = -calculate_csc(k, forward)

        return matrix.tocsc()


def calculate_radial_EF_pde(r, m, n, eigs_PDE):

    if m == 0:

        k = np.pi * (n + 1 / 2) / np.sqrt(2)
        radial_EF_pde = -np.cos(np.sqrt(2) * k * r)

    else:

        order = 1 / 2 * np.sqrt(1 + 4 * m ** 2)
        k = eigs_PDE[m, n]
        radial_EF_pde = np.sqrt(r) * scipy.special.jv(order, np.sqrt(2) * k * r)

    radial_EF_pde /= np.linalg.norm(radial_EF_pde)

    return radial_EF_pde


def tidy_visual_comparison(ax):
    ax.legend(fontsize=28, ncol=1, loc="lower right")
    ax.tick_params(axis='both', which='major', labelsize=17)
    ax.tick_params(axis='both', which='minor', labelsize=17)

    ticks = np.array([0 - 1, 2 - 2, 6 - 3, 12 - 4, 20 - 5]) + 0.5
    ax.set_xticks(ticks)

    ax.tick_params('x', length=7, width=3, which='major')
    ax.tick_params('y', length=7, width=3, which='major')


class Update_Eigenvalue_By_Newton:

    def __init__(self, g):

        self.g = g
        self.num_interior_Vs = len(self.g.interior_V_num)

    def __call__(self, k, num_rademacher_vectors, hutchinson_estimation=False):

        k = np.array([k]).flatten().astype(np.float64)

        for eigenvalue_num, eigenvalue in enumerate(k):

            L = self.g.construct_L(eigenvalue)

            dL = self.g.construct_L(eigenvalue, deriv=True)

            if hutchinson_estimation:

                L = scipy.sparse.linalg.splu(L)
                trace = self.hutchinson_estimator(L, dL, num_rademacher_vectors)

            else:

                trace = np.trace(np.linalg.solve(L.A, dL.A))

            k[eigenvalue_num] -= 1 / trace

        return k

    def hutchinson_estimator(self, L, dL, num_rademacher_vectors):

        trace = 0

        for _ in range(num_rademacher_vectors):
            rademacher_vector = np.random.choice((1, -1), self.num_interior_Vs)
            trace += L.solve(rademacher_vector) @ dL @ rademacher_vector

        return trace / num_rademacher_vectors


def Newton_Runner(g, k, max_iters=1000, atol=14,
                  num_rademacher_vectors=20, printerval=100,
                  hutchinson_estimation=False):

    eigs = Update_Eigenvalue_By_Newton(g)

    print(f"Step = 0")
    print(f"k    = {k}")
    print()

    if not isinstance(k, (list, np.ndarray)): k = np.array([k])
    unfinished_k_indices = np.arange(len(k))

    for step in range(max_iters):

        k_next = k.copy()

        k_next[unfinished_k_indices] = eigs(k[unfinished_k_indices],
                                            num_rademacher_vectors=num_rademacher_vectors,
                                            hutchinson_estimation=hutchinson_estimation)

        unfinished_k_indices = np.round(k_next, atol) != np.round(k, atol)

        k = k_next.copy()

        if sum(unfinished_k_indices) == 0:
            print("Finished")
            print(f"Step   = {step + 1}")
            print(f"k      = {k}")

            return k

        if (step + 1) % printerval == 0:
            print(f"Step = {step + 1}")
            print(f"k    = {k}\n")

        if step == max_iters - 1:
            print("Didn't converge\n")

            unfinished_k_indices = np.round(k_next, 8) != np.round(k, 8)

            return k[[not i for i in unfinished_k_indices]]


class Spherical_Graph:

    def convert_V_coords_to_spher(self):
        return self.cart_to_spher(self.V_coords[:, 0],
                                  self.V_coords[:, 1],
                                  self.V_coords[:, 2])

    def cart_to_spher(self, x, y, z):
        x, y, z = (convert_to_column_vector(i) for i in [x, y, z])

        xy = x ** 2 + y ** 2

        return np.hstack([np.arctan2(np.sqrt(xy), z), np.arctan2(y, x), np.sqrt(xy + z ** 2)])


class Icosphere(Graph, Spherical_Graph):
    """This class produces attributes of Icosahedron (seed) after successive Conway
        polyhedron operations: truncate (t) and dual (d): tdtdtdt...

        mesh_type   : type of graph we end up with--dual (triangular mesh) or truncate
        (hexagonal mesh with 12 pentagons (one for each of the 12 vertices of the Icosahedron))

        num_subdivs : the number of subdivisions. E.g. ("truncate", 2) will perform tdt
        to the icosahedron; ("dual", 3) will perform tdtdtd.

        The main attributes we end up with are:
            - the edge length-weighted adjacency matrix
            - a plotting function
    """

    def __init__(self, mesh_type_and_num_subdivs=("dual", 0), plot=True, figsize=4, view=[-58.4, 0]):

        self.geometry = "spherical"

        self.update_attributes_by_seed()

        self.mesh_type, self.num_subdivs = mesh_type_and_num_subdivs

        if self.mesh_type == "truncate" and self.num_subdivs == 0:
            raise ValueError("Seed cannot be of type truncate")

        self.initial_build = True

        if self.num_subdivs:

            for _ in range(self.num_subdivs - 1):
                self.update_attributes_by_truncate()
                self.update_attributes_by_dual()

            self.update_attributes_by_truncate()

            if self.mesh_type == "dual":
                self.update_attributes_by_dual()

        self.spher_V_coords = self.convert_V_coords_to_spher()
        (self.E_lengths,
         self.wadjacency_matrix) = self.construct_E_lengths_and_wadjacency_matrix()
        self.interior_V_num = np.arange(self.num_Vs)
        self.num_interior_Vs = self.num_Vs

        self.initial_build = False

        if plot:
            print(f"\n|V| = {self.num_Vs}")
            self.plot(figsize=figsize, view=view)

    def calculate_cell_area(self):

        if self.mesh_type == "truncate":
            V_by_f_num = self.construct_V_by_f_num()

        F_area = self.calculate_F_area()
        num_Vs_of_face = [len(f) for f in g.F_by_v_num]
        rescaled_F_area = np.array(F_area) / np.array(num_Vs_of_face)

        cell_area = []

        for v_by_f_num in V_by_f_num:
            cell_area.append(sum([rescaled_F_area[f_num] for f_num in v_by_f_num]))

        return cell_area

    def calculate_F_area(self):

        F_area = []
        F_by_v_coord = self.construct_F_by_v_coord()

        for f_by_v_coord in F_by_v_coord:
            f_vertex_pairs = list(zip(f_by_v_coord, np.roll(f_by_v_coord, -1, axis=0)))

            F_area.append(1 / 2 * np.linalg.norm(sum([np.cross(v_coord, w_coord)
                                                      for v_coord, w_coord in f_vertex_pairs])))

        return F_area

    def construct_curve(self, v, w):

        t = np.linspace(0, 1, 100).reshape(1, 100)
        l = v.reshape(3, 1) @ t + w.reshape(3, 1) @ (1 - t)
        norms = np.linalg.norm(l, axis=0, keepdims=True)

        l /= norms

        return l

    def construct_E_by_v_num(self):

        E_by_v_num = {}

        for f_by_v_num in self.F_by_v_num:
            edge = list(zip(f_by_v_num, np.roll(f_by_v_num, -1)))
            edge = [[min(e), max(e)] for e in edge]
            E_by_v_num = {**E_by_v_num, **{f"{i},{j}": [i, j] for i, j in edge}}

        return list(E_by_v_num.values())

    def construct_F_by_v_coord(self, purpose=""):

        F_by_v_coord = [self.V_coords[f_by_v_num] for f_by_v_num in self.F_by_v_num]

        if purpose == "plot" and self.mesh_type == "truncate":
            F_by_v_coord.sort(key=len)

            return F_by_v_coord[::-1]  # put pentagons at start to colour them red

        return F_by_v_coord

    def construct_next_graph(self):

        next_graph = copy.deepcopy(self)

        if next_graph.mesh_type == "dual":
            next_graph.update_attributes_by_truncate()
        else:
            next_graph.update_attributes_by_dual()

        return next_graph

    def construct_V_by_v_neighbours_num(self):

        V_by_v_neighbours_num = [[] for _ in range(self.num_Vs)]

        for v_a, v_b in self.E_by_v_num:
            V_by_v_neighbours_num[v_a].append(v_b)
            V_by_v_neighbours_num[v_b].append(v_a)

        return V_by_v_neighbours_num

    def order_F_by_v_num_and_construct_V_by_f_num(self, new_F_by_new_v_num):

        V_by_f_num = [[] for _ in range(self.num_Vs)]

        for f_num, f_by_v_num in enumerate(new_F_by_new_v_num):

            for v in f_by_v_num:
                V_by_f_num[v].append(f_num)

            # Calculates ordering of vertices of polygon using
            # angles of each w away from v using centroid:

            centroid = np.mean(self.V_coords[f_by_v_num], axis=0)
            disps = self.V_coords[f_by_v_num] - centroid

            costheta = np.dot(disps[1:], disps[0])
            sintheta = np.cross(disps[1:], disps[0])
            back = np.where(np.dot(sintheta, centroid) < 0)[0]
            sintheta = np.linalg.norm(np.cross(disps[1:], disps[0]), axis=1)
            sintheta[back] *= -1
            atan2 = np.arctan2(sintheta, costheta)
            back = np.where(atan2 < 0)[0]
            atan2[back] = 2 * np.pi + atan2[back]
            sort = np.argsort(atan2)
            new_F_by_new_v_num[f_num] = ([new_F_by_new_v_num[f_num][0]]
                                         + list(np.array(new_F_by_new_v_num[f_num][1:])[(sort)]))

        return new_F_by_new_v_num, V_by_f_num

    def construct_V_by_f_num(self):

        V_by_f_num = [[] for _ in range(self.num_Vs)]

        for f_num, f_by_v_num in enumerate(self.F_by_v_num):

            for v in f_by_v_num:
                V_by_f_num[v].append(f_num)

        return (V_by_f_num)

    def return_flipped_mesh_type(self):

        if self.mesh_type == "dual":
            return "truncate"

        return "dual"

    def plot(self, scatter=False, figsize=7, markersize=40,
             lim=0.7, view=[-58.4, 0], alpha=0.94, return_figax=False,
             show_pentagons=True, soccer_ball_colors=True):

        F_by_v_coord = self.construct_F_by_v_coord(purpose="plot")

        fig = plt.figure(figsize=(figsize, figsize))

        ax = fig.add_subplot(111, projection='3d')

        if self.mesh_type == "truncate":

            if show_pentagons:

                if soccer_ball_colors:

                    edgecolor = "black"
                    facecolors = [['white'], ['black']]
                    alpha = 1

                else:

                    edgecolor = 'blue'
                    facecolors = [['xkcd:light periwinkle'], ['red']]

                num_F_by_v_coord = len(F_by_v_coord)
                ax.add_collection3d(Poly3DCollection(F_by_v_coord, edgecolor=edgecolor,
                                                     facecolors=((num_F_by_v_coord - 12) * facecolors[0]
                                                                 + 12 * facecolors[1]),
                                                     linewidths=3, alpha=alpha))

            else:

                fac = 0.9842
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                x = fac * 1 * np.outer(np.cos(u), np.sin(v))
                y = fac * 1 * np.outer(np.sin(u), np.sin(v))
                z = fac * 1 * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z,
                                color='xkcd:light periwinkle', alpha=0.4,
                                shade=False)

                domain = np.hstack((self.construct_g_domain(points=100)))
                ax.scatter3D(domain[0], domain[1], domain[2], s=10, c='b')

        else:

            ax.add_collection3d(Poly3DCollection(F_by_v_coord, edgecolor='black',
                                                 facecolors='white',
                                                 linewidths=3, alpha=alpha))

        if scatter: ax.scatter3D(self.V_coords[:, 0], self.V_coords[:, 1], self.V_coords[:, 2],
                                 s=markersize)

        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
        ax.set_box_aspect([1, 1, 1])
        ax.axis('off')
        ax.view_init(*view)
        fig.tight_layout()

        if return_figax:
            return fig, ax
        else:
            plt.show()

    def truncate_V_by_v_neighbours_num(self):

        new_V_coord = []
        new_F_by_new_v_num = []
        old_E_by_new_v_num = {f"{v_a},{v_b}": [] for v_a, v_b in self.E_by_v_num}

        for v_num, v_neighbours_num in enumerate(self.V_by_v_neighbours_num):
            v_coord = self.V_coords[v_num]

            new_Vs = v_coord + 1 / 3 * (self.V_coords[v_neighbours_num] - v_coord)
            new_Vs /= np.linalg.norm(new_Vs, axis=1)[:, np.newaxis]
            new_V_coord += list(new_Vs)

            current_new_num_Vs = len(new_V_coord)

            new_F_by_new_v_num.append(np.arange(current_new_num_Vs - len(v_neighbours_num), current_new_num_Vs))

            edges = [[min([v_num, v_neighbour_num]), max([v_num, v_neighbour_num])]
                     for v_neighbour_num in v_neighbours_num]

            self.update_old_E_by_new_v_num(edges, new_F_by_new_v_num[-1], old_E_by_new_v_num)

        return new_V_coord, new_F_by_new_v_num, old_E_by_new_v_num

    def update_attributes_by_dual(self):

        new_V_coord = np.array([np.mean(self.V_coords[face], axis=0) for face in self.F_by_v_num])
        self.V_coords = new_V_coord / np.linalg.norm(new_V_coord, axis=1)[:, np.newaxis]
        self.num_Vs = len(self.V_coords)

        self.F_by_v_num = self.V_by_f_num.copy()
        del self.V_by_f_num

        self.E_by_v_num = self.construct_E_by_v_num()
        self.V_by_v_neighbours_num = self.construct_V_by_v_neighbours_num()

        if not self.initial_build:
            self.update_shared_attributes()

    def update_attributes_by_seed(self):

        """
        Update graph to seed--initial graph--the icosahedron (the largest
        Platonic solid, smallest geodesic polyhedron)
        """

        phi = (1 + np.sqrt(5)) / 2
        self.V_coords = np.array(([-1, phi, 0], [1, phi, 0], [-1, -phi, 0],
                                  [1, -phi, 0], [0, -1, phi], [0, 1, phi],
                                  [0, -1, -phi], [0, 1, -phi], [phi, 0, -1],
                                  [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]))
        self.V_coords /= np.linalg.norm(self.V_coords, axis=1)[:, np.newaxis]
        self.spher_V_coords = self.convert_V_coords_to_spher()
        self.num_Vs = len(self.V_coords)
        self.F_by_v_num = np.array(([0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                                    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                                    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                                    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]))
        self.F_by_v_num = np.sort(self.F_by_v_num, axis=1)
        self.E_by_v_num = self.construct_E_by_v_num()
        self.V_by_v_neighbours_num = self.construct_V_by_v_neighbours_num()
        (self.E_lengths,
         self.wadjacency_matrix) = self.construct_E_lengths_and_wadjacency_matrix()

    def update_attributes_by_truncate(self):

        new_V_coord, new_F_by_new_v_num, old_E_by_new_v_num = self.truncate_V_by_v_neighbours_num()
        del self.V_by_v_neighbours_num

        self.V_coords = np.array(new_V_coord)
        self.num_Vs = len(self.V_coords)

        new_F_by_new_v_num += [np.array(old_E_by_new_v_num[f"{v_a},{v_b}"]
                                        + old_E_by_new_v_num[f"{v_a},{v_c}"]
                                        + old_E_by_new_v_num[f"{v_b},{v_c}"])
                               for v_a, v_b, v_c in self.F_by_v_num]

        self.F_by_v_num, self.V_by_f_num = self.order_F_by_v_num_and_construct_V_by_f_num(new_F_by_new_v_num)
        self.E_by_v_num = self.construct_E_by_v_num()

        if not self.initial_build:
            self.update_shared_attributes()

    def update_shared_attributes(self):

        self.spher_V_coords = self.convert_V_coords_to_spher()
        self.mesh_type = self.return_flipped_mesh_type()
        self.num_subdivs += 1
        (self.E_lengths,
         self.wadjacency_matrix) = self.construct_E_lengths_and_wadjacency_matrix()
        self.interior_V_num = np.arange(self.num_Vs)
        self.num_interior_Vs = self.num_Vs

    def update_old_E_by_new_v_num(self, edges, new_f_by_new_v_num, old_E_by_new_v_num):

        for (v_a, v_b), new_v in zip(edges, new_f_by_new_v_num):
            old_E_by_new_v_num[f"{v_a},{v_b}"].append(new_v)