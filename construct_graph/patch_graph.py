"""
Patch takes a "total" graph. Takes out a 1x1 patch. Periodizes the patch. And solves the corresponding corrector problem.
The types of total_graph's are:
    - Truncated_Square
    - Random_Delaunay
These total_graph's need to have:
    - total_num_Vs    : total number of vertices (total = in and out of 1x1 square).
    - num_Vs          : number of vertices in 1x1 square
    - total_V_coords  : coordinates of total vertices. First num_Vs are those in the 1x1square
    - total_edges     : total edges between total vertices where edge (v_index, w_index) satisfies v_index < w_index

e.g.:
num_Vs = 100
total_graph = Random_Delaunay(num_Vs)
g = Patch(total_graph)
eq = Cell_Problem(g)
xi = eq.solve_corrector_equation()
coefficient = xi.construct_homogenized_tensor(xi)
print(coefficient)
"""

import numpy as np
import scipy
import pickle
import matplotlib
import collections
from construct_graph.graph import Graph, Flat
import os
# import sparseqr
# import sksparse

class Patch(Graph, Flat):

    """Parent class of more specific graph classes (spiderweb, delaunay triangulations etc.)
    These child classes need the attributes:
        - num_Vs
        - V_coords
        - E_lengths_by_v_num
        - interior_V_num
        - wadjacency_matrix
    """

    def __init__(self, total_graph):

        self.total_graph = total_graph
        self.box = np.array([1, 1])

        self.num_Vs = self.total_graph.num_Vs
        self.V_coords = self.total_graph.total_V_coords[:self.num_Vs]
        self.edges, self.bulk_edges, self.periodic_edges = self.construct_edges()
        self.interior_V_num = np.arange(self.num_Vs)

        self.E_lengths_by_v_num, self.wadjacency_matrices = self.construct_NEP_data()

        self.g_coords = self.construct_g_coords()

    def construct_g_coords(self, fixed_num_points=False, **kwargs):

        if fixed_num_points: 
            try: num_points = kwargs["num_points"]
            except: num_points = 33
            calculate_points_per_edge = lambda _: num_points
        else: 
            # Make sure the discretization size is roughly equal on each edge.
            # Also make sure that there is a minimum of one discretization point
            # on each edge
            min_l_vw = np.min(np.array(list(self.E_lengths_by_v_num.values())))
            num_points = 9 / min_l_vw
            calculate_points_per_edge = lambda l: min(max(9, int(l * num_points)), 33)

        g_coords = []

        for edge in self.edges:
            l_vw = edge["l_vw"]
            v_coord, w_coord = edge["vw_coords"]

            points_per_edge = calculate_points_per_edge(l_vw)
            x = np.linspace(0, 1, points_per_edge)
            l = (x * w_coord[:, np.newaxis] + (1 - x) * v_coord[:, np.newaxis])
            g_coords.append(l)

        return g_coords

    def construct_NEP_data(self):

        E_lengths_by_v_num = {}
        bulk_wadjacency_matrix = scipy.sparse.lil_matrix((self.num_Vs, self.num_Vs))
        for e in self.bulk_edges:
            v, w = e["vw"]
            E_lengths_by_v_num[v, w] = e["l_vw"]
            bulk_wadjacency_matrix[v, w] = e["l_vw"]
            bulk_wadjacency_matrix[w, v] = e["l_vw"]

        bulk_wadjacency_matrix = bulk_wadjacency_matrix.tocsc()

        periodic_wadjacency_matrices = []
        for e in self.periodic_edges:
            periodic_wadjacency_matrix = scipy.sparse.lil_matrix((self.num_Vs, self.num_Vs))
            v, w = e["vw"]
            E_lengths_by_v_num[v, w] = e["l_vw"]
            periodic_wadjacency_matrix[v, w] = e["l_vw"]
            periodic_wadjacency_matrix[w, v] = e["l_vw"]
            periodic_wadjacency_matrix = periodic_wadjacency_matrix.tocsc()
            periodic_wadjacency_matrices.append(periodic_wadjacency_matrix)

        wadjacency_matrices = [bulk_wadjacency_matrix] + periodic_wadjacency_matrices

        return E_lengths_by_v_num, wadjacency_matrices

    def construct_L(self, k, deriv=False):

        if not deriv:
            calculate_csc = self.calculate_csc
            calculate_cot = self.calculate_cot
        else:
            calculate_csc = self.calculate_dcsc
            calculate_cot = self.calculate_dcot

        L = scipy.sparse.csr_matrix((self.num_Vs, self.num_Vs))
        for wadjacency_matrix in self.wadjacency_matrices:
            matrix_csc = wadjacency_matrix.copy()
            matrix_csc.data = calculate_csc(k, matrix_csc.data)

            matrix_cot = wadjacency_matrix.copy()
            matrix_cot.data = calculate_cot(k, matrix_cot.data)
            matrix_cot = scipy.sparse.diags(matrix_cot.sum(axis=0).flat)

            L += (matrix_cot - matrix_csc).tocsr()[self.interior_V_num[:, None], self.interior_V_num]

        L = L.tocsc()

        return L

    def construct_edges(self):

        bulk_edges = []
        periodic_edges = []
        for v, w in self.total_graph.total_edges:
            v_isin_bulk = v < self.total_graph.num_Vs
            w_isin_bulk = w < self.total_graph.num_Vs

            if v_isin_bulk and w_isin_bulk:
                v_coords = self.total_graph.total_V_coords[v]
                w_coords = self.total_graph.total_V_coords[w]
                r_xy = np.array([0, 0])
                l_vw = np.linalg.norm(v_coords - w_coords)

                edge = {"vw": (v, w),
                        "l_vw": l_vw, 
                        "r_xy": r_xy,
                        "vw_coords": (v_coords, w_coords)}
                bulk_edges.append(edge)
                
            elif v_isin_bulk and not w_isin_bulk:
                v_coords = self.total_graph.total_V_coords[v]
                w_coords = self.total_graph.total_V_coords[w]

                # Calculate r_xy. v_patch_coords is assumed np.array([0, 0]).
                wrapped_w_coords = w_coords % self.box
                w_patch_coords = w_coords - wrapped_w_coords

                r_xy = w_patch_coords 
                distances_to_bulk_Vs = np.linalg.norm(self.V_coords - wrapped_w_coords, axis=1)
                w = np.argmin(distances_to_bulk_Vs)
                w_coords = self.V_coords[w] + r_xy
                l_vw = np.linalg.norm(v_coords - w_coords)

                edge = {"vw": (v, w),
                        "l_vw": l_vw, 
                        "r_xy": r_xy,
                        "vw_coords": (v_coords, w_coords)}
                periodic_edges.append(edge)

        periodic_edges = self.prune_edges(periodic_edges)

        edges = bulk_edges + periodic_edges

        return edges, bulk_edges, periodic_edges
    
    def prune_edges(self, edges_to_be_pruned):

        if self.total_graph.totally_periodic:
            edges = []
            for e in edges_to_be_pruned:
                v, w = e["vw"]
                if v < w:
                    edges.append(e.copy())
        else:
            edges = []
            for e in edges_to_be_pruned:
                if np.random.choice(2):
                    edges.append(e.copy())

        return edges
    
class Patch_from_Graph:

    def __init__(self, g, homogenized_V_coord, bounding_box_corners, box_rotation):

        (self.total_V_coords, 
        self.total_edges, 
        self.num_Vs) = self.extract_patch_graph(g, homogenized_V_coord, bounding_box_corners, box_rotation)

        self.totally_periodic = False
    
    def extract_patch_graph(self, g, homogenized_V_coord, bounding_box_corners, box_rotation):

        # Find g.V_coords within bounding_box_corners
        box_path = matplotlib.path.Path(bounding_box_corners)
        inside_mask = box_path.contains_points(g.V_coords, radius=1e-10)
        old_patch_V_inds = np.argwhere(inside_mask).flatten()

        # Find g.edges_by_v_num connected to these vertices
        edges_by_v_num = []
        for v, w in g.edges_by_v_num:
            if v in old_patch_V_inds or w in old_patch_V_inds:
                edges_by_v_num.append([v, w])

        # Reorder so that vertices within box are at the start of patch_V_coords
        unique_old_v_num = np.unique(edges_by_v_num)
        reorder_old_v_num = []
        for v in unique_old_v_num:
            if v in old_patch_V_inds:
                reorder_old_v_num = [v] + reorder_old_v_num
            else:
                reorder_old_v_num.append(v)

        # Construct old to new vertex index map
        old_to_new_map = {i: j for j, i in enumerate(reorder_old_v_num)}

        # Construct patch V_coords and edges_by_v_num
        patch_V_coords = g.V_coords[reorder_old_v_num]
        patch_edges_by_v_num = []
        for v, w in edges_by_v_num:
            v_new, w_new = old_to_new_map[v], old_to_new_map[w]
            patch_edges_by_v_num.append([v_new, w_new])

        # Number of vertices within box
        num_Vs = len(old_patch_V_inds)

        # Translate, rotate and rescale patch to have lower left corner at origin in unit box
        patch_V_coords = patch_V_coords.copy() 
        patch_V_coords -= homogenized_V_coord
        patch_V_coords = (box_rotation.T @ patch_V_coords.T).T
        box_side_length = np.linalg.norm(bounding_box_corners[0] - bounding_box_corners[1])
        patch_V_coords += np.array([box_side_length / 2, box_side_length / 2])
        patch_V_coords /= box_side_length

        return patch_V_coords, patch_edges_by_v_num, num_Vs
    
class Cell_Problem:

    def __init__(self, patch, rescale=1):

        self.patch = patch
        self.rescale = rescale # for different patch size lengths

    def construct_homogenized_tensor(self, xi):

        Q = np.zeros((2, 2))
        T = 0

        for e in self.patch.edges:
            v, w = e["vw"]
            l_vw = self.rescale * e["l_vw"]
            r_xy = self.rescale * e["r_xy"]

            a = r_xy + xi[w] - xi[v]

            Q += np.tensordot(a, a, axes=0) / l_vw
            T += l_vw

        c = np.trace(Q / T)

        return c, Q, T

    def solve_corrector_equation(self):

        LHS, RHS = self.construct_equation()
        # xi = sparseqr.solve(LHS, RHS)
        xi_0 = scipy.sparse.linalg.cg(LHS, RHS[:, 0].A, rtol=1e-14)[0]
        xi_1 = scipy.sparse.linalg.cg(LHS, RHS[:, 1].A, rtol=1e-14)[0]
        xi = np.vstack((xi_0, xi_1)).T

        residual = np.linalg.norm(LHS @ xi - RHS)
        if residual > 1e-10:
            print(f"Residual too large = {residual}")
            # return ValueError(f"Residual too large = {residual}")

        return xi

    def construct_equation(self):

        LHS = scipy.sparse.lil_matrix((self.patch.num_Vs, self.patch.num_Vs))
        RHS = scipy.sparse.lil_matrix((self.patch.num_Vs, 2))

        for e in self.patch.edges:
            v, w = e["vw"]
            l_vw = self.rescale * e["l_vw"] 
            r_xy = self.rescale * e["r_xy"]
            weight = 1 / l_vw

            LHS[v, v] += weight
            LHS[w, w] += weight

            LHS[v, w] -= weight
            LHS[w, v] -= weight

            RHS[v] += weight * r_xy
            RHS[w] -= weight * r_xy

        LHS = LHS.tocsc()
        RHS = RHS.tocsc()

        return LHS, RHS

class Truncated_Square:

    def __init__(self):

        self.total_num_Vs = 8
        self.total_V_coords, self.num_Vs = self.construct_total_V_coords()
        self.total_edges = self.construct_total_edges()
        self.totally_periodic = True

    def construct_total_V_coords(self):

        bulk_V_coords = np.array([[1 / 2 + np.sqrt(2) / 2, 1 / 2 + np.sqrt(2)],
                                  [1 / 2 + np.sqrt(2) / 2, 1 / 2],
                                  [1 / 2 + np.sqrt(2), 1 / 2 + np.sqrt(2) / 2],
                                  [1 / 2, 1 / 2 + np.sqrt(2) / 2]]) / (1 + np.sqrt(2))
        
        boundary_V_coords = np.array((bulk_V_coords[1] + np.array([0, 1]),
                                      bulk_V_coords[0] - np.array([0, 1]),
                                      bulk_V_coords[3] + np.array([1, 0]),
                                      bulk_V_coords[2] - np.array([1, 0])))
        
        num_Vs = 4
        total_V_coords = np.vstack((bulk_V_coords, boundary_V_coords))

        return total_V_coords, num_Vs

    def construct_total_edges(self):

        total_edges = np.array(([0, 2],
                                [0, 3],
                                [1, 2],
                                [1, 3],
                                [0, 4],
                                [1, 5],
                                [2, 6],
                                [3, 7]))
        
        boundary_boundary_edge_for_testing = np.array([6, 7])

        total_edges = np.vstack((total_edges, boundary_boundary_edge_for_testing))
        
        return total_edges
    
class Random_Delaunay:

    def __init__(self, num_Vs):

        self.num_Vs = num_Vs
        self.total_V_coords, self.total_num_Vs = self.construct_total_V_coords()
        self.total_edges = self.construct_total_edges()

        self.totally_periodic = False

    def construct_total_V_coords(self):

        num_Vs = 0
        bulk_V_coords = []
        boundary_V_coords = []
        while num_Vs < self.num_Vs:
            v_coords_x, v_coords_y = np.random.uniform(-1, 2, size=(2))
            v_coords = np.array([v_coords_x, v_coords_y])
            if (0 < v_coords_x < 1) and (0 < v_coords_y < 1):
                num_Vs += 1
                bulk_V_coords.append(v_coords)
            else:
                boundary_V_coords.append(v_coords)
        
        total_V_coords = np.vstack((bulk_V_coords, boundary_V_coords))
        total_num_Vs = total_V_coords.shape[0]

        return total_V_coords, total_num_Vs
    
    def construct_total_edges(self):

        triangulation = scipy.spatial.Delaunay(self.total_V_coords)
        V, W = triangulation.vertex_neighbor_vertices
        total_edges = []
        for v in range(self.total_num_Vs):
            w_inds = W[V[v]:V[v + 1]]
            for w in w_inds:
                if v < w:
                    total_edges.append([v, w])
        total_edges = np.array(total_edges)
        total_edges = np.sort(total_edges, axis=1)

        return total_edges
    
class Random_Voronoi:

    def __init__(self, N):

        self.total_num_Vs = N
        self.num_Vs, self.total_V_coords, self.total_edges = self.construct_graph_data()

        self.totally_periodic = False

    def construct_graph_data(self):

        x = np.random.uniform(-1, 2, self.total_num_Vs)
        y = np.random.uniform(-1, 2, self.total_num_Vs)

        delaunay_V_coords = np.vstack((x, y)).T

        delaunay_triangulation = scipy.spatial.Delaunay(delaunay_V_coords)
        triangles = delaunay_triangulation.simplices
        triangles = np.sort(triangles, axis=1)

        voronoi_V_coords = np.mean(delaunay_V_coords[triangles], axis=1)

        delaunay_edges = collections.defaultdict(list)
        for i, triangle in enumerate(triangles):
            for v, w in [(triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[0], triangle[2])]:
                delaunay_edges[(v, w)].append(i)

        voronoi_edges = []
        for H_edge in list(delaunay_edges.values()):
            if len(H_edge) != 1:
                voronoi_edges.append(H_edge)
        voronoi_edges = np.array(voronoi_edges)

        # Step 1: Determine which vertices are inside the unit square
        in_unit_square = np.all((voronoi_V_coords >= 0) & (voronoi_V_coords <= 1), axis=1)
        num_Vs = np.sum(in_unit_square)

        # Step 2: Get the indices that will sort the array with in-square vertices first
        in_indices = np.where(in_unit_square)[0]
        out_indices = np.where(~in_unit_square)[0]
        new_order = np.concatenate([in_indices, out_indices])

        # Step 3: Reorder V_coords
        voronoi_V_coords = voronoi_V_coords[new_order]

        # Step 4: Build a mapping from old indices to new indices
        index_map = np.zeros_like(new_order)
        index_map[new_order] = np.arange(len(new_order))

        # Step 5: Remap edges
        voronoi_edges = index_map[voronoi_edges]

        return num_Vs, voronoi_V_coords, voronoi_edges
    
class RGG:

    def __init__(self, num_Vs):

        self.num_Vs = num_Vs
        self.total_V_coords, self.total_num_Vs = self.construct_total_V_coords()
        self.total_edges = self.construct_total_edges()

        self.totally_periodic = False

    def construct_total_V_coords(self):

        num_Vs = 0
        bulk_V_coords = []
        boundary_V_coords = []
        while num_Vs < self.num_Vs:
            v_coords_x, v_coords_y = np.random.uniform(-1, 2, size=(2))
            v_coords = np.array([v_coords_x, v_coords_y])
            if (0 < v_coords_x < 1) and (0 < v_coords_y < 1):
                num_Vs += 1
                bulk_V_coords.append(v_coords)
            else:
                boundary_V_coords.append(v_coords)
        
        total_V_coords = np.vstack((bulk_V_coords, boundary_V_coords))
        total_num_Vs = total_V_coords.shape[0]

        return total_V_coords, total_num_Vs
    
    def construct_total_edges(self):

        tree = scipy.spatial.cKDTree(self.total_V_coords)

        max_coord = np.max(np.abs(self.total_V_coords))
        min_coord = np.min(np.abs(self.total_V_coords))
        mean_edge_length = np.abs(max_coord - min_coord) / np.sqrt(self.total_num_Vs)
        mean_edge_scaling_for_connection_radius = 3
        connection_radius = mean_edge_scaling_for_connection_radius * mean_edge_length

        total_edges = np.array(list(tree.query_pairs(r=connection_radius)))
        total_edges = np.sort(total_edges, axis=1)

        return total_edges
    
class Aperiodic_Monotile:

    def __init__(self, N, start_point=np.array([0, 0])):

        self.total_V_coords, self.total_edges, self.num_Vs = self.construct_V_coords(N, start_point)

        self.totally_periodic = False

    def construct_V_coords(self, N, start_point):

        script_dir = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(script_dir, "aperiodic_monotile_data", "V_coords.pkl"), "rb") as file:
            full_V_coords = pickle.load(file)

        with open(os.path.join(script_dir, "aperiodic_monotile_data", "E_lengths_by_v_num.pkl"), "rb") as file:
            full_E_lengths_by_v_num = pickle.load(file)

        translate = np.array([0.1, -50.1]) - start_point
        full_V_coords += translate

        self.original_coords = full_V_coords

        bulk_mask = ((0 < full_V_coords[:, 0]) & (0 < full_V_coords[:, 1]) 
                     & (full_V_coords[:, 0] < N) & (full_V_coords[:, 1] < N))

        full_V_coords = np.vstack((full_V_coords[bulk_mask], full_V_coords[~bulk_mask])) / N

        old_inds = np.concatenate((np.argwhere(bulk_mask).flatten(), np.argwhere(~bulk_mask).flatten()))
        new_inds = np.arange(full_V_coords.shape[0])
        remap = {i: j for i, j in zip(old_inds, new_inds)}

        old_edges = np.array(list(full_E_lengths_by_v_num.keys()))
        edges = []
        for v_ind, w_ind in old_edges:
            edges.append([remap[v_ind], remap[w_ind]])
        edges = np.array(edges)
        edges = np.sort(edges, axis=1)

        num_Vs = np.sum(bulk_mask)

        bulk_edges = []
        for i, j in edges:
            if i < num_Vs and j < num_Vs:
                bulk_edges.append([i, j])
        bulk_V_inds = np.unique(bulk_edges)
        disconnected_V_inds = set(np.arange(num_Vs)) - set(bulk_V_inds)
        if len(disconnected_V_inds) > 0:
            all_ok_inds = set(np.arange(full_V_coords.shape[0])) - disconnected_V_inds
            old_inds = np.concatenate((np.array(list(all_ok_inds)), np.array(list(disconnected_V_inds))))
            full_V_coords = full_V_coords[old_inds]
            new_inds = np.arange(full_V_coords.shape[0])
            remap = {i: j for i, j in zip(old_inds, new_inds)}
            new_edges = []
            for v_ind, w_ind in edges:
                new_edges.append([remap[v_ind], remap[w_ind]])
            new_edges = np.array(new_edges)
            edges = np.sort(new_edges, axis=1)
            num_Vs -= len(disconnected_V_inds)

        return full_V_coords, edges, num_Vs