import numpy as np
import scipy
from construct_graph.graph import Graph, Flat

class Random_Annulus(Graph, Flat):

    def __init__(self, num_Vs, radial_density, angular_density, angular_m, fixed_num_points=False, num_points=33):

        self.num_Vs = num_Vs
        self.radial_density = radial_density
        self.angular_density = angular_density
        self.angular_m = angular_m
        
        (self.V_coords, 
        self.edges_by_v_num, 
        self.interior_V_num, 
        self.boundary_V_num) = self.construct_edges()

        self.edges, self.E_lengths_by_v_num = self.construct_edge_data()
        self.wadjacency_matrix = self.construct_wadjacency_matrix()
        self.g_coords = self.construct_g_coords(fixed_num_points=fixed_num_points, num_points=num_points)

    def construct_edge_data(self):

        edges = []
        E_lengths_by_v_num = {}
        radius_to_length_aspect_ratio = 0.1
        mean_lvw = 1e-0#0.005696690975006936
        for v, w, in self.edges_by_v_num:
            l_vw = np.linalg.norm(self.V_coords[v] - self.V_coords[w])/mean_lvw
            E_lengths_by_v_num[v, w] = l_vw
            edge = {"vw": (v, w),
                    "l_vw": l_vw,
                    "vw_coords": np.array((self.V_coords[v], self.V_coords[w])),
                    "K": (radius_to_length_aspect_ratio * l_vw)**4,
                    "C": (radius_to_length_aspect_ratio * l_vw)**2,
                    "C/K": 1 / (radius_to_length_aspect_ratio * l_vw)**2,
                    "l_K": l_vw / (radius_to_length_aspect_ratio * l_vw)**4,
                    "l_C": l_vw * (radius_to_length_aspect_ratio * l_vw)**2}
            edges.append(edge)

        return edges, E_lengths_by_v_num
    
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

    def construct_edges(self):

        mu = lambda r, phi: self.radial_density(r) * self.angular_density(self.angular_m, phi)

        max_r, max_phi = 1, 0
        max_mu = mu(max_r, max_phi)
        V_coords = self.sample(self.num_Vs, mu, max_mu)

        # # For adding in inner rim
        # num_inner_rim = 5
        # inner_rim = 0.1 * np.array((np.cos(np.arange(num_inner_rim) * 2 * np.pi / num_inner_rim), 
        #                             np.sin(np.arange(num_inner_rim) * 2 * np.pi / num_inner_rim))).T

        # V_coords = np.vstack((inner_rim, V_coords[:-num_inner_rim]))

        V_coords = np.vstack((V_coords, np.array([0, 0])))
        triangulation = scipy.spatial.Delaunay(V_coords)
        V_coords = V_coords[:-1]
        
        V, W = triangulation.vertex_neighbor_vertices

        edges = []
        inner_boundary_V_num = []

        for v in range(self.num_Vs):
            w_inds = W[V[v]:V[v + 1]]
            for w in w_inds:
                if v < w:
                    if w > self.num_Vs - 1:
                        inner_boundary_V_num.append(v)
                    else:
                        edges.append([v, w])
        
        outer_boundary_V_num = np.unique(triangulation.convex_hull)

        boundary_V_num = np.concatenate([inner_boundary_V_num, outer_boundary_V_num])

        edges = np.array(edges)
        
        interior_V_num = np.setdiff1d(np.arange(self.num_Vs), boundary_V_num) 

        return V_coords, edges, interior_V_num, boundary_V_num
    
    def sample(self, num_samples, mu, max_mu, r_min=0.1, r_max=1):
        """Rejection method for sampling from distribution mu
        """

        # Sampling parameters
        samples = []
        n_accepted = 0

        while n_accepted < num_samples:
            # Sample r and φ uniformly over the annulus
            r_try = np.sqrt(np.random.uniform(r_min**2, r_max**2))  # area-correct sampling
            phi_try = np.random.uniform(0, 2 * np.pi)
            
            # Uniform sample for rejection
            u = np.random.uniform(0, max_mu)
            
            if u < mu(r_try, phi_try):
                samples.append((r_try * np.cos(phi_try), r_try * np.sin(phi_try)))
                n_accepted += 1

        # Convert to numpy array
        samples = np.array(samples)

        return samples