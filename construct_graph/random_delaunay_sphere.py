import numpy as np
import scipy
from collections import defaultdict
from construct_graph.graph import Graph, Spherical

class Random_Delaunay_Sphere(Graph, Spherical):

    """Parent class of more specific graph classes (spiderweb, delaunay triangulations etc.)
    These child classes need the attributes:
        - num_Vs DONE
        - V_coords DONE
        - E_lengths_by_v_num DONE
        - interior_V_num DONE
        - wadjacency_matrix DONE
    """

    def __init__(self, num_Vs):

        self.num_Vs = num_Vs
        self.construct_data()
        self.edges = self.construct_edges()
        self.g_coords = self.construct_g_coords()

    def construct_edges(self):

        edges = []

        for (v, w), l_vw in self.E_lengths_by_v_num.items():
            edges.append({"vw": (v, w), "l_vw": l_vw})

        return edges

    def construct_data(self):

        V_coords = np.random.normal(0, 1, size=(self.num_Vs, 3))
        V_coords /= np.linalg.norm(V_coords, axis=1, keepdims=True)

        # Construct edge set by Voronoi tesselation
        radius = 1
        center = np.array([0, 0, 0])
        sv = scipy.spatial.SphericalVoronoi(V_coords, radius, center)

        # Iterate over each Voronoi vertex
        vertex_to_regions = defaultdict(list)

        # Map each Voronoi vertex to the regions it belongs to
        for point_idx, region in enumerate(sv.regions):
            for vertex in region:
                vertex_to_regions[vertex].append(point_idx)

        edges = set()
        for region in vertex_to_regions.values():
            if len(region) != 3:
                raise ValueError(("Assumption that deg(Voronoi vertex) = 3 has been broken." 
                                  + "Probably not a problem, but there will be non-triangular faces in graph."))
            new_edges = [(region[0], region[1]), (region[1], region[2]), (region[0], region[2])]
            for edge in new_edges:
                edges.add(edge)

        E_lengths_by_v_num = {}
        for v_ind, w_ind in edges:
            E_lengths_by_v_num[v_ind, w_ind] = self.length_on_sphere(V_coords[v_ind], 
                                                                      V_coords[w_ind])
        
        # # Old (MUCH slower) method: Construct edge set by Delaunay triangulation
        # V_coords = np.vstack((V_coords, np.array(([0, 0, 0]))))

        # triangulation = scipy.spatial.Delaunay(V_coords)
        # self.triangulation = triangulation
        # print("triangulation built")

        # # Construct edges
        # triangulation = scipy.spatial.Delaunay(V_coords)
        # V, W = triangulation.vertex_neighbor_vertices
        # E_lengths_by_v_num = {}
        # for v_ind in range(self.num_Vs):
        #     w_inds = W[V[v_ind]:V[v_ind + 1]]
        #     for w_ind in w_inds:
        #         if (v_ind < w_ind) and (w_ind != self.num_Vs):
        #             E_lengths_by_v_num[v_ind, w_ind] = self.length_on_sphere(V_coords[v_ind], 
        #                                                                      V_coords[w_ind])
                    
        wadjacency_matrix = scipy.sparse.lil_matrix((self.num_Vs, self.num_Vs))
        for v_ind, w_ind in E_lengths_by_v_num:
            wadjacency_matrix[v_ind, w_ind] = E_lengths_by_v_num[v_ind, w_ind]
            wadjacency_matrix[w_ind, v_ind] = E_lengths_by_v_num[v_ind, w_ind]
        wadjacency_matrix = wadjacency_matrix.tocsc()

        self.V_coords = V_coords
        self.E_lengths_by_v_num = E_lengths_by_v_num
        self.interior_V_num = np.arange(self.num_Vs)
        self.wadjacency_matrix = wadjacency_matrix

    def length_on_sphere(self, p1, p2, r=1):

        x1, y1, z1 = p1
        x2, y2, z2 = p2

        # Calculate the central angle between the two points
        dot_product = x1 * x2 + y1 * y2 + z1 * z2
        cos_theta = dot_product / (r**2)
        theta = np.arccos(np.clip(cos_theta, -1, 1))

        # Calculate the arc length
        arc_length = r * theta

        return arc_length