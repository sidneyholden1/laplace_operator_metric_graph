import numpy as np 
import scipy
import pickle
from scipy.interpolate import RegularGridInterpolator

class Eigenvalue_Calculator:

    def __init__(self, g):

        self.num_interior_Vs = len(g.interior_V_num)
        self.L = lambda k: g.construct_L(k) 
        self.dL = lambda k: g.construct_L(k, deriv=True)
        self.type = str(type(g))
        if self.type == "<class 'construct_graph.spiderweb.Spiderweb'>":
            self.num_radial_Vs = g.num_radial_Vs

    def __call__(self, k, tol=1e-14, max_steps=1000, printerval=10, **kwargs):

        k = np.array([k]).flatten().astype(np.float64)

        eigenvalues = []
        for i in range(len(k)):
            if printerval != np.inf:
                print(f"\nCalculating eig number {i}\n")
            eigenvalues.append(self.run_Newton_iteration(k[i], tol=tol, max_steps=max_steps, printerval=printerval, **kwargs))

        return eigenvalues #np.array(eigenvalues)

    def calculate_SVD_iterate(self, k, u=None, tol=1e-10):

        Lk = self.L(k)
        dLk = self.dL(k)

        if u is not None:
            try:
                LU = scipy.sparse.linalg.splu(Lk)
        
                x = LU.solve(u)
                v = x / np.sqrt(np.linalg.norm(x))
                s = (u.T @ Lk @ v)[0, 0]
            except RuntimeError as e:
                if str(e) == "Factor is exactly singular":
                    print("Custom warning: " + str(e))
                    return np.nan, u

        else:
            if self.type == "<class 'construct_graph.spiderweb.Spiderweb'>":
                v0 = np.ones(self.num_radial_Vs)[:, None]
                s0 = 0
                s, u = scipy.sparse.linalg.eigs(Lk, k=1, sigma=s0, v0=v0, tol=tol)
            else:
                v0 = np.ones(self.num_interior_Vs)[:, None]
                s0 = 0
                try:
                    s, u = scipy.sparse.linalg.eigsh(Lk, k=1, sigma=s0, v0=v0, tol=tol)
                except RuntimeError as e:
                    if str(e) == "Factor is exactly singular":
                        print(str(e))
                        return 0, u
                    print(str(e))
                except Exception as e:
                    print(str(e))
                    
            s = np.real(np.abs(s[0]))
            u = np.real(u)
            v = u.copy()

        k_next = k - s / (u.T @ dLk @ v)[0, 0]

        return k_next, u

    def estimate_Newton_trace(self, k, num_vectors=20, estimation_type="Rademacher"):

        Lk = self.L(k)
        try:
            LU = scipy.sparse.linalg.splu(Lk)
            dLk = self.dL(k)

            if self.type == "<class 'construct_graph.spiderweb.Spiderweb'>":
                rand_vec_size = np.shape(Lk)[0]
            else: 
                rand_vec_size = self.num_interior_Vs

            if estimation_type=="Gaussian":
                rand_vec = lambda _: np.random.normal(0, 1, size=(rand_vec_size, 1))
            elif estimation_type=="Rademacher":
                rand_vec = lambda _: np.random.choice((-1, 1), size=(rand_vec_size, 1))

            tr = 0
            for _ in range(num_vectors):
                u = rand_vec(None)
                left = LU.solve(u)
                right = dLk @ u
                tr += (left.T @ right)[0, 0]

            return k - 1 / (tr / num_vectors)
        except RuntimeError as e:
            if str(e) == "Factor is exactly singular":
                return k
            print(str(e))
        except:
            print(str(e))
    
    def run_Newton_iteration(self, k, tol=1e-14, max_steps=1000, printerval=10, **kwargs):
        """
        """
        err = 1
        count = 0 

        if kwargs["solve_type"] == "SVD iterate":
            k_next, u = self.calculate_SVD_iterate(k, tol=tol)

        while err > tol and count < max_steps:

            if ((count + 1) % printerval) == 0:
                print(f"Count = {count + 1}")
                print(f"k = {k}\n")
            
            if kwargs["solve_type"] == "Newton trace estimation":
                k_next = self.estimate_Newton_trace(k, num_vectors=kwargs["num_vectors"], 
                                                    estimation_type=kwargs["estimation_type"])

            err = np.abs(k_next - k)

            if err < tol:
                if printerval != np.inf:
                    print(22*"-")
                    print(f"Converged at step {count}:\n")
                    print(f"k = {k_next}")
                    print(22*"-")
                return k_next, u
            
            k = k_next

            if kwargs["solve_type"] == "SVD iterate":
                k_next, u = self.calculate_SVD_iterate(k, u=u, tol=tol)

            count += 1

            if k == np.nan:
                return k

        return np.nan

class Graph_Function:

    def __init__(self, data, domain=None):

        self.data = data
        if domain is not None:
            self.domain = [edge_domain.copy() for edge_domain in domain]

    def __add__(self, other):
        
        if np.isscalar(other):
            result = [i + other for i in self.data]
            return Graph_Function(result, self.domain)
        
        elif isinstance(other, Graph_Function):
            result = [i + j for i, j in zip(self.data, other.data)]
            return Graph_Function(result, self.domain)
        
        else:
            return NotImplemented
        
    def __radd__(self, other):

        return self.__add__(other)

    def __sub__(self, other):

        if not isinstance(other, Graph_Function):
            return NotImplemented
        
        result = [i - j for i, j in zip(self.data, other.data)]

        return Graph_Function(result, self.domain)

    def __mul__(self, other):

        if isinstance(other, Graph_Function):
            result = [i * j for i, j in zip(self.data, other.data)]

        elif np.isscalar(other):  
            result = [arr * other for arr in self.data]

        else:
            return NotImplemented
        
        return Graph_Function(result, self.domain)

    def __rmul__(self, other):

        return self.__mul__(other)

    def __truediv__(self, other):

        if isinstance(other, Graph_Function):
            result = [i / j for i, j in zip(self.data, other.data)]

        elif np.isscalar(other):  
            result = [arr / other for arr in self.data]

        else:
            return NotImplemented
        
        return Graph_Function(result, self.domain)

    def __rtruediv__(self, other):

        if np.isscalar(other):
            result = [other / arr for arr in self.data]

        else:
            return NotImplemented
        
        return Graph_Function(result, self.domain)

    def __eq__(self, other):

        if not isinstance(other, Graph_Function):
            return NotImplemented
        
        return all(np.array_equal(i, j) for i, j in zip(self.data, other.data))

    def __repr__(self):

        return f"Graph_Function({self.data})"
    
    def norm(self):

        return np.sqrt(self.dot(self))
    
    def normalize(self):

        norm = self.norm()

        if norm == 0:
            raise ValueError("Cannot normalize a zero vector.")

        result = [arr / norm for arr in self.data]

        return Graph_Function(result, self.domain)
    
    def dot(self, other):
        
        if not isinstance(other, Graph_Function):
            return NotImplemented
        
        if (self.domain is None) and not (other.domain is None):
            self.domain = other.domain
        elif not (self.domain is None) and (other.domain is None):
            other.domain = self.domain
        elif (self.domain is None) and (other.domain is None):
            raise ValueError("Graph_Functions need domain attributes.")
        
        graph_inner_product = 0

        for f0_edge, f1_edge, edge in zip(self.data, other.data, self.domain):
            edge_length = np.linalg.norm([edge[0, 0] - edge[0, -1], edge[1, 0] - edge[1, -1]])
            edge_param = np.linspace(0, edge_length, edge.shape[1])
            graph_inner_product += scipy.integrate.trapezoid(f0_edge * f1_edge, edge_param)

        return graph_inner_product
    
class Graph_Eigenfunctions:

    def __init__(self, g, eigenvector_pairs, pde=False):

        dict_eigenvector_pairs = {"eigenvalues": [pair[0] for pair in eigenvector_pairs], 
                                  "eigenvectors": [np.zeros((g.num_Vs, 1)) for _ in eigenvector_pairs]}
        for en in range(len(eigenvector_pairs)):
            if not pde:
                dict_eigenvector_pairs["eigenvectors"][en][g.interior_V_num] = eigenvector_pairs[en][1]
            else:
                dict_eigenvector_pairs["eigenvectors"][en] = eigenvector_pairs[en][1]

        self.eigenvector_pairs = dict_eigenvector_pairs
        self.eigenfunction_pairs = self.construct_graph_eigenfunctions(g)

    def construct_graph_eigenfunctions(self, g):

        eigenfunction_pairs = {"eigenvalues": [], 
                               "eigenfunctions": []}

        for k, eigenvector in zip(self.eigenvector_pairs["eigenvalues"],
                                  self.eigenvector_pairs["eigenvectors"]):

            eigenfunction_pairs["eigenvalues"].append(k)
            eigenfunction_pairs["eigenfunctions"].append(self.construct_graph_eigenfunction(g, k, eigenvector))

        return eigenfunction_pairs
    
    def construct_graph_eigenfunction(self, g, eigenvalue, eigenvector):

        graph_eigenfunction = []

        if np.abs(eigenvalue) < 1e-10:
            for e_num, _ in enumerate(g.edges):
                edge_mode = np.ones(g.g_coords[e_num].shape[1])
                graph_eigenfunction.append(edge_mode)
        else:
            for e_num, edge in enumerate(g.edges):
                v, w = edge["vw"]
                l_vw = edge["l_vw"]
                parametrised_edge = np.linspace(0, l_vw, g.g_coords[e_num].shape[1])

                edge_mode = ((eigenvector[v] * np.sin(eigenvalue * parametrised_edge[::-1])
                              + eigenvector[w] * np.sin(eigenvalue * parametrised_edge)) 
                              / np.sin(eigenvalue * l_vw))
                
                graph_eigenfunction.append(edge_mode)

        graph_eigenfunction = Graph_Function(graph_eigenfunction, g.g_coords).normalize()

        return graph_eigenfunction
    
class Projector_Graph_Eigenfunctions:

    def __init__(self, g, eigenvalues):

        self.eigenvector_pairs = self.calculate_graph_eigenvectors(g, eigenvalues)
        self.eigenfunction_pairs = self.construct_graph_eigenfunctions(g)

    def construct_graph_eigenfunctions(self, g):

        eigenfunction_pairs = {"eigenvalues": [], 
                               "eigenfunctions": []}

        for k, eigenvector in zip(self.eigenvector_pairs["eigenvalues"],
                                  self.eigenvector_pairs["eigenvectors"]):

            eigenfunction_pairs["eigenvalues"].append(k**2)
            eigenfunction_pairs["eigenfunctions"].append(self.construct_graph_eigenfunction(g, k, eigenvector))

        return eigenfunction_pairs

    def calculate_graph_eigenvectors(self, g, eigenvalues):

        eigenvector_pairs = {"eigenvalues": [], 
                             "eigenvectors": []}

        for k in eigenvalues:

            L = g.construct_L(k)

            # eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(L, k=5, which='SM')
            # null_eigenvalues = np.argwhere(np.abs(eigenvalues) < 1e-8).flatten()

            eigs = Eigenvalue_Calculator(g)
            a = eigs(k, solve_type="SVD iterate", printerval=np.inf, tol=1e-9)
            eigenvectors = np.hstack(([i[1] for i in a]))

            residuals = np.linalg.norm(L @ eigenvectors, axis=0)
            bad = np.argwhere(residuals > 1e-8)
            if bad.size > 0:
                raise ValueError("No convergence to null vector (NEP eigenvector).")

            null_eigenvalues = np.arange(eigenvectors.shape[1])
            dim_null_space = len(null_eigenvalues)
            null_vectors = np.zeros((g.num_Vs, dim_null_space))
            for en, null_ind in enumerate(null_eigenvalues):
                null_vectors[g.interior_V_num, en] = eigenvectors[:, null_ind]

            for en in range(dim_null_space):
                eigenvector_pairs["eigenvalues"].append(k)
                eigenvector_pairs["eigenvectors"].append(null_vectors[:, en])

        return eigenvector_pairs
    
    def construct_graph_eigenfunction(self, g, eigenvalue, eigenvector):

        graph_eigenfunction = []

        if np.abs(eigenvalue) < 1e-10:
            for e_num, _ in enumerate(g.edges):
                edge_mode = np.ones(g.g_coords[e_num].shape[1])
                graph_eigenfunction.append(edge_mode)
        else:
            for e_num, edge in enumerate(g.edges):
                v, w = edge["vw"]
                l_vw = edge["l_vw"]
                parametrised_edge = np.linspace(0, l_vw, g.g_coords[e_num].shape[1])

                edge_mode = ((eigenvector[v] * np.sin(eigenvalue * parametrised_edge[::-1])
                              + eigenvector[w] * np.sin(eigenvalue * parametrised_edge)) 
                              / np.sin(eigenvalue * l_vw))
                
                graph_eigenfunction.append(edge_mode)

        graph_eigenfunction = Graph_Function(graph_eigenfunction, g.g_coords).normalize()

        return graph_eigenfunction
    
class PDE_Eigenfunctions:

    def __init__(self, problem, m, n, g, graph_type=None):

        self.problem = problem
        self.graph_type = graph_type

        self.eigenfunction_pairs = self.construct_PDE_eigenfunctions(m, n, g)

    def construct_PDE_eigenfunctions(self, m, n, g):

        eigenvalue = self.calculate_pde_eigenvalue(m, n)
        basis_functions = self.construct_basis_functions(m, n)

        eigenfunction_pairs = {"eigenvalues": [],
                               "eigenfunctions": []}

        for basis_function in basis_functions:
            pde_eigenfunction = []

            for edge in g.g_coords:
                pde_eigenfunction.append(basis_function(*edge))

            eigenfunction_pairs["eigenvalues"].append(eigenvalue)
            eigenfunction_pairs["eigenfunctions"].append(Graph_Function(pde_eigenfunction, g.g_coords).normalize())

        return eigenfunction_pairs

    def calculate_pde_eigenvalue(self, m, n):

        if self.graph_type==None:
            homogenization_coefficient = 1
        elif self.graph_type == "aperiodic_monotile":
            homogenization_coefficient = 0.6075027620250514#0.6073544524898717
        elif self.graph_type == "rgg":
            homogenization_coefficient = 0.6781071615369281
        elif self.graph_type == "random_delaunay":
            homogenization_coefficient = 0.9201245197443716

        if self.problem == "square_flat_torus":
            func_pde_eigenvalues = lambda m, n: ((2 * np.pi * m)**2 + (2 * np.pi * n)**2)

        elif self.problem == "inhomogeneous_square_flat_torus":
            func_pde_eigenvalues = lambda m, n: 29.51697331671992

        elif self.problem == "leaf":
            func_pde_eigenvalues = lambda m, n: (np.pi / 0.9)**2
        
        elif self.problem == "drum":
            func_pde_eigenvalues = lambda m, n: scipy.special.jn_zeros(m, n)[-1]**2
        
        elif self.problem == "sphere":
            func_pde_eigenvalues = lambda m, n: (m * (m + 1))

        return homogenization_coefficient / 2 * func_pde_eigenvalues(m, n)

    def construct_basis_functions(self, m, n):

        if self.problem == "square_flat_torus":

            if (m > 2) or (n > 2) or (m < 0) or (n < 0):
                raise ValueError("Projector is only set up for m, n \in [0, 1, 2]")

            if (m == 0) and (n == 0):
                return lambda x, y: np.ones(x.shape), 

            cc = lambda m, n: lambda x, y: np.cos(2 * np.pi * m * x) * np.cos(2 * np.pi * n * y)
            cs = lambda m, n: lambda x, y: np.cos(2 * np.pi * m * x) * np.sin(2 * np.pi * n * y)
            sc = lambda m, n: lambda x, y: np.sin(2 * np.pi * m * x) * np.cos(2 * np.pi * n * y)
            ss = lambda m, n: lambda x, y: np.sin(2 * np.pi * m * x) * np.sin(2 * np.pi * n * y)

            if m == 0:
                return cc(m, n), cs(m, n), cc(n, m), sc(n, m)
            elif n == 0:
                return cc(m, n), sc(m, n), cc(n, m), cs(n, m)
            elif m == n:
                return cc(m, n), cs(m, n), sc(m, n), ss(m, n),
            else:
                return (cc(m, n), cs(m, n), sc(m, n), ss(m, n), 
                        cc(n, m), cs(n, m), sc(n, m), ss(n, m))
            
        elif self.problem == "inhomogeneous_square_flat_torus":

            with open("/Users/sholden/repos/metric_graph/next_calculations/random_inhomogeneous/random_ncc_eigenmodes.pkl", "rb") as f:
                f0, f1 = pickle.load(f)
                x, y = np.linspace(0, 1, 64, endpoint=False), np.linspace(0, 1, 64, endpoint=False)
                f0_interp = RegularGridInterpolator((x, y), f0.T, bounds_error=False, fill_value=None)
                f1_interp = RegularGridInterpolator((x, y), f1.T, bounds_error=False, fill_value=None)

                f0_interp_func = lambda x, y: f0_interp(np.vstack((x, y)).T)
                f1_interp_func = lambda x, y: f1_interp(np.vstack((x, y)).T)

                return f0_interp_func, f1_interp_func
            
        elif self.problem == "leaf":

            with open("/Users/sholden/repos/metric_graph/next_calculations/random_inhomogeneous/disc/lowest_3_leaf_modes.pkl", "rb") as file:
                continuum_modes = pickle.load(file)

            if m==0:
                return lambda x, y: continuum_modes[0](np.array([np.sqrt(x**2 + y**2), np.arctan2(y, x) + np.pi/2]).T),
            else:
                return (lambda x, y: continuum_modes[1](np.array([np.sqrt(x**2 + y**2), np.arctan2(y, x) + np.pi/2]).T),
                        lambda x, y: continuum_modes[2](np.array([np.sqrt(x**2 + y**2), np.arctan2(y, x) + np.pi/2]).T))
            
        elif self.problem == "drum":

            if (m > 2) or (n > 2) or (m < 0) or (n < 0):
                raise ValueError("Projector is only set up for m, n \in [0, 1, 2]")

            r = lambda x, y: np.sqrt(x**2 + y**2)
            theta = lambda x, y: np.arctan2(y, x)

            c = lambda m, n: lambda x, y: scipy.special.jn(m, r(x, y) * scipy.special.jn_zeros(m, n)[-1]) * np.cos(m * theta(x, y))
            # s = lambda m, n: lambda x, y: scipy.special.jn(m, r(x, y) * np.sqrt(scipy.special.jn_zeros(m, n)[-1]**2 / 2)) * np.sin(m * theta(x, y))

            return c(m, n), #, s(m, n)

        elif self.problem == "sphere":

            y_lm_real = lambda n, m: lambda x, y, z: -np.real(scipy.special.sph_harm(n, m, *self.cartesian_to_spherical(x, y, z)))
            y_lm_zero = lambda n, m: lambda x, y, z: np.real(scipy.special.sph_harm(n, m, *self.cartesian_to_spherical(x, y, z)))
            y_lm_imag = lambda n, m: lambda x, y, z: -np.imag(scipy.special.sph_harm(n, m, *self.cartesian_to_spherical(x, y, z)))

            funcs = ()
            for n in range(-m, m + 1):
                if n < 0: funcs += (y_lm_imag(n, m), )
                elif n == 0: funcs += (y_lm_zero(n, m), )
                else: funcs += (y_lm_real(n, m), )

            return funcs
        
    def cartesian_to_spherical(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)  # Colatitude
        phi = np.arctan2(y, x)    # Azimuth
        return phi, theta
    
class Projector:

    def __init__(self, g, m, n, graph_ks, problem, graph_type=None):

        self.data = self.extract_eigendata(g, m, n, graph_ks, problem, graph_type)

    def extract_eigendata(self, g, m, n, graph_ks, problem, graph_type):

        graph_eigenfunction_pairs = Projector_Graph_Eigenfunctions(g, graph_ks)
        pde_eigenfunction_pairs = PDE_Eigenfunctions(problem, m, n, g, graph_type)

        projected_graph_eigenfunctions = []

        for graph_eigenfunction in graph_eigenfunction_pairs.eigenfunction_pairs["eigenfunctions"]:

            projected_graph_eigenfunction = self.project(graph_eigenfunction, 
                                                         pde_eigenfunction_pairs.eigenfunction_pairs["eigenfunctions"])
            
            projected_graph_eigenfunctions.append(projected_graph_eigenfunction)

        graph_eigenfunction_rel_errs = np.array([(i - j).norm() for i, j in zip(graph_eigenfunction_pairs.eigenfunction_pairs["eigenfunctions"],
                                                                                projected_graph_eigenfunctions)])
        
        pde_eigenvalue = pde_eigenfunction_pairs.calculate_pde_eigenvalue(m, n)
        graph_eigenvalue_rel_errs = np.abs((np.array(graph_eigenfunction_pairs.eigenfunction_pairs["eigenvalues"]) - pde_eigenvalue) / pde_eigenvalue)

        self.graph_eigenfunctions = graph_eigenfunction_pairs.eigenfunction_pairs["eigenfunctions"]
        self.pde_eigenfunctions = pde_eigenfunction_pairs.eigenfunction_pairs["eigenfunctions"]
        
        return graph_eigenvalue_rel_errs, graph_eigenfunction_rel_errs
            
    def project(self, graph_eigenfunction, pde_eigenfunctions):

        projection = 0

        for pde_eigenfunction in pde_eigenfunctions:

            projection = projection + graph_eigenfunction.dot(pde_eigenfunction) * pde_eigenfunction

        return projection