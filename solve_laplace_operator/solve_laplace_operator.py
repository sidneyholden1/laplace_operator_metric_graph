import numpy as np 
import scipy
import h5py
# import sparseqr
import warnings
import scipy

class Metric_Graph_Laplacian_Matrix:

    def calculate_csc(self, k, l):
        return 1 / np.sin(k * l)
    
    def calculate_sec(self, k, l):
        return 1 / np.cos(k * l)
    
    def calculate_cot(self, k, l):
        return 1 / np.tan(k * l)

    def calculate_dcsc(self, k, l):
        return -l * self.calculate_csc(k, l) * self.calculate_cot(k, l)

    def calculate_dcot(self, k, l):
        return -l * self.calculate_csc(k, l)**2

    def calculate_dsec(self, k, l):
        return l * self.calculate_sec(k, l) * np.tan(k * l)

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

        return np.array(eigenvalues)

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
                return k_next
            
            k = k_next

            if kwargs["solve_type"] == "SVD iterate":
                k_next, u = self.calculate_SVD_iterate(k, u=u, tol=tol)

            count += 1

            if k == np.nan:
                return k

        return np.nan

class Eigendata:

    def __init__(self, data):

        self.data = data
        self.V_nums = np.array(list(self.data.keys()))
        self.eigendata_indices = list(self.data[self.V_nums[0]].keys())
        self.graph_eigenvalues, self.PDE_eigenvalues = self.get_eigenvalues()
        self.relative_eigenvalue_differences = self.calculate_relative_eigenvalue_differences()
        self.relative_eigenfunction_differences = self.calculate_relative_eigenfunction_differences()

    def get_eigenvalues(self):

        graph_eigenvalues = {}
        PDE_eigenvalues = {}
        for ms in self.eigendata_indices:
            graph_eigenvalues[ms] = []
            PDE_eigenvalues[ms] = self.data[self.V_nums[0]][ms]['PDE']['eigenvalue']
            # for num_Vs in self.V_nums:
            #     graph_eigenvalues[ms].append(self.data[num_Vs][ms]['graph']['eigenvalue'])
            graph_eigenvalues[ms] = self.data[self.V_nums[0]][ms]['graph']['eigenvalue']

        return graph_eigenvalues, PDE_eigenvalues
        
    def calculate_relative_eigenvalue_differences(self):

        relative_eigenvalue_differences = {}
        for ms in self.eigendata_indices:
            pde_eig = self.PDE_eigenvalues[ms]
            # if np.abs(self.PDE_eigenvalues[ms]) < 1e-14:
            #     rel_errs = [np.mean(np.abs(i - pde_eig)) for i in self.graph_eigenvalues[ms]]
            # else:
            #     rel_errs = [np.mean(np.abs((i - pde_eig) / pde_eig)) for i in self.graph_eigenvalues[ms]]
            # relative_eigenvalue_differences[ms] = np.array(rel_errs)
            if np.abs(self.PDE_eigenvalues[ms]) < 1e-14:
                rel_errs = np.abs(self.graph_eigenvalues[ms]**2 - pde_eig)
            else:
                rel_errs = np.abs((self.graph_eigenvalues[ms]**2 - pde_eig) / pde_eig)
            relative_eigenvalue_differences[ms] = rel_errs

        return relative_eigenvalue_differences

    def calculate_relative_eigenfunction_differences(self):

        relative_eigenfunction_differences = {}
        for ms in self.eigendata_indices:
            relative_eigenfunction_differences[ms] = []
            # for num_Vs in self.V_nums:
            #     rel_errs = np.array([(i - j).norm() for i, j in zip(self.data[num_Vs][ms]['graph']['eigenfunction'], 
            #                                                         self.data[num_Vs][ms]['PDE']['eigenfunction'])])
                # relative_eigenfunction_differences[ms].append(np.mean(rel_errs))
            # relative_eigenfunction_differences[ms].append(rel_errs)
            rel_errs = np.array([(i - j).norm() for i, j in zip(self.data[self.V_nums[0]][ms]['graph']['eigenfunction'], 
                                                                self.data[self.V_nums[0]][ms]['PDE']['eigenfunction'])])
            relative_eigenfunction_differences[ms] = rel_errs

        return relative_eigenfunction_differences

class Graph_Modes:

    def __init__(self, g, graph_eigenvalues):

        (self.graph_eigenvalues,
         self.graph_eigenvectors, 
         self.null_space_dims) = self.calculate_graph_eigenvectors(g, graph_eigenvalues)
        self.graph_eigenfunctions = self.construct_graph_eigenfunctions(g)
    
    def construct_graph_eigenfunction(self, g, eigenvalue, eigenvector):

        graph_eigenfunction = []

        if np.abs(eigenvalue) < 1e-10:
            for e_num, _ in enumerate(g.edges):
                edge_mode = np.ones(g.g_coords[e_num].shape[1])
                graph_eigenfunction.append(edge_mode)
        else:
            # for e_num, edge in enumerate(g.edges):
            #     v, w = edge["vw"]
            #     l_vw = edge["l_vw"]
            for e_num, ((v, w), l_vw) in enumerate(g.E_lengths_by_v_num.items()):
                parametrised_edge = np.linspace(0, l_vw, g.g_coords[e_num].shape[1])

                edge_mode = ((eigenvector[v] * np.sin(eigenvalue * parametrised_edge[::-1])
                             + eigenvector[w] * np.sin(eigenvalue * parametrised_edge)) 
                             / np.sin(eigenvalue * l_vw))
                
                graph_eigenfunction.append(edge_mode)

        graph_eigenfunction = Graph_Function(graph_eigenfunction, g.g_coords).normalize()

        return graph_eigenfunction
    
    def construct_graph_eigenfunctions(self, g):

        graph_eigenfunctions = []
        for eigenvalue, eigenvector in zip(self.graph_eigenvalues, self.graph_eigenvectors.T):
            graph_eigenfunctions.append(self.construct_graph_eigenfunction(g, eigenvalue, eigenvector))
        
        return graph_eigenfunctions
    
    def calculate_graph_eigenvectors(self, g, graph_eigenvalues):

        graph_eigenvectors = []
        null_space_dims = []
        good_eigs = []

        for en, k in enumerate(graph_eigenvalues):
            print(f"{en} / {len(graph_eigenvalues)}")

            try:
                eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(g.construct_L(k), k=4, which='SM')
                null_eigenvalues = np.argwhere(np.abs(eigenvalues) < 1e-10).flatten()
                dim_null_space = len(null_eigenvalues)

                null_vectors = np.zeros((g.num_Vs, dim_null_space))
                for en, null_ind in enumerate(null_eigenvalues):
                    null_vectors[g.interior_V_num, en] = eigenvectors[:, null_ind]

                null_space_dims.append(dim_null_space)
                graph_eigenvectors.append(null_vectors)

                good_eigs.append(en)

            except:
                print(f"bad eig number = {en}")
                pass

        graph_eigenvalues = np.repeat(graph_eigenvalues[good_eigs], null_space_dims)
        graph_eigenvectors = np.hstack(graph_eigenvectors)

        return graph_eigenvalues, graph_eigenvectors, null_space_dims 
    
class Graph_Function:

    def __init__(self, data, domain=None):

        self.data = data
        if domain is not None:
            self.domain = [edge_domain.copy() for edge_domain in domain]

    def __add__(self, other):

        if not isinstance(other, Graph_Function):
            return NotImplemented
        
        result = [i + j for i, j in zip(self.data, other.data)]

        return Graph_Function(result, self.domain)

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

class Projector:

    def __init__(self, g, eigenvalues, continuum_eigendata):

        self.data = {}
        self.modes = Graph_Modes(g, eigenvalues)
        self.V_coords = g.V_coords
        self.g_coords = g.g_coords
        self.continuum_eigendata = continuum_eigendata

    def construct_pde_functions(self, m, n, function_domain="vertices"):

        basis_functions = self.continuum_eigendata.generate_basis_functions(m, n)

        if function_domain == "vertices":
            # x, y = self.V_coords[:, 0], self.V_coords[:, 1]

            pde_eigenvectors = [function(*self.V_coords.T) for function in basis_functions]
            pde_eigenvectors = [function / np.linalg.norm(function) for function in pde_eigenvectors]
            pde_eigenvectors = np.vstack((pde_eigenvectors)).T

            return pde_eigenvectors
        
        elif function_domain == "graph":
            pde_eigenfunctions = [[] for _ in range(len(basis_functions))]

            for edge in self.g_coords:
                # x, y = edge
                
                for i in range(len(basis_functions)):
                    pde_eigenfunctions[i].append(basis_functions[i](*edge))

            pde_eigenfunctions = [Graph_Function(pde_eigenfunction, self.g_coords).normalize() 
                                  for pde_eigenfunction in pde_eigenfunctions]
            
            return pde_eigenfunctions

        elif function_domain == "continuum":

            if self.continuum_eigendata.problem == "square_flat_torus":
                edge = np.linspace(0, 1, 256, endpoint=True)   
                x, y = np.meshgrid(edge, edge)
                pde_eigenfunctions = [function(x, y) for function in basis_functions]
                return x, y, pde_eigenfunctions
            
            elif self.continuum_eigendata.problem == "disc":
                points_per_dim = 128
                r = np.linspace(0, 1, points_per_dim)**0.5
                theta = np.linspace(0, 2 * np.pi, points_per_dim)
                r, theta = np.meshgrid(r, theta)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                pde_eigenfunctions = [function(x, y) for function in basis_functions]

                return x, y, pde_eigenfunctions
            
            elif self.continuum_eigendata.problem == "sphere":
                num_points = 256
                phi = np.linspace(0, 2 * np.pi, num_points)  
                theta = np.linspace(0, np.pi, num_points)   
                theta, phi = np.meshgrid(theta, phi)
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)

                pde_eigenfunctions = [function(x, y, z) for function in basis_functions]

                return x, y, z, pde_eigenfunctions

    def find_graph_eigenspace(self, m, n):

        pde_eigenvectors = self.construct_pde_functions(m, n, function_domain="vertices")
        num_basis_functions = pde_eigenvectors.shape[1]
        prod = pde_eigenvectors.T @ pde_eigenvectors
        invert_prod = np.linalg.inv(prod)
        proj = (pde_eigenvectors @ invert_prod @ pde_eigenvectors.T)

        keep_args = proj @ self.modes.graph_eigenvectors

        keep_args = np.argsort(np.linalg.norm(keep_args, axis=0))[-num_basis_functions:]

        # Handle when the graph eigenspace is larger than the PDE eigenspace
        uniques, counts = np.unique(np.round(self.modes.graph_eigenvalues[keep_args], 10), return_counts=True)

        new_keep_args = []
        for unique, count in zip(uniques, counts):
            where_equal = np.where(np.abs(self.modes.graph_eigenvalues - unique) < 1e-9)[0]
            if np.shape(where_equal)[0] != count:
                message = (f"For num_Vs={self.V_coords.shape[0]},m={m},n={n}, the graph eigenspace is larger than the PDE eigenspace. " + 
                            "This might be (a) a problem, (b) saying something interesting, or (c) nothing special.")
                warnings.warn(message)
            new_keep_args.append(where_equal)

        keep_args = np.concatenate((new_keep_args))

        return keep_args

    def __call__(self, m, n, splittings=False):

        self.data[m, n] = {"graph": {}, "PDE": {}}

        # Get PDE eigenvalues
        self.data[m, n]["PDE"]["eigenvalue"] = self.continuum_eigendata.calculate_pde_eigenvalues(m, n)

        # Get graph eigenvalues
        keep_args = self.find_graph_eigenspace(m, n)
        self.data[m, n]["graph"]["eigenvalue"] = self.modes.graph_eigenvalues[keep_args]

        # Get PDE eigenfunctions
        pde_eigenfunctions = self.construct_pde_functions(m, n, function_domain="graph")

        if splittings: # Project graph modes onto PDE modes

            # Get graph eigenfunctions
            self.data[m, n]["graph"]["eigenfunction"] = [self.modes.graph_eigenfunctions[i] for i in keep_args]

            # Get PDE eigenfunctions
            project_functions = []
            for graph_eigenfunction in self.data[m, n]["graph"]["eigenfunction"]:
                projection = []
                for pde_eigenfunction in pde_eigenfunctions:
                    ip = graph_eigenfunction.dot(pde_eigenfunction)
                    projection.append(ip * pde_eigenfunction)
                project_functions.append(np.sum(projection))
            self.data[m, n]["PDE"]["eigenfunction"] = project_functions 

        else: # Project PDE modes onto graph modes

            self.data[m, n]["PDE"]["eigenfunction"] = pde_eigenfunctions

            # Get graph eigenfunctions
            project_functions = []
            for pde_eigenfunction in pde_eigenfunctions:
                projection = []
                for arg in keep_args:
                    ip = pde_eigenfunction.dot(self.modes.graph_eigenfunctions[arg])
                    projection.append(ip * self.modes.graph_eigenfunctions[arg])
                project_functions.append(np.sum(projection))
            self.data[m, n]["graph"]["eigenfunction"] = project_functions 

class Continuum_Eigendata:

    def __init__(self, problem, graph_type=None):

        self.problem = problem
        self.graph_type = graph_type

    def calculate_pde_eigenvalues(self, m, n):

        if self.problem == "square_flat_torus":
            if self.graph_type==None:
                coefficient = 1
            elif self.graph_type=="aperiodic_monotile":
                coefficient = 0.6075027620250514#0.6073544524898717
            elif self.graph_type=="rgg":
                coefficient = 0.6781071615369281
            func_pde_eigenvalues = lambda m, n: coefficient * ((2 * np.pi * m)**2 + (2 * np.pi * n)**2) / 2
            if np.isscalar(m):
                m = np.array([m])
            if np.isscalar(n):
                n = np.array([n])
            pde_eigenvalues = np.zeros((len(m), len(n)))
            for eni, i in enumerate(m):
                for enj, j in enumerate(n):
                    pde_eigenvalues[eni, enj] = func_pde_eigenvalues(i, j)

            return pde_eigenvalues
        
        elif self.problem == "disc":
            func_pde_eigenvalues = lambda m, n: np.sqrt(scipy.special.jn_zeros(m, n)[-1]**2 / 2)
            if np.isscalar(m):
                m = np.array([m])
            if np.isscalar(n):
                n = np.array([n])
            pde_eigenvalues = np.zeros((len(m), len(n)))
            for eni, i in enumerate(m):
                for enj, j in enumerate(n):
                    pde_eigenvalues[eni, enj] = func_pde_eigenvalues(i, j)

            return pde_eigenvalues
        
        elif self.problem == "sphere":
            func_pde_eigenvalues = lambda m: np.sqrt((m * (m + 1)) / 2)
            # pde_eigenvalues = np.ones((1, (2 * m + 1))) * func_pde_eigenvalues(m)
            pde_eigenvalues = np.array([func_pde_eigenvalues(m)])

            return pde_eigenvalues

    def generate_basis_functions(self, m, n):

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
            
        elif self.problem == "disc":

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