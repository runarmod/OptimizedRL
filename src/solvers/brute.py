import time
from collections import deque
from copy import copy
from math import inf
from multiprocessing import Pool

import highspy
import numpy as np
import scipy

from solvers.solver_interface import Solver

# BFS brute force search for MILP solutions

class BruteForceMILP:
    """ Implements a brute force method to solve MILPs (only currently tested for mixed binary problems)
        Should be rewritten as requires one to reinstantiate solver for every problem, so can in practice just be pure function
    """
    # Implements a brute force method to solve an MILP.
    # Solves the relaxed problem and uses the rounded up solutions as an intial node for searching the solution space
    # For problems with defined bounds on variables it will test all combinations of solutions
    # For problems with no defined bounds it will test all solutions in a BFS search until the max number of iterations is completed

    def __init__(self,init_node):
        self.init_node = init_node
        self.integer = self.init_node["integer"]
        self.tree = []
        self.sol = None
        self.end_node = None
        self.queue = deque()
        self.pool = []


    def optimize_node(self,node):
        return scipy.optimize.linprog(
            node["c"],
            node["A_ub"],
            node["b_ub"],
            node["A_eq"],
            node["b_eq"],
            node["bounds"],
            )
    
    def manhattan_round(self,x, int_indices):
        """Find the closest integer solution by rounding the integer-constrained variables."""
        x_int = x.copy()
        for i, is_int in enumerate(int_indices):
            if is_int:
                x_int[i] = round(x[i])
        return x_int

    def generate_neighbors(self,x, is_integer,bounds):
        # Needs to handle that the original bounds shouldnt be breached
        """Generate neighboring integer solutions by varying each integer-constrained variable by ±1."""
        neighbors = []
        for i, is_int in enumerate(is_integer):
            if is_int:
                l_bound,r_bound = bounds[i]
                l_bound = l_bound if l_bound is not None else -inf
                r_bound = r_bound if r_bound is not None else inf
                x_new = x.copy()
                x_new[i] = min(x[i] + 1,r_bound)
                neighbors.append(x_new)
                x_new = x.copy()
                x_new[i] = max(x[i] - 1,l_bound)
                neighbors.append(x_new)
        return neighbors

    def solve(self,max_iter = 10000,store_pool = False,verbose = False):
        res = self.optimize_node(self.init_node)
        self.tree.append(self.init_node)
        if not res.success:
            return None
        
        initial_guess = self.manhattan_round(res.x, self.integer)
        self.queue.append(initial_guess)
        visited = set()
        visited.add(tuple(initial_guess))
        iter = 1
        best_value = float('inf')

        while len(self.queue) > 0 and  iter <= max_iter:
            iter += 1

            x_fixed = self.queue.popleft()
            new_bounds = self.init_node["bounds"].copy()
            for i, is_int in enumerate(self.integer):
                if is_int:
                    new_bounds[i] = (x_fixed[i], x_fixed[i])
            node = copy(self.init_node)
            node["bounds"] = new_bounds
            res = self.optimize_node(node)
            if verbose:
                print(res)
                print(new_bounds)
                print(x_fixed)


            for neighbor in self.generate_neighbors(x_fixed, self.integer,self.init_node["bounds"]):
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple not in visited:
                    self.queue.append(neighbor)
                    visited.add(neighbor_tuple)   
            
            if not res.success:
                # print("infeasible")
                continue

            if res.fun < best_value:
                self.sol = res
                best_value = res.fun
                
            if store_pool:
                self.pool.append(res)
                
        if verbose:
            print(f"Number of nodes explored: {iter}")
        return self.sol,self.pool
        



    
def translate_to_highspy(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """
    Translate a linear programming problem defined in scipy format to highspy.

    Parameters:
        c (np.array): Coefficient vector of the objective function.
        A_ub (np.array): Inequality constraint matrix (optional).
        b_ub (np.array): Inequality constraint vector (optional).
        A_eq (np.array): Equality constraint matrix (optional).
        b_eq (np.array): Equality constraint vector (optional).
        bounds (list of tuples): Bounds for each variable, e.g., [(0, None), (0, None)].

    Returns:
        h (highspy.Highs): A Highs object with the problem defined.
    """
    # Initialize the Highs object
    h = highspy.Highs()

    # Add variables (columns) to the model with their bounds
    num_vars = len(c)
    if bounds is None:
        # Default bounds: 0 <= x <= inf
        bounds = [(0, None)] * num_vars

    for i in range(num_vars):
        lb, ub = bounds[i]
        # Convert None to highspy.kHighsInf for unbounded variables
        lb = -highspy.kHighsInf if lb is None else lb
        ub = highspy.kHighsInf if ub is None else ub
        h.addVar(lb, ub)  # Add variable with bounds

    # Add the objective function
    for i in range(num_vars):
        h.changeColCost(i, c[i])

    # Add equality constraints (A_eq x = b_eq)
    if A_eq is not None and b_eq is not None:
        num_eq_constraints, num_vars_eq = A_eq.shape
        assert num_vars_eq == num_vars, "A_eq must have the same number of columns as variables"
        for i in range(num_eq_constraints):
            coefficients = A_eq[i, :]
            # Equality constraint: b_eq[i] <= A_eq[i] * x <= b_eq[i]
            h.addRow(b_eq[i], b_eq[i], num_vars, list(range(num_vars)), coefficients.tolist())

    # Add inequality constraints (A_ub x <= b_ub)
    if A_ub is not None and b_ub is not None:
        num_ub_constraints, num_vars_ub = A_ub.shape
        assert num_vars_ub == num_vars, "A_ub must have the same number of columns as variables"
        for i in range(num_ub_constraints):
            coefficients = A_ub[i, :]
            # Inequality constraint: -inf <= A_ub[i] * x <= b_ub[i]
            h.addRow(-highspy.kHighsInf, b_ub[i], num_vars, list(range(num_vars)), coefficients.tolist())

    return h
def optimize_node(node):
    return scipy.optimize.linprog(
        node["c"],
        node["A_ub"],
        node["b_ub"],
        node["A_eq"],
        node["b_eq"],
        node["bounds"],
        )
def optimize_node2(c,A_ub,b_ub,bounds):
    return scipy.optimize.linprog(
        c,
        A_ub,
        b_ub,
        None,
        None,
        bounds
        )
    
    
def optimize_node3(i):
    bound = bounds[i]
    return scipy.optimize.linprog(
        c,
        A_ub,
        b_ub,
        A_eq,
        b_eq,
        bound,
        options={"threads": 4, "parallel" : True,"simplex_max_concurrency": 8}
        )
def optimize_node4(i):
    bound = bounds[i]
    h = translate_to_highspy(
        c,
        A_ub,
        b_ub,
        A_eq,
        b_eq,
        bound
    )
    h.setOptionValue("log_to_console",False)
    h.setOptionValue("output_flag",False)
    h.silent()
    # h.writeModel("lp_problem.lp")
    h.run()
    res = h.getSolution()
    res = {
                "fun" : h.getInfo().objective_function_value,
                "x" : res.col_value ,
                "dual" : res.row_dual,
                "success" : res.value_valid
                  }

    return res
    # return scipy.optimize.linprog(
    #     c,
    #     A_ub,
    #     b_ub,
    #     A_eq,
    #     b_eq,
    #     bound,
    #     options={"threads": 4, "parallel" : True,"simplex_max_concurrency": 8}
    #     )
    

def optimize_node_para(node,results,index):
    res =  linprog(
        node["c"],
        node["A_ub"],
        node["b_ub"],
        node["A_eq"],
        node["b_eq"],
        node["bounds"],
        )
    results[index] = res
def manhattan_round(x, int_indices):
    """Find the closest integer solution by rounding the integer-constrained variables."""
    x_int = x.copy()
    for i, is_int in enumerate(int_indices):
        if is_int:
            x_int[i] = round(x[i])
    return x_int

def generate_neighbors(x, is_integer,bounds):
    # Needs to handle that the original bounds shouldnt be breached
    """Generate neighboring integer solutions by varying each integer-constrained variable by ±1."""
    neighbors = []
    for i, is_int in enumerate(is_integer):
        if is_int:
            l_bound,r_bound = bounds[i]
            l_bound = l_bound if l_bound is not None else -inf
            r_bound = r_bound if r_bound is not None else inf
            x_new = x.copy()
            x_new[i] = min(x[i] + 1,r_bound)
            neighbors.append(x_new)
            x_new = x.copy()
            x_new[i] = max(x[i] - 1,l_bound)
            neighbors.append(x_new)
    return neighbors
def worker_function(c_arr, A_ub_arr, b_ub_arr, bounds):
    c_local = np.frombuffer(c_arr, dtype=np.float64)  # Convert back to NumPy
    A_ub_local = np.frombuffer(A_ub_arr, dtype=np.float64)
    b_ub_local = np.frombuffer(b_ub_arr, dtype=np.float64)
    
    return optimize_node2(c_local, A_ub_local, b_ub_local, bounds)

def process_init(c_i,A_ub_i,b_ub_i,A_eq_i,b_eq_i,bounds_list):
    global c, A_ub,b_ub,A_eq,b_eq,bounds 
    c = c_i
    A_ub = A_ub_i
    b_ub = b_ub_i
    A_eq =A_eq_i
    b_eq =b_eq_i
    bounds = bounds_list
    
def bruteForceSolveMILP(node,max_iter=10000, store_pool=False, verbose=False,processes = None):
    """ Parallell Brute force solver implemented as a pure function.
        Should be rewritten so that takes a pool as input or just use object version

    Args:
        node (_type_): _description_
        max_iter (int, optional): _description_. Defaults to 10000.
        store_pool (bool, optional): _description_. Defaults to False.
        verbose (bool, optional): _description_. Defaults to False.
        processes (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    integer = node["integer"]
    pool = []
    queue = deque()
    res = optimize_node(node)
    # pool.append(node)
    if not res.success:
        return None

    initial_guess = manhattan_round(res.x, integer)
    queue.append(initial_guess)
    visited = set()
    # visited.add(tuple(initial_guess))
    # visited = set()
    # visited.add(tuple(initial_guess))
    iter = 1
    # best_value = float('inf')
    orig_node = node
    orig_bounds = node["bounds"].copy()
    while len(queue) > 0 and  iter <= max_iter:
        iter += 1

        x_fixed = queue.popleft()
        if tuple(x_fixed) in visited:
            continue
        visited.add(tuple(x_fixed))
        new_bounds = orig_bounds.copy()
        for i, is_int in enumerate(integer):
            if is_int:
                new_bounds[i] = (x_fixed[i], x_fixed[i])
        node = copy(orig_node)
        node["bounds"] = new_bounds
        pool.append(node)
        for neighbor in generate_neighbors(x_fixed, integer,orig_node["bounds"]):
            neighbor_tuple = tuple(neighbor)
            if neighbor_tuple not in visited:
                queue.append(neighbor)
                # visited.add(neighbor_tuple)
  
    c = orig_node["c"]
    A_ub = orig_node["A_ub"]
    b_ub = orig_node["b_ub"]
    A_eq = orig_node["A_eq"]
    b_eq = orig_node["b_eq"]
    bounds_list = [node["bounds"] for node in pool]
    indexes = [i for i in range(len(pool))]
    with Pool(processes,process_init,[c,A_ub,b_ub,A_eq,b_eq,bounds_list]) as p:
        results = p.map(optimize_node4, indexes)
    # results = [res for res in results if res.success]
    # results = [res.getSolution() for res in results]
    num_eq_constraints = A_eq.shape[0] if A_eq is not None else 0
    # num_ub_constraints, num_vars_ub = A_ub.shape


    results = [
                {
                "fun" : res["fun"],
                "x" : res["x"] ,
                "eqlin" : res["dual"][:num_eq_constraints],
                "ineqlin" : res["dual"][num_eq_constraints:]
                  }
                  for res in results if res["success"]
               ]
    return results


class BruteForcePara(Solver):
    """ Implements a parallell brute force algorithm
        Can be rewritten into pure function if pool is passed as argument. 
    """
    def __init__(self,processes = 0):
        self.p = Pool(processes)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['p']
        return self_dict
        
        
    def translate_to_highspy(self,c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
        """
        Translate a linear programming problem defined in scipy format to highspy.
        Chatgpt wrote this

        Parameters:
            c (np.array): Coefficient vector of the objective function.
            A_ub (np.array): Inequality constraint matrix (optional).
            b_ub (np.array): Inequality constraint vector (optional).
            A_eq (np.array): Equality constraint matrix (optional).
            b_eq (np.array): Equality constraint vector (optional).
            bounds (list of tuples): Bounds for each variable, e.g., [(0, None), (0, None)].

        Returns:
            h (highspy.Highs): A Highs object with the problem defined.
        """
        # Initialize the Highs object
        h = highspy.Highs()

        # Add variables (columns) to the model with their bounds
        num_vars = len(c)
        if bounds is None:
            # Default bounds: 0 <= x <= inf
            bounds = [(0, None)] * num_vars

        for i in range(num_vars):
            lb, ub = bounds[i]
            # Convert None to highspy.kHighsInf for unbounded variables
            lb = -highspy.kHighsInf if lb is None else lb
            ub = highspy.kHighsInf if ub is None else ub
            h.addVar(lb, ub)  # Add variable with bounds

        # Add the objective function
        for i in range(num_vars):
            h.changeColCost(i, c[i])

        # Add equality constraints (A_eq x = b_eq)
        if A_eq is not None and b_eq is not None:
            num_eq_constraints, num_vars_eq = A_eq.shape
            assert num_vars_eq == num_vars, "A_eq must have the same number of columns as variables"
            for i in range(num_eq_constraints):
                coefficients = A_eq[i, :]
                # Equality constraint: b_eq[i] <= A_eq[i] * x <= b_eq[i]
                h.addRow(b_eq[i], b_eq[i], num_vars, list(range(num_vars)), coefficients.tolist())

        # Add inequality constraints (A_ub x <= b_ub)
        if A_ub is not None and b_ub is not None:
            num_ub_constraints, num_vars_ub = A_ub.shape
            assert num_vars_ub == num_vars, "A_ub must have the same number of columns as variables"
            for i in range(num_ub_constraints):
                coefficients = A_ub[i, :]
                # Inequality constraint: -inf <= A_ub[i] * x <= b_ub[i]
                h.addRow(-highspy.kHighsInf, b_ub[i], num_vars, list(range(num_vars)), coefficients.tolist())

        return h
    def optimize_node(self,node):
        return scipy.optimize.linprog(
            node["c"],
            node["A_ub"],
            node["b_ub"],
            node["A_eq"],
            node["b_eq"],
            node["bounds"],
            )

    def optimize_node_process(self,c,A_ub,b_ub,A_eq,b_eq,bound):

        h = self.translate_to_highspy(
            c,
            A_ub,
            b_ub,
            A_eq,
            b_eq,
            bound
        )
        h.setOptionValue("log_to_console",False)
        h.silent()
        h.run()

        res = h.getSolution()
        res = {
                    "fun" : h.getInfo().objective_function_value,
                    "x" : res.col_value ,
                    "dual" : res.row_dual,
                    "success" : res.value_valid
                    }

        return res
    
    def manhattan_round(self,x, int_indices):
        """Find the closest integer solution by rounding the integer-constrained variables."""
        x_int = x.copy()
        for i, is_int in enumerate(int_indices):
            if is_int:
                x_int[i] = round(x[i])
        return x_int

    def generate_neighbors(self,x, is_integer,bounds):
        # Needs to handle that the original bounds shouldnt be breached
        """Generate neighboring integer solutions by varying each integer-constrained variable by ±1."""
        neighbors = []
        for i, is_int in enumerate(is_integer):
            if is_int:
                l_bound,r_bound = bounds[i]
                l_bound = l_bound if l_bound is not None else -inf
                r_bound = r_bound if r_bound is not None else inf
                x_new = x.copy()
                x_new[i] = min(x[i] + 1,r_bound)
                neighbors.append(x_new)
                x_new = x.copy()
                x_new[i] = max(x[i] - 1,l_bound)
                neighbors.append(x_new)
        return neighbors

    def solve(self,node,max_iter=10000):
        integer = node["integer"]
        pool = []
        queue = deque()
        res = self.optimize_node(node)
        # pool.append(node)
        if not res.success:
            return None

        initial_guess = self.manhattan_round(res.x, integer)
        queue.append(initial_guess)
        visited = set()
        # visited.add(tuple(initial_guess))
        # visited = set()
        # visited.add(tuple(initial_guess))
        iter = 1
        # best_value = float('inf')
        orig_node = node
        orig_bounds = node["bounds"].copy()
        while len(queue) > 0 and  iter <= max_iter:
            iter += 1

            x_fixed = queue.popleft()
            if tuple(x_fixed) in visited:
                continue
            visited.add(tuple(x_fixed))
            new_bounds = orig_bounds.copy()
            for i, is_int in enumerate(integer):
                if is_int:
                    new_bounds[i] = (x_fixed[i], x_fixed[i])
            node = copy(orig_node)
            node["bounds"] = new_bounds
            pool.append(node)
            for neighbor in self.generate_neighbors(x_fixed, integer,orig_node["bounds"]):
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple not in visited:
                    queue.append(neighbor)
        c = orig_node["c"]
        A_ub = orig_node["A_ub"]
        b_ub = orig_node["b_ub"]
        A_eq = orig_node["A_eq"]
        b_eq = orig_node["b_eq"]
        bounds_list = [node["bounds"] for node in pool]
        inputs = [[c,A_ub,b_ub,A_eq,b_eq,bounds] for bounds in bounds_list]
        results = self.p.starmap(self.optimize_node_process, inputs)

        num_eq_constraints = A_eq.shape[0] if A_eq is not None else 0


        results = [
                    {
                    "fun" : res["fun"],
                    "x" : res["x"] ,
                    "eqlin" : res["dual"][:num_eq_constraints],
                    "ineqlin" : res["dual"][num_eq_constraints:]
                        }
                        for res in results if res["success"]
                    ]

        return results

def main():
    values = np.array([1,2,2,5,1])
    weights = np.array([2,3,1,4,1])
#     # values = np.array([1,2,2,5,1,1,1,1,1,1,1])
#     # weights = np.array([2,3,1,4,1,1,1,1,1,1,1])     # Item weights
    capacity = 10                       # Knapsack capacity

    n = len(values)

#     # Define the linear programming matrices
    c = -values   # Maximize -> minimize negative value
    A = np.array([weights])  # Single inequality constraint for total weight
    b = np.array([capacity]) # Knapsack capacity
    bounds = [(0, 1) for _ in range(n)]  # Relaxed 0-1 constraint
    # integer  = [1,1,1,1,1,1,1,1,1,1,1]
    integer  = [1,1,1,1,1]

    node = {
        "c" : c,
        "A_ub" : A,
        "b_ub" : b,
        "A_eq" : None,
        "b_eq" :  None,
        "bounds" : bounds,
        "integer" : integer,
        "parent" : None,
        "children" : [],
        "sol" : None
    }
    # init_node = Node(c,A_ub=A,b_ub=b,bounds=bounds,integer=integer)
    start = time.time()
    solver = BruteForcePara(8)
    sol = solver.bruteForceSolveMILP(node,max_iter = 10000 )
    # print(sol)
    print(time.time()-start)
    start = time.time()
    
    sol = solver.bruteForceSolveMILP(node,max_iter = 10000 )
    # print(sol)
    print(time.time()-start)
    # print(sol)
    start = time.time()
    solver = BruteForceMILP(node)
    _,sol = solver.solve(store_pool = True ,verbose = False, max_iter = 10000)
    print(time.time()-start)
    start = time.time()
    solver = BruteForceMILP(node)
    _,sol = solver.solve(store_pool = True ,verbose = False, max_iter = 10000)
    print(time.time()-start)
    # print(len(sol))
    
    # sol = solver.sol
    # print(sol)

if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # stats = Stats(pr)
    # stats.sort_stats('time').print_stats(20)
    # cProfile.run("main()",sort = "time")
