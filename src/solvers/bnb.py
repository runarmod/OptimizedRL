from collections import deque
from copy import copy, deepcopy
from math import ceil, floor

import numpy as np
from scipy.optimize import linprog

# Node dictionary template

example_node = {
    "c" : None,
    "A_ub" : None,
    "b_ub" : None,
    "A_eq" : None,
    "b_eq" : None,
    "bounds" : None,
    "integer" : None,
}


def _normalize_linprog_duals(marginals, constraint_matrix):
    if constraint_matrix is None:
        return np.array([], dtype=float)

    duals = np.asarray(marginals, dtype=float)
    if duals.ndim == 0:
        n_constraints = constraint_matrix.shape[0]
        if n_constraints == 0:
            return np.array([], dtype=float)
        return duals.reshape(1)

    return duals.reshape(-1)






class BranchAndBound:
    """
        Implements a basic branch and bound algorithm, takes a dictionary defining the inital problem node as its input.
    
    """
    def __init__(self,init_node,sense):
        self.init_node = init_node
        self.integer = self.init_node["integer"]
        self.tree = []
        self.queue = deque()
        self.sol = None
        self.end_node = None
        self.pool = []


    def optimize_node(self,node):
        return linprog(
            node["c"],
            node["A_ub"],
            node["b_ub"],
            node["A_eq"],
            node["b_eq"],
            node["bounds"],
            )

# Handle infeasible sub sol
# Add parent and child relation

    def branch(self,node,x):
        # need to remove non integer
        # x = x[self.integer] 
        diff = np.abs(x) - np.floor(np.abs(x))
        diff = [val if isint else 0 for val,isint in zip(diff,self.integer)]
        # diff = [np.abs(xi - round(xi)) if isint else 0 for xi, isint in zip(x, self.integer)]
        if max(diff) < 1e-6:
            return None, None
        branch_var = np.argmax(diff)

        left_branch = deepcopy(node)
        right_branch = deepcopy(node)

        left_branch["bounds"][branch_var] = (left_branch["bounds"][branch_var][0],floor(x[branch_var]))
        right_branch["bounds"][branch_var] = (ceil(x[branch_var]),right_branch["bounds"][branch_var][1] )

        left_branch["parent"] = node
        right_branch["parent"] = node
        return left_branch,right_branch

        
    # def all_integer(self,x):
    #    return all( [not(var.is_integer() ^ bool(i)) for var,i in zip(x,self.integer)])
    def all_integer(self, x):
        return all((np.abs(var - round(var)) < 1e-6) if i else True for var, i in zip(x, self.integer))



    def solve(self,verbose = False):
        res = self.optimize_node(self.init_node)
        if not res.success:
            # print(res)
            # print("infeasible")
            return None
        self.tree.append(self.init_node)
        iter = 1
        if self.all_integer(res.x):
            # print(res.x[0].is_integer())
            self.sol = res
            self.init_node["sol"] = res
            self.end_node = self.init_node
            # print(f"Number of nodes explored: {iter}")


            return self.sol
        
        sol_rounded = np.floor(res.x) # Does this work for negative solutions?
        # sol_rounded = np.where(self.init_node["c"] >= 0,  np.ceil(res.x),np.floor(res.x))
        ub = self.init_node["c"] @ sol_rounded
        ub = float("inf")
        l,r = self.branch(self.init_node,res.x)
        self.queue.append(l)
        self.queue.append(r)
        self.init_node["children"].append(l)
        self.init_node["children"].append(r)
        while len(self.queue) > 0:
            iter += 1
            node = self.queue.popleft()
            self.tree.append(node)
            res = self.optimize_node(node)
            node["sol"] = res
            if verbose:
                print(node)

            if not res.success:
                print("infeasible")
                continue

            if self.all_integer(res.x):
                if res.fun <= ub:
                    ub = res.fun
                    self.sol = res
                    self.end_node = node
                    self.pool.append(res)

            elif res.fun <= ub:
                l,r = self.branch(node,res.x)

                self.queue.append(l)
                self.queue.append(r)
                node["children"].append(l)
                node["children"].append(r)
        if verbose:
            print(f"Number of nodes explored: {iter}")
        return self.sol





class BranchAndBoundRevamped:
    """
        Implements a basic branch and bound algorithm, takes a dictionary defining the inital problem node as its input.
    
    """
    def __init__(self,verbose = False):
        self.verbose = verbose
        self.last_tree_snapshot = None

    @staticmethod
    def _format_conds(conds):
        if not conds:
            return "root"
        return ", ".join(
            f"x_{int(var_id)} {op} {float(val):.6g}"
            for var_id, op, val in conds
        )

    def _store_tree_snapshot(self, node_records, results, incumbent):
        nodes = []
        pool_node_ids = []
        pool_node_id_set = set()

        for node_id in sorted(node_records):
            node = dict(node_records[node_id])
            node["in_pool"] = False
            nodes.append(node)

        node_lookup = {node["node_id"]: node for node in nodes}
        for entry in results or []:
            node_id = entry.get("node_id")
            if node_id is None or node_id not in node_lookup:
                continue
            node_lookup[node_id]["in_pool"] = True
            if node_id not in pool_node_id_set:
                pool_node_id_set.add(node_id)
                pool_node_ids.append(node_id)

        self.last_tree_snapshot = {
            "nodes": nodes,
            "pool_node_ids": pool_node_ids,
            "incumbent": None if incumbent == float("inf") else float(incumbent),
        }

    def format_last_tree_log(self):
        snapshot = self.last_tree_snapshot
        if not snapshot:
            return "[bnb-tree] unavailable"

        nodes = snapshot.get("nodes", [])
        children = {}
        for node in nodes:
            parent = node.get("parent")
            children.setdefault(parent, []).append(node)

        for siblings in children.values():
            siblings.sort(key=lambda item: item["node_id"])

        incumbent = snapshot.get("incumbent")
        incumbent_str = "None" if incumbent is None else f"{float(incumbent):.6f}"
        lines = [
            "[bnb-tree] "
            f"nodes={len(nodes)} "
            f"pool_nodes={len(snapshot.get('pool_node_ids', []))} "
            f"incumbent={incumbent_str}"
        ]

        def _render(node, prefix, is_last):
            branch = node.get("branching_decision")
            if branch is None:
                branch_str = "root"
            else:
                branch_str = f"x_{int(branch['var'])} {branch['type']} {float(branch['value']):.6g}"
            obj = node.get("lp_obj")
            obj_str = "None" if obj is None else f"{float(obj):.6f}"
            pool_str = "yes" if node.get("in_pool") else "no"
            parent = node.get("parent")
            parent_str = "None" if parent is None else str(parent)

            if node.get("parent") is None:
                line_prefix = "[root] "
                child_prefix = ""
            else:
                connector = "\\- " if is_last else "|- "
                line_prefix = prefix + connector
                child_prefix = prefix + ("   " if is_last else "|  ")

            lines.append(
                line_prefix
                + f"node={node['node_id']} "
                + f"parent={parent_str} "
                + f"branch={branch_str} "
                + f"obj={obj_str} "
                + f"status={node.get('status')} "
                + f"pool={pool_str} "
                + f"conds={self._format_conds(node.get('conds', []))}"
            )

            node_children = children.get(node["node_id"], [])
            for idx, child in enumerate(node_children):
                _render(child, child_prefix, idx == len(node_children) - 1)

        root_nodes = children.get(None, [])
        for idx, root in enumerate(root_nodes):
            _render(root, "", idx == len(root_nodes) - 1)

        return "\n".join(lines)

    def optimize_node(self,node):
        return linprog(
            node["c"],
            node["A_ub"],
            node["b_ub"],
            node["A_eq"],
            node["b_eq"],
            node["bounds"],
            )

# Handle infeasible sub sol
# Add parent and child relation

    def branch(self,bounds,x,integer):
        # need to remove non integer
        # x = x[self.integer] 
        diff = np.abs(x) - np.floor(np.abs(x))
        diff = [val if isint else 0 for val,isint in zip(diff,integer)]
        # diff = [np.abs(xi - round(xi)) if isint else 0 for xi, isint in zip(x, self.integer)]
        if max(diff) < 1e-6:
            return None, None
        branch_var = np.argmax(diff)

        left_branch = copy(bounds)
        right_branch = copy(bounds)

        left_branch[branch_var] =  (left_branch[branch_var][0],floor(x[branch_var]))
        right_branch[branch_var] = (ceil(x[branch_var]), right_branch[branch_var][1])

        return left_branch,right_branch,branch_var

        
    # def all_integer(self,x):
    #    return all( [not(var.is_integer() ^ bool(i)) for var,i in zip(x,self.integer)])
    def all_integer(self, x,integer):
        return all((np.abs(var - round(var)) < 1e-6) if i else True for var, i in zip(x,integer))



    def solve(self,init_node):
        verbose = self.verbose
        results = []
        queue = deque()
        self.last_tree_snapshot = None
        node_records = {}
        next_node_id = 0
        
        
        res = self.optimize_node(init_node)
        if not res.success:
            # print(res)
            # print("infeasible")
            return None
        # self.tree.append(self.init_node)
        iter = 1
        integer  = init_node["integer"]
        node_records[0] = {
            "node_id": 0,
            "parent": None,
            "depth": 0,
            "lp_obj": float(res.fun),
            "status": "root_integral" if self.all_integer(res.x,integer) else "branched",
            "branching_decision": None,
            "conds": [],
        }
        
        if self.all_integer(res.x,integer):
   
            # print(f"Number of nodes explored: {iter}")

            results.append(  
                           
                {       
                    "fun" : res.fun,
                    "x" : res.x ,
                    "eqlin" : _normalize_linprog_duals(res.eqlin.marginals, init_node["A_eq"]),
                    "ineqlin" : _normalize_linprog_duals(res.ineqlin.marginals, init_node["A_ub"]),
                    "lower" : res.lower.marginals,
                    "upper" : res.upper.marginals,
                    "fathomed" : False,
                    "node" : init_node,
                    "bounds" : init_node["bounds"],
                    "conds": [],
                    "node_id": 0,

                }
                
            )

            self._store_tree_snapshot(node_records, results, float(res.fun))

            return results
        
        ub = float("inf")
        
        l,r,branch_var = self.branch(init_node["bounds"],res.x,integer)
        if l is None or r is None:
            node_records[0]["status"] = "fractional_unbranched"
            self._store_tree_snapshot(node_records, results, ub)
            return results
        l_val = l[branch_var][1]
        r_val = r[branch_var][0]
        l_cond = [(branch_var,"<=",l_val)]
        r_cond = [(branch_var,">=",r_val)]
        next_node_id += 1
        left_node = {
            "node_id": next_node_id,
            "parent": 0,
            "depth": 1,
            "bounds": l,
            "conds": l_cond,
            "branching_decision": {"var": int(branch_var), "type": "<=", "value": float(l_val)},
        }
        node_records[left_node["node_id"]] = {
            "node_id": left_node["node_id"],
            "parent": 0,
            "depth": 1,
            "lp_obj": None,
            "status": "queued",
            "branching_decision": left_node["branching_decision"],
            "conds": list(left_node["conds"]),
        }
        next_node_id += 1
        right_node = {
            "node_id": next_node_id,
            "parent": 0,
            "depth": 1,
            "bounds": r,
            "conds": r_cond,
            "branching_decision": {"var": int(branch_var), "type": ">=", "value": float(r_val)},
        }
        node_records[right_node["node_id"]] = {
            "node_id": right_node["node_id"],
            "parent": 0,
            "depth": 1,
            "lp_obj": None,
            "status": "queued",
            "branching_decision": right_node["branching_decision"],
            "conds": list(right_node["conds"]),
        }
        queue.append(left_node)
        queue.append(right_node)
        
        while len(queue) > 0:
            iter += 1
            queued_node = queue.popleft()
            bounds = queued_node["bounds"]
            conds = queued_node["conds"]
            node = {
                "c" : init_node["c"],
                "A_ub" : init_node["A_ub"],
                "b_ub" : init_node["b_ub"],
                "A_eq" : init_node["A_eq"],
                "b_eq" : init_node["b_eq"],
                "bounds" : bounds
            }
            res = self.optimize_node(node)
            record = node_records[queued_node["node_id"]]
            
            if verbose:
                print(node)

            if not res.success:
                record["status"] = "infeasible"
                if verbose:
                    print("infeasible")
                continue

            record["lp_obj"] = float(res.fun)

            if self.all_integer(res.x,integer):
                if res.fun <= ub:
                    ub = res.fun
                    record["status"] = "integer_incumbent"
                    results.append(  
                                    
                        {       
                            "fun" : res.fun,
                            "x" : res.x ,
                            "eqlin" : _normalize_linprog_duals(res.eqlin.marginals, init_node["A_eq"]),
                            "ineqlin" : _normalize_linprog_duals(res.ineqlin.marginals, init_node["A_ub"]),
                            "lower" : res.lower.marginals,
                            "upper" : res.upper.marginals,
                            "fathomed" : False,
                            "conds" : conds,
                            "node" : node,
                            "bounds" : bounds,
                            "node_id": queued_node["node_id"]
                        }
                        
                    )
                else:
                    record["status"] = "integer_pruned"

            elif res.fun <= ub:
                record["status"] = "branched"
                l,r,branch_var = self.branch(bounds,res.x,integer)
                if l is None or r is None:
                    record["status"] = "fractional_unbranched"
                    continue
                l_val = l[branch_var][1]
                r_val = r[branch_var][0]
                l_cond = conds + [(branch_var,"<=",l_val)]
                r_cond =  conds + [(branch_var,">=",r_val)]
                next_node_id += 1
                left_node = {
                    "node_id": next_node_id,
                    "parent": queued_node["node_id"],
                    "depth": queued_node["depth"] + 1,
                    "bounds": l,
                    "conds": l_cond,
                    "branching_decision": {"var": int(branch_var), "type": "<=", "value": float(l_val)},
                }
                node_records[left_node["node_id"]] = {
                    "node_id": left_node["node_id"],
                    "parent": queued_node["node_id"],
                    "depth": left_node["depth"],
                    "lp_obj": None,
                    "status": "queued",
                    "branching_decision": left_node["branching_decision"],
                    "conds": list(left_node["conds"]),
                }
                next_node_id += 1
                right_node = {
                    "node_id": next_node_id,
                    "parent": queued_node["node_id"],
                    "depth": queued_node["depth"] + 1,
                    "bounds": r,
                    "conds": r_cond,
                    "branching_decision": {"var": int(branch_var), "type": ">=", "value": float(r_val)},
                }
                node_records[right_node["node_id"]] = {
                    "node_id": right_node["node_id"],
                    "parent": queued_node["node_id"],
                    "depth": right_node["depth"],
                    "lp_obj": None,
                    "status": "queued",
                    "branching_decision": right_node["branching_decision"],
                    "conds": list(right_node["conds"]),
                }
                queue.append(left_node)
                queue.append(right_node)
            else:
                record["status"] = "fathomed_by_bound"
                
                results.append(  
                                    
                    {       
                        "fun" : res.fun,
                        "x" : res.x ,
                        "eqlin" : _normalize_linprog_duals(res.eqlin.marginals, init_node["A_eq"]),
                        "ineqlin" : _normalize_linprog_duals(res.ineqlin.marginals, init_node["A_ub"]),
                        "lower" : res.lower.marginals,
                        "upper" : res.upper.marginals,
                        "fathomed" : True,
                        "conds" : conds,
                        "node" : node,
                        "bounds" : bounds,
                        "node_id": queued_node["node_id"]

                        
                        
                    }
                        
                )
                
                
                
        if verbose:
            print(f"Number of nodes explored: {iter}")
        self._store_tree_snapshot(node_records, results, ub)
        return results

