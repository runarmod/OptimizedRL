
# No handling of integer being none yet
class Node:
    def __init__(
            self,
            c,
            A_ub = None,
            b_ub = None,
            A_eq = None,
            b_eq = None,
            bounds = (0,None),
            integer = None
            ):
        
        self.c = c
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.bounds = bounds
        self.integer = integer

        self.sol_obj_val = None
        self.sol_obj_vars = None
        self.parent = None
        self.children = []
        

    
        