

import numpy as np


def gradient(node):
    """
    
        Calculates gradient of objective function w.r.t parameters of linear program.
        Returns vector of gradient in following order:
            
            dfdc
            dfdb_ub
            dfdA_ub
            dfdb_eq
            dfdA_eq


    
    """
    # Dont need node as input can just take in solution
    solution = node["sol"]
    if solution is None:
        raise Exception("Solution cannot be none")
    dfdtheta = []

    dfdtheta.extend(solution.x)

    rhs_ub_grads = solution.ineqlin.marginals
    if len(rhs_ub_grads) > 0:
        dfdtheta.extend(rhs_ub_grads)
        for var in solution.x:
            for lag_mult in rhs_ub_grads:
                dfdtheta.append(-var*lag_mult)
    rhs_eq_grads = solution.eqlin.marginals
    if len(rhs_eq_grads) > 0:
        dfdtheta.extend(rhs_eq_grads)
        for var in solution.x:
            for lag_mult in rhs_eq_grads:
                dfdtheta.append(-var*lag_mult)
    return np.array(dfdtheta)




    




