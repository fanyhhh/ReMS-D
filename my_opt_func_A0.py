from scipy.optimize import minimize
from scipy.linalg import lstsq
import numpy as np
import time
from scipy.optimize import Bounds
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def my_opt_func_A0(A0_init, A0_min, A0_max, A, B, C, D):    
    def cost(A0):
        distance = abs(A * A0[0] + B * A0[1] + C * A0[2] + D)
        distance = np.linalg.norm(distance)
        return distance

    constraints = [
        {'type': 'ineq', 'fun': lambda A0: A0_max.flatten() - A0.flatten()},
        {'type': 'ineq', 'fun': lambda A0: A0.flatten() - A0_min.flatten()},
    ]
    bounds = Bounds(A0_min.flatten(), A0_max.flatten())

    start_time = time.perf_counter()
    result = minimize(cost, A0_init, method='trust-constr', bounds = bounds, options={'disp': False, 'verbose': 0, 'maxiter': 1000}, tol = 1e-6) # 去掉边界和约束，都会导致计算速度变慢
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")

    return result.x