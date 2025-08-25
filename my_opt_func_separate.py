from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from scipy.linalg import lstsq
import numpy as np
import time
import os
from multiprocessing import Pool, cpu_count
import scipy.io as sio
import mat73
from scipy.optimize import Bounds
from scipy import optimize

def optimize_page(page):
    J_rate, A0, V, points = page

    def cost(J0):
        J_rate_shape = J_rate.shape
        J0 = np.expand_dims(J0, axis=0)
        J0_all = np.repeat(J0, J_rate_shape[0], axis=0) * J_rate  # 5*3

        cross0_all = np.cross(J0_all - np.repeat(A0, J_rate_shape[0], axis=0), V)

        distance = np.sqrt(np.sum(cross0_all ** 2, axis=1, keepdims=True))

        d_A0_I = np.sum((np.repeat(A0, J_rate_shape[0], axis=0) - points)**2, axis = 1)
        d_A0_J = np.sum((np.repeat(A0, J_rate_shape[0], axis=0) - J0_all)**2, axis = 1)
        t_result = d_A0_I / d_A0_J
        sorted_indices = np.argsort(t_result)[::-1]

        distance=distance[sorted_indices[:2]]

        distance = np.linalg.norm(cross0_all)**2

        return distance

    constraints = [
        {'type': 'ineq', 'fun': lambda J0: A0.flatten() - J0},
        {'type': 'ineq', 'fun': lambda J0: J0 - np.zeros(J0.shape)}
    ]

    bounds = Bounds([0, 0, 0], A0[0].flatten())

    J0_init = np.random.rand(*A0.shape)
    J0_init = J0_init.flatten()

    result = minimize(cost, J0_init, constraints=constraints, method='trust-constr', bounds = bounds, options={'disp': False, 'verbose': 0, 'maxiter': 100}, tol = 1e-6) # 去掉边界和约束，都会导致计算速度变慢

    return result.x

def main():
    opt_data_temp = mat73.loadmat('opt_data_temp.mat')

    J_rate_part = opt_data_temp['J_rate_part']
    A0 = opt_data_temp['A0']
    V_part = opt_data_temp['V_part']
    points_part = opt_data_temp['points_part']

    A0 = np.expand_dims(A0, axis=0)
    A0 = np.expand_dims(A0, axis=2)
    J_rate_shape = J_rate_part.shape
    A0 = np.repeat(A0, J_rate_shape[2], axis=2) # 变成396*1*3

    J_rate_part = np.transpose(J_rate_part, (2, 0, 1))
    A0 = np.transpose(A0, (2, 0, 1))
    V_part = np.transpose(V_part, (2, 0, 1))
    points_part = np.transpose(points_part, (2, 0, 1))

    start_time = time.perf_counter()
    count_use = min(cpu_count(), 60)
    with Pool(count_use) as pool:
        J0_result = pool.map(optimize_page, zip(J_rate_part, A0, V_part, points_part))
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    J0_result = np.array(J0_result)
    J0_result = np.transpose(J0_result, (1, 0))
    J0_result = np.expand_dims(J0_result, axis=0)

    data_dict = {'J0_result': J0_result}
    sio.savemat('opted_data_temp.mat', data_dict)

if __name__ == '__main__':
    main()
