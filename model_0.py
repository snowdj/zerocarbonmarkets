import numpy as np
from interpolation import interp
#from scipy.optimize import brentq
from quantecon.optimize.scalar_maximization import brent_max
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from itertools import product
from numba import njit, prange
from pathos.multiprocessing import ProcessingPool
from fixedpoint import fixed_point

from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
from spotmarkets import EmarketModel, time_operator_factory


import dill as pickle 




@njit
def pr(e,x):
    "Inverse of the demand function"
    rho_demand = -1
    return np.exp(e)*x**rho_demand


@njit
def p_inv(e,x):
    "The demand function. Gives demand for price and demand shock"
    rho_demand = -1
    return np.exp(-e/rho_demand)*x**(1/rho_demand)


@njit
def phi(x):
    "Investment cost function"
    return 2*x


@njit
def phi_prime(x):
    "Derivative of the cost function"
    return 2

"""
Model with high low supply variance and no demand shocks.
Demand shock set to 1
No upper bound on storage
"""

if __name__ == '__main__':

    og = EmarketModel(s_supply =.1, 
                        grid_size = 350,
                        grid_max_x = 300, 
                        S_bar_flag = 0, 
                        p=pr, 
                        p_inv=p_inv, 
                        phi=phi, 
                        phi_prime=phi_prime, 
                        demand_shocks = [1,1.5],
                        demand_P = [1, 0])

    TC_star =time_operator_factory(og) 


    # n = 15
    sigma = np.ones(og.grid.shape)  # Set initial condition
    rho = np.ones(og.grid.shape)
    v_init = np.array([rho,sigma])

    og.sol_func= fixed_point( lambda v: TC_star(v, tol = 1e-5), v_init, error_flag = 1, tol = 1e-5,  error_name = "main", maxiter =20)


    pickle.dump(og, open("/home/akshay_shanker/model_0.mod","wb"))
