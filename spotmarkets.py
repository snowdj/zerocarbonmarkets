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


class EmarketModel:

    def __init__(self,
                 p,
                 p_inv,
                 phi,
                 phi_prime,
                 beta=0.96,
                 delta_storage=.05,
                 delta_cap = .05,
                 grid_max_k= 100,
                 grid_max_x  =500, 
                 mu_supply = 0.5,
                 s_supply = 0.5,
                 S_bar = 100,
                 S_bar_flag = 0,
                 grid_size=1000,
                 supply_shock_size=150,
                 demand_shocks = [1,1.5],
                 demand_P = [1, 0], 
                 solved_flag = 0,
                 sol_func = np.array(0)):

        self.beta, self.delta_storage, self.delta_cap = beta, delta_storage, delta_cap
        self.p, self.p_inv = p, p_inv
        self.phi, self.phi_prime = phi, phi_prime
        self.demand_shocks, self.demand_P = demand_shocks, demand_P
        self.grid_size = grid_size
        self.s_supply = s_supply
        self.S_bar, self.S_bar_flag = S_bar, S_bar_flag
        # Set up grid for capital 

        alpha = mu_supply*(((mu_supply*(1-mu_supply))/s_supply)-1)
        beta = (1-mu_supply)*(((mu_supply*(1-mu_supply))/s_supply)-1)

        self.shocks = np.random.beta(a = alpha, b = beta, size =supply_shock_size )  # Store supply shocks
        self.gridI  = ((1e-5, grid_max_k, grid_size), (1e-3, grid_max_x, grid_size))
        self.grid   =  nodes(self.gridI) 

def time_operator_factory(og, parallel_flag=True):
    """
    A function factory for building the Coleman-Reffett and timer operator.
    Here og is an instance of EmarketModel
    """
    beta, delta_storage, delta_cap= og.beta, og.delta_storage, og.delta_cap
    p, p_inv = og.p, og.p_inv
    phi, phi_prime = og.phi, og.phi_prime
    gridI, grid, shock, grid_size= og.gridI, og.grid, og.shocks, og.grid_size
    demand_shocks, demand_P = og.demand_shocks, og.demand_P
    S_bar, S_bar_flag = og.S_bar, og.S_bar_flag

    #@njit
    def objective_price(c, rho, sigma, y):

        """
        The right hand side of the pricing operator
        """
        # First turn w into a function via interpolation
        sigma_func = lambda x,z: eval_linear(gridI, sigma[:,int(y[2])].reshape(grid_size, grid_size),np.array([x,z]))
        rho_func = [lambda points: eval_linear(gridI, rho[:,0].reshape(grid_size, grid_size), points), \
                    lambda points: eval_linear(gridI, rho[:,1].reshape(grid_size, grid_size), points)]

        x_prime =  (1-delta_storage)*(y[1] - p_inv(demand_shocks[int(int(y[2]))],c)) + shock*sigma_func(y[0],y[1])
        k_prime =  x_prime.copy()
        k_prime.fill(sigma_func(y[0],y[1]))
        prime = np.column_stack((k_prime,x_prime))
        integrand = demand_P[0]*rho_func[0](prime)\
                +demand_P[1]*rho_func[1](prime)
        Eprice = np.mean(integrand)

        if S_bar_flag == 1:
            return  np.min([np.max([beta*Eprice, p(demand_shocks[int(y[2])], y[1])]), p(demand_shocks[int(y[2])], y[1]- S_bar)])  -c
        elif S_bar_flag == 0:
            return  np.max([beta*Eprice, p(demand_shocks[int(y[2])], y[1])])  -c
    

    #@njit(nop)
    def objective_cap(c, rho, sigma, y):
        """
        The right hand side of the pricing operator
        """
        # First turn w into a function via interpolation
        #print(c)
        sigma_func = [lambda points: eval_linear(gridI, sigma[:,0].reshape(grid_size, grid_size),points),\
                         lambda points: eval_linear(gridI, sigma[:,1].reshape(grid_size, grid_size),points)]
        rho_func = [lambda points: eval_linear(gridI, rho[:,0].reshape(grid_size, grid_size), points),\
                     lambda points: eval_linear(gridI, rho[:,1].reshape(grid_size, grid_size), points)]

        x_prime =  (1-delta_storage)*(y[1]- p_inv(demand_shocks[int(y[2])], rho_func[int(y[2])](y[0:2]))) + shock*c 
        I = c - (1-delta_cap)*y[0]
        c_col = x_prime.copy()
        c_col.fill(c)
        prime_xc = np.column_stack((c_col, x_prime))
        I_prime = [(sigma_func[0](prime_xc) - (1-delta_cap)*c), (sigma_func[1](prime_xc) - (1-delta_cap)*c)]
        
        #k_prime= x_prime.copy()
        #k_prime.fill(sigma_func[int(y[2])](y[0:2]))
        prime = np.column_stack((c_col,x_prime))

        integrand = demand_P[0]*(rho_func[0](prime)*shock\
                                +(1-delta_cap)*(2*phi_prime(I_prime[0]**2)*I_prime[0] +1))\
                    + demand_P[1]*(rho_func[1](prime)*shock\
                                +(1-delta_cap)*(2*phi_prime(I_prime[1]**2)*I_prime[1] +1))
        Eprof = beta*np.mean(integrand)
        #print(np.max([beta*Eprof, (2*phi_prime((-(1-delta_cap)*y[0])**2))*(-(1-delta_cap)*y[0])]) - 2*phi_prime(I**2)*I)
        return np.max([beta*Eprof, (2*phi_prime((-(1-delta_cap)*y[0])**2))*(-(1-delta_cap)*y[0])]) - 2*phi_prime(I**2)*I


    #@njit(parallel=parallel_flag)
    def T(rho,sigma):
        """
        The time operator
        """
        rho_new = np.empty_like(rho)
        rang = np.arange(len(rho[0:,]))

        for j in range(len(demand_shocks)):
                Pool = ProcessingPool(96)
                def do_T(i):
                    y = np.append(grid[i],j)
                    #print(y)
                    # Solve for optimal c at y
                    c_star = brentq(objective_price, 1e-12, 1000000, args=(rho,sigma, y))
                    return c_star 

                rho_new[:,j]= Pool.map(do_T, rang)
                #print(i)
        return rho_new

    #@njit(parallel=parallel_flag)
    def C(rho,sigma):
        """
        The Coleman-Reffett operator
        """
        sigma_new = np.empty_like(sigma)
        rang = np.arange(len(sigma[0:,]))
        for j in range(len(demand_shocks)):
            Pool = ProcessingPool(96)
            def do_C(i):
                y = np.append(grid[i],j)
                # Solve for optimal c at y
                #print(y)
                s_star = brentq(objective_cap, 1e-12, 1000000, args=(rho,sigma, y), full_output = True)
                return s_star[0]
            sigma_new[:,j]= Pool.map(do_C, rang)
        return sigma_new

    def TC_star(v_init, tol):
        rho = v_init[0]
        sigma = v_init[1]
        rho_updated= fixed_point( lambda v: T(v, sigma), rho, error_flag = 1, tol = tol, error_name = "pricing")
        sigma_updated= fixed_point( lambda v: C(rho_updated, v), sigma,error_flag = 1, tol = tol,  error_name = "Coleman-Reffett")
        return np.array([rho_updated, sigma_updated])

    return TC_star

    #Define functions


