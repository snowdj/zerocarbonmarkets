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
from tabulate import tabulate


from interpolation.splines import UCGrid, CGrid, nodes, eval_linear

import dill as pickle


"""
Unpacks model policy functions and convets to timeseries results and plots
"""


def runres(filein):

    alpha = og.mu_supply*(((og.mu_supply*(1-og.mu_supply))/og.s_supply)-1)
    beta = (1-og.mu_supply)*(((og.mu_supply*(1-og.mu_supply))/og.s_supply)-1)

	og =pickle.load(open("/home/akshay_shanker/model_{}.mod".format(filein),"rb"))

	sigma = og.sol_func[1]
	rho = og.sol_func[0]

	sigma_func = [lambda points: eval_linear(og.gridI, sigma[:,0].reshape(og.grid_size, og.grid_size),points),\
                 lambda points: eval_linear(og.gridI, sigma[:,1].reshape(og.grid_size, og.grid_size),points)]
	rho_func = [lambda points: eval_linear(og.gridI, rho[:,0].reshape(og.grid_size, og.grid_size), points),\
             lambda points: eval_linear(og.gridI, rho[:,1].reshape(og.grid_size, og.grid_size), points)]


	T= 10000000
	time = np.arange(0,T,1)
	x,k,price, d, s = np.zeros(T),np.zeros(T),np.zeros(T), np.zeros(T), np.zeros(T)
	e = np.random.randint(0, 2, size=T) 
	z = np.random.beta(a = alpha, b = beta, size =T )

	k[0] = 37
	x[0] = 100

	for i in range(T):
	    price[i] = rho_func[e[i]](np.array([k[i],x[i]]))
	    d[i] = og.p_inv(e[i],price[i])
	    s[i] = x[i] - k[i]*z[i]
	    if i<T-1:
		    k[i+1] = sigma_func[e[i]](np.array([k[i],x[i]]))
		    x[i+1] = (1-og.delta_storage)*x[i] - og.p_inv(e[i],price[i]) + z[i+1]*k[i+1] 


	results = {}

	results['model'] = og
	results['mean_price'] = np.mean(price)
	results['mean_capacity'] = np.mean(k)
	results['mean_stor'] = np.mean(s)
	results['mean_demand'] = np.mean(d)
	results['mean_supply'] = np.mean(z)

	results['var_price']= np.std(price)
	results['var_capacity'] = np.std(k)
	results['var_stor'] = np.std(s)
	results['var_demand'] = np.std(d)
	results['var_supply'] = np.std(z)

	results['cov_zd'] = np.corrcoef(z,d)
	results['cov_pd'] = np.corrcoef(price,d)
	results['cov_pz'] = np.corrcoef(price,z)
	results['cov_sz'] = np.corrcoef(s,z)

	results['capital'] = k
	results['stock'] = x
	results['price'] = price
	results['demand'] = d
	results['stored'] = s
	results['demshock'] = e
	results['stockout'] =  (s< (np.mean(d))).sum()/T

	f, axarr = plt.subplots(3,2)
	axarr[0,0].plot(time, price,  linewidth=.6)
	axarr[0,0].set_ylabel('Price', fontsize = 8)
	axarr[1,0].plot(time, d,  linewidth=.6)
	axarr[1,0].set_ylabel('Eqm. demand', fontsize = 8)
	axarr[2,0].plot(time, k,  linewidth=.6)
	axarr[2,0].set_ylabel('Capacity', fontsize = 8)
	axarr[0,1].plot(time, x,  linewidth=.6)
	axarr[0,1].set_ylabel('Stored power', fontsize = 8)
	axarr[1,1].plot(time, z,  linewidth=.6)
	axarr[1,1].set_ylabel('Supply shock ', fontsize = 8)
	axarr[2,1].plot(time, e,  linewidth=.6)
	axarr[2,1].set_ylabel('Demand shock', fontsize = 8)
	f.tight_layout()


	plt.savefig("/home/akshay_shanker/model_{}_sim.png".format(filein))



 
	return results 



