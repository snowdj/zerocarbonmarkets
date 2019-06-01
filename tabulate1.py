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

from results import runres


from interpolation.splines import UCGrid, CGrid, nodes, eval_linear

import dill as pickle


resmod1 = runres("var_{}".format(0.1))
resmod0 = runres("var_{}".format(0.5))
resmod2 = runres("var_{}".format(1))
resmod3 = runres("var_{}".format(1.25))





mod1 = ["Low shock var", resmod1['mean_supply'],resmod1['mean_price'],resmod1['mean_capacity'], resmod1['mean_demand'],resmod1['mean_stor'], resmod1['stockout'] ]
mod2 = ["Medium shock var", resmod0['mean_supply'],resmod0['mean_price'],resmod0['mean_capacity'], resmod0['mean_demand'],resmod0['mean_stor'], resmod0['stockout'] ]
mod4 = ["High shock var", resmod3['mean_supply'],resmod3['mean_price'],resmod3['mean_capacity'], resmod3['mean_demand'],resmod3['mean_stor'], resmod3['stockout'] ]



header = ["Supply", "Price", "Capacity", "Eqm. Dem.", "Storage", "Stockout %"]


table= [mod1, mod2, mod4]

print(tabulate(table, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))

restab = open("vresultsmean_tab.tex", 'w')

restab.write(tabulate(table, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))

restab.close()


mod1 = ["Low shock var", resmod1['var_supply'],resmod1['var_price'],resmod1['var_capacity'], resmod1['var_demand'],resmod1['var_stor']]
mod2 = ["Medium shock var", resmod0['var_supply'],resmod0['var_price'],resmod0['var_capacity'], resmod0['var_demand'],resmod0['var_stor']]
mod4 = ["High shock var", resmod3['var_supply'],resmod3['var_price'],resmod3['var_capacity'], resmod3['var_demand'],resmod3['var_stor']]


header = ["Supply", "Price", "Capacity", "Eqm. Dem.", "Storage"]


table= [mod1, mod2, mod4]

print(tabulate(table, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))

restab = open("vresultsvar_tab.tex", 'w')

restab.write(tabulate(table, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))
restab.close()

