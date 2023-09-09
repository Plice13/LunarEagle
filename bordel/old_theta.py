#from all pair of stars computes ra,dec-axis rotation with respect to x,y-axis 

from scipy.optimize import fsolve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def equation_to_solve(x):
    return 3.53961 * np.cos(x) + 0.881903 * np.sin(x) - 2.98910707445551

# Initial guess for the solution (you can change this if needed)
initial_guess = 0.8

# Using fsolve to find the solution
solution = fsolve(equation_to_solve, initial_guess)

print("Approximate solution for x:", solution)
