from scipy.optimize import fsolve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def equation_to_solve(x):
    return 3.53961 * np.cos(x) + 0.881903 * np.sin(x) - 2.98910707445551

# Initial guess for the solution (you can change this if needed)
initial_guess = 0.0

# Using fsolve to find the solution
solution = fsolve(equation_to_solve, initial_guess)

print("Approximate solution for x:", solution[0])

df = pd.read_csv(r'fits\corr_small.csv')

# Extract X and Y columns from DataFrame
ra = df['index_ra']
dec = df['index_dec']
x = df['index_x'] 
y = -df['index_y']

print(dec[1], dec[0])
print(dec[1]-dec[0])