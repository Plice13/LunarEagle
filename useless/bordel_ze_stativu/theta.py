from scipy.optimize import fsolve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def equation_to_solve(theta):
    return 3.53961 * np.cos(theta) + 0.881903 * np.sin(theta) - 2.98910707445551

df = pd.read_csv(r'fits\corr_small.csv')

# Extract X and Y columns from DataFrame
ra = df['field_ra']
dec = df['field_dec']
x = df['field_x'] 
y = -df['field_y']

# Initial guess for the solution (you can change this if needed)
initial_guess = 0.8

for i in range(len(ra)):
    for j in range(i + 1, len(ra)):
        global x_equation
        x_equation = x[i]-x[j]
        global y_equation
        y_equation = y[i]-y[j]
        global dec_equation
        dec_equation = dec[i]-y[j]
        solution = fsolve(equation_to_solve, initial_guess)
        print(solution)
