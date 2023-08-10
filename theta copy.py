from scipy.optimize import fsolve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dec_equation_to_solve(x):
    return multiplier*x_equation * np.cos(x) + multiplier*y_equation * np.sin(x) - dec_equation

def ra_equation_to_solve(x):
    return multiplier*y_equation * np.cos(x) - multiplier*x_equation * np.sin(x) - ra_equation

def degrees_to_hours_minutes_seconds(degrees):
    total_seconds = degrees * 3600  # Total seconds in the angle
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), seconds

def deg2HMS(ra='', dec='', round=False):
  RA, DEC, rs, ds = '', '', '', ''
  if dec:
    if str(dec)[0] == '-':
      ds, dec = '-', abs(dec)
    deg = int(dec)
    decM = abs(int((dec-deg)*60))
    if round:
      decS = int((abs((dec-deg)*60)-decM)*60)
    else:
      decS = (abs((dec-deg)*60)-decM)*60
    DEC = '{0}{1} {2} {3}'.format(ds, deg, decM, decS)
  
  if ra:
    if str(ra)[0] == '-':
      rs, ra = '-', abs(ra)
    raH = int(ra/15)
    raM = int(((ra/15)-raH)*60)
    if round:
      raS = int(((((ra/15)-raH)*60)-raM)*60)
    else:
      raS = ((((ra/15)-raH)*60)-raM)*60
    RA = '{0}{1} {2} {3}'.format(rs, raH, raM, raS)
  
  if ra and dec:
    return (RA, DEC)
  else:
    return RA or DEC
  
df = pd.read_csv(r'fits\corr.fits.csv')

# Extract X and Y columns from DataFrame
ra = df['field_ra']
dec = df['field_dec']
x = df['field_x'] 
y = -df['field_y']

# Initial guess for the solution (you can change this if needed)
initial_guess = 0.8
multiplier = 0.003764988004863472
solutions = []

for i in range(len(ra)):
    for j in range(i + 1, len(ra)):
        global x_equation
        x_equation = x[i]-x[j]
        global y_equation
        y_equation = y[i]-y[j]
        global dec_equation
        dec_equation = dec[i]-dec[j]
        global ra_equation
        ra_equation = ra[i]-ra[j]
        solution_dec = fsolve(dec_equation_to_solve, initial_guess)
        solution_ra = fsolve(ra_equation_to_solve, initial_guess)
       
        print(solution_dec[0],solution_ra[0])
        solutions.append(solution_dec)
        solutions.append(solution_ra)

mean_solution = np.mean(solutions)
median_solution = np.median(solutions)
print('Mean', mean_solution, 'Median', median_solution)
a=multiplier*(3898.43017578125-x[0]) * np.cos(median_solution)
b=multiplier*(-1495.77490234375-y[0]) * np.sin(median_solution)
pollux_ra = multiplier*(2685-x[0]) * np.cos(median_solution) - multiplier*(-1818-y[0]) * np.sin(median_solution)
print('Rektascenze Polluxe je', pollux_ra, x[0], y[0])

# Example usage
degrees_ra = pollux_ra
hours, remainder = divmod(degrees_ra * 24, 1)
minutes, seconds = divmod(remainder * 60, 1)
seconds *= 60
print(f"{degrees_ra:.10f}° is equivalent to {int(hours)}h {int(minutes)}m {seconds:.1f}s")
print(deg2HMS(pollux_ra, pollux_ra))