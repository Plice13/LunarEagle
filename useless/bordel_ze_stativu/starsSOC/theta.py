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

df = pd.read_csv(r'fits\corr.fits.csv')

# Extract X and Y columns from DataFrame
ra = df['field_ra']
dec = df['field_dec']
x = df['field_x'] 
y = -df['field_y']

multipliers = []

for i in range(len(ra)):
    for j in range(i + 1, len(ra)):
        multiplier = np.sqrt(((ra[j] - ra[i]) ** 2 + (dec[j] - dec[i]) ** 2) / ((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2))
        multipliers.append(multiplier)
        plt.plot(multiplier, 1 , marker='o', color='b', alpha=0.05)


# Plot the multipliers on a scatter plot
plt.xlabel('Pair Index')
plt.ylabel('Multiplier Value')
plt.title('Multipliers of All Possible Pairs')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the mean of all multipliers
mean_multiplier = np.mean(multipliers)

print(multipliers)
print("Mean Multiplier:", mean_multiplier)