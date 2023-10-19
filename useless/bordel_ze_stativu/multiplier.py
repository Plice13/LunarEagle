#from all pair of stars computes ra,dec-axis rotation with respect to x,y-axis 

from scipy.optimize import fsolve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r'fits\corr.fits.csv')

# Extract X and Y columns from DataFrame
ra = df['index_ra']
dec = df['index_dec']
x = df['index_x'] 
y = -df['index_y']

multipliers = []

for i in range(len(ra)):
    for j in range(i + 1, len(ra)):
        distance_in_px = np.sqrt(((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2))
        multiplier = np.sqrt(((ra[j] - ra[i]) ** 2 + (dec[j] - dec[i]) ** 2) / ((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2))
        #multipliers.append((multiplier,(i,j)))
        multipliers.append(multiplier)
        red = round(distance_in_px/5700,1)
        color = (red, 0.0, 0.0)  
        plt.plot(multiplier, 1 , marker='o', c = (color), alpha=0.25)


# Plot the multipliers on a scatter plot
plt.xlabel('Pair Index')
plt.ylabel('Multiplier Value')
plt.title('Multipliers of All Possible Pairs')
plt.legend()
plt.grid(True)
plt.show()
print(multipliers)
# Calculate the mean of all multipliers
mean_multiplier = np.mean(multipliers)
median_multiplier = np.median(multipliers)
print("Mean Multiplier:", mean_multiplier)
print("Median Multiplier:", median_multiplier)