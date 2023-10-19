#plot 3 is better

import matplotlib.pyplot as plt

# Coordinates in the format (RA, DEC)
coordinates = [(113.64947124999998, 31.888282222222223), (116.32895749999999, 28.02619888888889), (99.42796041666666, 16.399280277777777), (110.03072708333332, 21.982303888888886), (100.98302583333333, 25.131125277777777), (106.02721333333332, 20.570298333333334), (93.719355, 22.506787222222222), (111.43164833333333, 27.798079444444447), (116.11188958333332, 24.397996388888888), (109.52324541666665, 16.540385833333335), (95.74011166666665, 22.5135825), (97.24077541666665, 20.21213472222222), (101.32235124999998, 12.895591944444444)]
# Separate RA and DEC values from the list of coordinates
ra_values = [-coord[0] for coord in coordinates]
dec_values = [coord[1] for coord in coordinates]

# Plot the coordinates
plt.scatter(ra_values, dec_values, marker='o', color='b', label='Coordinates')

# Customize the plot
plt.xlabel('Right Ascension (RA)')
plt.ylabel('Declination (DEC)')
plt.title('Coordinates Scatter Plot')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
