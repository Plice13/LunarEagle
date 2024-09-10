import pandas as pd
import matplotlib.pyplot as plt

def plot_data_from_csv_and_coordinates(csv_file, coordinates):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract X and Y columns from DataFrame
    x_values = -df['RA']
    y_values = df['DEC']

    # Create the scatter plot for the CSV data
    plt.scatter(x_values, y_values, marker='o', s=20, c='black', label='CSV Data')

    # Separate RA and DEC values from the list of coordinates
    ra_values = [-coord[0] for coord in coordinates]
    dec_values = [coord[1] for coord in coordinates]

    # Create the scatter plot for the coordinates data
    plt.scatter(ra_values, dec_values, marker='o', s=2, c='red', label='Coordinates')

    # Customize the plot (optional)
    plt.xlabel('Right Ascension (RA)')
    plt.ylabel('Declination (DEC)')
    plt.title('Scatter Plot of CSV Data and Coordinates')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

# Example usage:
csv_file_path = r'fits\rdls.fits.csv'
coordinates = [(113.64947124999998, 31.888282222222223), (116.32895749999999, 28.02619888888889), (99.42796041666666, 16.399280277777777), (110.03072708333332, 21.982303888888886), (100.98302583333333, 25.131125277777777), (106.02721333333332, 20.570298333333334), (93.719355, 22.506787222222222), (111.43164833333333, 27.798079444444447), (116.11188958333332, 24.397996388888888), (109.52324541666665, 16.540385833333335), (95.74011166666665, 22.5135825), (97.24077541666665, 20.21213472222222), (101.32235124999998, 12.895591944444444)]

plot_data_from_csv_and_coordinates(csv_file_path, coordinates)
