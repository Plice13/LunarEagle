#plots star in x, y and write their RA, DEC
#build on plot 1

import pandas as pd
import matplotlib.pyplot as plt

def plot_data_from_csv(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract X and Y columns from DataFrame
    x_values = df['field_x']
    y_values = -df['field_y']
    ra, dec = df['field_ra'], df['field_dec']
    print(len(x_values))
    flux = df['FLUX']
    # Create the scatter plot
    for i in range(len(x_values)):
        plt.scatter(x_values[i], y_values[i], marker='o', s=1.01**flux[i]*8, c='black')  # 'marker' sets the marker style, 's' sets the marker size
        plt.text(x_values[i], y_values[i], str(round(ra[i],2))+'\n'+str(round(dec[i],2)), fontsize=10, ha='center', va='center', c='blue')    # Customize the plot (optional)
      

#    plt.scatter(x_values, y_values, marker='o', s=10)  # 'marker' sets the marker style, 's' sets the marker size

    # Customize the plot (optional)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of X and Y')
    plt.grid(True)

    plt.xlim(0, 5616)  # Example values, adjust as needed
    plt.ylim(-3744, 0)   # Example values, adjust as needed



    # Show the plot
    plt.show()

# Example usage:
csv_file_path = r'fits\corr.fits.csv'
plot_data_from_csv(csv_file_path)
