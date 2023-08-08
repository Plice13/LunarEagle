#for plotting 1000 stars detected on webside


import pandas as pd
import matplotlib.pyplot as plt

def plot_data_from_csv(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract X and Y columns from DataFrame
    x_values = df['X']
    y_values = -df['Y']
    print(len(x_values))
    flux = df['FLUX']
    # Create the scatter plot
    for i in range(80):
        if flux[i] >= 20:
            plt.scatter(x_values[i], y_values[i], marker='o', s=20, c='black')  # 'marker' sets the marker style, 's' sets the marker size
        

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
csv_file_path = r'fits\axy.fits.csv'
plot_data_from_csv(csv_file_path)
