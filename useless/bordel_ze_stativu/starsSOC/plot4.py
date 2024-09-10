import pandas as pd
import matplotlib.pyplot as plt


def make_lines(pairs,pairs_not):
    pairs_dict = dict(pairs_not)
    print(pairs_dict)
    for pair in pairs:
        try:
            x1, y1 = pairs_dict[pair[0]]
            x2, y2 = pairs_dict[pair[1]]
            x1=-x1
            x2=-x2
            plt.plot((x1, x2), (y1, y2), 'g-')
        except:
            print('Cannot make line between', pair)


def plot_data_from_csv_and_coordinates(csv_file, coordinates):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract X and Y columns from DataFrame
    x_values = -df['RA']
    y_values = df['DEC']

    # Create the scatter plot for the CSV data
    plt.scatter(x_values, y_values, marker='o', s=20, c='black', label='CSV Data')

    # Separate RA and DEC values from the list of coordinates
    
def make_constelation(coordinates):
    for coord in coordinates:
        x,y = coord[1]
        x=-x
        plt.scatter(x, y, marker='o', s=8, c='red')  # 'marker' sets the marker style, 's' sets the marker size
        plt.text(x, y, str(coord[0]), fontsize=10, ha='center', va='center', c='blue')    # Customize the plot (optional)

def show():
    plt.xlabel('Right Ascension (RA)')
    plt.ylabel('Declination (DEC)')
    plt.title('Scatter Plot of CSV Data and Coordinates')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


# Example usage:
csv_file_path = r'fits\rdls.fits.csv'
constelation = [('alf', (113.64947124999998, 31.888282222222223)), ('bet', (116.32895749999999, 28.02619888888889)), ('gam', (99.42796041666666, 16.399280277777777)), ('del', (110.03072708333332, 21.982303888888886)), ('eps', (100.98302583333333, 25.131125277777777)), ('zet', (106.02721333333332, 20.570298333333334)), ('eta', (93.719355, 22.506787222222222)), ('tet', (103.197245, 33.96125666666667)), ('iot', (111.43164833333333, 27.798079444444447)), ('kap', (116.11188958333332, 24.397996388888888)), ('lam', (109.52324541666665, 16.540385833333335)), ('mu', (95.74011166666665, 22.5135825)), ('nu', (97.24077541666665, 20.21213472222222)), ('xi', (101.32235124999998, 12.895591944444444)), ('omi', (114.79139416666666, 34.58430611111111)), ('pi', (116.87634916666666, 33.41569666666666)), ('rho', (112.27799499999999, 31.784549166666668)), ('sig', (115.82803166666666, 28.88350861111111)), ('tau', (107.78487624999998, 30.245163611111114)), ('ups', (113.98062499999997, 26.895744444444443)), ('phi', (118.37420041666664, 26.76578138888889)), ('chi', (120.87958166666667, 27.794350555555557)), ('ome', (105.60324958333331, 24.21544611111111))]
parky = [('alf','rho'),('rho', 'tau'),('tau','tet'),('tau','eps'),('eps','nu'),('eps','mu'),('mu','eta'),('tau','iot'),('iot','ups'),('ups','bet'),('ups','kap'),('ups','del'),('del','zet'),('zet','gam'),('del','lam'),('lam','xi')]
parky = [('alf','rho'),('rho', 'tau'),('tau','tet'),('tau','eps'),('eps','nu'),('eps','mu'),('mu','eta'),('tau','iot'),('iot','ups'),('ups','bet'),('ups','kap'),('ups','del'),('del','zet'),('zet','gam'),('del','lam'),('lam','xi')]
plot_data_from_csv_and_coordinates(csv_file_path, constelation)
make_lines(parky,constelation)
make_constelation(constelation)
show()