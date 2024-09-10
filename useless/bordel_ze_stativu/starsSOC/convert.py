import pandas as pd
from astropy.io import fits

def fits_to_csv(input_file, output_file):
    # Read the FITS file
    with fits.open(input_file) as hdul:
        data = hdul[1].data

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_file, index=False)

def fits_to_excel(input_file, output_file):
    # Read the FITS file
    with fits.open(input_file) as hdul:
        data = hdul[1].data

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save to Excel
    df.to_excel(output_file, index=False)

# Example usage:
input_fits_file = 'axy2.fits'
output_csv_file = 'output_table.csv'
output_excel_file = 'output_table.xlsx'

fits_to_csv(input_fits_file, output_csv_file)
fits_to_excel(input_fits_file, output_excel_file)
