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

# Example usage:
input_fits_file = 'wcs.fits'
output_csv_file = 'output_table.csv'

fits_to_csv(input_fits_file, output_csv_file)
