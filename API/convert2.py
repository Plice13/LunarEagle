#absolutně k ničemu

import os
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

def fits_to_csv(hdul, output_csv_file):
    # Print the header information to understand the structure of the FITS file
    hdul.info()

    # Read the FITS data
    try:
        data = hdul[1].data
    except IndexError:
        print(f"Error: The FITS file does not have the required HDU.")
        hdul.close()
        return

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_csv_file, index=False)

def fits_to_jpg(hdul, output_jpg_file):
    # Read the FITS data
    data = hdul[0].data

    # Check if data is None
    if data is None:
        print(f"Error: Image data in FITS file is None.")
        return

    # Check the data type
    if data.dtype.name != 'float64':
        print(f"Error: The image data in FITS file is not of type 'float64'.")
        return

    # Save to JPG
    plt.imshow(data, cmap='gray')  # Assuming the data is a 2D grayscale image
    plt.axis('off')  # Turn off axis
    plt.savefig(output_jpg_file, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

# Example usage:
input_fits_file = r'C:\Users\Uživatel\Desktop\starsSOC\fits\wcs.fits'
output_csv_file = input_fits_file + '.csv'
output_jpg_file = input_fits_file + '.jpg'

# Check if the input file exists
if os.path.exists(input_fits_file):
    # Open the FITS file
    hdul = fits.open(input_fits_file)

    # Convert FITS to CSV
    fits_to_csv(hdul, output_csv_file)

    # Convert FITS to JPG
    fits_to_jpg(hdul, output_jpg_file)

    # Close the FITS file
    hdul.close()

    print("Conversion completed successfully.")
else:
    print("Input FITS file not found.")
