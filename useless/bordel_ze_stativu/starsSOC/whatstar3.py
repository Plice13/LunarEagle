from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pandas as pd


def get_star_name(ra_deg, dec_deg):
    # Convert RA and DEC to radians
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)

    # Create SkyCoord object with radians
    coords = SkyCoord(ra=ra_rad * u.rad, dec=dec_rad * u.rad, frame='icrs')

    # Use Simbad to query the star by its coordinates
    result_table = Simbad.query_region(coord.SkyCoord(ra_deg, dec_deg,

                                   unit=(u.deg, u.deg), frame='icrs'),

                                   radius=0.05 * u.deg)
    #print(result_table)
    if result_table is not None:
        # Get the star name from the result
        star_name = result_table['MAIN_ID'][0]
        return star_name
    else:
        return "Star not found in the catalog."

def get_ra_dec(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract X and Y columns from DataFrame
    x_values = df['index_ra']
    y_values = df['index_dec']
    return x_values, y_values

csv_file_path = r'fits\corr.fits.csv'
ra, dec = get_ra_dec(csv_file_path)
for i in range(len(ra)):
    star_name = get_star_name(ra[i], dec[i])
    print(f"The star at RA {ra[i]} and DEC {dec[i]} is named: {star_name}")

