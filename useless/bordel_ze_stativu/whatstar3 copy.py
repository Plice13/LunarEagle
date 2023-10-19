from astroquery.simbad import Simbad
import astropy.coordinates as coord
import astropy.units as u
import pandas as pd


def get_star_name(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract X and Y columns from DataFrame
    ra_deg = df['index_ra']
    dec_deg = df['index_dec']
    name_and_coordinations =[]
    for i in range(len(ra_deg)):
        # Use Simbad to query the star by its coordinates
        result_table = Simbad.query_region(coord.SkyCoord(ra_deg[i], dec_deg[i],

                                    unit=(u.deg, u.deg), frame='icrs'),

                                    radius=0.05 * u.deg)
        #print(result_table)
        if result_table is not None:
            # Get the star name from the result
            star_name = result_table['MAIN_ID'][0]
        else:
            star_name = "---" 
        name_and_coordinations.append((star_name, (ra_deg[i],dec_deg[i])))
    return name_and_coordinations

csv_file_path = r'fits\corr.fits.csv'
get_star_name(csv_file_path)

