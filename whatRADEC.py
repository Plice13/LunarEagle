from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u

def get_ra_dec_from_star_name(star_name):
    # Query the Simbad database for the star information
    result_table = Simbad.query_object(star_name)

    if result_table is not None:
        # Get RA and DEC from the result
        ra_str = result_table['RA'][0]
        dec_str = result_table['DEC'][0]

        # Convert RA and DEC from sexagesimal format to degrees
        coords = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))

        return coords.ra.deg, coords.dec.deg
    else:
        return None, None

# Example usage
star_name = "Polaris"
ra, dec = get_ra_dec_from_star_name(star_name)

if ra is not None and dec is not None:
    print(f"The star '{star_name}' has RA: {ra} degrees and DEC: {dec} degrees.")
else:
    print(f"Star '{star_name}' not found in the catalog.")
