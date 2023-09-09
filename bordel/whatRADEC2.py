from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u

def get_ra_dec_from_star_name(star_name):
    # Query the Simbad database for the star information
    result_table = Simbad.query_object(star_name)
    print(result_table)
    if result_table is not None:
        # Get RA and DEC from the result
        ra_str = result_table['RA'][0]
        dec_str = result_table['DEC'][0]

        # Convert RA and DEC from sexagesimal format to degrees
        coords = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))

        return coords.ra.deg, coords.dec.deg
    else:
        return None, None

idk = []
#greek = ['alf', 'bet', 'gam', 'del', 'eps', 'zet', 'eta', 'tet', 'iot', 'kap', 'lam', 'mu', 'nu', 'xi', 'omi', 'pi', 'rho', 'sig', 'tau', 'ups', 'phi', 'chi', 'psi', 'ome']
# Example usage
greek = ['alf']
for i in greek:
    try:
        star_name = i+" Gem "
        print('SN', star_name)
        ra, dec = get_ra_dec_from_star_name(star_name)
        if ra is not None and dec is not None:
            #print(f"The star '{star_name}' has RA: {ra} degrees and DEC: {dec} degrees.")
            idk.append((i,(ra, dec)))
        else:
            #print(f"Star '{star_name}' not found in the catalog.")
            print(f"Star '{star_name}' not exist.")
    except:    
        print(f"Star '{star_name}' not exist.")

print(idk)
            

