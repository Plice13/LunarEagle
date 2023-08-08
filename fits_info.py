from astropy.io import fits

def print_hdu_info(input_file):
    with fits.open(input_file) as hdul:
        hdul.info()

input_fits_file = 'fits\wcs.fits'
print_hdu_info(input_fits_file)
