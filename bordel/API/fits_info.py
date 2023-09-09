from astropy.io import fits

# Open the FITS file
fits_file_path = 'wcs.fits'
hdul = fits.open(fits_file_path)

# Get header of the primary HDU (Header Data Unit)
header = hdul[0].header

# Close the FITS file
hdul.close()

# List of keywords you are interested in
keywords_of_interest = ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2']

# Print only the selected keywords
print("Selected header information:")
for key in keywords_of_interest:
    if key in header:
        print(key, header[key])
    else:
        print(key, "Not present in header")
