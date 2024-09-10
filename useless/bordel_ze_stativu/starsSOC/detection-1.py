from astropy.stats import sigma_clipped_stats

from photutils.datasets import load_star_image

hdu = load_star_image()  

data = hdu.data[0:401, 0:401]  

mean, median, std = sigma_clipped_stats(data, sigma=3.0)  

print((mean, median, std))

from photutils.detection import DAOStarFinder

daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)  

sources = daofind(data - median)  

for col in sources.colnames:  

    if col not in ('id', 'npix'):

        sources[col].info.format = '%.2f'  # for consistent table output

sources.pprint(max_width=76)  