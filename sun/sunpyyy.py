import sunpy.map
from sunpy.data.sample import AIA_171_IMAGE
from astropy import units as u

aiamap = sunpy.map.Map(AIA_171_IMAGE)
smap = aiamap.submap([-1200, -1000]*u.arcsec, [-200, 0]*u.arcsec)
smap.peek(draw_grid=True)
