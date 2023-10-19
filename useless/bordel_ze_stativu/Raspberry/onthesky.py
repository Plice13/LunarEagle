import os
from astrometry.blind import plotstuff as ps
from astrometry.util import util as anutil

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot location of a WCS on the sky')
    parser.add_argument('wcs_file', help='WCS input file')
    parser.add_argument('--out', default='zoom-',
                        help='Output plot filename base; default: "zoom-"')
    parser.add_argument('--hd', help='Henry Draper catalog filename (fetch from http://data.astrometry.net/hd.fits)')
    parser.add_argument('--tycho2', help='Tycho-2 catalog filename (fetch from http://data.astrometry.net/tycho2.kd)')
    opt = parser.parse_args()

    wcsfn = opt.wcs_file
    plotbase = opt.out
    wcsfn = 'wcs.fits'

    kwargs = dict()
    if opt.hd:
        kwargs.update(hd_cat=opt.hd)
    if opt.tycho2:
        kwargs.update(tycho2_cat=opt.tycho2)
    
    # this is assumed in the plot_aitoff_wcs_outline and plot_wcs_outline calls
    wcsext = 0

    wcs = anutil.Tan(wcsfn, wcsext)

    # zoom 0
    zoomin = wcs.radius() < 15.
    plotfn = plotbase + '0.png'
    plot_aitoff_wcs_outline(wcsfn, plotfn, zoom=zoomin)

    # zoom 1
    zoomin = wcs.radius() < 1.5
    plotfn = plotbase + '1.png'
    plot_wcs_outline(wcsfn, plotfn, zoom=zoomin, **kwargs)

    # zoom 2
    zoomin = wcs.radius() < 0.15
    plotfn = plotbase + '2.png'
    plot_wcs_outline(wcsfn, plotfn, width=3.6, grid=1, zoom=zoomin,
                     zoomwidth=0.36, hd=True, hd_labels=False,
                     tycho2=False, **kwargs)
    # hd=True is too cluttered at this level

    # zoom 3
    plotfn = plotbase + '3.png'
    plot_wcs_outline(wcsfn, plotfn, width=0.36, grid=0.1, zoom=False,
                     hd=True, hd_labels=True, tycho2=True, **kwargs)
    

def plot_wcs_outline(wcsfn, plotfn, W=256, H=256, width=36, zoom=True,
                     zoomwidth=3.6, grid=10, hd=False, hd_labels=False,
                     tycho2=False,
                     hd_cat=None,
                     tycho2_cat=None,
                     ):
    wcs = anutil.Tan(wcsfn, 0)
    ra,dec = wcs.radec_center()

    plot = ps.Plotstuff(outformat='png', size=(W, H), rdw=(ra,dec,width))
    plot.linestep = 1.
    plot.color = 'verydarkblue'
    plot.plot('fill')

    plot.fontsize = 12
    #plot.color = 'gray'
    # dark gray
    plot.rgb = (0.3,0.3,0.3)
    if grid is not None:
        plot.plot_grid(*([grid]*4))

    plot.rgb = (0.4, 0.6, 0.4)
    ann = plot.annotations
    ann.NGC = ann.bright = ann.HD = 0
    ann.constellations = 1
    ann.constellation_labels = 1
    ann.constellation_labels_long = 1
    plot.plot('annotations')
    plot.stroke()
    ann.constellation_labels = 0
    ann.constellation_labels_long = 0
    ann.constellation_lines = 0

    ann.constellation_markers = 1
    plot.markersize = 3
    plot.rgb = (0.4, 0.6, 0.4)
    plot.plot('annotations')
    plot.fill()
    ann.constellation_markers = 0

    ann.bright_labels = False
    ann.bright = True
    plot.markersize = 2
    if zoom >= 2:
        ann.bright_labels = True
    plot.plot('annotations')
    ann.bright = False
    ann.bright_labels = False
    plot.fill()

    if hd and hd_cat:
        ann.HD = True
        ann.HD_labels = hd_labels
        ps.plot_annotations_set_hd_catalog(ann, hd_cat)
        plot.plot('annotations')
        plot.stroke()
        ann.HD = False
        ann.HD_labels = False

    if tycho2 and tycho2_cat:
        from astrometry.libkd.spherematch import tree_open, tree_close, tree_build_radec, tree_free, trees_match
        from astrometry.libkd import spherematch_c
        from astrometry.util.starutil_numpy import deg2dist, xyztoradec
        import numpy as np
        import sys
        kd = tree_open(tycho2_cat)
        # this is a bit silly: build a tree with a single point, then do match()
        kd2 = tree_build_radec(np.array([ra]), np.array([dec]))
        r = deg2dist(width * np.sqrt(2.) / 2.)
        #r = deg2dist(wcs.radius())
        I,nil,nil = trees_match(kd, kd2, r, permuted=False)
        del nil
        #print 'Matched', len(I)
        xyz = kd.get_data(I.astype(np.uint32))
        del I
        #print >>sys.stderr, 'Got', xyz.shape, xyz
        tra,tdec = xyztoradec(xyz)
        #print >>sys.stderr, 'RA,Dec', ra,dec
        plot.apply_settings()
        for r,d in zip(tra,tdec):
            plot.marker_radec(r,d)
        plot.fill()

    ann.NGC = 1
    plot.plot('annotations')
    ann.NGC = 0

    plot.color = 'white'
    plot.lw = 3
    out = plot.outline
    out.wcs_file = wcsfn
    plot.plot('outline')

    if zoom:
        # MAGIC width, height are arbitrary
        zoomwcs = anutil.anwcs_create_box(ra, dec, zoomwidth, 1000,1000)
        out.wcs = zoomwcs
        plot.lw = 1
        plot.dashed(3)
        plot.plot('outline')

    plot.write(plotfn)

def plot_aitoff_wcs_outline(wcsfn, plotfn, W=256, zoom=True):
    #anutil.log_init(3)
    H = int(W//2)
    # Create Hammer-Aitoff WCS of the appropriate size.
    wcs = anutil.anwcs_create_allsky_hammer_aitoff(0., 0., W, H)
 
    plot = ps.Plotstuff(outformat='png', size=(W, H))
    plot.wcs = wcs
    plot.linestep = 1.
    plot.color = 'verydarkblue'
    plot.apply_settings()
    plot.line_constant_ra(180, -90, 90)
    plot.line_constant_ra(-180, 90, -90)
    plot.fill()

    #plot.plot_grid(60, 30, 60, 30)
    plot.fontsize = 12
    ras = [-180, -120, -60, 0, 60, 120, 180]
    decs = [-60, -30, 0, 30, 60]
    # dark gray
    plot.rgb = (0.3,0.3,0.3)
    plot.apply_settings()
    for ra in ras:
        plot.line_constant_ra(ra, -90, 90)
        plot.stroke()
    for dec in decs:
        plot.line_constant_dec(dec, -180, 180)
        plot.stroke()

    plot.color = 'gray'
    plot.apply_settings()
    for ra in ras:
        plot.move_to_radec(ra, 0)
        plot.text_radec(ra, 0, '%i'%((ra+360)%360))
        plot.stroke()
    for dec in decs:
        if dec != 0:
            plot.move_to_radec(0, dec)
            plot.text_radec(0, dec, '%+i'%dec)
            plot.stroke()
    
    plot.color = 'white'
    plot.lw = 3
    out = plot.outline
    #out.fill = 1
    out.wcs_file = wcsfn
    anutil.anwcs_print_stdout(out.wcs)
    plot.plot('outline')

    # Not helpful to add constellations in this view
    #ann = plot.annotations
    #ann.NGC = ann.bright = ann.HD = 0
    #ann.constellations = 1
    #plot.plot('annotations')

    if zoom:
        owcs = anutil.Tan(wcsfn, 0)
        # MAGIC 15 degrees radius
        #if owcs.radius() < 15.:
        if True:
            ra,dec = owcs.radec_center()
            # MAGIC 36-degree width zoom-in
            # MAGIC width, height are arbitrary
            zoomwcs = anutil.anwcs_create_box(ra, dec, 36, 1000,1000)
            out.wcs = zoomwcs
            #plot.color = 'gray'
            plot.lw = 1
            plot.dashed(3)
            plot.plot('outline')

    plot.write(plotfn)
    
    
    
if __name__ == '__main__':
    main()

