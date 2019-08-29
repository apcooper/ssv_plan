import healpy as hp
import matplotlib
import matplotlib.pyplot as pl
import desimodel
import desiutil
import numpy as np
import warnings
import os
import sys
import glob

from   astropy.io import fits
import astropy.units as u
import astropy.coordinates as coord
from   astropy.table import Table

try:
    import spherical_geometry
    from spherical_geometry import polygon as spoly
    WITH_SPHERICAL_GEOMETRY = True
except ImportError:
    WITH_SPHERICAL_GEOMETRY = False
    

SSV_AREAS = dict()

_FIELD_TABLE_UNUSED = """
M53;	  13 12 55.25;  +18 10 05.4; icrs; b; s; hour,deg
NGC 5053; 13 16 27.09;  +17 42 00.9; icrs; b; s; hour,deg
M62;      17 01 12.80; 	-30 06 49.4; icrs; b; s; hour,deg
"""

_OLD_FIELD_TABLE = """
name;          lon;          lat;        frame;    c; s; unit
DRACO;         260.051625;   57.915361;  icrs;     b; x; deg
MWS1;          145.0;        30.0;       icrs;     m; s; deg
MWS2;          148.0;        32.25;       icrs;     m; s; deg
ORPHAN-A;      158.5; 	10.0;  icrs;     m; s; deg
G15;           217.5;    1.0;  icrs;     m; s; deg
G12;           180.0;   -1.0;  icrs;     m; s; deg
M53+N5053;     199.0;   18.3;  icrs;     m; s; deg
NGP;             0.0;   90.0;  galactic; b; +; deg
M92;   17 17 07.39;     +43 08 09.4; icrs; b; s; hour,deg
M13;   16 41 41.24;  	+36 27 35.5; icrs; b; s; hour,deg
M71;   19 53 46.49;     +18 46 45.1; icrs; b; s; hour,deg
M2;    21 33 27.02; 	-00 49 23.7; icrs; b; s; hour,deg
M5;         15 18 33.22;  	+02 04 51.7; icrs; b; s; hour,deg
M15;        21 29 58.33;  	+12 10 01.2; icrs; b; s; hour,deg
NGC 4147;   12 10 06.30; 	+18 32 33.5; icrs; b; s; hour,deg
M67;        08 51 18.00;	+11 48 00; icrs; g; o; hour,deg
HYADES;	    04 26 54.00; 	+15 52 00.0; icrs; g; o; hour,deg
NGC 6791;	19 20 53.00;	+37 46 18.0; icrs; g; o; hour,deg
NGC 7789;	23 57 24.00; 	+56 42 30.0; icrs; g; o; hour,deg
GD1-A;       220.2;	26.9; galactic;     m; s; deg
GD1-B;       186.1;	52.7; galactic;     m; s; deg
GD1-C;       138.3;	60.9; galactic;     m; s; deg
GD1-C-BACKUP;       138.3;	60.9; galactic;     m; s; deg
BOSS7456;    198.039;	0.0; icrs;      lime; s; deg
LOWB;       126.0;	0; icrs;     k; s; deg
"""

_FIELD_TABLE = """
name;          lon;          lat;        frame;    c; s; unit
DRACO;         260.051625;   57.915361;  icrs;     b; x; deg
MWS1;          145.0;        30.0;       icrs;     m; s; deg
MWS2;          148.0;        32.25;       icrs;     m; s; deg
ORPHAN-A;      158.5; 	10.0;  icrs;     m; s; deg
G15;           217.5;    1.0;  icrs;     m; s; deg
G12;           180.0;   -1.0;  icrs;     m; s; deg
M53+N5053;     199.0;   18.3;  icrs;     m; s; deg
NGP;             0.0;   90.0;  galactic; b; +; deg
M92;   17 17 07.39;     +43 08 09.4; icrs; b; s; hour,deg
M13;   16 41 41.24;  	+36 27 35.5; icrs; b; s; hour,deg
M71;   19 53 46.49;     +18 46 45.1; icrs; b; s; hour,deg
M2;    21 33 27.02; 	-00 49 23.7; icrs; b; s; hour,deg
M5;         15 18 33.22;  	+02 04 51.7; icrs; b; s; hour,deg
M15;        21 29 58.33;  	+12 10 01.2; icrs; b; s; hour,deg
NGC 4147;   12 10 06.30; 	+18 32 33.5; icrs; b; s; hour,deg
M67;        08 51 18.00;	+11 48 00; icrs; g; o; hour,deg
HYADES;	    04 26 54.00; 	+15 52 00.0; icrs; g; o; hour,deg
NGC 6791;	19 20 53.00;	+37 46 18.0; icrs; g; o; hour,deg
NGC 7789;	23 57 24.00; 	+56 42 30.0; icrs; g; o; hour,deg
GD1-A;       220.2;	26.9; galactic;     m; s; deg
GD1-B;       186.1;	52.7; galactic;     m; s; deg
GD1-C;       138.3;	60.9; galactic;     m; s; deg
GD1-C-BACKUP;       138.3;	60.9; galactic;     m; s; deg
BOSS7456;    198.039;	0.0; icrs;      lime; s; deg
LOWB;       126.0;	0; icrs;     k; s; deg
"""
FIELD_TABLE = Table.read(_FIELD_TABLE,format='csv',delimiter=';')

other_cols = FIELD_TABLE.colnames.copy()
other_cols.remove('name')
SSV_AREAS = dict(zip(FIELD_TABLE['name'],
                     FIELD_TABLE[other_cols].to_pandas().to_dict(orient='records')))

for area,d in SSV_AREAS.items():
    unit  = d.get('unit','deg')
    frame = d.get('frame','icrs')
    SSV_AREAS[area]['coord'] = coord.SkyCoord(d['lon'],d['lat'],unit=unit,frame=frame).icrs
    
bmap_cache = dict()
stdict     = None

############################################################
def plot_streams(m,streams=None):
    """
    'GD-1'
    """
    global bmap_cache
    global stdict
    
    #  = '/global/homes/a/apcooper/software/modules/galstreams/master/lib/python3.6/site-packages/galstreams-1.0.0-py3.6.egg'
    _ = '/global/homes/a/apcooper/software/src/git/galstreams'
    if _ not in sys.path:
        sys.path.append(_)
   
    import galstreams
    
    if stdict is None:
        footprints_dir   = os.path.join(galstreams.__path__[0],'../footprints/individual_footprints')
        footprints_glob  = os.path.join(footprints_dir,'galstreams*.dat')
        footprints_files = glob.glob(footprints_glob)
        stdict           = dict()
        for f in footprints_files:
            name = os.path.basename(f).split('.')[2]
            stdict[name] = Table.read(f,format='ascii.commented_header')
    
    if streams is not None:
    
        if isinstance(streams,str):
            _streams = [streams]
        else:
            _streams = streams
            
        scat_kwargs = dict(vmin=0.,vmax=80.,cmap='viridis', alpha=0.3)

        for stream in _streams:
            v = stdict[stream]
            if not stream in bmap_cache:
                bmap_cache[stream] = m(v['RA_deg'],v['DEC_deg'])
            x,y = bmap_cache[stream]
            m.scatter(x,y,s=1,c=v['Rhel_kpc'],**scat_kwargs)

    return

############################################################
def plot_gd1_ting(m):
    """
    """
    ra       = np.linspace(123.860301355618, 183.37915611130998, 1000)
    dec_func = np.poly1d([-1.80670575e-07,  1.40950934e-04, -4.36119783e-02,  6.67035494e+00,  -5.01780078e+02, 1.47751518e+04])
    dec      = dec_func(ra)

    x,y = m(ra,dec)
    m.scatter(x,y,s=5,c='r',alpha=0.3)
    return

############################################################
def plot_orphan_sergey(m):
    """
    """
    d   = Table(fits.getdata('./py/ssv/data/gdr2_orphan_sergey.fits'))
    x,y = m(d['ra'], d['dec']) 
    m.scatter(x,y,s=1,c='k',alpha=1)
    return
    
############################################################
def plot_boss_segue(m):
    """
    SEGUE GES fields from
    https://www.sdss.org/dr13/algorithms/ancillary/boss/starsacross/
    """
    boss = Table(fits.getdata('./py/ssv/data/boss_platelist.fits'))

    sel  = np.ones(len(boss),dtype=np.bool)
    sel &= (boss['PLATEQUALITY'] == 'good')
    sel &= (boss['PLATESN2'] > 0)
    sel &= ((boss['PROGRAMNAME'] == 'SEGUE_GES'.ljust(27)))

    x,y = m(boss['RACEN'][sel], boss['DECCEN'][sel]) 
    m.scatter(x,y,s=32,marker='s',c='c',alpha=1)
   
    for i in range(0,len(x)):
        x2,y2 = 1,1
        pl.annotate('GES{}'.format(i), xy=(x[i], y[i]),  xycoords='data',
                    xytext=(x2, y2), textcoords='offset points',
                    color='k')
    return

############################################################
def plot_sweeps(m):
    """
    """
    import desiutil.plots
    
    color_cycle = pl.rcParams['axes.prop_cycle']
    
    plot_all_sweeps = False
            
    for sweep_poly,c in zip(sweep_polys,color_cycle):
        sweep_poly.draw(m, c=c['color'])

    if plot_all_sweeps:
        for sweep_poly in all_sweep_polys:
            sweep_poly.draw(m,c='lime') # c=c['color']) 
        
    return None

############################################################
def plot_kepler(m):
    """
    """
    import json

    kepler = json.load(open("k2-footprint.json"))
    
    kepler_polys = list()
    for c,d in kepler.items():
        for channel in d['channels'].values():
            m.scatter(channel['corners_ra'],channel['corners_dec'],c='lime',s=2,latlon=True)

        x,y = m(np.median(list(d['channels'].values())[0]['corners_ra']),
                np.median(list(d['channels'].values())[0]['corners_dec']))
        x2,y2 = 1,1
        
        pl.annotate('K{}'.format(c), xy=(x, y),  xycoords='data',
                xytext=(x2, y2), textcoords='offset points',
                color='r')
    return

############################################################
def plot_gama(m,name='g15'):
    """
    G12	12.0	174.0 to 186.0	-3.0 to +2.0	full
    G15	14.5	211.5 to 223.5	-2.0 to +3.0	full	r < 19.8
    """
    corners = {'g15': (211.5,223.5,-2.0,3.0),
               'g12': (174.0,186.0,-3.0,2.0)}
    
    if WITH_SPHERICAL_GEOMETRY:
        rmin,rmax,dmin,dmax = corners[name]
        corners_ra   = [rmin,rmin,rmax,rmax]
        corners_dec  = [dmin,dmax,dmax,dmin]
        poly  = spoly.SphericalPolygon.from_radec(corners_ra,corners_dec)
        poly.draw(m,c='k')
    else:
        poly = None
    
    x,y = m(rmin,dmin)
    x2,y2 = 1,1
    
    pl.annotate(name, xy=(x, y),  xycoords='data',
        xytext=(x2, y2), textcoords='offset points',
        color='k')
        
    return poly


############################################################
def plot_constant_b(bval,m,**kwargs):
    """
    """

    galactic_l     = np.linspace(0, 2 * np.pi, 1000)
    galactic_plane = coord.SkyCoord(l=galactic_l*u.radian,
                                  b=np.deg2rad(bval)*np.ones_like(galactic_l)*u.radian,
                                  frame='galactic').fk5
    # Project to map coordinates and display.  Use a scatter plot to
    # avoid wrap-around complications.
    galactic_x, galactic_y = m(galactic_plane.ra.degree,
                               galactic_plane.dec.degree)

    paths = m.scatter(galactic_x, galactic_y, marker='.', s=10, lw=0, alpha=0.75, zorder=20, **kwargs)
    return

############################################################
def plot_footprint(nside=16,
                   kepler=False,
                   gama=False,
                   streams=None,
                   streams_extra=False,
                   sweep_polys=None,
                   dither_polys=None,
                   boss_segue=False,
                   only_areas=None,
                   bcontours=[20,40,60,80]):
    """
    Show our SV fields on the sky, and highlight overlaping bricks of the sweep files.
    """
    import desimodel.footprint
    import desiutil.plots
    import json
    import matplotlib.patheffects as PathEffects

    # Healpix map of pixels in DESI
    in_desi_pix = desimodel.footprint.tiles2pix(nside)
    in_desi     = np.zeros(hp.nside2npix(nside))
    in_desi[in_desi_pix] = 1.0

    f = pl.figure(figsize=(15,10))
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
        m = desiutil.plots.init_sky(ecliptic_color='w')
        
        desiutil.plots.plot_healpix_map(in_desi,nest=True,colorbar=False) 
    
        if kepler:
            plot_kepler(m)
            
        if gama:
            plot_gama(m,'g15')
            plot_gama(m,'g12')
        
        if sweep_polys is not None:
            for sweep_poly in sweep_polys:
                sweep_poly.draw(m, c='lime')

        if dither_polys is not None:
            for dither_poly in dither_polys:
                dither_poly.draw(m,c='r')
                
            #color_cycle = pl.rcParams['axes.prop_cycle']
            #for sweep_poly,c in zip(sweep_polys,color_cycle):
            #    sweep_poly.draw(m, c=c['color'])
        
        if boss_segue:
            plot_boss_segue(m)
            
        if streams is not None:
            plot_streams(m,streams)
       
            if streams_extra:
                if 'GD-1' in streams:
                    plot_gd1_ting(m)
                if 'Orphan' in streams:
                    plot_orphan_sergey(m)
        
        if bcontours is not None:
            for b in bcontours:
                plot_constant_b( b,m,c='k')
                plot_constant_b(-b,m,c='k')
                
        for name,area in SSV_AREAS.items():
            
            if only_areas is not None:
                if name not in only_areas:
                    continue
                    
            ac  = area['coord']
            ra  = ac.ra.value
            dec = ac.dec.value
            x,y = m(ra,dec)
            
            m.scatter(ac.ra.value,
                      ac.dec.value,
                      latlon=True,
                      c=area.get('c','k'),
                      s=20,
                      marker=area.get('marker','s')) 
            
            x2, y2 = 3, 3
            txt = pl.annotate(name, xy=(x, y),  xycoords='data',
                xytext=(x2, y2), textcoords='offset points',
                color='k',zorder=30)
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
