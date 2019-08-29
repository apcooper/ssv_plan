import numpy as np
import desimodel
import astropy.coordinates as coord
import astropy.units as u

from astropy.table import Table, Column, vstack

try:
    import spherical_geometry
    from spherical_geometry import polygon as spoly
    WITH_SPHERICAL_GEOMETRY = True
except ImportError:
    WITH_SPHERICAL_GEOMETRY = False
    

############################################################
class Tile():
    """
    Single desi tile.
    """
    def __init__(self,ra,dec,**kwargs):
        self.ra            = ra
        self.dec           = dec
        self.tileid        = kwargs.get('tileid',0)
        self.tilepass      = kwargs.get('tilepass',0)
        self.in_desi       = kwargs.get('in_desi',0)
        self.ebv           = kwargs.get('ebv',0.0)
        self.airmass       = kwargs.get('airmass',1.0)
        self.star_density  = kwargs.get('star_density',1.0)
        self.exposefac     = kwargs.get('exposefac',1.0)
        self.program       = kwargs.get('program','UNKNOWN')
        self.obsconditions = kwargs.get('obsconditions')

############################################################
class Dither():
    """
    Dither pattern with ndither tiles. Tile centres are evenly
    spaced around a circle of rdither_max centred on (ra,dec).
    """
    def __init__(self,ra,dec,ndither,rdither_max,**kwargs):
        self.ra            = ra
        self.dec           = dec
        self.coord         = coord.SkyCoord(self.ra,self.dec,frame='icrs')
        self.rdither_max   = rdither_max
        self.tiles         = list()
        self.r_dither      = rdither_max
        self.theta_dither  = np.cumsum(np.repeat(2*np.pi/ndither,ndither))
        self.ra_dither     = self.r_dither*np.sin(self.theta_dither)
        self.dec_dither    = self.r_dither*np.cos(self.theta_dither)

        self.one_pass_per_dither =  kwargs.get('one_pass_per_dither',True)

        self.ditherpass    = kwargs.get('ditherpass',0)
        self.program       = kwargs.get('program','MWSV')
        self.obsconditions = kwargs.get('obsconditions',4)
        self.startpass     = kwargs.get('startpass',0)
        self.starttile     = kwargs.get('starttile',0)
        
        # Each dither step is a separate pass
        for idither in range(0,ndither):
            if self.one_pass_per_dither:
                ipass = self.startpass + self.ditherpass+idither
            else:
                ipass = self.ditherpass
                
            t = Tile(self.ra+self.ra_dither[idither],
                     self.dec+self.dec_dither[idither],
                     in_desi  = True,
                     tileid   = self.starttile+ idither,
                     tilepass = ipass,
                     program  = self.program,
                     obsconditions = self.obsconditions)
            self.tiles.append(t)
    
    def __len__(self):
        return len(self.tiles)
    
    def extent(self):
        """
        Bounding box of the pattern (used to cut out areas for
        targeting)
        """
        r   = desimodel.focalplane.get_tile_radius_deg()*u.deg
        ra  = coord.Angle([_.ra for _ in self.tiles])
        dec = coord.Angle([_.dec for _ in self.tiles])
        
        rmax = (ra.max() + coord.Angle(r)).wrap_at(360*u.deg)
        rmin = (ra.min() - coord.Angle(r)).wrap_at(360*u.deg)
        
        dmax = np.minimum( 90*u.deg,(np.max(dec) + coord.Angle(r)))
        dmin = np.maximum(-90*u.deg,(np.min(dec) - coord.Angle(r)))
        
        return rmin,rmax,dmin,dmax
    
    def to_spoly(self):
        if WITH_SPHERICAL_GEOMETRY:
            rmin,rmax,dmin,dmax = self.extent()
            corners_ra   = [_.value for _ in [rmin,rmin,rmax,rmax]]
            corners_dec  = [_.value for _ in [dmin,dmax,dmax,dmin]]
            dither_poly  = spoly.SphericalPolygon.from_radec(corners_ra,corners_dec)
        else:
            dither_poly = None
    
        return dither_poly
    
    def to_table(self):
        """
        Write the dither pattern in the data model for desi tile
        files.
        """
        dmodel = {
        'TILEID': '>i4',
        'RA': '>f8',
        'DEC': '>f8',
        'PASS': '>i2',
        'IN_DESI': '>i2',
        'EBV_MED': '>f4',
        'AIRMASS': '>f4',
        'STAR_DENSITY': '>f4',
        'EXPOSEFAC': '>f4',
        'PROGRAM': '<U6',
        'OBSCONDITIONS': '>i4',
        }
        t = Table()
        
        data = {
            'TILEID': np.array([_.tileid for _ in self.tiles]),
            'RA': np.array([_.ra.deg for _ in self.tiles]),
            'DEC': np.array([_.dec.deg for _ in self.tiles]), 
            'PASS': np.array([_.tilepass for _ in self.tiles]),
            'IN_DESI': np.array([_.in_desi for _ in self.tiles]),
            'EBV_MED': np.array([_.ebv for _ in self.tiles]),
            'AIRMASS': np.array([_.airmass for _ in self.tiles]),
            'STAR_DENSITY': np.array([_.star_density for _ in self.tiles]),
            'EXPOSEFAC': np.array([_.exposefac for _ in self.tiles]),
            'PROGRAM': np.array([_.program for _ in self.tiles]),
            'OBSCONDITIONS': np.array([_.obsconditions for _ in self.tiles]),
        }
        
        for k,v in data.items():
            t.add_column(Column(data=v,dtype=dmodel[k],name=k))

        return t
    
############################################################
class DitherSet():
    """
    Collection of dither patterns.
    """
    def __init__(self,dithers):
        """
        dithers : collection of Dither objects.
        """
        self.dithers = list(dithers)
    
    def extent(self):
        rmin,rmax,dmin,dmax = np.inf,0,np.inf,0
        for _ in self.dithers:
            _rmin,_rmax,_dmin,_dmax = _.extent()
            rmin = np.minimum(rmin,_rmin)
            rmax = np.maximum(rmax,_rmax)
            dmin = np.minimum(dmin,_dmin)
            dmax = np.maximum(dmax,_dmax)
        return rmin,rmax,dmin,dmax
    
    def to_table(self):
        return vstack([_.to_table() for _ in self.dithers])
    
