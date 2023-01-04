from bluesky import core, stack, traf  #, settings, navdb, sim, scr, tools
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
from bluesky.tools import geo, aero, areafilter, plotter

import bluesky as bs
import numpy as np
import area
import random
import math

from sink_alt import poly_arc


def init_plugin():
    
    source_alt = Source_alt()
    
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'SOURCE_ALT',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config


class Source_alt(core.Entity):
    
    # Define border of airspace with center at AMS
    circlelat           = 52.3322
    circlelon           = 4.75
    circlerad           = 150
    
    speed               = 250

    ac_nmbr = 1

    def __init__(self):
        super().__init__()
        stack.stack(f'Circle source {self.circlelat} {self.circlelon} {self.circlerad}')

        poly_arc(self.circlelat,self.circlelon,25,30,-30)
    
    def create_ac(self, radiusmax = 0): 
        acid                    = 'KL' + str(self.ac_nmbr)
        heading                 = random.randint(0,359)
        lat,lon,alt             = self.get_spawn(heading, radiusmax)
        groundspeed             = 200 + (alt/30000) * 200
        speed                   = bs.tools.aero.tas2cas(groundspeed*kts,alt*ft)
        
        traf.cre(acid,'a320',lat,lon,heading,alt*ft,speed)
        
        #stack.stack(f'CRE {acid} a320 {lat} {lon} {heading} {altitude} {speed}')
        stack.stack(f'ADDWPTMODE {acid} FLYOVER')
        self.ac_nmbr += 1
                
    
    def get_spawn(self, heading, radiusmax):

        enterpoint = random.randint(-9999,9999)/10000
       
        bearing     = np.deg2rad(heading + 180) + math.asin(enterpoint)

        lat         = np.deg2rad(self.circlelat)
        lon         = np.deg2rad(self.circlelon)

        """ b1 """
        #radius = self.circlerad * 1.852 
        
        """ b2 """
        radius = self.circlerad * 1.852 * random.random()

        """ b3 """
        #radius = self.circlerad * 1.852 * math.sqrt(random.random())

        """ b4 """
        # radius = radiusmax * 1.852 * random.random()

        latspawn    = np.rad2deg(self.get_new_latitude(bearing,lat, radius))
        lonspawn    = np.rad2deg(self.get_new_longitude(bearing, lon, lat, np.deg2rad(latspawn), radius))
        
        altmin = 5000 + radius * (20000-5000) / (150*1.852)
        altmax = 5000 + radius * (30000-5000) / (150*1.852)

        alt = random.randint(int(altmin),int(altmax))

        return latspawn, lonspawn, alt
        
    def get_new_latitude(self,bearing,lat,radius):
        R = Rearth/1000.
        return math.asin( math.sin(lat)*math.cos(radius/R) +\
               math.cos(lat)*math.sin(radius/R)*math.cos(bearing))
        
    def get_new_longitude(self,bearing,lon,lat1,lat2,radius):
        R   = Rearth/1000.
        return lon + math.atan2(math.sin(bearing)*math.sin(radius/R)*\
                     math.cos(lat1),math.cos(radius/R)-math.sin(lat1)*math.sin(lat2))
        