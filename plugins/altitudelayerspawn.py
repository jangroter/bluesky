# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:20:34 2021

This script automatically creates aircraft at a set interval based on the density of the airspace,
The aircraft generated adhere to the altitude layers and are generated at the edge of the circle.

Aircraft are initialized with a random heading every x seconds

an altitude is matched to that heading 

speed constant

aircraft are then created on the edge of the circle with defined constraints


@author: Jan
"""
from bluesky import core, stack, traf  #, settings, navdb, sim, scr, tools
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
import bluesky as bs
import numpy as np
import area
import random
import math


def init_plugin():
    
    altitudelayerspawn = Altitudelayerspawn()
    
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'ALTITUDELAYERSPAWN',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config


class Altitudelayerspawn(core.Entity):
    
    headinglayers       = 16
    
    circlelat           = 48.86
    circlelon           = 2.37
    circlerad           = 1.62
    
    aircraftnumber      = 1 
    speed               = 20
    
    spawntime           = 3.05
    
    degreesperheading   = 2*(360/headinglayers)
    
    basealtitude        = 200
    maxaltitude         = 975
    
    altperlayer         = (maxaltitude - basealtitude)/(headinglayers)
    
    def __init__(self):
        super().__init__()
        stack.stack(f'Circle altitudelayer {self.circlelat} {self.circlelon} {self.circlerad}')
        stack.stack('Area altitudelayer')
    
    @core.timed_function(name='testplugin', dt=spawntime)
    def update(self): 
        self.aircraftnumber    += 1 
        acid                    = 'kl00' + str(self.aircraftnumber)
        heading                 = 30.0
        altitude                = self.get_altitude(heading)
        lat,lon                 = self.get_spawn(heading)
        speed                   = self.speed
        
        stack.stack(f'CRE {acid} Amzn {lat} {lon} {heading} {altitude} {speed}')
                
    def get_altitude(self, heading):
        #uncomment if using a single set of layers instead of duplicate
        # layer = 1
        # extraheight   = 0
        
        #uncomment if using a pair set of layers instead of duplicate
        layer           = random.randint(0,1)
        extraheight     = layer * (self.maxaltitude - self.basealtitude) / 2.0
        
        return self.basealtitude + ( self.altperlayer * (heading // self.degreesperheading)) + extraheight + self.altperlayer/4
    
    def get_spawn(self, heading):

        enterpoint = random.randint(-9999,9999)/10000
       
        bearing     = np.deg2rad(heading + 180) + math.asin(enterpoint)

        lat         = np.deg2rad(self.circlelat)
        lon         = np.deg2rad(self.circlelon)
        radius      = self.circlerad * 1.852
        
        latspawn    = np.rad2deg(self.get_new_latitude(bearing,lat, radius))
        lonspawn    = np.rad2deg(self.get_new_longitude(bearing, lon, lat, np.deg2rad(latspawn), radius))
        
        return latspawn, lonspawn
        
    def get_new_latitude(self,bearing,lat,radius):
        R = Rearth/1000.
        return math.asin( math.sin(lat)*math.cos(radius/R) +\
               math.cos(lat)*math.sin(radius/R)*math.cos(bearing))
        
    def get_new_longitude(self,bearing,lon,lat1,lat2,radius):
        R   = Rearth/1000.
        return lon + math.atan2(math.sin(bearing)*math.sin(radius/R)*\
                     math.cos(lat1),math.cos(radius/R)-math.sin(lat1)*math.sin(lat2))
        
