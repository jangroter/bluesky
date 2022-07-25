from bluesky import core, stack, traf, tools, settings 
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
from bluesky.tools import geo
import bluesky as bs
import numpy as np
import math

import plugins.SAC.sac_agent as sac
from plugins.source import Source
import plugins.functions as fn

timestep = 1.5
state_size = 2
action_size = 2

def init_plugin():
    
    experiment_drl = Experiment_drl()
    
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'EXPERIMENT_DRL',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config

class Experiment_drl(core.Entity):  
    def __init__(self):
        super().__init__()

        self.agent = sac.SAC(action_size,state_size)

        self.rewards = np.array([])
        self.state_size = state_size
        self.action_size = action_size

        self.first = True
        self.action_required = False
        self.AC_present = False
        
        self.wptdist = 0

        self.source = Source()

        ## Initialize Experiment Area with Sink & Source ##

        with self.settrafarrays():
            self.totreward = np.array([])  # Total reward this AC has accumulated

            self.state = []    # Latest state information
            self.action = []    # Latest selected actions

    def create(self, n=1):
        super().create(n)
        self.totreward[-n:] = 0

        self.state[-n:] = list(np.zeros(self.state_size))
        self.action[-n:] = list(np.zeros(self.action_size))

    @core.timed_function(name='experiment_main', dt=timestep)
    def update(self):
        self.check_ac()
        
        if self.action_required:
            state = self.get_state(0)
            self.first = False

            action = [[1,0]]

            self.do_action(action,0)

            #action = self.agent.step(state)

            # self.do_action()

        
        # if ! AC
            # Create AC
        # check action required
        # if action required:
            # state

    def check_ac(self):
        if len(bs.traf.id) == 0:
            self.source.create_ac()
            self.first = True
            self.action_required = True
        else: 
            if self.check_past_wpt(0):
                self.action_required = True
            else:
                self.action_required = False
        
    def check_past_wpt(self, idx):
        if self.first:
            dis = fn.haversine(bs.traf.lat[idx], bs.traf.lon[idx], bs.traf.actwp.lat[idx], bs.traf.actwp.lon[idx])
            self.wptdist = dis
        else:
            dis = fn.haversine(bs.traf.lat[idx], bs.traf.lon[idx], bs.traf.actwp.lat[idx], bs.traf.actwp.lon[idx])
            olddis = self.wptdist
            self.wptdist = dis
            if olddis - dis < 0:
                return True
        return False

    def get_state(self,idx):
        brg, dist = geo.kwikqdrdist(52.3322, 4.75, bs.traf.lat[idx], bs.traf.lon[idx])
        x = np.cos(np.radians(brg))*dist / 150.
        y = np.sin(np.radians(brg))*dist / 150.

        print(x,y)
        return [x,y]

    def do_action(self,action,idx):
        action = action[0]
        distance = math.sqrt(action[0]**2 + action[1]**2)*20
        bearing = math.atan2(action[1],action[0])

        ac_lat = np.deg2rad(bs.traf.lat[idx])
        ac_lon = np.deg2rad(bs.traf.lon[idx])

        new_lat = self.get_new_latitude(bearing,ac_lat,distance)
        new_lon = self.get_new_longitude(bearing,ac_lon,ac_lat,new_lat,distance)

        stack.stack(f'ADDWPT KL001 {np.rad2deg(new_lat)},{np.rad2deg(new_lon)}')

    def get_new_latitude(self,bearing,lat,radius):
        R = Rearth/1000.
        return math.asin( math.sin(lat)*math.cos(radius/R) +\
               math.cos(lat)*math.sin(radius/R)*math.cos(bearing))
        
    def get_new_longitude(self,bearing,lon,lat1,lat2,radius):
        R   = Rearth/1000.
        return lon + math.atan2(math.sin(bearing)*math.sin(radius/R)*\
                     math.cos(lat1),math.cos(radius/R)-math.sin(lat1)*math.sin(lat2))