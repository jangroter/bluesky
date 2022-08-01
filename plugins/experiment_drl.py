from ctypes.wintypes import WPARAM
from bluesky import core, stack, traf, tools, settings 
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
from bluesky.tools import geo, aero, areafilter, plotter
from bluesky.traffic import Route
from bluesky import stack
from bluesky.network.npcodec import encode_ndarray, decode_ndarray
import msgpack

import bluesky as bs
import numpy as np
import math

import plugins.SAC.sac_agent as sac
from plugins.source import Source
import plugins.source as sc
import plugins.functions as fn

import plugins.custom_events as ce

timestep = 1.5
state_size = 2
action_size = 2

nm2km = nm/1000

circlerad = 150. * nm2km
max_action = 25. * nm2km

max_episode_length = 15

circle_lat = 52.3322
circle_lon = 4.75

def init_plugin():
    
    experiment_drl = Experiment_drl()
    
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'EXPERIMENT_DRL',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return these two dicts.
    return config


class Experiment_drl(core.Entity):  
    def __init__(self):
        super().__init__()

        self.count = 0

        self.rewards = np.array([])
        self.state_size = state_size
        self.action_size = action_size

        self.first = True
        self.action_required = False
        self.AC_present = False
        self.new_wpt_set = True
        self.print = False
        self.waiting_for_action = False
        
        self.source = Source()

        self.wptdist = 0
        self.wptdist_old = 0

        with self.settrafarrays():
            self.totreward = np.array([])  # Total reward this AC has accumulated
            self.nactions = np.array([])  # Total reward this AC has accumulated

            self.state = [] # Previous timestep state information   
            self.state_ = [] # Latest state information
            self.action = [] # Previous timestep selected action

    def create(self, n=1):
        super().create(n)
        self.count += 1
        self.totreward[-n:] = 0
        self.nactions[-n:] = 0

        self.state[-n:] = list(np.zeros(self.state_size))
        self.state_[-n:] = list(np.zeros(self.state_size))
        self.action[-n:] = list(np.zeros(self.action_size))

    @core.timed_function(name='experiment_drl', dt=timestep)
    def update(self):

        idx = 0
        self.check_ac(idx)

        if not self.first:
           self.check_done(idx)
        
        if self.waiting_for_action:
            new_data, action = self.get_action(idx)
            if new_data:
                self.action[idx] = action
                self.do_action(idx,action)
                self.waiting_for_action = False
            else:
                return

        if self.action_required:
            state_ = self.get_state(idx)
            acid = bs.traf.id[idx]
            self.request_action(state_,acid) # use acid for requesting action as traf arrays can change between data transfers
            if not self.first:
                reward, done = self.get_reward(idx,self.state[idx],state_)
                self.store_transition(self.state[idx],self.action[idx],reward,state_,done)
            self.state[idx] = state_
            self.first = False
            self.action_required = False
            self.waiting_for_action = True

            # bs.sim.hold()

        # if len(self.rewards) % 50 == 0 and self.print == True:
        #    self.print = False
        #    print(np.mean(self.rewards[-500:]))

    def check_ac(self,idx):
        if bs.traf.ntraf == 0:
            self.source.create_ac()
            self.first = True
            self.action_required = True
        else: 
            if self.check_past_wpt(idx):
                self.action_required = True
            else:
                self.action_required = False
    
    def request_action(self, state, acid):
        bs.net.send_event(b'REQUESTACTION', (state,acid))

    def get_action(self,idx):
        action = None
        new_data = False
        if ce.custom_events.data_available[idx] == 1:
            action = ce.custom_events.action[idx]
            new_data = True
            ce.custom_events.data_available[idx] = 0
        
        return new_data, action
    
    def do_action(self,idx,action):

        acid = bs.traf.id[idx]

        distance = max(math.sqrt(action[0]**2 + action[1]**2)*max_action,2)
        bearing = math.atan2(action[1],action[0])

        ac_lat = np.deg2rad(bs.traf.lat[idx])
        ac_lon = np.deg2rad(bs.traf.lon[idx])

        new_lat = self.get_new_latitude(bearing,ac_lat,distance)
        new_lon = self.get_new_longitude(bearing,ac_lon,ac_lat,new_lat,distance)

        self.action_required = False
        self.new_wpt_set = True

        if not self.first:
            stack.stack(f'DELRTE {acid}')

        stack.stack(f'ADDWPT {acid} {np.rad2deg(new_lat)},{np.rad2deg(new_lon)}')

        self.nactions[idx] += 1

    def store_transition(self,state,action,reward,state_,done):
        bs.net.send_event(b'SETRESULT', (state,action,reward,state_,done))

    def check_past_wpt(self, idx):
        if self.new_wpt_set:
            dis = fn.haversine(bs.traf.lat[idx], bs.traf.lon[idx], bs.traf.actwp.lat[idx], bs.traf.actwp.lon[idx])
            self.wptdist = dis
            self.wptdist_old = dis
            self.new_wpt_set = False
        else:
            dis = fn.haversine(bs.traf.lat[idx], bs.traf.lon[idx], bs.traf.actwp.lat[idx], bs.traf.actwp.lon[idx])
            if self.wptdist - dis < 0 and self.wptdist_old - self.wptdist > 0:
                return True
            if self.wptdist - dis < 0 and self.wptdist_old - self.wptdist < 0:
                wptlat = bs.traf.actwp.lat[idx]
                wptlon = bs.traf.actwp.lon[idx]
                stack.stack(f'DELRTE {bs.traf.id[idx]}')
                stack.stack(f'ADDWPT {bs.traf.id[idx]} {wptlat},{wptlon}')

            self.wptdist_old = self.wptdist
            self.wptdist = dis
            
        return False

    def check_done(self,idx):
        finish, d_f = self.get_rew_finish(idx,self.state[idx])
        oob, d_oob = self.get_rew_outofbounds(idx)

        done = min(d_f + d_oob, 1)
        reward = finish+oob

        if done:
            state_ = self.get_state(idx)
            self.totreward[idx] += reward
            self.rewards = np.append(self.rewards, self.totreward[idx])

            if self.nactions[idx] > 0: 
                self.store_transition(self.state[idx],self.action[idx],reward,state_,done)
                bs.net.send_event(b'FINALREWARD', (self.totreward[idx]))

            self.action_required = False
            self.waiting_for_action = False
            ce.custom_events.data_available[idx] = 0
            self.print = True

            bs.traf.delete(idx)
        
        elif self.nactions[idx] > max_episode_length:
            self.rewards = np.append(self.rewards, self.totreward[idx])
            bs.net.send_event(b'FINALREWARD', (self.totreward[idx]))
            self.print = True
            self.action_required = False
            self.waiting_for_action = False
            ce.custom_events.data_available[idx] = 0
            bs.traf.delete(idx)
        

    def get_state(self,idx):
        brg, dist = geo.kwikqdrdist(circle_lat, circle_lon, bs.traf.lat[idx], bs.traf.lon[idx])

        x = np.cos(np.radians(brg))*dist*nm2km / circlerad
        y = np.sin(np.radians(brg))*dist*nm2km / circlerad

        return [x,y]

    def get_latlon_state(self,state):
        distance = np.sqrt(state[0]**2 + state[1]**2)*circlerad
        bearing = math.atan2(state[1],state[0])

        lat = self.get_new_latitude(bearing,np.deg2rad(circle_lat),distance)
        lon = self.get_new_longitude(bearing,np.deg2rad(circle_lon),np.deg2rad(circle_lat),lat,distance)   

        return np.rad2deg(lat), np.rad2deg(lon)

    def get_new_latitude(self,bearing,lat,radius):
        R = Rearth/1000.
        return math.asin( math.sin(lat)*math.cos(radius/R) +\
               math.cos(lat)*math.sin(radius/R)*math.cos(bearing))
        
    def get_new_longitude(self,bearing,lon,lat1,lat2,radius):
        R   = Rearth/1000.
        return lon + math.atan2(math.sin(bearing)*math.sin(radius/R)*\
                     math.cos(lat1),math.cos(radius/R)-math.sin(lat1)*math.sin(lat2))

    def get_reward(self,idx,state,state_):
        dis = self.get_rew_distance(state,state_)
        step = self.get_rew_step()
        fuel = self.get_rew_fuel()
        finish, d_f = self.get_rew_finish(idx,state)
        oob, d_oob = self.get_rew_outofbounds(idx)

        done = min(d_f + d_oob, 1)
        reward = dis+step+fuel+finish+oob

        return reward, done

    def get_rew_distance(self,state,state_, coeff = 0.005):
        old_distance = np.sqrt(state[0]**2 + state[1]**2)*circlerad
        new_distance = np.sqrt(state_[0]**2 + state_[1]**2)*circlerad

        d_dis = old_distance - new_distance

        return d_dis * coeff

    def get_rew_step(self, coeff = -0.01):
        return coeff

    def get_rew_fuel(self, coeff = 0):
        return coeff

    def get_rew_finish(self, idx, state, coeff = 15):
        lat, lon = self.get_latlon_state(state)

        lat_ = bs.traf.lat[idx]
        lon_ = bs.traf.lon[idx]

        if areafilter.checkIntersect('SINK', lat, lon, lat_, lon_):
            return coeff, 1

        if areafilter.checkIntersect('RESTRICT', lat, lon, lat_, lon_):
            return -2, 1

        else:
            return 0, 0

    def get_rew_outofbounds(self, idx, coeff = -5):
        dis_origin = fn.haversine(circle_lat, circle_lon, bs.traf.lat[idx], bs.traf.lon[idx])
        if dis_origin > circlerad*1.01:
            return coeff, 1
        else:
            return 0, 0
