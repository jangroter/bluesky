from ctypes.wintypes import WPARAM
from bluesky import core, stack, traf, tools, settings 
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
from bluesky.tools import geo, aero, areafilter, plotter
from bluesky.traffic import Route

import bluesky as bs
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path

import plugins.SAC.sac_agent as sac
from plugins.source_alt import Source_alt
import plugins.functions as fn
import plugins.fuelconsumption as fc 
import plugins.noisepollution as noisepol

timestep = 5
state_size = 3
action_size = 3

nm2km = nm/1000

circlerad = 150. * nm2km
max_action = 25. * nm2km

max_episode_length = 15

circle_lat = 52.3322
circle_lon = 4.75

# '\\' for windows, '/' for linux or mac
dir_symbol = '\\'
model_path = 'output' + dir_symbol + 'model_b4c3' + dir_symbol + 'model'

# Make folder for logfiles
path = model_path
Path(path).mkdir(parents=True, exist_ok=True)

def init_plugin():
    
    experiment_drl_alt = Experiment_drl_alt()
    
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'EXPERIMENT_DRL_ALT',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config

class Experiment_drl_alt(core.Entity):  
    def __init__(self):
        super().__init__()

        self.agent = sac.SAC(action_size,state_size,model_path)

        self.rewards = np.array([])
        self.state_size = state_size
        self.action_size = action_size

        self.first = True

        self.finished = np.array([])

        self.action_required = False
        self.AC_present = False
        self.new_wpt_set = True
        self.print = False
        
        self.wptdist = 0
        self.wptdist_old = 0

        self.nac = 0

        self.source = Source_alt()

        self.fuel = np.array([])
        self.noise = np.array([])

        with self.settrafarrays():
            self.totreward = np.array([])  # Total reward this AC has accumulated
            self.nactions = np.array([])  # Total reward this AC has accumulated

            self.state = []    # Latest state information
            self.action = []    # Latest selected actions

    def create(self, n=1):
        super().create(n)
        self.totreward[-n:] = 0
        self.nactions[-n:] = 0

        self.state[-n:] = list(np.zeros(self.state_size))
        self.action[-n:] = list(np.zeros(self.action_size))

    @core.timed_function(name='experiment_main', dt=timestep)
    def update(self):
        fc.fuelconsumption.update(timestep)
        noisepol.noisepollution.update(timestep)
        
        idx = 0
        
        self.check_ac()
        
        if not self.first:
            self.check_done(idx)
        
        if self.action_required:
            state = self.get_state(idx)
            action = self.agent.step(state)
            self.do_action(action,idx)

            if not self.first:
                reward, done = self.get_reward(idx,self.state[idx],state)
                self.totreward[idx] += reward
                self.agent.store_transition(self.state[idx],self.action[idx][0],reward,state,done)
                self.agent.train()
            
            self.state[idx] = state
            self.action[idx] = action

            self.nactions[idx] += 1

            self.first = False

        if len(self.rewards) % 50 == 0 and self.print == True:
            self.print = False
            print(np.mean(self.rewards[-500:]))
            # print(np.mean(self.finished[-500:]*100.))
            # print(f'Fuel: {np.mean(self.fuel)}')
            # print(f'Noise: {np.mean(self.noise)}')

            fig, ax = plt.subplots()
            ax.plot(self.agent.qf1_lossarr, label='qf1')
            ax.plot(self.agent.qf2_lossarr, label='qf2')
            fig.savefig('qloss.png')
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(self.rewards, label='total reward')
            fig.savefig('reward.png')
            plt.close(fig)

            self.log()

    def check_ac(self):
        if bs.traf.ntraf == 0:
            self.source.create_ac()
            """ b4 """
            # rmax = 50 # min(((self.nac//500)+1)*25,150)
            # self.source.create_ac(radiusmax = rmax)

            self.first = True
            self.action_required = True

            self.nac += 1
        else: 
            if self.check_past_wpt(0):
                self.action_required = True
            else:
                self.action_required = False
        
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

        fuel = self.get_rew_fuel(idx,-0.03/17.)
        finish, d_f = self.get_rew_finish(idx,self.state[idx])
        oob, d_oob = self.get_rew_outofbounds(idx)
        noise = self.get_rew_noise(idx, coeff = -0.0)
        done = min(d_f + d_oob, 1)
        reward = finish+oob+fuel+noise

        if done:
            # terminate episode, but dont let model know about it if out-of-bounds,
            # this limits cheesing of episodes by quickly flying out of bounds/
            if d_oob == 1:
                self.finished = np.append(self.finished,0)
                done = 0
            state = self.get_state(idx)
            self.totreward[idx] += reward
            self.rewards = np.append(self.rewards, self.totreward[idx])

            self.agent.store_transition(self.state[idx],self.action[idx][0],reward,state,done)
            self.agent.train()
 
            self.action_required = False
            self.print = True

            bs.traf.delete(idx)
        
        elif self.nactions[idx] > max_episode_length:
            self.rewards = np.append(self.rewards, self.totreward[idx])
            self.print = True
            self.finished = np.append(self.finished,0)
            bs.traf.delete(idx)

        

    def get_state(self,idx):
        brg, dist = geo.kwikqdrdist(circle_lat, circle_lon, bs.traf.lat[idx], bs.traf.lon[idx])

        x = np.sin(np.radians(brg))*dist*nm2km / circlerad
        y = np.cos(np.radians(brg))*dist*nm2km / circlerad

        alt = (bs.traf.alt[idx] / (30000*ft))*2 - 1 

        return [x,y,alt]

    def get_latlon_state(self,state):
        distance = np.sqrt(state[0]**2 + state[1]**2)*circlerad
        bearing = math.atan2(state[0],state[1])

        lat = self.get_new_latitude(bearing,np.deg2rad(circle_lat),distance)
        lon = self.get_new_longitude(bearing,np.deg2rad(circle_lon),np.deg2rad(circle_lat),lat,distance)   

        return np.rad2deg(lat), np.rad2deg(lon)

    def do_action(self,action,idx):
        acid = bs.traf.id[idx]

        action = action[0]
        distance = max(math.sqrt(action[0]**2 + action[1]**2)*max_action,2)
        bearing = math.atan2(action[0],action[1])

        slope = max(action[2],0)
        slope = np.deg2rad(slope*-3)
        
        alt = (bs.traf.alt[idx] + np.tan(slope)*distance*1000)/ft
        alt = max(alt,5000)

        groundspeed = 200 + (alt/30000) * 200
        speed = bs.tools.aero.tas2cas(groundspeed*kts,alt*ft)

        ac_lat = np.deg2rad(bs.traf.lat[idx])
        ac_lon = np.deg2rad(bs.traf.lon[idx])

        new_lat = self.get_new_latitude(bearing,ac_lat,distance)
        new_lon = self.get_new_longitude(bearing,ac_lon,ac_lat,new_lat,distance)

        self.action_required = False
        self.new_wpt_set = True

        if not self.first:
            stack.stack(f'DELRTE {acid}')

        stack.stack(f'ADDWPT {acid} {np.rad2deg(new_lat)},{np.rad2deg(new_lon)} {alt} {speed/kts}')
        stack.stack(f'LNAV {acid} ON')
        stack.stack(f'VNAV {acid} ON')

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
        step = self.get_rew_step(coeff = 0)
        fuel = self.get_rew_fuel(idx, coeff = -0.03/17.) #Fuel ~ Noise * 17 
        noise = self.get_rew_noise(idx, coeff = 0) #Coeff ~ -0.002
        finish, d_f = self.get_rew_finish(idx,state)
        oob, d_oob = self.get_rew_outofbounds(idx)

        self.fuel = np.append(self.fuel,fuel)
        self.noise = np.append(self.noise,noise)

        fc.fuelconsumption.fuelconsumed[idx] = 0
        noisepol.noisepollution.noise[idx] = 0
        
        done = min(d_f, 1)
        reward = dis+step+fuel+noise+finish+oob

        return reward, done

    def get_rew_distance(self,state,state_, coeff = 0.005):
        old_distance = np.sqrt(state[0]**2 + state[1]**2)*circlerad
        new_distance = np.sqrt(state_[0]**2 + state_[1]**2)*circlerad

        d_dis = old_distance - new_distance

        """ c1 """
        # d_dis = min(d_dis,0)
        # return d_dis * coeff

        """ c2 """
        # return d_dis * coeff

        """ c3 """
        return 0

    def get_rew_step(self, coeff = -0.01):
        return coeff

    def get_rew_fuel(self, idx, coeff = -0.005):
        fuel = fc.fuelconsumption.fuelconsumed[idx]
        reward = coeff * fuel
                
        return reward
    
    def get_rew_noise(self,idx, coeff = -1):
        noise = noisepol.noisepollution.noise[idx]
        reward = coeff * noise

        return reward

    def get_rew_finish(self, idx, state, coeff = 0):
        lat, lon = self.get_latlon_state(state)

        lat_ = bs.traf.lat[idx]
        lon_ = bs.traf.lon[idx]

        if areafilter.checkIntersect('SINK', lat, lon, lat_, lon_) and bs.traf.alt[idx] < 10000*ft:
            self.finished = np.append(self.finished,1)
            return 1, 1
        
        if areafilter.checkIntersect('SINK', lat, lon, lat_, lon_):
            self.finished = np.append(self.finished,1)
            return 1, 1

        if areafilter.checkIntersect('RESTRICT', lat, lon, lat_, lon_) and bs.traf.alt[idx] < 10000*ft:
            self.finished = np.append(self.finished,1)
            return 0, 1

        else:
            return 0, 0

    def get_rew_outofbounds(self, idx, coeff = -0.5):
        dis_origin = fn.haversine(circle_lat, circle_lon, bs.traf.lat[idx], bs.traf.lon[idx])
        if dis_origin > circlerad*1.10:
            return coeff, 1
        else:
            return 0, 0

    def log(self):
        self.agent.save_models()