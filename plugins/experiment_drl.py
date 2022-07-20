from re import L
from bluesky import core, stack, traf, tools, settings 
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
import bluesky as bs
import numpy as np
import math

import plugins.SAC.sac_agent as sac
import plugins.state as st
import plugins.functions as fn
import plugins.reward as rw

timestep = 1.5
state_size, _ = fn.get_statesize()
action_size = 3

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

        self.print = False

        self.rewards = np.array([])
        self.state_size = state_size
        self.action_size = action_size

        with self.settrafarrays():
            self.targetalt = np.array([])  # Target altitude of the AC   
            self.control = np.array([])  # Is AC controlled by external algorithm
            self.first = np.array([])  # Is it the first time AC is called
            self.distinit = np.array([])  # Initial (vertical) distance from target state
            self.totreward = np.array([])  # Total reward this AC has accumulated
            self.call = np.array([])  # Number of times AC has been comanded

            self.state = []    # Latest state information
            self.action = []    # Latest selected actions
            self.choice = []    # Latest intended actions (e.g no noise)

    def create(self, n=1):
        super().create(n)
        
        self.targetalt[-n:] = traf.alt[-n:]  
        self.control[-n:] = 0
        self.totreward[-n:] = 0
        self.first[-n:] = 1
        self.distinit[-n:] = 0
        self.call[-n:] = 0
        
        self.state[-n:] = list(np.zeros(self.state_size))
        self.action[-n:] = list(np.zeros(self.action_size))
        self.choice[-n:] = list(np.zeros(self.action_size))

    @core.timed_function(name='experiment_main', dt=timestep)
    def update(self):
        for acid in traf.id:
            ac_idx = traf.id2idx(acid)
            self.update_AC_control(ac_idx)

            if self.control[ac_idx] == 1:
                self.call[ac_idx] += 1
                
                state = self.state[ac_idx]  # set the old state to the previous state
                action = self.action[ac_idx] # set the old action to the previous action
                
                state_, _ = st.get_state(ac_idx, self.targetalt[ac_idx])

                staten_ = fn.normalize_state(np.array(state_))

                if self.first[ac_idx]:
                    self.first[ac_idx] = 0
                    reward = 0 # Cant get a reward in the first step
                    done = 0
                    
                else:
                    staten = fn.normalize_state(np.array(state)) 
                    reward, done = rw.calc_reward(state_)
                    self.totreward[ac_idx] += reward
                    self.agent.store_transition(staten, action[0], reward, staten_, done)
                    self.agent.train()

                if done:
                    self.control[ac_idx] = 0
                    self.rewards = np.append(self.rewards, self.totreward[ac_idx])
                    stack.stack(f'alt {acid} {self.targetalt[ac_idx]}')
                    self.print = True
                else:
                    action = self.agent.step(staten_)
                    self.do_action(action[0],acid,ac_idx)

                self.state[ac_idx] = state_
                self.action[ac_idx] = action

                if len(self.rewards) % 50 == 0 and self.print == True:
                    self.print = False
                    print(np.mean(self.rewards[-500:]))



    
    def update_AC_control(self,ac_idx):   
        if self.first[ac_idx]:                
            # Checks if the AC is inside the bounds of the delivery area
            inbounds    = fn.checkinbounds(traf.lat[ac_idx], traf.lon[ac_idx])               
            if inbounds:
                
                targetalt, control, layer   = fn.get_targetalt(ac_idx)
                self.control[ac_idx]        = control
                self.targetalt[ac_idx]      = targetalt                                 #ft
                self.distinit[ac_idx]       = (abs(targetalt*ft - traf.alt[ac_idx]))/ft #ft

                # Ensure that each aircraft only has one chance at getting
                # an altitude command, dont reset first if control == 1
                # because that information is still required in that case.
                if control == 0:
                    self.first[ac_idx]      = 0

    def do_action(self,action,acid,ac_idx):
        self.do_vz(action,acid,ac_idx)     
        self.do_vh(action,acid,ac_idx) 
        self.do_hdg(action,acid,ac_idx)

    def do_vz(self,action,acid,ac_idx):
        vz = ((action[0] + 1.)*2.5) /fpm
        if vz < 20:
            target = traf.alt[ac_idx] / ft
            vz = 0        
        elif self.targetalt[ac_idx]*ft < traf.alt[ac_idx]:
            target = 0               
        else:
            target = 5000
        
        stack.stack(f'alt {acid} {target} {vz}') 

    def do_vh(self,action,acid,ac_idx):       
        vh = action[1]*2.5
        
        targetvelocity  = (traf.cas[ac_idx] + vh)/kts
        targetvelocity  = np.clip(targetvelocity,10,30)
        
        stack.stack(f'SPD {acid} {targetvelocity}')
    
    def do_hdg(self,action,acid,ac_idx):
        heading = traf.hdg[ac_idx]
        hdg = action[2]*45 
        
        targetheading   = heading + hdg

        if targetheading < 0:
            targetheading = 360 + targetheading
        
        if targetheading > 360:
            targetheading = targetheading - 360
        
        stack.stack(f'HDG {acid} {targetheading}')


