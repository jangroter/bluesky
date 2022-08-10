from re import L
from bluesky import core, stack, traf, tools, settings 
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
from bluesky.tools import geo
import bluesky as bs
import numpy as np
import math
import pandas as pd
from pathlib import Path

import plugins.SAC.sac_agent as sac
import plugins.state as st
import plugins.functions as fn
import plugins.reward as rw

timestep = 1.5
state_size, _, _ = fn.get_statesize()
action_size = 3

default_vz = 2.5/fpm
default_speed = 10/kts

onsetperiod = 600 # Number of seconds before experiment starts

n_aircraft = bs.settings.num_aircraft


def init_plugin():
    
    experiment_mvp = Experiment_mvp()
    
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'EXPERIMENT_MVP',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config

class Experiment_mvp(core.Entity):  
    def __init__(self):
        super().__init__()

        self.print = False

        self.controlac = 0 # int type variable to keep track of the number of ac controlled

        self.rewards = np.array([])
        self.state_size = state_size
        self.action_size = action_size

        self.logfile = None
        self.lognumber = 0
        self.init_logfile()

        with self.settrafarrays():
            self.targetalt = np.array([])  # Target altitude of the AC   
            self.control = np.array([])  # Is AC controlled by external algorithm
            self.first = np.array([])  # Is it the first time AC is called
            self.distinit = np.array([])  # Initial (vertical) distance from target state
            self.totreward = np.array([])  # Total reward this AC has accumulated
            self.call = np.array([])  # Number of times AC has been comanded
            self.acnum = np.array([]) # ith aircraft 

            self.confint = [] # callsigns of all current and old intruders that are still active

    def create(self, n=1):
        super().create(n)
        
        self.targetalt[-n:] = traf.alt[-n:]  
        self.control[-n:] = 0
        self.totreward[-n:] = 0
        self.first[-n:] = 1
        self.distinit[-n:] = 0
        self.call[-n:] = 0
        self.acnum[-n:] = 0
        
        self.confint[-n:] = [[]]

    @core.timed_function(name='experiment_mvp', dt=timestep)
    def update(self):
        for acid in traf.id:
            ac_idx = traf.id2idx(acid)
            self.update_AC_control(ac_idx)

            if self.control[ac_idx] == 1:
                self.call[ac_idx] += 1
                state_, logstate, int_idx = st.get_state(ac_idx, self.targetalt[ac_idx])
                
                if self.first[ac_idx]:
                    self.reset_action(acid,ac_idx)
                    self.first[ac_idx] = 0
                    self.controlac += 1

                else:
                    v = np.array([traf.cas[ac_idx]*np.sin(np.deg2rad(traf.hdg[ac_idx])), traf.cas[ac_idx]*np.cos(np.deg2rad(traf.hdg[ac_idx])), traf.vs[ac_idx]])
                    v_update = False
                    for j, tLos, int_id in int_idx:
                        dv, _ = self.MVP(ac_idx, int(j), float(tLos))
                        v = v-dv
                        if int_id not in self.confint[ac_idx]:
                            self.confint[ac_idx].append(int_id)
                        v_update = True

                    reward, done = rw.calc_reward(state_)
                    self.totreward[ac_idx] += reward
                    
                    action = self.do_action(acid,ac_idx,v,v_update)

                    if done:
                        self.control[ac_idx] = 0
                        self.rewards = np.append(self.rewards, self.totreward[ac_idx])
                        stack.stack(f'alt {acid} {self.targetalt[ac_idx]}')
                        self.print = True

                    else:
                        self.update_conflist(ac_idx)
                        self.reset_action(acid,ac_idx)
                    
                    if len(self.rewards) % 50 == 0 and self.print == True:
                        self.print = False
                        print(np.mean(self.rewards[-500:]))
                        
                    
                    self.log(logstate,action,acid,ac_idx)

    def update_AC_control(self,ac_idx):   
        if self.first[ac_idx]:                
            # Checks if the AC is inside the bounds of the delivery area
            inbounds    = fn.checkinbounds(traf.lat[ac_idx], traf.lon[ac_idx])               
            if inbounds:
                
                targetalt, control, layer = fn.get_targetalt(ac_idx)

                if bs.sim.simt > onsetperiod:
                    self.control[ac_idx] = control
                    self.targetalt[ac_idx] = targetalt                                 #ft
                    self.distinit[ac_idx] = (abs(targetalt*ft - traf.alt[ac_idx]))/ft #ft

                # Ensure that each aircraft only has one chance at getting
                # an altitude command, dont reset first if control == 1
                # because that information is still required in that case.
                if control == 0:
                    self.first[ac_idx] = 0

    def do_action(self,acid,ac_idx,newv,v_update):
        action = np.array([0,0,0])
        if v_update:
            newtrack = (np.arctan2(newv[0],newv[1])*180/np.pi) %360
            newhs    = np.sqrt(newv[0]**2 + newv[1]**2) / kts
            newvs    = newv[2] / fpm

            stack.stack(f'SPD {acid} {newhs}')

            if newvs < 20:
                target = traf.alt[ac_idx] / ft
                newvs = 0        
            elif self.targetalt[ac_idx]*ft < traf.alt[ac_idx]:
                target = 0               
            else:
                target = 5000

            stack.stack(f'ALT {acid} {target} {newvs}')
            stack.stack(f'HDG {acid} {newtrack}')

            action[0] = newvs
            action[1] = bs.traf.cas[ac_idx] - newhs
            if newtrack - bs.traf.hdg[ac_idx] > 180:
                action[2] = -1*((newtrack - bs.traf.hdg[ac_idx] + 180) % 360 - 180)
            elif newtrack - bs.traf.hdg[ac_idx] < -180:
                action[2] = ((newtrack - bs.traf.hdg[ac_idx] + 180) % 360 - 180)
            else:
                action[2] = newtrack - bs.traf.hdg[ac_idx]

        return action
            
    def update_conflist(self,ac_idx):
        if self.confint[ac_idx]:
            for i in self.confint[ac_idx]:
                int_idx = bs.traf.id2idx(i)
                tcpa = st.get_tcpa(ac_idx,int_idx)
                if tcpa < 0:
                    self.confint[ac_idx].remove(i)

    def reset_action(self,acid,ac_idx):
        if not self.confint[ac_idx]:
            stack.stack(f'SPD {acid} {default_speed}')

            if self.targetalt[ac_idx]*ft < traf.alt[ac_idx]:
                target = 0               
            else:
                target = 5000
                
            stack.stack(f'ALT {acid} {target} {default_vz}')
            
    def log(self,logstate,action,acid,ac_idx):
        data = [acid, self.acnum[ac_idx], self.call[ac_idx]] + list(logstate) + list(action)
        self.logfile.loc[len(self.logfile)] = data

        if len(self.logfile) == 10000:
            lognumber = str(self.lognumber)    

            if self.lognumber == 0:
                path = bs.settings.experiment_path + '\\' + bs.settings.experiment_name
                Path(path).mkdir(parents=True, exist_ok=True)

            logsavename = bs.settings.experiment_path +'\\'+ bs.settings.experiment_name+ '\\'+ 'logdata_'+lognumber+'.csv'
            self.logfile.to_csv(logsavename)

            self.lognumber += 1
            self.init_logfile()

    def init_logfile(self):
        header = ['acid','acnum','callnum','alt_dif','vz','vh','d_hdg']
        intruder_header = ['int_','intrusion_','conflict_','tcpa_','dcpa_','dalt_','du_','dv_','dx_','dy_','dis_']
        tail_header = ['dvz','dvh','dhdg']

        for i in range(n_aircraft):
            header_append = [s + str(i) for s in intruder_header]
            header = header + header_append

        header = header + tail_header

        self.logfile = pd.DataFrame(columns = header)

    def MVP(self, own_idx, int_idx, tLOS):
        
        tcpa = st.get_tcpa(own_idx,int_idx)

        pz      = bs.settings.asas_pzr * nm
        hpz     = bs.settings.asas_pzh * ft
        safety  = bs.settings.asas_marh
        
        int_idx = int(int_idx)
        
        brg, dist   = geo.kwikqdrdist(bs.traf.lat[own_idx], bs.traf.lon[own_idx],\
                                    bs.traf.lat[int_idx], bs.traf.lon[int_idx],)
        
        dist = dist * nm

        qdr         = np.radians(brg)
        
        
        drel        = np.array([np.sin(qdr)*dist, \
                                np.cos(qdr)*dist, \
                                bs.traf.alt[int_idx]-bs.traf.alt[own_idx]])
            
            
        # Write velocities as vectors and find relative velocity vector
        v1      = np.array([traf.tas[own_idx]*np.sin(np.deg2rad(traf.hdg[own_idx])), traf.tas[own_idx]*np.cos(np.deg2rad(traf.hdg[own_idx])), traf.vs[own_idx]])
        v2      = np.array([traf.tas[int_idx]*np.sin(np.deg2rad(traf.hdg[int_idx])), traf.tas[int_idx]*np.cos(np.deg2rad(traf.hdg[int_idx])), traf.vs[int_idx]])
        vrel    = np.array(v2-v1)

        dcpa    = drel + vrel*tcpa
        dabsH   = np.sqrt(dcpa[0]*dcpa[0]+dcpa[1]*dcpa[1])
        
        iH = (pz * safety) - dabsH
        
        # Exception handlers for head-on conflicts
        # This is done to prevent division by zero in the next step
        if dabsH <= 10.:
            dabsH = 10.
            dcpa[0] = drel[1] / dist * dabsH
            dcpa[1] = -drel[0] / dist * dabsH
        
        # If intruder is outside the ownship PZ, then apply extra factor
        # to make sure that resolution does not graze IPZ
        if (pz * safety) < dist and dabsH < dist:
            # Compute the resolution velocity vector in horizontal direction.
            # abs(tcpa) because it bcomes negative during intrusion.
            erratum=np.cos(np.arcsin((pz * safety)/dist)-np.arcsin(dabsH/dist))
            dv1 = (((pz * safety)/erratum - dabsH)*dcpa[0])/(abs(tcpa)*dabsH)
            dv2 = (((pz * safety)/erratum - dabsH)*dcpa[1])/(abs(tcpa)*dabsH)
        else:
            dv1 = (iH * dcpa[0]) / (abs(tcpa) * dabsH)
            dv2 = (iH * dcpa[1]) / (abs(tcpa) * dabsH)
        
        # Vertical resolution------------------------------------------------------

        # Compute the  vertical intrusion
        # Amount of vertical intrusion dependent on vertical relative velocity
        iV = (hpz * safety) if abs(vrel[2])>0.0 else (hpz * safety)-abs(drel[2])

        # Get the time to solve the conflict vertically - tsolveV
        tsolV = abs(drel[2]/vrel[2]) if abs(vrel[2])>0.0 else tLOS

        # If the time to solve the conflict vertically is longer than the look-ahead time,
        # because the the relative vertical speed is very small, then solve the intrusion
        # within tinconf
        if tsolV>50:
            tsolV = tLOS
            iV    = (hpz * safety)

        dv3 = np.where(abs(vrel[2])>0.0,  (iV/tsolV)*(-vrel[2]/abs(vrel[2])), (iV/tsolV))
        
        dv = np.array([dv1,dv2,dv3])
        
        return dv, tsolV