from bluesky import core, stack, traf, sim, tools, scr
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
import math as m
import plugins.functions as fn
import bluesky as bs


def calc_reward(state_):
    reward_a, done_a = calc_reward_a(state_)
    reward_b, done_b = calc_reward_b(state_)
    
    done                = ((done_a * done_b) -  1) * -1 
    
    current_reward      = reward_a + reward_b
    
    return current_reward, done
    
def calc_reward_a(state_):
    headinglayers       = bs.settings.num_headinglayers
    
    basealtitude        = bs.settings.lower_alt
    maxaltitude         = bs.settings.upper_alt
    
    altperlayer         = (maxaltitude - basealtitude)/(headinglayers)
    
    if state_[0] < altperlayer:
        return 0, 0
    else:
        return 0, 1

def calc_reward_b(state_):
    n_aircraft = bs.settings.num_aircraft
    
    state_start = 3
    state_per_ac = 8

    reward = 0
    
    for i in range(0,n_aircraft):
        if state_[state_start+2+state_per_ac*i] == 0 and state_[state_start+1+state_per_ac*i] == 1:
            reward += -0.25
            
    return reward, 1
