"""
Plugin to keep track of all the fuel used by the different aircraft according to the OpenAP performance models
"""

import numpy as np
import bluesky as bs
from bluesky import core, stack, traf


timestep = 1

def init_plugin():
    fuelconsumption = Fuelconsumption()

    config = {
        'plugin_name': 'FUELCONSUMPTION',
        'plugin_type': 'sim',
        }
    
    return config

class Fuelconsumption(core.Entity):
    def __init__(self):
        super().__init__()
        
        with self.settrafarrays():
            self.fuelconsumed = np.array([])
        
    def create(self, n=1):
        super().create(n)
        self.fuelconsumed[-n:] = [0] * n

    @core.timed_function(name='fuelconsumption', dt = timestep)
    def update(self):
        self.fuelconsumed += traf.perf.fuelflow * timestep
        for acid in traf.id:
            idx = traf.id2idx(acid)
            print(f'AC {acid}, fuel consumed: {self.fuelconsumed[idx]} [kg]')