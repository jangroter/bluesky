from bluesky import core
import bluesky as bs
import numpy as np

custom_events = None

def init_plugin():
    global custom_events
    custom_events = Custom_events()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'CUSTOM_EVENTS',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    return config

class Custom_events(core.Entity):  
    def __init__(self):
        super().__init__()

        with self.settrafarrays():
            self.data_available = np.array([])
            self.action = []
            self.state = []
    
    def create(self, n=1):
        super().create(n)
        self.data_available[-n:] = 0
        self.state[-n:] = list(np.zeros(2))
        self.action[-n:] = list(np.zeros(2))
        
    def process(self,eventdata,eventname):
        if eventname == b'SETACTION':
            acid = eventdata[2]
            if acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                self.action[idx] = eventdata[0][0]
                self.state[idx] = eventdata[1]     
                self.data_available[idx] = 1
                # bs.sim.op()
                # bs.sim.fastforward()

            return True
        else:
            return False
