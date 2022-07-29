''' BlueSky simulation server. '''
from multiprocessing import cpu_count
import plugins.SAC.sac_agent as sac

import zmq
import msgpack

# Local imports
import bluesky as bs
from bluesky.network.server import Server, split_scenarios
from bluesky.network.npcodec import encode_ndarray, decode_ndarray
import numpy as np


state_size = 2
action_size = 2

# Register settings defaults
bs.settings.set_variable_defaults(max_nnodes=cpu_count(),
                                  event_port=9000, stream_port=9001,
                                  simevent_port=10000, simstream_port=10001,
                                  enable_discovery=False)

class CustomServer(Server):
    ''' Implementation of the BlueSky simulation server. '''

    def __init__(self, discovery, altconfig=None, startscn=None):

        self.agent = sac.SAC(state_size,action_size)

        super().__init__(discovery, altconfig, startscn)

    def customevent(self, eventname, src, dest, msg, route, data, sender_id):
        msgpassed = False

        if eventname == b'GETACTION':
            state, idx = msgpack.unpackb(data, raw=False)
            action = self.agent.step(state)

            data = data = msgpack.packb((action), use_bin_type=False)
            self.sendaction(sender_id, data)
            
            # command = f'SETACTION {idx}'
            # for i in range(0,len(action[0])):
            #     command += ' '+str(action[0][i])

            # stackcommand = command # transform into stackcommand here

            # naction = np.array(action).flatten()
            # nstate = np.array(state).flatten()
            # nidx = np.array([idx])
            # data = np.concatenate((naction,nstate,nidx))

            # encdata = encode_ndarray(data)
            # print(encdata)

            # stackcommand = f"SETACTION {encdata.get('data')}"

            # data = msgpack.packb(stackcommand, use_bin_type=True)
            # self.sendaction(sender_id, data)
            # msgpassed = True  
        
        elif eventname == b'SETRESULT':
            state, action, reward, state_, done = msgpack.unpackb(data, raw=False)
            self.agent.store_transition(state,action,reward,state_,done)
            self.agent.train()

        return msgpassed
    
    def sendaction(self, worker_id, data):
        self.be_event.send_multipart([worker_id, self.host_id, b'MYEVENT', data])