from bluesky import core


custom_events = None

def init_plugin():
    global custom_events
    custom_events = Custom_events()
    custom_events.process(1,"MY_EVENT")
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'CUSTOM_EVENTS',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    return config

class Custom_events():  
    def process(self,eventdata,eventname):
        if eventname == "MY_EVENT":
            print(eventdata)
            return True
        else:
            return False
