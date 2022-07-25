import openap
import numpy as np

def compute_eng_ff_coeff(ffidl, ffapp, ffco, ffto):
    """Compute fuel flow based on engine icao fuel flow model

    Args:
        thrust_ratio (1D-array): thrust ratio between 0 and 1
        n_engines (1D-array): number of engines on the aircraft
        ff_idl (1D-array): fuel flow - idle thrust
        ff_app (1D-array): fuel flow - approach
        ff_co (1D-array): fuel flow - climb out
        ff_to (1D-array): fuel flow - takeoff

    Returns:
        list of coeff: [a, b, c], fuel flow calc: ax^2 + bx + c
    """

    # standard fuel flow at test thrust ratios
    y = [0, ffidl, ffapp, ffco, ffto]
    x = [0, 0.07, 0.3, 0.85, 1.0]

    a, b, c = np.polyfit(x, y, 2)
    
    return a, b, c

ffidl = 107
ffapp = 326
ffco = 961
ffto = 1166

print(compute_eng_ff_coeff(ffidl, ffapp, ffco, ffto))

fuelflow = openap.FuelFlow(ac='a320')
drag = openap.Drag(ac='a320')

thr = drag.clean(mass=60000,tas=448, alt=36000)

ffcorrect = 5.142857*10**-7 * (thr / 1000) * (36000 * 0.3048)
FF = fuelflow.enroute(mass=60000, tas=448, alt=36000, path_angle=0)

print(FF - ffcorrect)