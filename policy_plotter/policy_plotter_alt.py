import agent
import functions as fn
import areafilter as af
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.cm as cm

pop_array = np.genfromtxt('population_1km.csv',delimiter=',')

x_index_min = int(((500000)/1000)-300)
x_index_max = int(((500000)/1000)+300)
y_index_min = int(((500000)/1000)-300)
y_index_max = int(((500000)/1000)+300)

x_array = np.genfromtxt('x_1km.csv',delimiter=',')
y_array = np.genfromtxt('y_1km.csv',delimiter=',')

pop_array = pop_array[y_index_min:y_index_max,x_index_min:x_index_max]

actor = agent.Agent(action_dim = 3, state_dim = 3,modelname="actor_3dof.pt")

circle_lat = 52.3322
circle_lon = 4.75

lat_0 = 0
lon_0 = 0

lat_1 = 0
lon_1 = 0

bearing = np.arange(0,360,10)
distance = np.zeros(36)+250

# bearing = np.array([180])
# distance = np.array([250])
alt_start = 25000

# bearing = np.random.randint(0,360,720)
# distance = np.sqrt(np.random.random(720))*275

c_map = cm.get_cmap("cividis").copy()
c_map.set_bad('k')

# Polderbaan 30,160,20
# East 20,60,-60
line = af.poly_arc(circle_lat,circle_lon,25,30,-30)
fig, ax = plt.subplots(figsize=(15, 15))
fig_alt, ax_alt = plt.subplots(figsize=(7, 5))

ax_alt.invert_xaxis()
ax_alt.set_xlabel('Horizontal Distance Remaining (km)')
ax_alt.set_ylabel('Altitude (ft)')
#ax.imshow(pop_array,extent=[np.min(x_array),np.max(x_array),np.min(y_array),np.max(y_array)],norm=LogNorm(vmin=1000,vmax=100000))
ax.imshow(pop_array,cmap=c_map,extent=[-300000,300000,-300000,300000],norm=LogNorm(vmin=100,vmax=100000))

for i,j in zip(bearing,distance):
    # for j in distance:
    alt_start = np.random.randint(20000,30000)
    make_line = True
    count = 0

    lat = np.array([])
    lon = np.array([])

    x = np.array([])
    y = np.array([])
    z = np.array([])

    while make_line:
        if count == 0:
            lat_0, lon_0 = fn.get_spawn(i, radius = j)
            lat = np.append(lat,lat_0)
            lon = np.append(lon,lon_0)
            x_, y_ = fn.get_xy(lat_0,lon_0)
            z_ = fn.get_z(alt_start)
            x = np.append(x,x_)
            y = np.append(y,y_)
            z = np.append(z,z_)
            alt = alt_start

        state = fn.get_state(lat_0, lon_0)
        z_ = fn.get_z(alt)
        state = state + [z_]
        print(state)
        action = actor.step(state)
        lat_1, lon_1, alt = fn.do_action_alt(action,lat_0,lon_0,alt)
        lat_1, lon_1 = np.rad2deg(np.array([lat_1, lon_1]))
        lat = np.append(lat,lat_1)
        lon = np.append(lon,lon_1)
        x_, y_ = fn.get_xy(lat_1,lon_1)
        z_ = fn.get_z(alt)
        x = np.append(x,x_)
        y = np.append(y,y_)
        z = np.append(z,z_)
        if count > 35:
            make_line = False
        if af.checkIntersect(line,lat_0,lon_0,lat_1,lon_1):
            make_line = False
        
        count += 1
        lat_0 = lat_1
        lon_0 = lon_1
    ax.plot(x,y,'r',alpha=1)

    x_flip = np.flip(x)
    y_flip = np.flip(y)

    distance = np.zeros(len(x_flip))
    count = 0
    for i,j in zip(x_flip,y_flip):
        if count == 0:
            distance_temp = np.sqrt(i**2 + j**2)
            distance[count] = distance_temp
            count += 1
            i_old = i
            j_old = j
        else:
            distance_temp = np.sqrt((i_old-i)**2+(j_old-j)**2)+distance_temp
            distance[count] = distance_temp
            count += 1
            i_old = i
            j_old = j

    alt = fn.get_alt(z)
    alt = np.flip(alt)

    ax_alt.plot(distance,alt,'black',alpha=0.2)



fig.savefig('path3dof.png')
plt.close(fig)



fig_alt.savefig('alt.png')
plt.close(fig_alt)