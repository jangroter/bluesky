import agent_n_layers as agent
import functions as fn
import areafilter as af
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.cm as cm

# pop_array = np.genfromtxt('population_1km.csv',delimiter=',')

# x_index_min = int(((500000)/1000)-300)
# x_index_max = int(((500000)/1000)+300)
# y_index_min = int(((500000)/1000)-300)
# y_index_max = int(((500000)/1000)+300)

# x_array = np.genfromtxt('x_1km.csv',delimiter=',')
# y_array = np.genfromtxt('y_1km.csv',delimiter=',')

# pop_array = pop_array[y_index_min:y_index_max,x_index_min:x_index_max]

value = agent.Value()

x0 = np.arange(-1, 1, 0.01)
y0 = np.arange(-1, 1, 0.01)
x,y = np.meshgrid(x0,y0)

index = np.arange(len(x0))

z = np.zeros(np.shape(x))

for i in index:
    for j in index:
        state = [x0[i],y0[j]]
        z[i,j] = value.step(state)[0].detach().cpu().numpy() * -1

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig, ax = plt.subplots()
# # Plot the surface.
# surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
#                         linewidth=0, antialiased=True,rcount=100,ccount=100,alpha=0.9)

ax.imshow(np.rot90(z),extent=[-277800,277800,-277800,277800], cmap=cm.coolwarm)

# ax.invert_yaxis()
# Add a color bar which maps values to colors.
# fig.colorbar(shrink=0.5, aspect=5)

# plt.show()


# circle_lat = 52.3322
# circle_lon = 4.75

# lat_0 = 0
# lon_0 = 0

# lat_1 = 0
# lon_1 = 0

# bearing = np.arange(0,360,1)
# distance = np.zeros(360)+275

# # bearing = np.random.randint(0,360,720)
# # distance = np.sqrt(np.random.random(720))*275

# c_map = cm.get_cmap("cividis").copy()
# c_map.set_bad('k')

# # Polderbaan 30,160,20
# # East 20,60,-60
# line = af.poly_arc(circle_lat,circle_lon,15,30,-30)
# fig, ax = plt.subplots(figsize=(15, 15))
# #ax.imshow(pop_array,extent=[np.min(x_array),np.max(x_array),np.min(y_array),np.max(y_array)],norm=LogNorm(vmin=1000,vmax=100000))
# ax.imshow(pop_array,cmap=c_map,extent=[-300000,300000,-300000,300000],norm=LogNorm(vmin=100,vmax=100000))

# for i,j in zip(bearing,distance):
#     # for j in distance:
#     make_line = True
#     count = 0

#     lat = np.array([])
#     lon = np.array([])

#     x = np.array([])
#     y = np.array([])

#     while make_line:
#         if count == 0:
#             lat_0, lon_0 = fn.get_spawn(i, radius = j)
#             lat = np.append(lat,lat_0)
#             lon = np.append(lon,lon_0)
#             x_, y_ = fn.get_xy(lat_0,lon_0)
#             x = np.append(x,x_)
#             y = np.append(y,y_)

#         state = fn.get_state(lat_0, lon_0)
#         action = actor.step(state)
#         lat_1, lon_1 = np.rad2deg(fn.do_action(action,lat_0,lon_0))
#         lat = np.append(lat,lat_1)
#         lon = np.append(lon,lon_1)
#         x_, y_ = fn.get_xy(lat_1,lon_1)
#         x = np.append(x,x_)
#         y = np.append(y,y_)
#         if count > 35:
#             make_line = False
#         if af.checkIntersect(line,lat_0,lon_0,lat_1,lon_1):
#             make_line = False
        
#         count += 1
#         lat_0 = lat_1
#         lon_0 = lon_1
#     ax.plot(x,y,'r',alpha=0.4)

fig.savefig('value_E_4_1024.png')
plt.close(fig)