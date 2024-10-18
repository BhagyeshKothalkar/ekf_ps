import pickle
import numpy as np
import matplotlib.pyplot as plt

with open(r'data\data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']  # timestamps [s]

x_init  = data['x_init'] # initial x position [m]
y_init  = data['y_init'] # initial y position [m]
th_init = data['th_init'] # initial theta position [rad]

# input signal
v  = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]

# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]

# print(r[:, 0])
# var = np.var(r[:, 0]**2)
# print(var)
# #2825873.69
def wraptopi(x):
    while(x > np.pi or x <= -np.pi):
        if(x>np.pi):
            x = x - 2*np.pi
        elif(x < np.pi):
            x = x + 2*np.pi
    return x
# #6.082762741013452
x = np.float32(input(""))
print(wraptopi(x))