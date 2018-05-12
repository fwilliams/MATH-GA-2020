
# Numerical Methods II, Courant Institute, NYU, spring 2018
# http://www.math.nyu.edu/faculty/goodman/teaching/NumericalMethodsII2018/index.html
#  written by Jonathan Goodman (instructor)
#  see class notes Part 3 for more discussion


import matplotlib
# matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from time_step import step


#    Physical parameters
R = 3.    # box size, linear dimension
D = 1.    # diffusion coefficient
T = 3.   # final time for the computation
Tf = .1   # time between movie frames (may be changed slightly)


#    Numerical parameters
n = 50             # number of grid intervals in each dimension
lam = 25           # the CFL ratio = D*dt/(dx^2)
dx = R/n
dt = dx*dx*lam/D   # time step may be adjusted down slightly for the movie

nf = int(T/Tf)+1    # The number of movie frames = total time/frame time + adjustment
Tf = T/nf           # A slightly smaller time per frame to get exactly to time T at the end
ntf = int(Tf/dt)+1  # The number of time steps per frame = (time per frame)/dt
ntf = ((ntf+1)/2)*2
dt = Tf/ntf

# Store all solution at the plot times
# A real computation would not do this, but
# I am sick of trying to get Python to
# make movies the "right" way.
uFrames = np.ndarray([nf, n-1, n-1])

u = np.ndarray([n+1, n+1])
v = np.ndarray([n+1, n+1])
for i in range(n+1):
    u[0, i] = 0.
    u[n, i] = 0.
    u[i, 0] = 0.
    u[i, n] = 0.

for i in range(1, n):
    for j in range(1, n):
        u[i, j] = 1.

for i in range(n+1):
    for j in range(n+1):
        v[i, j] = u[i, j]


for frame in range(nf):
    print("frame " + str(frame))
    for i in range(int(ntf/2)):        # there are two steps per trip through the loop

        step(v, u, D, dx, dt)     # copy u to v
        step(u, v, D, dx, dt)     # copy v back to u

    uFrames[frame, :, :] = u[1:n,1:n]

fig = plt.figure()
for i in range(nf):
    im = plt.imshow(uFrames[i,:,:], animated=True)
    plt.show()

# im = plt.imshow(u[1:n, 1:n], animated=True)
# plt.colorbar()


# def updatefig(*args):
#     global u, v
#     print("frame!")
#     for _ in range(int(ntf/2)):        # there are two steps per trip through the loop
#         step(v, u, D, dx, dt)     # copy u to v
#         step(u, v, D, dx, dt)     # copy v back to u
#     im.set_array(u[1:n, 1:n])
#     return im,
#
#
# ani = animation.FuncAnimation(fig, updatefig, interval=30, blit=True)
# plt.show()