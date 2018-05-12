import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Pretty plots
try:
    import seaborn as sns
    sns.set()
except ImportError:
    pass

# Load the time-stepping functions from cython
from ab2_allen_cahn_timestep import step, rk_step, f_allen_cahn, step_linear, step_cubic


# Calculate the order of convergence of a time stepping method as dt -> 0
# This uses the fact that log_2   |u_h - u_{h/2}|     -> p as h -> 0
#                               -------------------
#                               |u_{h/2} - u_{h/4}|
# where p is the order of convergence
def convergence_test():
    global dt, u_prev, u

    dt0 = 0.00001
    dt = dt0

    # Initialize u(x, t) with random data between -1 and 1 on the interior and 0 on the boundary
    u_0 = np.zeros([n, n])
    u_0[1:n - 1, 1:n - 1] = 2 * (np.random.rand(n - 2, n - 2) - 0.5)

    res = []
    for i in range(10):
        dt_nxt = dt0/2
        dt_nxt_nxt = dt0/4

        u_prev = u_0
        u = rk_step(lambda u_: f_allen_cahn(u_, dx), np.array(u_0), dt0)
        _, u_h = strang_step(dt0, ret_stuff=True)

        u_prev = u_0
        u = rk_step(lambda u_: f_allen_cahn(u_, dx), np.array(u_0), dt_nxt_nxt)
        _, u_h2 = strang_step(dt_nxt, ret_stuff=True)

        u_prev = u_0
        u = rk_step(lambda u_: f_allen_cahn(u_, dx), np.array(u_0), dt_nxt_nxt)
        _, u_h4 = strang_step(dt_nxt_nxt, ret_stuff=True)

        res.append(np.log2(np.max(u_h - u_h2)/np.max(u_h2 - u_h4)))

        dt0 = dt_nxt
    return res


# The strang step
def strang_step(dt_, ret_stuff=False):
    global u, u_prev, u_nxt, dx
    step_linear(u, u_prev, dx, 0.5 * dt_, u_nxt[0])
    step_cubic(u, u_prev, dx, dt_, u_nxt[1])
    step_linear(u_nxt[0], u_nxt[1], dx, 0.5*dt_, u_nxt[2])
    u_prev = u
    u = u_nxt[2]
    if ret_stuff:
        return np.array(u_prev), np.array(u)


n = 100
width = 3.0

dt = 0.00001
dx = width/n

# Initialize u(x, t) with random data between -1 and 1 on the interior and 0 on the boundary
u_0 = np.zeros([n, n])
u_0[1:n-1, 1:n-1] = 2*(np.random.rand(n-2, n-2)-0.5)

# Use RK4 to compute u_1 so we have two values to do AB2 timesteps
u_prev = np.array(u_0)
u = rk_step(lambda u_: f_allen_cahn(u_, dx), np.array(u_0), dt)
u_nxt = np.array(u)

it = 0
t = 0

# Print the approximate order of accuracy of the time stepping method
print(convergence_test()[-1])

# Record a movie of the flow map
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = plt.imshow(u_0, animated=True)
cb = fig.colorbar(mappable=im)
plt.title("Allen-Cahn Equation with AB2 timestep")
time_text = ax.text(0.01, 0.01, "Time = %0.4f s" % t, transform=ax.transAxes, color=[0, 1, 0])


# This function is called on every frame to update the plot
def update_fig(*args):
    global u, u_prev, u_nxt, it, t
    for _ in range(1):
        step(u, u_prev, dx, dt, u_nxt)
        u_prev = u
        u = u_nxt

    im.set_data(u)
    im.set_clim(np.min(u), np.max(u))
    it += 1
    time_text.set_text("Time = %0.4f s" % t)
    t += dt
    if it % 10 == 0:
        print(it)
        print(np.min(u), np.max(u))

    return im, time_text,


ani = animation.FuncAnimation(fig, update_fig, interval=30, blit=True, frames=1000)
# ani.save("allen_cahn.mp4", fps=30, extra_args=['-vcodec', 'libx264']) # <- Uncomment this to save the video file
plt.show()





