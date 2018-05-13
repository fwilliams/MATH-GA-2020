import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Pretty plots
try:
    import seaborn as sns
    sns.set()
except ImportError:
    pass

# Load the time-stepping functions from cython
from ab2_allen_cahn_timestep import step, rk_step, f_allen_cahn


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
        u_h = np.array(u)
        u_h2 = np.array(u)
        u_h4 = np.array(u)

        step(u_0, rk_step(lambda u_: f_allen_cahn(u_, dx), np.array(u_0), dt0), dx, dt0, u_h)
        step(u_0, rk_step(lambda u_: f_allen_cahn(u_, dx), np.array(u_0), dt_nxt), dx, dt_nxt, u_h2)
        step(u_0, rk_step(lambda u_: f_allen_cahn(u_, dx), np.array(u_0), dt_nxt_nxt), dx, dt_nxt_nxt, u_h4)

        res.append(np.log2(np.abs(u_h[4, 3] - u_h2[4, 3])/np.abs(u_h2[4, 3] - u_h4[4, 3])))

        dt0 = dt_nxt
    return res


movie_or_pdf = 'movie'

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

if movie_or_pdf == 'movie':
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
elif movie_or_pdf == 'pdf':
    pp = PdfPages('multipage.pdf')
    for i in range(5):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        im = plt.imshow(u, animated=True)
        cb = fig.colorbar(mappable=im)
        plt.title("Allen-Cahn Equation with AB2 timestep")
        time_text = ax.text(0.01, 0.01, "Time = %0.4f s" % t, transform=ax.transAxes, color=[0, 1, 0])
        pp.savefig()
        for x in range(150):
            step(u, u_prev, dx, dt, u_nxt)
            u_prev = u
            u = u_nxt
            t += dt
    pp.close()
else:
    assert False, "movie_or_pdf must be one of 'movie' or 'pdf'"

