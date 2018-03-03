import numpy as np
import matplotlib.pyplot as plt
from spectral_derivative import spectral_derivative
import matplotlib.animation as animation


# Time derivative of the Burgers Sivashinsky Equation
def f_bs(u):
    global R

    # Numerical hack to make sure roundoff in the zero term is always zero
    u = np.fft.fft(u)
    u[0] = 0
    u = np.fft.ifft(u)

    dudx = spectral_derivative(u, R, len(u), der=1)
    dudx2 = spectral_derivative(u, R, len(u), der=2)
    return u + dudx2 - u*dudx


# Time derivative of the Kuramoto Sivashinsky Equation
def f_ks(u):
    global R
    dudx = spectral_derivative(u, R, len(u), der=1)
    dudx2 = spectral_derivative(u, R, len(u), der=2)
    dudx4 = spectral_derivative(u, R, len(u), der=4)
    return -u*dudx - dudx2 - dudx4


# Fourth order Runge-Kutta time stepping function
def rk_step(fun, u, h):
    k1 = fun(u)
    k2 = fun(u + h * k1 / 2)
    k3 = fun(u + h * k2 / 2)
    k4 = fun(u + h * k3)
    return u + (h/6) * (k1 + 2*k2 + 2*k3 + k4)


# Hack to plot a final point which equals the first
def wraparound(x):
    ret = np.ndarray([len(x)+1,], dtype=x.dtype)
    ret[0:-1] = x
    ret[-1] = x[0]
    return ret


# Simulation parameters
time_step = 0.001
R = 30.0
n = 60
n_nodes = n
assert n % n_nodes == 0, "n_nodes must be a multiple of n"

xs = np.linspace(0, R, n, endpoint=False)
u0 = 5*np.cos((2*np.pi/R)*xs)  # The initial condition
assert np.fft.fft(u0)[0] < 1e-15, "u0 does not integrate to zero"

u = u0  # The current function value
t = 0  # The current time
f = f_ks  # The evolution function to use
fname = "Burgers Sivashinsky" if f == f_bs else "Kuramoto Sivashinsky"


# Video recording parameters
record_seconds = 5  # Record this many seconds of video
video_filename = "ks2.mp4"  # Save the video to this file
fps = 33  # Use this framerate for the video
total_frames = int(record_seconds / (timesteps_per_frame*time_step))  # Total number of video frames to record
timesteps_per_frame = np.round(1 / (fps * time_step))  # How many time steps to do between frames
ms_per_frame = (time_step * timesteps_per_frame)*1000

# Plot for animation
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_xlim(0, R)
ax1.axis([0, R, -6, 6])
ax1.grid()
ax1.set_title("Trajectory of %s over time starting from $u(x, 0) = cos(x)$" % fname)
ax1.set_xlabel("x")
ax1.set_ylabel("u(x, t)")
time_text = ax1.text(0.01, 0.1, "Time = %0.4f" % t, transform=ax1.transAxes)
ut_line, = ax1.plot(np.concatenate([xs, np.array([R])]), wraparound(u0))

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title("Discrete Fourier coefficients")
ax2.axis([-len(u)//2, len(u)//2-1, -5, 5])
ax2.set_xticks(np.arange(-len(u)//2, len(u)//2))
ax2.set_xlabel("Fourier coefficient, $\\alpha$")
ax2.set_ylabel("$\\frac{2}{N} log_{10}(|\hat{u}_\\alpha|)$")
ax2.grid()
log_power_line, = ax2.plot(np.arange(-len(u)//2, len(u)//2), np.zeros(len(u)))
log_power_pts = ax2.scatter(np.arange(-len(u)//2, len(u)//2), np.zeros(len(u)))

fig.tight_layout()


def animate(i):
    global u, time_step, t, total_frames, timesteps_per_frame
    for _ in range(timesteps_per_frame):
        t += time_step
        u_nxt = rk_step(f, u, time_step)
        u = u_nxt

    ut_line.set_ydata(wraparound(u))  # update the data
    time_text.set_text("Time = %0.4f" % t)

    uh = np.fft.fftshift(np.fft.fft(u))
    y_data = 2/len(u) * np.log(np.abs(uh))
    log_power_line.set_ydata(y_data)

    positions = np.zeros([len(u), 2])
    positions[:, 0] = np.arange(-len(u)//2, len(u)//2)
    positions[:, 1] = y_data
    log_power_pts.set_offsets(positions)

    print("%d/%d" % (i, total_frames))
    return ut_line, log_power_line, log_power_pts, time_text


# Init only required for blitting to give a clean slate.
def init():
    global u0, log_power_line
    ut_line.set_ydata(np.array([u0, u0[0]]))
    log_power_line.set_ydata(np.zeros(len(u0)))
    positions = np.zeros([len(u), 2])
    positions[:, 0] = np.arange(-len(u)//2, len(u)//2)
    positions[:, 1] = np.zeros(len(u))
    log_power_pts.set_offsets(positions)
    time_text.set_text("Time = %0.4f" % t)
    return ut_line, log_power_line, log_power_pts, time_text


ani = animation.FuncAnimation(fig, animate, frames=total_frames, init_func=init,
                              interval=ms_per_frame, blit=True, repeat=False)
# ani.save(video_filename, fps=fps, extra_args=['-vcodec', 'libx264'])
plt.show()


