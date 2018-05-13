import numpy as np
import scipy as sp
import scipy.fftpack
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def spectral_neumann_laplacian(u, period):
    N = u.shape[0]
    assert u.shape[0] == u.shape[1], "Only squares please"
    u_f = sp.fftpack.dctn(u, norm='ortho')
    alphas, betas = np.meshgrid(np.fft.fftfreq(N, 1 / N), np.fft.fftfreq(N, 1 / N))
    u_f *= (-(np.pi*alphas/period)**2 - (np.pi*betas/period)**2)
    return sp.fftpack.idctn(u_f, norm='ortho')


def time_step(u_i, u_i_prev):
    l_u = spectral_neumann_laplacian(u_i, R)
    return 2*u_i - u_i_prev + dt**2 * c**2 * l_u


# Simulation parameters
N = 512       # Number of grid cells
R = 1.0       # Width and height of the box in real units
r = R/8.0     # Width of the "slow" region
c_min = 0.5   # Minimum c value
dt = 0.005/3  # Time Step

# Generate the c field
xs, ys = np.meshgrid(np.linspace(-R/2, R/2, N), np.linspace(-R/2, R/2, N))
c = 1.0 - (1.0 - c_min)*np.exp(-(xs**2 + ys**2) / (2*r**2))

# Place a finite supported bump on the left side of the domain as the initial condition
bump_64 = np.zeros([64, 64])
xs64, ys64 = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
r = xs64**2 + ys64**2
bump_64 = (1.0-r)**4 * (1.0 + 4*r)
bump_64[r >= 1] = 0.0

u_prev = np.zeros([N, N])
u_prev[224:224 + 64, 32:32 + 64] = bump_64
u_cur = u_prev


# Plot the speed of sound in the domain
plt.imshow(c**2)
plt.show()

# Make a video
t = 0
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = plt.imshow(u_prev, animated=True)
cb = fig.colorbar(mappable=im)
plt.title("Scalar Wave Equation")
time_text = ax.text(0.01, 0.01, "Time = %0.4f s" % t, transform=ax.transAxes, color=[0, 1, 0])


# This function is called on every frame to update the plot
def update_fig(*args):
    global u_cur, u_prev, t
    for _ in range(1):
        u_nxt = time_step(u_cur, u_prev)
        u_prev = u_cur
        u_cur = u_nxt

    im.set_data(u_cur)
    im.set_clim(0.0, 0.5)
    time_text.set_text("Time = %0.4f s" % t)
    t += dt

    return im, time_text,


ani = animation.FuncAnimation(fig, update_fig, interval=30, blit=True, frames=1000)
# ani.save("wave_eqn.mp4", fps=30, extra_args=['-vcodec', 'libx264']) # <- Uncomment this to save the video file
plt.show()

