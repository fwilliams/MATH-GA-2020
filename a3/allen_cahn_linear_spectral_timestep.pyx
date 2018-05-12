import numpy as np

def spectral_f(u, R):
    [nxp1, nyp1] = u.shape
    alphas = np.fft.fftfreq(nxp1, 1.0/nxp1)
    betas = np.fft.fftfreq(nyp1, 1.0/nyp1)

    out = np.array(u)

    for i in range(len(alphas)):
        for j in range(len(betas)):
            alpha = alphas[i]
            beta = betas[j]
            out[i, j] = u[i, j] * (1 + ((np.pi*1j*alpha)/R)**2 + ((np.pi*1j*beta)/R)**2)


def step_linear(u, u_prev, dx, dt, out):
    # TODO: Use a spectral Laplacian here
    [nxp1, nyp1] = u.shape
    nx = nxp1 - 1
    ny = nyp1 - 1

    [nxp1v, nyp1v] = u_prev.shape
    if out.shape != u.shape or u.shape != u_prev.shape:
        print("u_n and u_n_prev have different shapes")
        return False

    R = dx * nxp1
    k = 1.0 / (dx**2)

    u_f = np.fft.fft2(u)
    # u_f_prev = np.fft.fft2(u_prev)
    f_ij = spectral_f(u_f, R)
    # f_ij_prev = spectral_f(u_f_prev, R)
    out[1:nx, 1:ny] = np.fft.ifft2(f_ij)[1:nx, 1:ny]
    # def l_ij(x, y, u_n):
    #     return -4*u_n[x, y] + u_n[x+1, y] + u_n[x-1, y] + u_n[x, y+1] + u_n[x, y-1]
    #
    # def f_ij(x, y, u_n):
    #     return l_ij(x, y, u_n) + u_n[x, y]

    # for i in range(1, nx):
    #     for j in range(1, ny):
    #         out[i, j] = u[i, j] + dt * k * (1.5*f_ij(i, j, u) - 0.5*f_ij(i, j, u_prev))


def step_cubic(u, u_prev, dx, dt, out):
    [nxp1, nyp1] = u.shape
    nx = nxp1 - 1
    ny = nyp1 - 1

    [nxp1v, nyp1v] = u_prev.shape
    if out.shape != u.shape or u.shape != u_prev.shape:
        print("u_n and u_n_prev have different shapes")
        return False

    k = 1.0 / (dx**2)

    def f_ij(x, y, u_n):
        return - u_n[x, y]**3

    for i in range(1, nx):
        for j in range(1, ny):
            out[i, j] = u[i, j] + dt * k * (1.5*f_ij(i, j, u) - 0.5*f_ij(i, j, u_prev))
