import numpy as np

# Adams Bashforth 2 timestep of finite difference Allen Cahn equation
def step(u, u_prev, dx, dt, out):
    [nxp1, nyp1] = u.shape
    nx = nxp1 - 1
    ny = nyp1 - 1

    [nxp1v, nyp1v] = u_prev.shape
    if out.shape != u.shape or u.shape != u_prev.shape:
        print("u_n and u_n_prev have different shapes")
        return False

    # k = 1.0 / (dx**2)

    def l_ij(x, y, u_n):
        return (u_n[x+1, y] + u_n[x-1, y] + u_n[x, y+1] + u_n[x, y-1] - 4*u_n[x, y]) / dx**2

    def f_ij(x, y, u_n):
        return l_ij(x, y, u_n) + u_n[x, y]/dx**2 - (u_n[x, y]/dx**2)**3

    for i in range(1, nx):
        for j in range(1, ny):
            out[i, j] = u[i, j] + dt * (1.5*f_ij(i, j, u) - 0.5*f_ij(i, j, u_prev))


# Adams Bashforth 2 timestep of finite difference linear part of the Allen Cahn equation
def step_linear(u, u_prev, dx, dt, out):
    print(u.shape, u_prev.shape)
    [nxp1, nyp1] = u.shape
    nx = nxp1 - 1
    ny = nyp1 - 1

    [nxp1v, nyp1v] = u_prev.shape
    if out.shape != u.shape or u.shape != u_prev.shape:
        print("u_n and u_n_prev have different shapes")
        return False

    k = 1.0 / (dx**2)

    def l_ij(x, y, u_n):
        return (u_n[x+1, y] + u_n[x-1, y] + u_n[x, y+1] + u_n[x, y-1] - 4*u_n[x, y]) / dx**2

    def f_ij(x, y, u_n):
        return l_ij(x, y, u_n) + u_n[x, y]

    for i in range(1, nx):
        for j in range(1, ny):
            out[i, j] = u[i, j] + dt * (1.5*f_ij(i, j, u) - 0.5*f_ij(i, j, u_prev))


# Adams Bashforth 2 timestep of finite difference cubic part of the Allen Cahn equation
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
        return -u_n[x, y]**3

    for i in range(1, nx):
        for j in range(1, ny):
            out[i, j] = u[i, j] + dt * k * (1.5*f_ij(i, j, u) - 0.5*f_ij(i, j, u_prev))


# The RK4 step used to get x_1 from x_0
def rk_step(fun, u, h):
    k1 = fun(u)
    k2 = fun(u + h * k1 / 2)
    k3 = fun(u + h * k2 / 2)
    k4 = fun(u + h * k3)
    return u + (h/6) * (k1 + 2*k2 + 2*k3 + k4)


def f_allen_cahn(u, dx):
    [nxp1, nyp1] = u.shape
    nx = nxp1 - 1
    ny = nyp1 - 1

    out = np.array(u)

    k = 1.0 / (dx**2)

    def l_ij(x, y, u_n):
        return (u_n[x+1, y] + u_n[x-1, y] + u_n[x, y+1] + u_n[x, y-1] - 4*u_n[x, y]) / dx**2

    def f_ij(x, y, u_n):
        return l_ij(x, y, u_n) + u_n[x, y]/dx**2 - (u_n[x, y]/dx**2)**3

    for i in range(1, nx):
        for j in range(1, ny):
            out[i, j] = f_ij(i, j, u)
    return out