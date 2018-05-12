# Adams Bashforth 2 timestep of finite difference Laplacian
def step(u, u_prev, D, dx, dt, out):
    [nxp1, nyp1] = u.shape
    nx = nxp1 - 1
    ny = nyp1 - 1

    [nxp1v, nyp1v] = u_prev.shape
    if out.shape != u.shape or u.shape != u_prev.shape:
        print("u_n and u_n_prev have different shapes")
        return False

    def l_ij(x, y, u_n):
        return -4*u_n[x, y] + u_n[x+1, y] + u_n[x-1, y] + u_n[x, y+1] + u_n[x, y-1]

    k = (dt * D / dx**2)
    for i in range(1, nx):
        for j in range(1, ny):
            out[i, j] = u[i, j] + k * (1.5*l_ij(i, j, u) - 0.5*l_ij(i, j, u_prev))
