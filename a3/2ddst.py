import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack


def make_it_super_big(u_):
    nr, nc = u_.shape[0], u_.shape[1]

    ret = np.ndarray([2*nr, 2*nc], dtype=float)
    ret[0:nr, 0:nc] = u_
    ret[nr:, 0:nc] = -np.flip(u_, axis=0)
    ret[0:nr, nc:] = -np.flip(u_, axis=1)
    ret[nr:, nc:] = np.flip(np.flip(u_, axis=0), axis=1)

    return ret


def dst(u):
    # modes are 0, 1, 2, 3, ..., n/2, -n/2, -n/2-1, ..., -1
    ny, nx = u.shape[0], u.shape[1]

    alphas = np.fft.fftfreq(nx, 1.0/nx)
    betas = np.fft.fftfreq(ny, 1.0/ny)

    ret = np.array(u)
    for b in range(len(betas)):
        for a in range(len(alphas)):
            u_ab = 0.0
            alpha = alphas[a]
            beta = betas[b]
            for k in range(ny):
                fac = np.sin(np.pi*beta*k/ny)
                for j in range(nx):
                    u_ab += u[k, j] * np.sin(np.pi*alpha*j/nx) * fac
            ret[b, a] = u_ab
    return ret / (nx*ny)


def idst(u_f):
    ny, nx = u_f.shape[0], u_f.shape[1]

    ret = np.array(u_f)

    alphas = np.fft.fftfreq(nx, 1.0/nx)
    betas = np.fft.fftfreq(ny, 1.0/ny)

    for k in range(ny):
        for j in range(nx):
            u_ij = 0.0
            for a in range(len(alphas)):
                for b in range(len(betas)):
                    alpha = alphas[a]
                    beta = betas[b]
                    u_ij += u_f[b, a] * np.sin(np.pi*alpha*j/nx)*np.sin(np.pi*beta*k/ny)
            ret[k, j] = u_ij
    return ret