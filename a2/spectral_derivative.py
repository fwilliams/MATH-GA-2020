import numpy as np
import matplotlib.pyplot as plt


def spectral_derivative(node_values, period, n_pts, der=0):
    n = len(node_values)
    m = n // 2
    uh = np.fft.fftshift(np.fft.fft(node_values))
    alphas = 2j * np.pi * np.arange(-m, m) / period

    uh *= alphas ** der
    mg = n_pts // 2
    uhp = np.zeros(n_pts, dtype=np.complex)
    start = mg - m
    end = start + n
    uhp[start:end] = uh

    shifted_back = np.fft.ifftshift(uhp)
    return (n_pts / n) * np.real(np.fft.ifft(shifted_back))


if __name__ == "__main__":
    R = 2 * np.pi
    alpha = 4
    k = 2 * np.pi * alpha / R

    def f(x): return np.sin(x)**2 + np.cos(x)/0.5 + np.cos(x+0.2)**4

    def df_exact(x): return 2*np.sin(x)*np.cos(x) - np.sin(x)/0.5 - 4*np.sin(x+0.2)*np.cos(x+0.2)**3

    nodes_x = np.linspace(0, R, 10, endpoint=False)
    nodes_y = f(nodes_x)  # np.cos(k * nodes_x)

    xs = np.linspace(0, R, 100, endpoint=False)
    df = spectral_derivative(nodes_y, R, n_pts=len(xs), der=1)

    print(np.max(np.abs(df - df_exact(xs))))

    plt.scatter(nodes_x, nodes_y)
    plt.plot(xs, f(xs))
    plt.plot(xs, df_exact(xs))
    plt.plot(xs, df)
    # plt.plot(xs, df_exact)
    plt.show()
