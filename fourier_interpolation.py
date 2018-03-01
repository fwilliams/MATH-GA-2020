# Numerical Methods II, Courant Institute, NYU, spring 2018
# http://www.math.nyu.edu/faculty/goodman/teaching/NumericalMethodsII2018/index.html
#  written by Jonathan Goodman (instructor)
#  see class notes Part 2 for more discussion

#   Illustrate Fourier interpolation ...
#   ... and how the Python FFT works ...
#   ... and some Python vector instructions that run faster than loops.

import numpy as np
import matplotlib.pyplot as plt


#     Function to be interpolated

# A symmetric hat function with uf(0) = 1 and u(x)=0 for |x|>r

def u1(x):
    return np.cos(x)
    r = 1.
    ax = np.abs(x)
    if ax > r:
        return 0.
    #    if x < -r/2.:
    #        return 0.
    else:  # uf(x) = 1-|x|/r if |x| < r
        return 1. - (ax / r)


u1name = "continuous"


# The hat function for x>0, but discontinuous jump to zero at x=0

def u2(x):
    r = 1.
    if x < 0:
        return 0.
    if x > r:
        return 0.
    return 1. - x / r


u2name = "discontinuous"


# A gaussian that is (almost) smooth if r << L

def u3(x):
    r = .4
    sx = x / r
    return np.exp(-sx * sx / 2.)


u3name = "smooth_big"


# A gaussian that is (almost) smooth if r << L

def u4(x):
    r = .1
    sx = x / r
    return np.exp(-sx * sx / 2.)


u4name = "smooth_small"

# --------------------------------------------------------------------------------


m = 10  # so n is even
n = 2 * m  # number of sample points
L = 4.  # length of the physical interval
dx = L / n  # for comments only, not used in the code

#    Uniformly spaced points in [-L/2,L/2], for periodic functions:
#          x[0] = -L/2 and x[n-1] = L/2 - dx.
#    Could be done with a scalar loop: xa[j] = (-L/2) + j*dx, ...
#    ... but that would be slower, and possibly less clear?

xa = np.linspace(0, 2*np.pi, num=n, endpoint=False)  # array of x values

mg = 30
ng = 2 * mg  # of points for graphics
xg = np.linspace(0, 2*np.pi, num=ng, endpoint=False)  # array of x values, for plotting

#    Apply the function u1 (or u2 or u3) to every entry in the array xa
#    ... and put the result in the array u.

u = list(map(u1, xa))  # choose functions 1, 2, or 3, AND !!!
ug = list(map(u1, xg))
name = u1name  # !!! MUST change the name when you change the function!!!

uh = np.fft.fft(u)  # "u hat", the Python "forward" FFT routine, ...
#  ... produces a length n complex array

#    "roll" from numpy does a circular shift every entry of uh moves left (or right") ...
#    ... by m-1 and the ones that fall off are copied onto the other end.

uh = np.roll(uh, m - 1)  # put frequency k=0 in the middle of the array

#    Evaluate the Fourier interpolant using the inverse FFT.  Must pad by zeros
#    on the left and right to make a longer vector.

uhp = np.zeros(ng, dtype=complex)  # "u hat padded", an array with  ...
# ... uh padded by zeros left and right.
# "dtype = complex" makes them complex zeros

js = mg - m  # "j start", location for first uDft[0] to go
je = js + n  # "j end", there are n entries in uDFT

#       Vectorized copy commands give ranges using colons.
#       You can find this under "slicing" in the documentation.

uhp[js:je] = uh[:]  # Also converts the type of uh to the type of uhp ..
# ... that's why we had to make uhp be complex zeros.

print(uhp)
uhp = np.roll(uhp, -mg + 1)  # Circular shift to get ready for the inverse FFT
ugi = (ng / n) * np.fft.ifft(uhp)  # The Python inverse FFT has a factor of (1/n) ...
# ... built in.  We want (1/n), not (1/ng).

plt.plot(xa, u, "o", xg, np.real(ugi), '.', xg, ug)
titleString = name + ", n = " + str(n)
plt.title(titleString)
fileName = "FourierInterp_" + name + ".pdf"
plt.savefig(fileName)
plt.show()
