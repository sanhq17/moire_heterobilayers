import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

###########################################################################
# Parameters
###########################################################################

a = 1.0                      # lattice constant
m = 1.0                      # electron mass (hbar=1)
V0 = 0.5                     # Fourier coefficient of crystal potential

Nk = 31                      # k-point mesh
Nx = 81                      # real-space mesh
Ny = 81

###########################################################################
# Reciprocal lattice vectors
###########################################################################

b1 = 2*np.pi*np.array([1,0])/a
b2 = 2*np.pi*np.array([0,1])/a

# Five plane waves:
#
# G = (0,0)
# ±b1
# ±b2
#

Glist = np.array([
    [0,0],
    [1,0],
    [-1,0],
    [0,1],
    [0,-1]
],dtype=int)

NG = len(Glist)

###########################################################################
# Fourier coefficients of the potential
#
# V(r)=2V0(cos(Gx x)+cos(Gy y))
###########################################################################

def VFourier(delta):

    if np.array_equal(delta,[1,0]):
        return V0

    if np.array_equal(delta,[-1,0]):
        return V0

    if np.array_equal(delta,[0,1]):
        return V0

    if np.array_equal(delta,[0,-1]):
        return V0

    return 0.0


###########################################################################
# Hamiltonian
###########################################################################

def Hk(k):

    H = np.zeros((NG,NG),dtype=complex)

    for i,G in enumerate(Glist):

        q = k + G[0]*b1 + G[1]*b2

        H[i,i] = np.dot(q,q)/(2*m)

        for j,Gp in enumerate(Glist):

            dG = G-Gp

            H[i,j] += VFourier(dG)

    return H


###########################################################################
# k mesh
###########################################################################

kx = np.linspace(-np.pi/a,np.pi/a,Nk,endpoint=False)
ky = np.linspace(-np.pi/a,np.pi/a,Nk,endpoint=False)

bands = np.zeros((Nk,Nk,NG))
eigvecs = np.zeros((Nk,Nk,NG,NG),dtype=complex)

for ix,k1 in enumerate(kx):

    for iy,k2 in enumerate(ky):

        H = Hk(np.array([k1,k2]))

        E,U = eigh(H)

        bands[ix,iy] = E
        eigvecs[ix,iy] = U

###########################################################################
# Plot band energies
###########################################################################

# plt.figure(figsize=(6,4))

# for n in range(NG):

#     plt.imshow(bands[:,:,n].T,
#                origin='lower')

#     plt.colorbar()

#     plt.title("Band {}".format(n))

#     plt.show()

###########################################################################
# Wannier function of one band
###########################################################################

x = np.linspace(-2*a,2*a,Nx)
y = np.linspace(-2*a,2*a,Ny)

X,Y = np.meshgrid(x,y,indexing='ij')

###########################################################################
# top two bands
###########################################################################

topbands = [0, 1]

wannier = []

for band in topbands:

    W = np.zeros_like(X,dtype=complex)

    for ix,k1 in enumerate(kx):

        for iy,k2 in enumerate(ky):

            psi = np.zeros_like(X,dtype=complex)

            coeff = eigvecs[ix,iy,:,band]

            for c,G in zip(coeff,Glist):

                q = np.array([k1,k2]) + G[0]*b1 + G[1]*b2

                psi += c*np.exp(1j*(q[0]*X+q[1]*Y))

            W += psi

    W /= Nk**2

    wannier.append(W)

###########################################################################
# Plot
###########################################################################

for i,W in enumerate(wannier):

    plt.figure(figsize=(6,5))

    plt.imshow(np.abs(W)**2,
               extent=[x.min(),x.max(),y.min(),y.max()],
               origin='lower')

    plt.colorbar()

    plt.title("Wannier function band {}".format(topbands[i]))

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()