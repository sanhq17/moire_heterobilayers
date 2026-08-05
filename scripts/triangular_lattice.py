import numpy as np
import matplotlib.pyplot as plt

# Reciprocal lattice vectors
b1 = np.array([4*np.pi/np.sqrt(3), 0.0])
b2 = np.array([-2*np.pi/np.sqrt(3), 2*np.pi])

N = 5

# Generate lattice
points = {}
for m in range(-N, N+1):
    for n in range(-N, N+1):
        points[(m, n)] = m*b1 + n*b2

plt.figure(figsize=(8, 4))

# Faint triangular nearest-neighbour bonds (underlying lattice)
tri_shifts = [(1, 0), (0, 1), (1, 1)]
for (m, n), p in points.items():
    for dm, dn in tri_shifts:
        if (m+dm, n+dn) in points:
            q = points[(m+dm, n+dn)]
            plt.plot([p[0], q[0]], [p[1], q[1]],
                     color='k', lw=1.3, zorder=1)

# Rectangular bonds within each colour.
# (1,0) -> b1 = (4pi/sqrt3, 0)   horizontal
# (1,2) -> b1 + 2*b2 = (0, 4pi)  vertical
# Both shifts preserve parity of n, so they never mix colours.
rect_shifts = [(1, 0), (1, 2)]
for (m, n), p in points.items():
    colour = 'royalblue' if n % 2 == 0 else 'crimson'
    for dm, dn in rect_shifts:
        if (m+dm, n+dn) in points:
            q = points[(m+dm, n+dn)]
            plt.plot([p[0], q[0]], [p[1], q[1]],
                     color=colour, lw=0.5, ls='--', alpha=0.8, zorder=2)

# Split into two rectangular sublattices
A, B = [], []
for (m, n), p in points.items():
    (A if n % 2 == 0 else B).append(p)
A, B = np.array(A), np.array(B)

plt.scatter(A[:, 0], A[:, 1], color='royalblue', s=50,
            label='Rectangular sublattice A', zorder=3)
plt.scatter(B[:, 0], B[:, 1], color='crimson', s=50,
            label='Rectangular sublattice B', zorder=3)

# Draw the three vectors from the origin
ax = plt.gca()
b3 = b1 + 2*b2   # = (0, 4pi)

vectors = [
    (b1,      'darkgreen',  r'$\widetilde{\mathbf{b}}_1$'),
    (b2,      'darkorange', r'$\widetilde{\mathbf{b}}_3$'),
    (b3,      'purple',     r'$\widetilde{\mathbf{v}}$'),
]

for vec, col, lbl in vectors:
    ax.annotate("", xy=(vec[0], vec[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5,
                                mutation_scale=20, shrinkA=0, shrinkB=0),
                zorder=5)
    ax.text(vec[0], vec[1]*1.1, lbl, color=col,
            fontsize=13, fontweight='bold', zorder=6)

ax.set_aspect('equal')
plt.xlabel(r'$k_x$')
plt.ylabel(r'$k_y$')
plt.grid(alpha=0.3)
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.yticks([])
plt.xticks([])
plt.savefig("rectangular_lattices.pdf", bbox_inches='tight', dpi=300)
plt.show()