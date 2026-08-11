import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ----------------------------------------------------------------------
# Triangular lattice in the z = 0 plane
# ----------------------------------------------------------------------
a = 1.0
a1 = a * np.array([1.0, 0.0])
a2 = a * np.array([0.5, np.sqrt(3) / 2])

R = 2
pts = np.array([m * a1 + n * a2
                for m in range(-R, R + 1)
                for n in range(-R, R + 1)])
X, Y = pts[:, 0], pts[:, 1]
Z = np.zeros_like(X)

# ----------------------------------------------------------------------
# Solid 3D arrow (capped cylinder shaft + cone head) along +z
# ----------------------------------------------------------------------
def cylinder(z0, z1, radius, n=40):
    th = np.linspace(0, 2 * np.pi, n)
    zz = np.linspace(z0, z1, 2)
    TH, ZZ = np.meshgrid(th, zz)
    return radius * np.cos(TH), radius * np.sin(TH), ZZ

def cone(z0, z1, r_base, n=40):
    th = np.linspace(0, 2 * np.pi, n)
    zz = np.linspace(z0, z1, 20)
    TH, ZZ = np.meshgrid(th, zz)
    r = r_base * (1 - (ZZ - z0) / (z1 - z0))
    return r * np.cos(TH), r * np.sin(TH), ZZ

def disk(z, r_out, r_in=0.0, n=40, nr=2):
    """Filled disk (or annulus if r_in>0) in the plane z=const."""
    th = np.linspace(0, 2 * np.pi, n)
    rr = np.linspace(r_in, r_out, nr)
    TH, RR = np.meshgrid(th, rr)
    return RR * np.cos(TH), RR * np.sin(TH), np.full_like(TH, z)

L        = 1.0
shaft_f  = 0.72
r_shaft  = 0.04
r_head   = 0.12
zc = shaft_f * L

parts = [
    cylinder(0.0, zc, r_shaft),          # shaft wall
    disk(0.0, r_shaft),                  # bottom cap
    disk(zc, r_head, r_in=r_shaft),      # annular underside of the head
    cone(zc, L, r_head),                 # head
]

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(X, Y, Z, color="steelblue", s=10, depthshade=True)
nn = a * 1.001
for i in range(len(pts)):
    for j in range(i + 1, len(pts)):
        if np.linalg.norm(pts[i] - pts[j]) < nn:
            ax.plot([X[i], X[j]], [Y[i], Y[j]], [0, 0],
                    color="lightgray", lw=0.8, zorder=1)

ls = LightSource(azdeg=315, altdeg=45)
base_rgb = "darkorange"#np.array([0.86, 0.08, 0.24])

for (Xc, Yc, Zc_) in parts:
    ax.plot_surface(
        Xc, Yc, Zc_,
        color=base_rgb, shade=True, lightsource=ls,
        rstride=1, cstride=1, linewidth=0,
        antialiased=True, zorder=5,
    )

# ax.scatter([0], [0], [0], color="crimson", s=60)

# ----------------------------------------------------------------------
# Strip axes, panes, gridlines
# ----------------------------------------------------------------------
span = R * a
ax.set_xlim(-span, span)
ax.set_ylim(-span, span)
ax.set_zlim(0, L * 1.1)
try:
    ax.set_box_aspect((2 * span, 2 * span, L * 1.1))
except Exception:
    pass

ax.set_axis_off()
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.set_pane_color((1, 1, 1, 0))
    axis.line.set_color((1, 1, 1, 0))
    axis._axinfo["grid"]["color"] = (1, 1, 1, 0)

ax.view_init(elev=25, azim=-60)
plt.tight_layout()
plt.show()