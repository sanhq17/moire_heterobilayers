import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as mplPath
from HeteroBilayer import HeteroBilayer
import utils

# LaTeX rendering parameters
plt.rcParams.update({
    'figure.figsize': (6, 5),
    'figure.dpi': 120,          # crisper on screen
    'savefig.dpi': 300,         # high-res when saving
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.spines.top': True,   # remove top/right borders
    'axes.spines.right': True,
    'axes.grid': False,
    'lines.linewidth': 2,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
})


# ------------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------------

def plot_bands(ax, bands, kdist, ticks, max_bands=4, linestyle='-', lw=1, color='k'):
    ax.plot(kdist, bands[:, 0], color=color, lw=lw, ls=linestyle)
    ax.plot(kdist, bands[:, 1:max_bands], color=color, lw=lw, ls=linestyle)

    tick_positions = [t[0] for t in ticks]
    tick_labels    = [t[1] for t in ticks]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=12)
    for x in tick_positions:
        ax.axvline(x, color='k', lw=0.5, zorder=0)

    ax.set_xlim(kdist[0], kdist[-1])
    ax.set_ylabel(r"$E$ [meV]", fontsize=18)

    return ax

def plot_moire_potential(sys, ax, extent, resolution=200):
    x = np.linspace(extent[0], extent[1], resolution)
    y = np.linspace(extent[2], extent[3], resolution)
    X, Y = np.meshgrid(x, y)

    pot = sys.moire_potential(X, Y)
    im = ax.imshow(pot, extent=extent, origin='lower', cmap='RdBu_r')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Delta(\mathbf{r})$ [meV]", fontsize=18)

    return im

def plot_triangular_lattice(sys, ax, extent):
    r1 = sys.a1 
    r2 = sys.a2 
    r3 = r2 - r1

    x_min, x_max, y_min, y_max = extent

    # Build a grid of lattice sites large enough to cover the plotting extent.
    n_max = int(np.ceil(max(x_max - x_min, y_max - y_min))) + 4
    lattice_pts = []
    for n1 in range(-n_max, n_max + 1):
        for n2 in range(-n_max, n_max + 1):
            lattice_pts.append(n1 * r1 + n2 * r2)
    lattice_pts = np.array(lattice_pts)

    # Draw nearest-neighbor bonds as dashed lines.
    offsets = [r1, r2, r3]
    for p in lattice_pts:
        for off in offsets:
            q = p + off
            x0, y0 = p
            x1, y1 = q
            if (min(x0, x1) <= x_max and max(x0, x1) >= x_min and
                min(y0, y1) <= y_max and max(y0, y1) >= y_min):
                #ax.scatter([x0, x1], [y0, y1], color='r', s=10, alpha=0.9, edgecolors='k', linewidth=1.0)
                ax.plot([x0, x1], [y0, y1], color='#33221E', ls='--', lw=0.8, alpha=0.9)
    
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

def plot_k_mesh(self, ax, Nx, Ny):
    k_points, _, _, _ = self.mesh_k_hex(Nx=Nx, Ny=Ny)
    k_points_bz = self.map_to_first_BZ(k_points)
    ax.scatter(k_points_bz[:, 0], k_points_bz[:, 1], color='#FF216B', s=30, alpha=0.9, edgecolors='k', linewidth=1.0)
    self.plot_hexagonal_bz(ax)

def plot_wavefunction(ax, wavefunction, extent, label="", cmap='RdBu_r', norm=None, levels=12,
                      boundary=None):
    # im = ax.imshow(np.abs(wavefunction)**2, extent=extent, origin='lower', 
    #             interpolation='none', cmap=cmap, norm=norm)
    # cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label(label, fontsize=18)
    # return im
    density = np.abs(wavefunction)**2
    
    ny, nx = density.shape
    x = np.linspace(extent[0], extent[1], nx)
    y = np.linspace(extent[2], extent[3], ny)
    X, Y = np.meshgrid(x, y)

    # Filled contours
    cf = ax.contourf(X, Y, density, levels=levels, cmap=cmap, norm=norm)

    # Thin black line on every level for the crisp outlined look
    ax.contour(X, Y, density, levels=cf.levels, colors='k',
            linewidths=0.5)

    # Optional dashed boundary overlay (e.g. Brillouin zone / unit cell)
    if boundary is not None:
        bx, by = boundary  # arrays of vertex coords, closed polygon
        ax.plot(bx, by, '--', color='red', linewidth=1.5)

    # Boxed corner label
    if label is not None:
        ax.text(0.05, 0.95, label, transform=ax.transAxes,
                va='top', ha='left', fontsize=14,
                bbox=dict(boxstyle='square', facecolor='white',
                        edgecolor='black', linewidth=1.5))

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    return cf

def plot_wavefunction_phase(ax, wavefunction, extent, cmap='RdBu_r'):
    im = ax.imshow(np.abs(np.angle(wavefunction)), extent=extent, origin='lower', 
                interpolation='none', cmap=cmap)
    return im

def plot_wannier_function(sys, axs, X, Y, k_points, trial_centers=np.array([np.zeros(2)]), bands=(-1,), R0=np.zeros(2), norm=None, cmap='RdBu_r'):
    wanniers = sys.wannier_function(X, Y, k_points, trial_centers=trial_centers, bands=bands, R0=R0)
    extent = [X.min(), X.max(), Y.min(), Y.max()]

    if np.isscalar(axs) or isinstance(axs, plt.Axes):
        plot_wavefunction(axs, wanniers, extent, label=r"$|w_0(\mathbf{r})|^2$", norm=norm, cmap=cmap)
        plot_triangular_lattice(sys, axs, extent)
        axs.set_xlabel(r'$x/a_M$', fontsize=22)
        axs.set_ylabel(r'$y/a_M$', fontsize=22)
    else:
        labels = [r"$|w_{0A}(\mathbf{r})|^2$", r"$|w_{0B}(\mathbf{r})|^2$"]
        for ax, wannier, label in zip(np.ravel(axs), wanniers, labels):
            plot_wavefunction(ax, wannier, extent, label=label, norm=norm, cmap=cmap)
            plot_triangular_lattice(sys, ax, extent)
            ax.set_xlabel(r'$x/a_M$', fontsize=22)
            ax.set_ylabel(r'$y/a_M$', fontsize=22)

def plot_wannier_function2(self, ax, X, Y, k_points, R0=np.zeros(2), norm=None, cmap='RdBu_r'):
    return 0

def plot_berry_curvature(sys, ax, KX, KY, abs_val=False, mask_bz=True):
    k_points = np.column_stack([KX.ravel(), KY.ravel()])
    berry = sys.berry_curvature(k_points).reshape(KX.shape)
    if abs_val:
        berry = np.abs(berry)
        cmap = 'Reds'
    else:
        cmap = 'RdBu_r'

    # rebuild matching extent
    extent = [KX.min() / sys.ktheta, KX.max() / sys.ktheta, KY.min() / sys.ktheta, KY.max() / sys.ktheta]

    if mask_bz:
        dk = KX[0, 1] - KX[0, 0]
        mask = sys.BZ_path.contains_points(k_points, radius=dk)              # Test whether points are inside
        mask = mask.reshape(KX.shape)                                           # Reshape back to grid shape
    else:
        mask = np.ones_like(berry, dtype=bool)

    berry_masked = np.where(mask, berry, 0.0)

    plot_hexagonal_bz(sys, ax)
    im = ax.imshow(berry_masked, extent=extent, origin='lower', 
                interpolation='none', cmap=cmap)
    ax.set_title(r"$\Omega(\mathbf{k})$")
    plt.colorbar(im, ax=ax)
    
    return im

def plot_hexagonal_bz(sys, ax, lw=3.0):
    verts = sys.BZ_path.vertices          
    corners = np.vstack([verts, verts[0]]) / sys.ktheta  # close the loop for plotting
    ax.plot(corners[:, 0], corners[:, 1], color='k', lw=lw, zorder=3)
    ax.text((sys.kappa[0] + sys.kappa_pp[0]) / (2 * sys.ktheta), sys.kappa[1] / sys.ktheta,
            "1BZ", fontsize=16, color='k')

def plot_metric(sys, ax, KX, KY, mask_bz=True):
    k_points = np.column_stack([KX.ravel(), KY.ravel()])
    g_xx, g_xy, g_yy = sys.quantum_metric(k_points)
    g_xx = g_xx.reshape(KX.shape)
    g_xy = g_xy.reshape(KX.shape)
    g_yy = g_yy.reshape(KX.shape)

    if mask_bz:
        dk = KX[0, 1] - KX[0, 0]
        mask = sys.BZ_path.contains_points(k_points, radius=dk)             # Test whether points are inside
        mask = mask.reshape(KX.shape)                                        # Reshape back to grid shape
    else:
        mask = np.ones_like(g_xx, dtype=bool)

    g_xx_masked = np.where(mask, g_xx, 0.0)
    g_xy_masked = np.where(mask, g_xy, 0.0)
    g_yx_masked = g_xy_masked
    g_yy_masked = np.where(mask, g_yy, 0.0)

    # rebuild matching extent
    extent = [KX.min() / sys.ktheta, KX.max() / sys.ktheta, KY.min() / sys.ktheta, KY.max() / sys.ktheta]

    lw = 3.0

    im00 = ax[0, 0].imshow(g_xx_masked, origin='lower', extent=extent, cmap='RdBu_r', interpolation='none')
    ax[0, 0].set_title(r"$g_{xx}$")
    ax[0, 0].set_xlabel(r"$k_x / k_\theta$")
    ax[0, 0].set_ylabel(r"$k_y / k_\theta$")
    plot_hexagonal_bz(sys, ax[0, 0], lw=lw)
    plt.colorbar(im00, ax=ax[0, 0])

    im01 = ax[0, 1].imshow(g_xy_masked, origin='lower', extent=extent, cmap='RdBu_r', interpolation='none')
    ax[0, 1].set_title(r"$g_{xy}$")
    ax[0, 1].set_xlabel(r"$k_x / k_\theta$")
    ax[0, 1].set_ylabel(r"$k_y / k_\theta$")
    plot_hexagonal_bz(sys, ax[0, 1], lw=lw)
    plt.colorbar(im01, ax=ax[0, 1])

    im10 = ax[1, 0].imshow(g_yx_masked, origin='lower', extent=extent, cmap='RdBu_r', interpolation='none')
    ax[1, 0].set_title(r"$g_{yx}$")
    ax[1, 0].set_xlabel(r"$k_x / k_\theta$")
    ax[1, 0].set_ylabel(r"$k_y / k_\theta$")
    plot_hexagonal_bz(sys, ax[1, 0], lw=lw)
    plt.colorbar(im10, ax=ax[1, 0])

    im11 = ax[1, 1].imshow(g_yy_masked, origin='lower', extent=extent, cmap='RdBu_r', interpolation='none')
    ax[1, 1].set_title(r"$g_{yy}$")
    ax[1, 1].set_xlabel(r"$k_x / k_\theta$")
    ax[1, 1].set_ylabel(r"$k_y / k_\theta$")
    plot_hexagonal_bz(sys, ax[1, 1], lw=lw)
    plt.colorbar(im11, ax=ax[1, 1])

def plot_metric_band(sys, ax, KX, KY, band, mask_bz=True):
    k_points = np.column_stack([KX.ravel(), KY.ravel()])
    g_xx, g_xy, g_yy = sys.quantum_metric(k_points)
    g_xx = g_xx[:, band].reshape(KX.shape)
    g_xy = g_xy[:, band].reshape(KX.shape)
    g_yy = g_yy[:, band].reshape(KX.shape)

    if mask_bz:
        dk = KX[0, 1] - KX[0, 0]
        mask = sys.BZ_path.contains_points(k_points, radius=dk)             # Test whether points are inside
        mask = mask.reshape(KX.shape)                                        # Reshape back to grid shape
    else:
        mask = np.ones_like(g_xx, dtype=bool)

    g_xx_masked = np.where(mask, g_xx, 0.0)
    g_xy_masked = np.where(mask, g_xy, 0.0)
    g_yx_masked = g_xy_masked
    g_yy_masked = np.where(mask, g_yy, 0.0)

    # rebuild matching extent
    extent = [KX.min() / sys.ktheta, KX.max() / sys.ktheta, KY.min() / sys.ktheta, KY.max() / sys.ktheta]

    lw = 3.0

    if band == 0:
        band_label = "A"
    elif band == 1:
        band_label = "B"

    im00 = ax[0, 0].imshow(g_xx_masked, origin='lower', extent=extent, cmap='RdBu_r', interpolation='none')
    ax[0, 0].set_title(rf"$g^{{({band_label})}}_{{xx}}$")
    ax[0, 0].set_xlabel(r"$k_x / k_\theta$")
    ax[0, 0].set_ylabel(r"$k_y / k_\theta$")
    plot_hexagonal_bz(sys, ax[0, 0], lw=lw)
    plt.colorbar(im00, ax=ax[0, 0], fraction=0.046, pad=0.04)

    im01 = ax[0, 1].imshow(g_xy_masked, origin='lower', extent=extent, cmap='RdBu_r', interpolation='none')
    ax[0, 1].set_title(rf"$g^{{({band_label})}}_{{xy}}$")
    ax[0, 1].set_xlabel(r"$k_x / k_\theta$")
    ax[0, 1].set_ylabel(r"$k_y / k_\theta$")
    plot_hexagonal_bz(sys, ax[0, 1], lw=lw)
    plt.colorbar(im01, ax=ax[0, 1], fraction=0.046, pad=0.04)

    im10 = ax[1, 0].imshow(g_yx_masked, origin='lower', extent=extent, cmap='RdBu_r', interpolation='none')
    ax[1, 0].set_title(rf"$g^{{({band_label})}}_{{yx}}$")
    ax[1, 0].set_xlabel(r"$k_x / k_\theta$")
    ax[1, 0].set_ylabel(r"$k_y / k_\theta$")
    plot_hexagonal_bz(sys, ax[1, 0], lw=lw)
    plt.colorbar(im10, ax=ax[1, 0], fraction=0.046, pad=0.04)

    im11 = ax[1, 1].imshow(g_yy_masked, origin='lower', extent=extent, cmap='RdBu_r', interpolation='none')
    ax[1, 1].set_title(rf"$g^{{({band_label})}}_{{yy}}$")
    ax[1, 1].set_xlabel(r"$k_x / k_\theta$")
    ax[1, 1].set_ylabel(r"$k_y / k_\theta$")
    plot_hexagonal_bz(sys, ax[1, 1], lw=lw)
    plt.colorbar(im11, ax=ax[1, 1], fraction=0.046, pad=0.04)

def plot_trace_metric(sys, ax, KX, KY, mask_bz=True):
    k_points = np.column_stack([KX.ravel(), KY.ravel()])
    g_xx, _, g_yy = sys.quantum_metric(k_points)
    g_xx = g_xx.reshape(KX.shape)
    g_yy = g_yy.reshape(KX.shape)

    if mask_bz:
        dk = KX[0, 1] - KX[0, 0]
        mask = sys.BZ_path.contains_points(k_points, radius=dk)             # Test whether points are inside
        mask = mask.reshape(KX.shape)                                        # Reshape back to grid shape
    else:
        mask = np.ones_like(g_xx, dtype=bool)

    g_xx_masked = np.where(mask, g_xx, 0.0)
    g_yy_masked = np.where(mask, g_yy, 0.0)

    trace = g_xx_masked + g_yy_masked
    extent = [KX.min() / sys.ktheta, KX.max() / sys.ktheta, KY.min() / sys.ktheta, KY.max() / sys.ktheta]

    lw = 3.0

    im00 = ax.imshow(trace, origin='lower', extent=extent, cmap='Reds')
    ax.set_title(r"$\text{Tr}(g(\mathbf{k}))$")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    plot_hexagonal_bz(sys, ax, lw=lw)
    plt.colorbar(im00, ax=ax, fraction=0.046, pad=0.04)

def plot_trace_metric_band(self, ax, KX, KY, band, mask_bz=True):
    k_points = np.column_stack([KX.ravel(), KY.ravel()])
    g_xx, _, g_yy = self.quantum_metric(k_points)
    g_xx = g_xx[:, band].reshape(KX.shape)
    g_yy = g_yy[:, band].reshape(KX.shape)

    if mask_bz:
        dk = KX[0, 1] - KX[0, 0]
        mask = self.BZ_path.contains_points(k_points, radius=dk)             # Test whether points are inside
        mask = mask.reshape(KX.shape)                                        # Reshape back to grid shape
    else:
        mask = np.ones_like(g_xx, dtype=bool)

    g_xx_masked = np.where(mask, g_xx, 0.0)
    g_yy_masked = np.where(mask, g_yy, 0.0)

    trace = g_xx_masked + g_yy_masked
    extent = [KX.min(), KX.max(), KY.min(), KY.max()]

    lw = 0.6

    if band == 0:
        band_label = "A"
    elif band == 1:
        band_label = "B"

    im00 = ax.imshow(trace, origin='lower', extent=extent, cmap='Reds')
    ax.set_title(rf"$\text{{Tr}}(g^{{({band_label})}}(\mathbf{{k}}))$")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    self.plot_hexagonal_bz(ax, lw=lw)
    plt.colorbar(im00, ax=ax, fraction=0.046, pad=0.04)

def plot_det_metric(self, ax, KX, KY, mask_bz=True):
    k_points = np.column_stack([KX.ravel(), KY.ravel()])
    g_xx, g_xy, g_yy = self.quantum_metric(k_points)
    g_xx = g_xx.reshape(KX.shape)
    g_xy = g_xy.reshape(KX.shape)
    g_yy = g_yy.reshape(KX.shape)

    if mask_bz:
        dk = KX[0, 1] - KX[0, 0]
        mask = self.BZ_path.contains_points(k_points, radius=dk)             # Test whether points are inside
        mask = mask.reshape(KX.shape)                                        # Reshape back to grid shape
    else:
        mask = np.ones_like(g_xx, dtype=bool)

    g_xx_masked = np.where(mask, g_xx, 0.0)
    g_xy_masked = np.where(mask, g_xy, 0.0)
    g_yy_masked = np.where(mask, g_yy, 0.0)

    det = g_xx_masked * g_yy_masked - g_xy_masked * g_xy_masked
    extent = [KX.min(), KX.max(), KY.min(), KY.max()]

    lw = 0.6

    im = ax.imshow(2 * np.sqrt(det), origin='lower', extent=extent, cmap='Reds')
    ax.set_title(r"$2\sqrt{\text{det}(g(\mathbf{k}))}$")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    self.plot_hexagonal_bz(ax, lw=lw)
    plt.colorbar(im, ax=ax)

def plot_trace_det_metric(self, ax, KX, KY, mask_bz=True):
    k_points = np.column_stack([KX.ravel(), KY.ravel()])
    g_xx, g_xy, g_yy = self.quantum_metric(k_points)
    g_xx = g_xx.reshape(KX.shape)
    g_xy = g_xy.reshape(KX.shape)
    g_yy = g_yy.reshape(KX.shape)

    if mask_bz:
        dk = KX[0, 1] - KX[0, 0]
        mask = self.BZ_path.contains_points(k_points, radius=dk)             # Test whether points are inside
        mask = mask.reshape(KX.shape)                                        # Reshape back to grid shape
    else:
        mask = np.ones_like(g_xx, dtype=bool)

    g_xx_masked = np.where(mask, g_xx, 0.0)
    g_xy_masked = np.where(mask, g_xy, 0.0)
    g_yy_masked = np.where(mask, g_yy, 0.0)

    trace = g_xx_masked + g_yy_masked
    det = g_xx_masked * g_yy_masked - g_xy_masked * g_xy_masked
    extent = [KX.min(), KX.max(), KY.min(), KY.max()]

    lw = 0.6

    im0 = ax[0].imshow(trace, origin='lower', extent=extent, cmap='Reds')
    ax[0].set_title(r"$\text{Tr}(g(\mathbf{k}))$")
    ax[0].set_xlabel(r"$k_x$")
    ax[0].set_ylabel(r"$k_y$")
    self.plot_hexagonal_bz(ax[0], lw=lw)
    plt.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(2 * np.sqrt(det), origin='lower', extent=extent, cmap='Reds')
    ax[1].set_title(r"$2\sqrt{\text{det}(g(\mathbf{k}))}$")
    ax[1].set_xlabel(r"$k_x$")
    ax[1].set_ylabel(r"$k_y$")
    self.plot_hexagonal_bz(ax[1], lw=lw)
    plt.colorbar(im1, ax=ax[1])

