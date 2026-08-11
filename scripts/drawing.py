import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from math import factorial, pi

# --- natural units: hbar = m = omega = 1 ---

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


def psi(n, x):
    """Normalized eigenstate of the 1D quantum harmonic oscillator."""
    norm = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(pi))
    Hn = hermite(n)                      # physicists' Hermite polynomial
    return norm * Hn(x) * np.exp(-x**2 / 2)

x = np.linspace(-4, 4, 1000)
V = 0.5 * x**2                           # potential V(x) = 1/2 x^2

n_states = 1
amp = 0.6                                # amplitude scale for visibility
colors = ['#8C0F2A', '#0F8C24', '#8C6C0F', '#8C0F2A']

fig, ax = plt.subplots(figsize=(7, 5))

# potential well
ax.plot(x, V - .5, color='#555555', lw=2.5, label=r'$V(x)=\frac{1}{2}x^2$')

# eigenstates, each offset by its energy E_n = n + 1/2
for n in range(n_states):
    E = n + 0.5
    ax.axhline(E-.5, color='0.7', ls='--', lw=1)                 # energy level
    ax.plot(x, E + amp * psi(n, x) - .5, color=colors[n], lw=4)

for n in range(3):
    E = n + 0.5
    ax.axhline(E-.5, color='0.7', ls='--', lw=1)                 # energy level

# --- mark the Gaussian width sigma on the ground state (n=0) ---
sigma = 1.0                              # psi_0 ~ exp(-x^2 / 2 sigma^2)  ->  sigma = 1
E0 = 0.5
y_sig = E0 + amp * psi(0, sigma)         # curve height at x = ±sigma  (peak * e^-1/2)

# # vertical droplines at x = ±sigma
# for xs_ in (-sigma, sigma):
#     ax.plot([xs_, xs_], [E0-.5, E0 + amp * psi(0, xs_)-1-.5],
#             color=colors[0], ls=':', lw=1)

# double-headed arrow spanning the full width 2*sigma
# ax.annotate('', xy=(-sigma, y_sig-.5), xytext=(sigma, y_sig-.5),
#             arrowprops=dict(arrowstyle='<->', color=colors[0], lw=1.3))
# ax.text(0, y_sig - 0.3 -.5, r'$2\sigma$',
#         color=colors[0], ha='center', va='bottom', fontsize=18)

ax.set_xlim(-4, 4)
ax.set_ylim(-.7, 2)
ax.set_yticks([0, 1, 2])
ax.set_xticks([])
ax.set_ylabel(r'$n$', fontsize=26)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.tight_layout()
plt.savefig("notebooks/plots/harmonic_oscillator.pdf", dpi=300, bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()