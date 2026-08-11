import numpy as np
import matplotlib.pyplot as plt

def wavefunction_spread_schematic(x0=0.0, sigma=1.0, k=6.0):
    x = np.linspace(-5, 5, 1000)

    envelope = np.exp(-(x - x0)**2 / (2 * sigma**2))      # |psi|^2 (unnormalized)
    psi = envelope * np.cos(k * (x - x0))                  # oscillating amplitude

    fig, ax = plt.subplots(figsize=(5, 3.2))

    # Probability envelope
    ax.fill_between(x, envelope, color='0.5', alpha=0.12, zorder=1)
    ax.plot(x, envelope, color='0.25', lw=1.4, zorder=3,
            label=r'$|\psi(x)|^2$')

    # Oscillating amplitude
    # ax.plot(x, psi, color='0.55', lw=1.0, zorder=2,
    #         label=r'$\psi(x)$')

    # Center line
    # ax.axvline(x0, color='0.7', lw=0.8, ls=(0, (4, 4)), zorder=0)
    # ax.text(x0, -0.06, r'$x_0$', ha='center', va='top', fontsize=10)

    # Spread marker at half height
    h = 0.5
    ax.annotate('', xy=(x0 + sigma, h), xytext=(x0 - sigma, h),
                arrowprops=dict(arrowstyle='<->', color='0.3', lw=0.9))
    ax.text(x0, h + 0.04, r'$\sim\mathcal{L}$', ha='center', va='bottom',
            fontsize=10, color='0.3')

    ax.set_ylim(-1.05, 1.15)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines[['top', 'bottom', 'right', 'left']].set_visible(False)

    fig.tight_layout()
    return fig, ax


if __name__ == '__main__':
    fig, ax = wavefunction_spread_schematic(sigma=1.0, k=6.0)
    fig.savefig('wavefunction_spread.pdf', bbox_inches='tight')
    plt.show()