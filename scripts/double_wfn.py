import numpy as np
import matplotlib.pyplot as plt

def two_overlapping_wavefunctions(sep=10, sigma=1.0):
    x = np.linspace(-6, 6, 1000)

    g1 = np.exp(-(x + sep/2)**2 / (2 * sigma**2))
    g2 = np.exp(-(x - sep/2)**2 / (2 * sigma**2))
    overlap = np.minimum(g1, g2)

    fig, ax = plt.subplots(figsize=(6, 3.0))

    # Light fill under each curve
    ax.fill_between(x, g1, color='0.5', alpha=0.10, zorder=1)
    ax.fill_between(x, g2, color='0.5', alpha=0.10, zorder=1)

    # Darker shaded overlap region
    ax.fill_between(x, overlap, color='0.35', alpha=0.35, zorder=2)

    # Thin dark outlines
    ax.plot(x, g1, color='0.25', lw=1.4, zorder=3)
    ax.plot(x, g2, color='0.25', lw=1.4, zorder=3)

    # Baseline
    ax.axhline(0, color='0.4', lw=0.8, zorder=0)

    # Separation marker between the two centers
    h = 1.06
    # ax.annotate('', xy=(sep/2, h), xytext=(-sep/2, h),
    #             arrowprops=dict(arrowstyle='<->', color='0.3', lw=0.9))
    # ax.text(0, h + 0.03, r'$\sim \mathcal{L}$', ha='center', va='bottom',
    #         fontsize=11, color='0.3')

    ax.set_ylim(-0.05, 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

    fig.tight_layout()
    return fig, ax


if __name__ == '__main__':
    fig, ax = two_overlapping_wavefunctions(sep=3.0, sigma=1.0)
    fig.savefig('overlap.pdf', bbox_inches='tight')
    plt.show()