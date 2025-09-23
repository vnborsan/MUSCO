import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_filters(summary: pd.DataFrame, COLORS, HATCHES, group1, group2, save_path=None):
    """
    summary: DataFrame with rows in the order you want to display and two columns: 'Ciciban' and 'SLP'
    COLORS:  list of 2 colors [Ciciban, SLP]
    HATCHES: list of 2 hatches [Ciciban, SLP]
    group1:  list of row labels for the TOP panel (melodic)
    group2:  list of row labels for the BOTTOM panel (rhythm + mixed)
    save_path: optional path to save PNG
    """
    # Ensure row order exists in summary (reindex safely)
    top  = summary.reindex(group1)
    bottom = summary.reindex(group2)

    # Figure & axes
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)

    def _barpanel(ax, block, title):
        # indices
        y_pos = np.arange(len(block))
        height = 0.38

        # fetch values, default to 0 if NaN
        c1 = block.get('Ciciban', pd.Series([0]*len(block), index=block.index)).fillna(0).values
        c2 = block.get('SLP', pd.Series([0]*len(block), index=block.index)).fillna(0).values

        # bars (side-by-side for each row)
        ax.barh(y_pos + height/2, c1, height=height, color=COLORS[0], hatch=HATCHES[0], edgecolor='black', label='Ciciban')
        ax.barh(y_pos - height/2, c2, height=height, color=COLORS[1], hatch=HATCHES[1], edgecolor='black', label='SLP')

        # labels
        for i, (v1, v2) in enumerate(zip(c1, c2)):
            if v1 > 0:
                ax.text(v1 + 1, i + height/2, f"{v1:.1f}%", va='center', fontsize=9)
            if v2 > 0:
                ax.text(v2 + 1, i - height/2, f"{v2:.1f}%", va='center', fontsize=9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(block.index)
        ax.set_xlim(0, max(100, np.nanmax([c1, c2]) + 10))
        ax.set_xlabel("Percentage of songs (%)")
        ax.set_title(title, loc='left', fontsize=12, pad=6)
        ax.grid(axis='x', alpha=0.2)
        ax.legend(loc='lower right')

    _barpanel(axes[0], top,    "A) Melodic filters")
    _barpanel(axes[1], bottom, "B) Rhythmic & mixed filters")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()