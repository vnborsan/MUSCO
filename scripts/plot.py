import numpy as np
import matplotlib.pyplot as plt
import filter_df

def plot_filters(summary_df, COLORS, HATCHES, group1, group2):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.5)

    bar_width = 0.4
    label_fontsize = 8
    title_fontsize = 10
    label_offset = 3

    # --- Top panel ---
    df1 = summary_df.loc[group1]
    x1 = np.arange(len(df1))
    for i, corpus in enumerate(df1.columns):
        bars = ax1.bar(
            x1 + i*bar_width, df1[corpus], width=bar_width,
            label=corpus, color=COLORS[i], edgecolor='black', linewidth=0.5, hatch=HATCHES[i]
        )
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f'{h:.0f}%', (bar.get_x()+bar.get_width()/2, h),
                         xytext=(0, label_offset), textcoords="offset points",
                         ha='center', va='bottom', fontsize=label_fontsize,
                         bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    ax1.set_title('A. Melody Filters', fontsize=title_fontsize, pad=10, loc='left')
    ax1.set_xticks(x1 + bar_width/2); ax1.set_xticklabels(df1.index, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Percentage (%)', fontsize=9)
    ax1.grid(axis='y', linestyle=':', alpha=0.5)
    ax1.set_ylim(0, max(df1.max().max() * 1.15, 100))
    ax1.legend(title='Corpus', fontsize=8, title_fontsize=9, loc='upper left', frameon=True, framealpha=0.9)

    # --- Bottom panel ---
    df2 = summary_df.loc[group2]
    x2 = np.arange(len(df2))
    for i, corpus in enumerate(df2.columns):
        bars = ax2.bar(
            x2 + i*bar_width, df2[corpus], width=bar_width,
            color=COLORS[i], edgecolor='black', linewidth=0.5, hatch=HATCHES[i]
        )
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f'{h:.0f}%', (bar.get_x()+bar.get_width()/2, h),
                         xytext=(0, label_offset), textcoords="offset points",
                         ha='center', va='bottom', fontsize=label_fontsize,
                         bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    ax2.set_title('B. Rhythm + Mixed Filters', fontsize=title_fontsize, pad=10, loc='left')
    ax2.set_xticks(x2 + bar_width/2); ax2.set_xticklabels(df2.index, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Percentage (%)', fontsize=9)
    ax2.grid(axis='y', linestyle=':', alpha=0.5)
    ax2.set_ylim(0, max(df2.max().max() * 1.15, 100))

    for ax in (ax1, ax2):
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color('#808080'); spine.set_linewidth(0.5)

    plt.savefig('filter_analysis_labels_above.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('filter_analysis_labels_above.png', bbox_inches='tight', dpi=600)
    plt.show()