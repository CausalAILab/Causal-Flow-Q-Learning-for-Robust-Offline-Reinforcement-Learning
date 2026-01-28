import os
import json
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

from matplotlib.ticker import FormatStrFormatter
from decimal import Decimal

orig_data = pd.DataFrame({
    'Task': ['cube-double', 'scene-play', 'puzzle-3x3', 'puzzle-4x4'],
    'State-based Learner': [29, 56, 30, 17],
    'Pixel-based Learner':  [6, 41, 20, 10],
    'CFQL': [11, 43, 26, 13]
})

data = pd.DataFrame({
    'Task': ['cube-double', 'scene-play', 'puzzle-3x3', 'puzzle-4x4'],
    'Pixel-based Learner':  np.array([6, 41, 20, 10])/np.array([29, 56, 30, 17])*100,
    'State-based Learner': [100, 100, 100, 100],
})

fe = fm.FontEntry(
    fname=os.path.expanduser('~/prima_serif_roman_bt.ttf'),
    name='primaserif')
fm.fontManager.ttflist.insert(0, fe) # or append is fine
matplotlib.rcParams['font.family'] = fe.name # = 'your custom ttf font name'
palette = sns.color_palette('Set3')

COLORS = [palette[0], palette[2], palette[5], palette[6], palette[8], palette[4], palette[9], palette[7]]

sns.reset_defaults()
fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.barplot(data=data, x='Task', y='State-based Learner', legend=False, linewidth=2, palette=COLORS[:4], alpha=0.3, hue='Task')
ax = sns.barplot(data=data, x='Task', y='Pixel-based Learner', legend=False, linewidth=2, palette=COLORS[:4], hue='Task')
# Add values on top of each bar
for i, label in enumerate((100 - data['Pixel-based Learner'])):
    ax.text(i, data['Pixel-based Learner'][i] + label/2 - 3, f'-{label:.1f}%',
            ha='center', va='bottom', fontsize=14, fontname='primaserif', color=palette[3])

ax.set_xticks([0, 1, 2, 3], labels=['cube-double', 'scene-play', 'puzzle-3x3', 'puzzle-4x4'], fontsize=12, fontname='primaserif')
ax.set_yticks([0, 25, 50, 75, 100], labels=["0", "0.25", "0.5", "0.75", "1.0"], fontsize=12, fontname='primaserif')
ax.set_xlabel('Task', fontsize=13, labelpad=0, fontname='primaserif')
ax.set_ylabel('Normalized Success Rate', fontsize=13, labelpad=0, fontname='primaserif')
# plt.setp(ax.get_legend().get_texts(), fontsize='10', fontname='primaserif')

sns.despine(offset=5)
plt.xticks(fontname='primaserif')
plt.yticks(fontname='primaserif')
sns.set_theme(font_scale=8)
plt.tight_layout()
# plt.show()
fig = plt.gcf()
fig.set_size_inches(8, 6)
imgPath = f'single_bar_comparison.png'
fig.savefig(imgPath, dpi=800, bbox_inches='tight', pad_inches=0)
plt.close()
