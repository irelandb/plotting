#!/usr/bin/env python
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from helper_functions import Node, parse_dataframe, get_nodes, gelx, gely, mm2inch, width_mm2fig, height_mm2fig
sns.set(color_codes=True)
import gauge

df_matrix = pd.read_csv('models/full-wt_IM_rnap.txt',delim_whitespace=True)

# Remove position column
del df_matrix['pos']

# Rename columns by dinucleotide
df_matrix.columns = [x.split('_')[-1] for x in df_matrix.columns]

# Put df in canonical gauge
df_matrix.iloc[:,:] = gauge.fix_matrix(np.array(df_matrix), normalize=True)

# sns.heatmap(df_matrix.transpose())
# xticks = np.arange(0,df_matrix.shape[0],5)
# plt.xticks(xticks+.5,xticks);
# plt.yticks(rotation=0);

df_neighbor = pd.read_csv('models/full-wt_IM_NBR_rnap.txt',delim_whitespace=True)

# Remove position column
del df_neighbor['pos']

# Rename columns by dinucleotide
df_neighbor.columns = [x.split('_')[-1] for x in df_neighbor.columns]

# Put df in canonical gauge
df_neighbor.iloc[:,:] = gauge.fix_neighbor(np.array(df_neighbor), normalize=True)

# sns.heatmap(df_neighbor.transpose())
# xticks = np.arange(0,df_neighbor.shape[0],5)
# plt.xticks(xticks+.5,xticks);
# plt.yticks(rotation=0);

width=80
height=130
bottom = 5
fig = plt.figure(figsize=(mm2inch(width),mm2inch(height+bottom)))
sns.set(font_scale=0.8, style='dark')

# Set colormap
cmap = "coolwarm"

# Set positions
left = width_mm2fig(9,fig)
level0 = height_mm2fig(128+bottom,fig)
level1 = height_mm2fig(78+bottom,fig)
level2 = height_mm2fig(0+bottom,fig)

# Set plot sizes
width = width_mm2fig(70,fig)
matrix_height = height_mm2fig(12.5,fig)
neighbor_height = height_mm2fig(65,fig)

# Set font sizes
labelsize = 8
panelsize = 12

## Matrix modl parameters
vmin = -.8
vmax = .8

ax = fig.add_axes([left, level1, width, matrix_height])
sns.heatmap(df_matrix.transpose(),cmap=cmap,cbar=None, vmax=vmax, vmin=vmin)
xticks = np.arange(0,df_matrix.shape[0],5)
plt.xticks(xticks+.5,xticks);
plt.yticks(rotation=0);
plt.xticks(rotation=0);
ax.set_title('RNAP mat')

## Neighbor model parameters
err_dict = {'linewidth':0.5}
ax = fig.add_axes([left, level2, width, neighbor_height])
sns.heatmap(df_neighbor.transpose(),cmap=cmap,cbar_kws={"orientation": "horizontal","pad":.15},vmax=vmax, vmin=vmin)
xticks = np.arange(0,df_neighbor.shape[0],5)
plt.xticks(xticks+.5,xticks);
plt.yticks(rotation=0);
plt.xticks(rotation=0);
ax.set_title('RNAP nbr')


# Add panel labels
plt.figtext(left-.1, level0 ,'A', va='top',ha='left',fontsize=panelsize)
plt.figtext(left-.1, level1+matrix_height+.05 ,'B', va='top',ha='left',fontsize=panelsize)
plt.figtext(left-.1, level2+neighbor_height+.05 ,'C', va='top',ha='left',fontsize=panelsize)

# Add figure label
plt.figtext(0.5, 0.00 ,'Figure 3', \
    va='bottom',ha='center',fontsize=10)

#plt.subplots_adjust(left=0.1, right=1.0)
plt.savefig('plots/fig_03_models.pdf')