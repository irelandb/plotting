#!/usr/bin/env python
import pandas as pd
import numpy as np
import scipy as sp
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from helper_functions import Node, parse_dataframe, get_nodes, gelx, gely, mm2inch, width_mm2fig, height_mm2fig
import MPAthic.gauge as gauge
sns.set(color_codes=True)

# NEED THIS FOR HEATMAPS ANNOTATION TO WORK
plt.ion()

df = pd.read_csv('comparisons/compare_mpra_submission.txt',delim_whitespace=True)
df2 = df.copy()

# Remove unwanted rows
#pdb.set_trace()
df2 = df2.iloc[[1,2,3,4,7,8,9],:]

names = df2['Training/Test']
# Set training column
df2['training'] = df2['Training/Test']
df2['training'] = ['replicate 1' if ('Rep1' in x) else x for x in df2['training']]
df2['training'] = ['replicate 2' if ('Rep2' in x) else x for x in df2['training']]
df2['training'] = ['' if ('QSAM' in x) else x for x in df2['training']]

# Set fitting column
df2['fitting'] = df2['Training/Test']
df2['fitting'] = ['LS' if ('_LS_' in x) else x for x in df2['fitting']]
df2['fitting'] = ['IM' if ('MCMC' in x) else x for x in df2['fitting']]
df2['fitting'] = ['DT' if ('DT' in x) else x for x in df2['fitting']]
df2['fitting'] = ['Pub' if ('QSAM' in x) else x for x in df2['fitting']]

# Rename columns, reorder rows, and select columns
df2 = df2.rename(columns={'MPRA_Train':'replicate 1', 'MPRA_Test':'replicate 2'})
#pdb.set_trace()
df2 = df2.iloc[[3,0,1,2,4,5,6],:]
#df2 = df2.iloc[[2,0,1,3,4,5,6],:]
df2 = df2.reset_index()[['training','fitting','replicate 1','replicate 2']]

# Normalize information values
block = df2.iloc[:,2:]
df2.iloc[:,2:] = ((100.0000001)*block/block.max()).astype(int)

# Extract comparison data
df_mpra_comparison = df2.iloc[:,2:]

# Create x annotation
df_mpra_xannotation = df2.iloc[:,:2]

# Create y annotation
df_mpra_yannotation = df2.iloc[:,2:].transpose().reset_index()[['index']]

# # Plot
# plt.figure(figsize=(6.5,2))
# ax = sns.heatmap(df_mpra_comparison.transpose(),annot=True,fmt="d")
# gelx(ax,df_mpra_xannotation,annotation_spacing=0.4)
# gely(ax,df_mpra_yannotation,annotation_spacing=0.4)

df = pd.read_csv('comparisons/compare_dms_submission.txt',delim_whitespace=True)
df2 = df.copy()

# Set training column
df2['training'] = df2['Training/Test']
df2['training'] = \
	['rounds 0,3' if ('dms1' in x) else x for x in df2['training']]
df2['training'] = \
	['rounds 3,6' if ('dms2' in x) else x for x in df2['training']]

# Delete neighbor rows
df2['fitting'] = df2['Training/Test']
indices = [not 'Neighbor' in x for x in df2['fitting']]
df2 = df2[indices]

# Set fitting column
df2['fitting'] = ['LS' if ('_LS_' in x) else x for x in df2['fitting']]
df2['fitting'] = ['IM' if ('MCMC' in x) else x for x in df2['fitting']]
df2['fitting'] = ['DT' if ('DT' in x) else x for x in df2['fitting']]

# Rename and select columns
df2 = df2.rename(columns={'DMS_Train':'rounds 0,3', 'DMS_Test':'rounds 3,6'})
df2 = df2.reset_index()[['training','fitting','rounds 0,3','rounds 3,6']]

# Normalize information values
block = df2.iloc[:,2:]
df2.iloc[:,2:] = ((100.0000001)*block/block.max()).astype(int)

# Extract comparison data
df_dms_comparison = df2.iloc[:,2:]

# Create x annotation
df_dms_xannotation = df2.iloc[:,:2]

# Create y annotation
df_dms_yannotation = df2.iloc[:,2:].transpose().reset_index()[['index']]

# # Plot
# plt.figure(figsize=(6.5,2))
# ax = sns.heatmap(df_dms_comparison.transpose(),annot=True,fmt="d")
# gelx(ax,df_dms_xannotation,annotation_spacing=0.4)
# gely(ax,df_dms_yannotation,annotation_spacing=0.4)

width=85
height=52
bottom=5
fig = plt.figure(figsize=(mm2inch(width),mm2inch(height+bottom)))
#fig = plt.figure(figsize=(mm2inch(85),mm2inch(50)))
sns.set(font_scale=0.8)

left = width_mm2fig(20,fig)
middle = width_mm2fig(50,fig)

level1 = height_mm2fig(30+bottom,fig)
level2 = height_mm2fig(5+bottom,fig)

hm_width = width_mm2fig(60,fig)
hm_height = height_mm2fig(10,fig)

labelsize = 8
panelsize = 12

param_lims = [-1,1]
param_ticks = [-1,-.5,0,.5,1]

# Set colormaps
cmap = sns.cubehelix_palette(8, start=1.2, rot=0.0, reverse=True, as_cmap=True) 
vmax = 100
vmin = 70

## MPRA heatmap

# Plot results for MPRA data
ax = fig.add_axes([left, level1, hm_width, hm_height])
sns.heatmap(
    df_mpra_comparison.transpose(), annot=True, fmt="d", vmin=vmin, vmax=vmax, 
    annot_kws={"size": 7}, cmap=cmap, cbar_kws={"pad":.05})
gelx(ax,df_mpra_xannotation,annotation_spacing=1.0,fontsize=labelsize)
gely(ax,df_mpra_yannotation,annotation_spacing=0.5,fontsize=labelsize,rotation=0, ha='right')

# Set ticks on colorbar
cax = plt.gcf().axes[-1]
cax.set_yticklabels(['70','','80','','90','','100'],fontsize=labelsize)

# Draw white lines 
(num_cols,num_rows) = df_mpra_comparison.shape
for y in range(num_rows):
    plt.plot([0,num_cols],[y,y],color='white',linewidth=2)
for x in [1,4]:
    plt.plot([x,x],[0,num_rows],color='white',linewidth=2)

## DMS heatmap

# Set colormap
cmap = sns.cubehelix_palette(8, start=2.0, rot=0.0, reverse=True, as_cmap=True) 

    
# Plot results for DMS data
ax = fig.add_axes([left, level2, hm_width, hm_height])
sns.heatmap(
    df_dms_comparison.transpose(), annot=True, fmt="d", vmin=vmin, vmax=vmax, 
    annot_kws={"size": 7}, cmap=cmap,cbar_kws={"pad":.05})
gelx(ax,df_dms_xannotation,annotation_spacing=1.0,fontsize=labelsize)
gely(ax,df_dms_yannotation,annotation_spacing=0.5,fontsize=labelsize,rotation=0, ha='right')

# Set ticks on colorbar
cax = plt.gcf().axes[-1]
cax.set_yticklabels(['70','','80','','90','','100'],fontsize=labelsize)

# Draw white lines 
(num_cols,num_rows) = df_dms_comparison.shape
for y in range(num_rows):
    plt.plot([0,num_cols],[y,y],color='white',linewidth=2)
for x in [3]:
    plt.plot([x,x],[0,num_rows],color='white',linewidth=2)

# Add panel labels
plt.figtext(left-.22, level1+hm_height+.18 ,'A', va='top',ha='left',fontsize=panelsize)
plt.figtext(left-.22, level2+hm_height+.18 ,'B', va='top',ha='left',fontsize=panelsize)

# Add figure label
plt.figtext(0.5, 0.01 ,'Figure 6', \
    va='bottom',ha='center',fontsize=10)

#plt.subplots_adjust(left=0.1, right=1.0)
plt.savefig('plots/fig_06_mpra_dms.pdf')
