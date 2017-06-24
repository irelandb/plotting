#!/usr/bin/env python
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from helper_functions import Node, parse_dataframe, get_nodes, gelx, gely, mm2inch, width_mm2fig, height_mm2fig
import mpathic.gauge as gauge
import pdb
from labeler import Labeler
sns.set(color_codes=True)

# NEED THIS FOR HEATMAPS ANNOTATION TO WORK
plt.ion()

df0 = pd.read_csv('comparisons/compare_sim_RNAP_submission.txt_linfoot',delim_whitespace=True)

# Relable rows
df2 = df0.copy()
names = df0['Training/Test']
df2 = df2.rename(columns={'Training/Test':'training'})

df2['fitting'] = [name.split('_')[-1] for name in names] 
df2['training'] = ['sim-'+name.split('_')[0]+' train' if len(name.split('_')) > 1 else '' for name in names]
df2['model'] = ['nbr' if ('eighbor' in name) else 'mat' for name in names]
df2['fitting'] = ['IM' if ('eighbor' in x or 'MI' in x) else x for x in df2['fitting']]
df2['fitting'] = ['ER' if x=='BVH' else x for x in df2['fitting']]
df2['fitting'] = ['IM' if x=='MCMC' else x for x in df2['fitting']]
df2['fitting'] = ['LS' if 'LS' in x else x for x in df2['fitting']]
df2['fitting'] = ['' if x=='True' else x for x in df2['fitting']]
df2.loc[df2['training']=='',['model']] = 'true'
#df2.loc[df2['training']=='True',['training']] = ''
df2 = df2[df2['fitting'] != 'dms']

#pdb.set_trace()

# Reorder columns and rows
df3 = df2.reset_index()[["training","model","fitting","10_bin_train","10_bin_test","2_bin_train","2_bin_test"]]
df3 = df3.iloc[range(10,-1,-1),:].reset_index(drop=True)

#Normalize values
block = df3.iloc[:,3:]
df3.iloc[:,3:] = ((100.0000001)*block/block.max()).astype(int)

# Create x annotation
df_rnap_xannotation = df3.iloc[:,:3]

# Create y annotation
df_rnap_yannotation = df3.transpose().reset_index().loc[3:,:]
df_rnap_yannotation['type'] = [name.split('_')[-1] for name in df_rnap_yannotation['index']]
df_rnap_yannotation['data'] = ['sim-'+name.split('_')[0] for name in df_rnap_yannotation['index']]
df_rnap_yannotation['label'] = ''
df_rnap_yannotation = df_rnap_yannotation[['label','data','type']]

# Retrieve data
df_rnap_comparison = df3.iloc[:,3:]

# plt.figure(figsize=(6.5,2))

# ax = sns.heatmap(df_rnap_comparison.transpose(),annot=True,fmt="d")
# gelx(ax,df_rnap_xannotation,annotation_spacing=0.6)
# gely(ax,df_rnap_yannotation,annotation_spacing=0.6)

df = pd.read_csv('comparisons/compare_sim_CRP_submission.txt_linfoot',delim_whitespace=True)

# Relable rows
df2 = df.copy()
names = df['Training/Test']
df2 = df2.rename(columns={'Training/Test':'training'})
df2['fitting'] = [name.split('_')[-1] for name in names] 
df2['training'] = ['sim-'+name.split('_')[0] + ' train' if len(name.split('_')) > 1 else '' for name in names]
df2['model'] = ['nbr' if ('eighbor' in name) else 'mat' for name in names]
df2['fitting'] = ['IM' if ('eighbor' in x or 'MI' in x) else x for x in df2['fitting']]
df2['fitting'] = ['ER' if x=='BVH' else x for x in df2['fitting']]
df2['fitting'] = ['IM' if x=='MCMC' else x for x in df2['fitting']]
df2['fitting'] = ['LS' if 'LS' in x else x for x in df2['fitting']]
df2['fitting'] = ['' if x=='True' else x for x in df2['fitting']]
df2.loc[df2['training']=='',['model']] = 'true'
#df2.loc[df2['training']=='True',['training']] = ''
df2 = df2[df2['fitting'] != 'dms']

# Reorder columns and rows
df3 = df2.reset_index()[["training","model","fitting","10_bin_train","10_bin_test","2_bin_train","2_bin_test"]]
df3 = df3.iloc[range(10,-1,-1),:].reset_index(drop=True)

#Normalize values
block = df3.iloc[:,3:]
df3.iloc[:,3:] = ((100.0000001)*block/block.max()).astype(int)

# Create x annotation
df_crp_xannotation = df3.iloc[:,:3]

# Create y annotation
df_crp_yannotation = df3.transpose().reset_index().loc[3:,:]
df_crp_yannotation['type'] = [name.split('_')[-1] for name in df_crp_yannotation['index']]
df_crp_yannotation['data'] = ['sim-'+name.split('_')[0] for name in df_crp_yannotation['index']]
df_crp_yannotation['label'] = ''
df_crp_yannotation = df_crp_yannotation[['label','data','type']]

# Retrieve data
df_crp_comparison = df3.iloc[:,3:]

# plt.figure(figsize=(6.5,2))

# ax = sns.heatmap(df_crp_comparison.transpose(),annot=True,fmt="d")
# gelx(ax,df_crp_xannotation,annotation_spacing=0.6)
# gely(ax,df_crp_yannotation,annotation_spacing=0.6)

# RNAP models to compare
rnap_true_file = 'models/true_model_rnap.txt'
rnap_learned_file = 'models/RNAP_10_bin_IM_NBR.txt'

# CRP models to compare
crp_true_file = 'models/true_model_crp.txt'
crp_learned_file = 'models/10_bin_IM_NBR_crp.txt'

# plt.figure(figsize=(6,3))

# Load models as numpy arrays

# rnap_true
df = pd.read_csv(rnap_true_file,delim_whitespace=True)
del df['pos']
model = np.array(df)
rnap_true_model = gauge.fix_neighbor(model, normalize=True)

# rnap_learned
df = pd.read_csv(rnap_learned_file,delim_whitespace=True)
del df['pos']
model = -np.array(df)
rnap_learned_model = gauge.fix_neighbor(model, normalize=True)

# crp_true
df = pd.read_csv(crp_true_file,delim_whitespace=True)
del df['pos']
model = np.array(df)
crp_true_model = gauge.fix_neighbor(model, normalize=True)

# crp_learned
df = pd.read_csv(crp_learned_file,delim_whitespace=True)
del df['pos']
model = -np.array(df)
crp_learned_model = gauge.fix_neighbor(model, normalize=True)

# # Plot entries against one another
# plt.subplot(121)
# sns.regplot(rnap_true_model.flatten(),rnap_learned_model.flatten())
# plt.xlabel('RNAP True')
# plt.ylabel('RNAP Learned (Matrix)')

# plt.subplot(122)
# sns.regplot(crp_true_model.flatten(),crp_learned_model.flatten())
# plt.xlabel('CRP True')
# plt.ylabel('CRP Learned')

# plt.tight_layout()

width=160
height=140
bottom=5
fig = plt.figure(figsize=(mm2inch(width),mm2inch(height+bottom)))

#fig = plt.figure(figsize=(mm2inch(170),mm2inch(155)))
sns.set(font_scale=0.8)

left = width_mm2fig(20,fig)
stat_left = width_mm2fig(25,fig)
middle = width_mm2fig(100,fig)

level1 = height_mm2fig(105+bottom,fig)
level2 = height_mm2fig(65+bottom,fig)
level3 = height_mm2fig(10+bottom,fig)

hm_width = width_mm2fig(145,fig)
hm_height = height_mm2fig(20,fig)
stat_width = width_mm2fig(40,fig)
stat_height = height_mm2fig(40,fig)

labelsize = 8
panelsize = 12

param_lims = [-1,1]
param_ticks = [-1,-.5,0,.5,1]

# Set colormaps
cmap = sns.cubehelix_palette(8, start=0.0, rot=0.0, reverse=True, as_cmap=True) 
vmax = 100
vmin = 75

sns.set_style('white')

# Make a labler to add labels to subplots
labeler = Labeler(xpad=.07,ypad=0.02,fontsize=10)

## RNAP heatmap

# Plot results for real RNAP data
ax = fig.add_axes([left, level1, hm_width, hm_height])
labeler.label_subplot(ax,'A',xpad_adjust=0.03,ypad_adjust=0.04)
sns.heatmap(
    df_rnap_comparison.transpose(), annot=True, fmt="d", vmin=vmin, vmax=vmax, 
    annot_kws={"size": 7}, cmap=cmap, cbar_kws={"pad":.03})
gelx(ax,df_rnap_xannotation,annotation_spacing=0.8,fontsize=labelsize)
gely(ax,df_rnap_yannotation,annotation_spacing=0.8,fontsize=labelsize,rotation=0)

# Draw white lines 
(num_cols,num_rows) = df_rnap_comparison.shape
for y in range(num_rows):
    plt.plot([0,num_cols],[y,y],color='white',linewidth=2)
for x in [1,6]:
    plt.plot([x,x],[0,num_rows],color='white',linewidth=2)

## CRP heatmap
    
# Plot results for real CRP data
ax = fig.add_axes([left, level2, hm_width, hm_height])
labeler.label_subplot(ax,'B',xpad_adjust=0.03,ypad_adjust=0.04)
sns.heatmap(
    df_crp_comparison.transpose(), annot=True, fmt="d", vmin=vmin, vmax=vmax, 
    annot_kws={"size": 7}, cmap=cmap,cbar_kws={"pad":.03})
gelx(ax,df_crp_xannotation,annotation_spacing=0.8,fontsize=labelsize)
gely(ax,df_crp_yannotation,annotation_spacing=0.8,fontsize=labelsize,rotation=0)
#plt.show()

# Draw white lines 
(num_cols,num_rows) = df_crp_comparison.shape
for y in range(num_rows):
    plt.plot([0,num_cols],[y,y],color='white',linewidth=2)
for x in [1,6]:
    plt.plot([x,x],[0,num_rows],color='white',linewidth=2)
    
## RNAP parameter values

xs = rnap_true_model.flatten()
ys = rnap_learned_model.flatten()
ax = fig.add_axes([stat_left, level3, stat_width, stat_height])
labeler.label_subplot(ax,'C')
R,P = stats.pearsonr(xs,ys)
sns.regplot(rnap_true_model.flatten(),rnap_learned_model.flatten(),fit_reg=False)
plt.xlabel('true',fontsize=labelsize)
plt.ylabel('learned',fontsize=labelsize)
plt.xlim(param_lims)
plt.xticks(param_ticks)
plt.ylim(param_lims)
plt.yticks(param_ticks)
lims = ax.get_xlim()
plt.plot(lims,lims,':k',linewidth=1)
plt.title('RNAP: %d nbr parameters'%len(xs),fontsize=labelsize)
signal = np.var(xs)
noise = np.var(ys - xs)
plt.text(-0.85,0.6,'$S/N=$ %0.1f\n$R^2=$ %0.2f'%(signal/noise,R**2),fontsize=labelsize)

## CRP parameter values

xs = crp_true_model.flatten()
ys = crp_learned_model.flatten()
ax = fig.add_axes([middle, level3, stat_width, stat_height])
labeler.label_subplot(ax,'D')
R,P = stats.pearsonr(xs,ys)
sns.regplot(xs,ys,fit_reg=False)
plt.xlabel('true',fontsize=labelsize)
plt.ylabel('learned',fontsize=labelsize)
plt.xlim(param_lims)
plt.xticks(param_ticks)
plt.ylim(param_lims)
plt.yticks(param_ticks)
lims = ax.get_xlim()
plt.plot(lims,lims,':k',linewidth=1)
plt.title('CRP: %d nbr parameters'%len(xs),fontsize=labelsize)
signal = np.var(xs)
noise = np.var(ys - xs)
plt.text(-0.85,0.6,'$S/N=$ %0.1f\n$R^2=$ %0.2f'%(signal/noise,R**2),fontsize=labelsize)

# Add figure label
plt.figtext(0.5, 0.00 ,'Figure 5', \
    va='bottom',ha='center',fontsize=10)
#plt.show()

#plt.subplots_adjust(left=0.1, right=1.0)
plt.savefig('plots/fig_05_sim_linfoot.pdf')
