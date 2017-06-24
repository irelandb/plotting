#!/usr/bin/env python
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pdb
from helper_functions import Node, parse_dataframe, get_nodes, gelx, gely, mm2inch, width_mm2fig, height_mm2fig
from labeler import Labeler
sns.set(color_codes=True)

# NEED THIS FOR HEATMAPS ANNOTATION TO WORK
plt.ion()

rnap_real_raw = pd.read_csv('comparisons/compare_sortseq_RNAP_submission.txt',delim_whitespace=True)
#rnap_real_raw = pd.read_csv('comparisons/old/compare_sortseq_RNAP.txt',delim_whitespace=True)
rnap_real = rnap_real_raw.copy()

# Clean annotations
names = rnap_real['Training/Test']
rnap_real = rnap_real.rename(columns={'Training/Test':'training'})

rnap_real['fitting'] = [name.split('_')[-1] for name in names] 
rnap_real['training'] = [name.split('_')[0] for name in names]
rnap_real['model'] = ['nbr' if 'eighbor' in name else 'mat' for name in names]
rnap_real['fitting'] = ['IM' if ('eighbor' in x or 'MI' in x) else x for x in rnap_real['fitting']]
rnap_real['fitting'] = ['ER' if x=='BVH' else x for x in rnap_real['fitting']]
rnap_real['fitting'] = ['DT' if x=='DT' else x for x in rnap_real['fitting']]

# Reorder test datasets
rnap_real['label'] = '' #'RNAP model'
rnap_real = rnap_real[['label','training','model','fitting','rnap-wt','full-wt','full-500','full-150','full-0']]

#pdb.set_trace()

# Reorder list of models
indices = [(not 'iterLS' in name) for name in rnap_real['fitting']]
rnap_real = rnap_real[indices]
rnap_real = rnap_real.reset_index(drop=True)

indices = [
    20,21,22,23,24,
    0,1,2,3,4,
    15,16,17,18,19,
    10,11,12,13,14,
    5,6,7,8,9
]
rnap_real = rnap_real.iloc[indices,:]
rnap_real = rnap_real.reset_index(drop=True)

# Normalize information values across test data sets
block = rnap_real.iloc[:,4:]
rnap_real.iloc[:,4:] = (100.001*block/block.max()).astype(int)

# Separate annotations and data
rnap_real_data = rnap_real.iloc[:,4:]
rnap_real_annotation = rnap_real.iloc[:,:4]

rnap_real_ylabel = rnap_real_data.transpose().reset_index()
rnap_real_ylabel['label'] = '' #dataset'
rnap_real_ylabel = rnap_real_ylabel[['label','index']]

# # Draw heatmap
# plt.figure(figsize=(16,6))
# sns.set(font_scale=1)
# ax = sns.heatmap(rnap_real_data.transpose(), annot=True, fmt="d",annot_kws={"size": 12})

# # Draw x- and y- annotations
# gelx(ax,rnap_real_annotation,annotation_spacing=0.3,fontsize=12)
# gely(ax,rnap_real_ylabel,annotation_spacing=1.0,fontsize=12)

# # Draw white lines 
# (num_cols,num_rows) = rnap_real_data.shape
# for y in range(num_rows):
#     plt.plot([0,num_cols],[y,y],color='white',linewidth=2)
# for x in range(0,num_cols,5):
#     plt.plot([x,x],[0,num_rows],color='white',linewidth=2)

# Load data
crp_real_raw = pd.read_csv('comparisons/compare_sortseq_CRP_submission.txt',delim_whitespace=True)
#crp_real_raw = pd.read_csv('comparisons/old/compare_sortseq_CRP.txt',delim_whitespace=True)
crp_real = crp_real_raw.copy()

# Remove full-0 data
indices = [not 'full-0' in x for x in crp_real['Training/Test']]
crp_real = crp_real.loc[indices,:]
crp_real

# Clean annotations
names = crp_real['Training/Test']
crp = crp_real.rename(columns={'Training/Test':'training'})
crp_real['fitting'] = [name.split('_')[-1] for name in names] 
crp_real['training'] = [name.split('_')[0] for name in names]
crp_real['model'] = ['nbr' if 'eighbor' in name else 'mat' for name in names]
crp_real['fitting'] = ['IM' if ('eighbor' in x or 'MI' in x) else x for x in crp_real['fitting']]
crp_real['fitting'] = ['ER' if x=='BVH' else x for x in crp_real['fitting']]
crp_real['fitting'] = ['DT' if x=='DT' else x for x in crp_real['fitting']]

# Reorder columns
crp_real['label'] = ''#CRP model'
crp_real = crp_real[['label','training','model','fitting','crp-wt','full-wt','full-500','full-150']]
crp_real = crp_real.reset_index(drop=True)

# Reorder rows
indices = [
    5,6,7,8,9,
    0,1,2,3,4,
    15,16,17,18,19,
    10,11,12,13,14
]
crp_real = crp_real.loc[indices,:]

# Normalize information values across test data sets
block = crp_real.iloc[:,4:]
crp_real.iloc[:,4:] = (100.001*block/block.max()).astype(int)

# Separate annotations and data
crp_real_data = crp_real.iloc[:,4:]
crp_real_annotation = crp_real.iloc[:,:4]

crp_real_ylabel = crp_real_data.transpose().reset_index()
crp_real_ylabel['label'] = '' #dataset'
crp_real_ylabel = crp_real_ylabel[['label','index']]

# # Draw heatmap
# plt.figure(figsize=(16,6))
# sns.set(font_scale=1)
# ax = sns.heatmap(crp_real_data.transpose(), annot=True, fmt="d",annot_kws={"size": 12})

# # Draw x- and y- annotations
# gelx(ax,crp_real_annotation,annotation_spacing=0.3,fontsize=12)
# gely(ax,crp_real_ylabel,annotation_spacing=1.0,fontsize=12)

# # Draw white lines 
# (num_cols,num_rows) = crp_real_data.shape
# for y in range(num_rows):
#     plt.plot([0,num_cols],[y,y],color='white',linewidth=2)
# for x in range(0,num_cols,5):
#     plt.plot([x,x],[0,num_rows],color='white',linewidth=2)

# # Create figure
# plt.figure(figsize=(mm2inch(150),mm2inch(50)))

# Isolate appropriate rows
mat_im_indices = (rnap_real['model'] == 'mat') & (rnap_real['fitting'] == 'IM')
mat_ls_indices = (rnap_real['model'] == 'mat') & (rnap_real['fitting'] == 'LS')
mat_er_indices = (rnap_real['model'] == 'mat') & (rnap_real['fitting'] == 'DT')
nbr_im_indices = (rnap_real['model'] == 'nbr') & (rnap_real['fitting'] == 'IM')

# Create off-diagonal indices
M = sum(mat_im_indices)
nondiag = (1-np.identity(M)).astype(bool)

# Extract non-diagonal values
rnap_vals_im = np.array(rnap_real[mat_im_indices].iloc[:,-M:]).astype(float)[nondiag]
rnap_vals_ls = np.array(rnap_real[mat_ls_indices].iloc[:,-M:]).astype(float)[nondiag]
rnap_vals_er = np.array(rnap_real[mat_er_indices].iloc[:,-M:]).astype(float)[nondiag]
rnap_vals_mat = rnap_vals_im
rnap_vals_nbr = np.array(rnap_real[nbr_im_indices].iloc[:,-M:]).astype(float)[nondiag]

# Compute mean and std of info ratios: im v. er
rnap_ratios_im_er = rnap_vals_im / rnap_vals_er
ratio_mean = np.mean(rnap_ratios_im_er)
ratio_std = np.std(rnap_ratios_im_er)
print 'Off-diagonal ratio of predictive info for im vs. er rnap matrics = %f +- %f'%(ratio_mean,ratio_std)

# # Histogram info ratios
# plt.subplot(121)
# ax = sns.distplot(rnap_ratios_im_er,bins=10,kde=False,hist_kws={'linewidth':0})
# plt.plot([1,1],ax.get_ylim(),'--k')
# plt.xlabel('Ratio of predictive info for im vs. er rnap matrics', fontsize=12)
# plt.title('PI_im / PI_er for RNAP models on non-training data')
# ax.set_xlim([0.8,1.2])

# Perform a nonparametric sign test
n_success = sum(rnap_ratios_im_er > 1)
n_trials = len(rnap_ratios_im_er)
nst_pval = stats.binom_test(n_success,n_trials)
print 'Nonparametric sign test for im > er: P = %f'%nst_pval

# Compute mean and std of info ratios
rnap_ratios_nbr_mat = rnap_vals_nbr / rnap_vals_mat
ratio_mean = np.mean(rnap_ratios_nbr_mat)
ratio_std = np.std(rnap_ratios_nbr_mat)
print 'Off-diagonal ratio of predictive info for nbr vs. mat rnap models = %f +- %f'%(ratio_mean,ratio_std)

# Perform a nonparametric sign test
n_success = sum(rnap_ratios_nbr_mat  > 1)
n_trials = len(rnap_ratios_nbr_mat )
nst_pval = stats.binom_test(n_success,n_trials)
print 'Nonparametric sign test for nbr > mat: P = %f'%nst_pval

# Isolate appropriate rows
mat_im_indices = (crp_real['model'] == 'mat') & (crp_real['fitting'] == 'IM')
mat_ls_indices = (crp_real['model'] == 'mat') & (crp_real['fitting'] == 'LS')
mat_er_indices = (crp_real['model'] == 'mat') & (crp_real['fitting'] == 'DT')
nbr_im_indices = (crp_real['model'] == 'nbr') & (crp_real['fitting'] == 'IM')

# Create off-diagonal indices
M = sum(mat_im_indices)
nondiag = (1-np.identity(M)).astype(bool)

# Extract non-diagonal values
crp_vals_im = np.array(crp_real[mat_im_indices].iloc[:,-M:]).astype(float)[nondiag]
crp_vals_ls = np.array(crp_real[mat_ls_indices].iloc[:,-M:]).astype(float)[nondiag]
crp_vals_er = np.array(crp_real[mat_er_indices].iloc[:,-M:]).astype(float)[nondiag]
crp_vals_mat = crp_vals_im
crp_vals_nbr = np.array(crp_real[nbr_im_indices].iloc[:,-M:]).astype(float)[nondiag]

# Compute mean and std of info ratios: im v. er
crp_ratios_im_er = crp_vals_im / crp_vals_er
ratio_mean = np.mean(crp_ratios_im_er)
ratio_std = np.std(crp_ratios_im_er)
print 'Off-diagonal ratio of predictive info for im vs. er rnap matrics = %f +- %f'%(ratio_mean,ratio_std)

# Perform a nonparametric sign test
n_success = sum(crp_ratios_im_er > 1)
n_trials = len(crp_ratios_im_er)
nst_pval = stats.binom_test(n_success,n_trials)
print 'Nonparametric sign test for im > er: P = %f'%nst_pval

# Compute mean and std of info ratios
crp_ratios_nbr_mat = crp_vals_nbr / crp_vals_mat
ratio_mean = np.mean(crp_ratios_nbr_mat)
ratio_std = np.std(crp_ratios_nbr_mat)
print 'Off-diagonal ratio of predictive info for nbr vs. mat CRP models = %f +- %f'%(ratio_mean,ratio_std)

# Perform a nonparametric sign test
n_success = sum(crp_ratios_nbr_mat  > 1)
n_trials = len(crp_ratios_nbr_mat )
nst_pval = stats.binom_test(n_success,n_trials)
print 'Nonparametric sign test for nbr > mat: P = %f'%nst_pval

#
# Make figure
#

width=160
height=175
bottom=5
fig = plt.figure(figsize=(mm2inch(width),mm2inch(height+bottom)))
sns.set(font_scale=0.8)

# Make a labler to add labels to subplots
labeler = Labeler(xpad=.07,ypad=0.02,fontsize=10)

left = width_mm2fig(15,fig)
stat_left = left
middle = width_mm2fig(70,fig)
right = width_mm2fig(125,fig)

level1 = height_mm2fig(140+bottom,fig)
level2 = height_mm2fig(105+bottom,fig)
level3 = height_mm2fig(60+bottom,fig)
level4 = height_mm2fig(10+bottom,fig)

hm_width = width_mm2fig(160,fig)
hm_height = height_mm2fig(20,fig)
stat_width = width_mm2fig(30,fig)
stat_height = height_mm2fig(30,fig)

labelsize = 8
panelsize = 12

# Set colormaps
cmap = sns.cubehelix_palette(8, start=0.5, rot=0.0, reverse=True, as_cmap=True) 
vmax = 100
vmin = 75

sns.set_style('white')

## RNAP heatmap

# Plot results for real RNAP data
ax = fig.add_axes([left, level1, hm_width, hm_height])
labeler.label_subplot(ax,'A',ypad_adjust=0.02)
sns.heatmap(
    rnap_real_data.transpose(), annot=True, fmt="d", vmin=vmin, vmax=vmax, 
    annot_kws={"size": 7}, cmap=cmap, cbar_kws={"pad":.01})
gelx(ax,rnap_real_annotation,annotation_spacing=0.8,fontsize=labelsize)
gely(ax,rnap_real_ylabel,annotation_spacing=0.5,\
    fontsize=labelsize,rotation=0,ha='right')

# Draw white lines 
(num_cols,num_rows) = rnap_real_data.shape
for y in range(num_rows):
    plt.plot([0,num_cols],[y,y],color='white',linewidth=2)
for x in range(0,num_cols,5):
    plt.plot([x,x],[0,num_rows],color='white',linewidth=2)

## RNAP summary statistics

lims = [50,100]

ax = fig.add_axes([stat_left, level3, stat_width, stat_height])
labeler.label_subplot(ax,'C')
ax.set_xlim(lims)
ax.set_ylim(lims)
sns.regplot(rnap_vals_er,rnap_vals_im,ci=95,ax=ax,fit_reg=False)
lims = ax.get_xlim()
plt.plot(lims,lims,':k',linewidth=1)
#plt.xticks([60, 70, 80, 90, 100])
#plt.yticks([60, 70, 80, 90, 100])
plt.xlabel('$I_\mathrm{mat,DT}$',fontsize=labelsize)
plt.ylabel('$I_\mathrm{mat,IM}$',fontsize=labelsize)
plt.title('RNAP mat: IM vs. DT',fontsize=labelsize)
n_success = sum(rnap_vals_im > rnap_vals_er)
n_trials = len(rnap_vals_im)
nst_pval = stats.binom_test(n_success,n_trials)
plt.text(70,55,'P = %1.1E'%nst_pval, fontsize=labelsize)


ax = fig.add_axes([middle, level3, stat_width, stat_height])
labeler.label_subplot(ax,'E')
ax.set_xlim(lims)
ax.set_ylim(lims)
sns.regplot(rnap_vals_ls,rnap_vals_im,ci=95,ax=ax,fit_reg=False)
lims = ax.get_xlim()
plt.plot(lims,lims,':k',linewidth=1)
#plt.xticks([60, 70, 80, 90, 100])
#plt.yticks([60, 70, 80, 90, 100])
plt.xlabel('$I_\mathrm{mat,LS}$',fontsize=labelsize)
plt.ylabel('$I_\mathrm{mat,IM}$',fontsize=labelsize)
plt.title('RNAP mat: IM vs. LS',fontsize=labelsize)
n_success = sum(rnap_vals_im > rnap_vals_ls)
n_trials = len(rnap_vals_im)
nst_pval = stats.binom_test(n_success,n_trials)
plt.text(70,55,'P = %1.1E'%nst_pval, fontsize=labelsize)


ax = fig.add_axes([right, level3, stat_width, stat_height])
labeler.label_subplot(ax,'G')
ax.set_xlim(lims)
ax.set_ylim(lims)
sns.regplot(rnap_vals_mat,rnap_vals_nbr,ci=95,ax=ax,line_kws={'linewidth':1})
lims = ax.get_xlim()
plt.plot(lims,lims,':k',linewidth=1)
#plt.xticks([60, 70, 80, 90, 100])
#plt.yticks([60, 70, 80, 90, 100])
plt.xlabel('$I_\mathrm{mat,IM}$',fontsize=labelsize)
plt.ylabel('$I_\mathrm{nbr,IM}$',fontsize=labelsize)
plt.title('RNAP IM: nbr vs. mat',fontsize=labelsize)
n_success = sum(rnap_vals_nbr > rnap_vals_mat)
n_trials = len(rnap_vals_nbr)
nst_pval = stats.binom_test(n_success,n_trials)
plt.text(55,90,'P = %1.1E'%nst_pval, fontsize=labelsize)
    
## CRP heatmap
    
# Plot results for real CRP data
ax = fig.add_axes([left, level2, hm_width, hm_height])
labeler.label_subplot(ax,'B',ypad_adjust=0.02)
sns.heatmap(
    crp_real_data.transpose(), annot=True, fmt="d", vmin=vmin, vmax=vmax, 
    annot_kws={"size": 7}, cmap=cmap,cbar_kws={"pad":.01})
gelx(ax,crp_real_annotation,annotation_spacing=0.70,fontsize=labelsize)
gely(ax,crp_real_ylabel,annotation_spacing=0.5,\
    fontsize=labelsize,rotation=0,ha='right')

# Draw white lines 
(num_cols,num_rows) = rnap_real_data.shape
for y in range(num_rows):
    plt.plot([0,num_cols],[y,y],color='white',linewidth=2)
for x in [5,10,15,20]:
    plt.plot([x,x],[0,num_rows],color='white',linewidth=2)
    
## CRP Summary statistics

lims = [50,100]

ax = fig.add_axes([stat_left, level4, stat_width, stat_height])
labeler.label_subplot(ax,'D')
ax.set_xlim(lims)
ax.set_ylim(lims)
sns.regplot(crp_vals_er,crp_vals_im,ci=95,ax=ax,fit_reg=False)
lims = ax.get_xlim()
plt.plot(lims,lims,':k',linewidth=1)
#plt.xticks([60, 70, 80, 90, 100])
#plt.yticks([60, 70, 80, 90, 100])
plt.xlabel('$I_\mathrm{mat,DT}$',fontsize=labelsize)
plt.ylabel('$I_\mathrm{mat,IM}$',fontsize=labelsize)
plt.title('CRP mat: IM vs. DT',fontsize=labelsize)
n_success = sum(crp_vals_im > crp_vals_er)
n_trials = len(crp_vals_im)
nst_pval = stats.binom_test(n_success,n_trials)
plt.text(70,55,'P = %1.1E'%nst_pval, fontsize=labelsize)

ax = fig.add_axes([middle, level4, stat_width, stat_height])
labeler.label_subplot(ax,'F')
ax.set_xlim(lims)
ax.set_ylim(lims)
sns.regplot(crp_vals_ls,crp_vals_im,ci=95,ax=ax,fit_reg=False)
lims = ax.get_xlim()
plt.plot(lims,lims,':k',linewidth=1)
#plt.xticks([60, 70, 80, 90, 100])
#plt.yticks([60, 70, 80, 90, 100])
plt.xlabel('$I_\mathrm{mat,DT}$',fontsize=labelsize)
plt.ylabel('$I_\mathrm{mat,IM}$',fontsize=labelsize)
plt.title('CRP mat: IM vs. LS',fontsize=labelsize)
n_success = sum(crp_vals_im > crp_vals_ls)
n_trials = len(crp_vals_im)
nst_pval = stats.binom_test(n_success,n_trials)
plt.text(70,55,'P = %1.1E'%nst_pval, fontsize=labelsize)


ax = fig.add_axes([right, level4, stat_width, stat_height])
labeler.label_subplot(ax,'H')
ax.set_xlim(lims)
ax.set_ylim(lims)
sns.regplot(crp_vals_mat,crp_vals_nbr,ci=95,ax=ax,line_kws={'linewidth':1})
#plt.xticks([70, 80, 90, 100])
#plt.yticks([70, 80, 90, 100])
lims = ax.get_xlim()
plt.plot(lims,lims,':k',linewidth=1)
plt.xlabel('$I_\mathrm{mat,IM}$',fontsize=labelsize)
plt.ylabel('$I_\mathrm{nbr,IM}$',fontsize=labelsize)
plt.title('CRP IM: nbr vs. mat',fontsize=labelsize)
n_success = sum(crp_vals_nbr > crp_vals_mat)
n_trials = len(crp_vals_nbr)
nst_pval = stats.binom_test(n_success,n_trials)
plt.text(55,90,'P = %1.1E'%nst_pval, fontsize=labelsize)


# Add figure label
plt.figtext(0.5, 0.00 ,'Figure 4', \
    va='bottom',ha='center',fontsize=10)

#plt.subplots_adjust(left=0.1, right=1.0)
plt.savefig('plots/fig_04_sortseq.pdf')
plt.close()
