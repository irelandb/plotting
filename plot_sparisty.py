#!/usr/bin/env python
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy import stats
import pdb
from helper_functions import Node, parse_dataframe, get_nodes, gelx, gely, mm2inch, width_mm2fig, height_mm2fig
from labeler import Labeler
import sortseq.utils as utils
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import scipy



df_model = pd.io.parsers.read_csv('downsample_model_ER_v2',delim_whitespace=True)
del df_model['pos']
#df_model = np.array(df_model)
df_counts = pd.io.parsers.read_csv('rep0_ct',delim_whitespace=True)
del df_counts['pos']
del df_counts['ct']
#fix gauge so that average of ct=0 entries have values equal to 0
#mask out the values where ct=0
weighted_avg = (np.array(df_counts == 0) * np.array(df_model)).sum(axis=1)

entries = np.array(df_counts == 0).sum(axis=1)
weighted_avg = weighted_avg/entries
wt = 'SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK*'

seq_dict,inv_dict = utils.choose_dict('protein')

wt_model = np.transpose(utils.seq2mat(wt,seq_dict))

#wt_row =  (wt_model * df_model).sum(axis=1)

df_model =  df_model.subtract(weighted_avg, axis=0)
test = np.array(df_counts <50) * np.array(df_model)
print test.max()

print df_model.max()
model_raveled = np.ravel(df_model)
wt_ravel = wt_model.ravel()


ct_raveled = np.ravel(df_counts)
print len(model_raveled)
model_raveled = [model_raveled[i] for i in range(len(model_raveled)) if wt_ravel[i] == 0] 
print len(model_raveled)
ct_raveled = [ct_raveled[i] for i in range(len(ct_raveled)) if wt_ravel[i] == 0]
old_ct_raveled = ct_raveled 

fig, ax = plt.subplots()
#plt.scatter(ct_raveled,model_raveled)

ct_raveled = np.log(np.add(ct_raveled,1))
heatmap, xedges, yedges = np.histogram2d(ct_raveled,model_raveled, bins=50)
print xedges
print yedges
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.clf()
#plt.imshow(heatmap.T, extent=extent, origin='lower')
#heatmap = scipy.ndimage.gaussian_filter(heatmap,5)
print np.exp(xedges[0:50]) - 1
#ax.set_xlim([1,10000000])
ax.set_xticks(np.arange(0,50,5)+0.5, minor=False)
ax.set_yticks(np.arange(0,50,5)+0.5, minor=False)
ax.set_xticklabels(np.exp(xedges[0:50]) - 1,minor=False)
ax.set_yticklabels((yedges[0:50]),minor=False)
plt.imshow(heatmap.T, cmap = cm.coolwarm, interpolation='spline16', norm=LogNorm())
'''
ax.set_xlim([-10,10])
#ax.set_xscale('log')
plt.xlabel('Counts')
plt.ylabel('Value of parameter')
plt.title('Sparsity Analysis ER')
plt.show()
'''
plt.savefig(open('test2','w'),format='pdf')
#plt.show()
