import sys
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import imageio
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
import json

'''

Format for files:

one file per preset

filenames:
    {presets separated by '-'}-{generation number}.dat

files must all contain 10 columns each

[ hand x, hand y, stick_1 x, stick_1 y, stick_2 x, stick_2 y, magnet x, magnet y, scratch 1, scratch 2 ]
all are end positions
'''



ARCHIVE_PATH = "./" #"../results/"

new_dir = "./Cmaps_scatter"

labels = ["Hand", "Magnetic Stick", "Velcro Stick", "Magnet", "magnet_2", "magnet_3", "Velcro", "scratch_2", "scratch_3", "cat", "dog", "static_1", "static_2", "static_3", "static_4s"]
columns = ["Hand", "Magnetic Stick", "Velcro Stick", "Magnet", "Velcro"]


# create results directory
try:
    os.mkdir(new_dir)
except OSError as error:
    pass

with open('values.json', 'r') as f:
    values = json.load(f)


LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)
NUM_COLORS = 0
TOTAL_COLORS = 0

pre_list = []
for obj in values:

    fig, ax = plt.subplots()
    sns.reset_orig()

    NUM_COLORS = 0
    TOTAL_COLORS = 0
    for pre in values[obj]:
        values[obj][pre][1], values[obj][pre][0] = zip(*sorted(zip(values[obj][pre][1], values[obj][pre][0])))
        if ( 'uniform' in pre): #values['Magnetic Stick'][pre][0][-1] >    4000):
            if pre not in pre_list:
                pre_list.append(pre)
            NUM_COLORS+=1
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    i = 0
    clrs = sns.color_palette('husl', n_colors=NUM_COLORS)
    for pre in values[obj]:
        if ( 'uniform' in pre): #values['Magnetic Stick'][pre][0][-1] >    4000):
            #if (True):# "batch" not in pre or "SSP" in pre):#"gaussian" not in pre and "random" not in pre):#(values["Magnet"][pre][0][-1] >= 500 or values["Magnetic Stick"][pre][0][-1] >= 2000 or values["Velcro"][pre][0][-1] >= 400 or values["Velcro Stick"][pre][0][-1] >= 3500)):# and ("random" not in pre) and ("gaussian" not in pre)):
            #if ("batch" in pre):
            #    i = 0
            #    if ("curiosity" in pre):
            #        i = 2
            #else:
            #    i = 1
            lines = ax.plot(values[obj][pre][1], values[obj][pre][0], label = pre)
            lines[0].set_color(clrs[i])
            lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
            i+=1
    lg = ax.legend(bbox_to_anchor=(1.05,1.0), loc='upper left')
    plt.title(obj)
    plt.xlabel('Evaluation')
    plt.ylabel('Population')
    plt.savefig(new_dir + "/" + obj + ".png", dpi=None, facecolor='w', edgecolor='w',
                orientation='landscape', papertype=None, format='png',
                transparent=False, bbox_inches="tight", pad_inches=0.1, metadata=None, bbox_extra_artists=(lg,))
    plt.close()

general = {}
for obj in values:
    for pre in values[obj]:
        if pre not in general:
            general[pre] = [[], []]
            general[pre][0] = list(values[obj][pre][0])
            general[pre][1] = list(values[obj][pre][1])
        else: 
            for i in range(len(values[obj][pre][0])):
                general[pre][0][i] += values[obj][pre][0][i]

fig, ax = plt.subplots()
sns.reset_orig()
TOTAL_COLORS=len(pre_list)
clrs = sns.color_palette('husl', n_colors=TOTAL_COLORS)
i = 0
for pre in pre_list:
    lines = ax.plot(general[pre][1], general[pre][0], label=pre)
    print(i)
    lines[0].set_color(clrs[i])
    lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
    i+=1
lg = ax.legend(bbox_to_anchor=(1.05,1.0), loc='upper left')
plt.savefig(new_dir + "/" + 'general' + ".png", dpi=None, facecolor='w', edgecolor='w',
                orientation='landscape', papertype=None, format='png',
                transparent=False, bbox_inches="tight", pad_inches=0.1, metadata=None, bbox_extra_artists=(lg,))
plt.close()
    


'''
filenames = []
try:
    os.mkdir(new_dir)
xcept OSError as error:
    pass

joints = [ "J_" + str(x) for x in range(20)]

plottable_values = {}
    gen_? : {
        presets : {
            hand:       ?
            stick 1:    ?
            ...
        } 
    }

for i in range(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])):

    # get list of files with this generation number (has to be end of file name) 
    for FILE in FILES:
        presets = 

    data = pd.read_csv( ARCHIVE_PATH + "archive_" + str(i) + ".dat", delim_whitespace=True, \
                names = ["Fitness", "Cluster_1", "Cluster_2", "Cluster_3", \
                "BD_1", "BD_2", "BD_3", "BD_4", "BD_5"] + joints)

    data = data.drop(["BD_4", "BD_5", "Fitness","Cluster_1", "Cluster_2", "Cluster_3"] + joints\
                , axis = 1)
    
    # discretizing the data into bins
    data['z_bin']=pd.cut(x = data['BD_1'],
                        bins = [p/15 for p in range(16)], 
                        labels = [p for p in range(15)])
    data['x_bin']=pd.cut(x = data['BD_2'],
                        bins = [p/100 for p in range(101)], 
                        labels = [p for p in range(100)])
    data['y_bin']=pd.cut(x = data['BD_3'],
                        bins = [p/100 for p in range(101)],
                        labels = [p for p in range(100)])
    
    arr = np.empty((15, 100, 100))
    arr[:] = np.NaN
    
    for index, row in data.iterrows(): 
        if np.isnan(row['x_bin']):
            row['x_bin'] = 0
        if np.isnan(row['y_bin']):
            row['y_bin'] = 0
        if np.isnan(row['z_bin']):
            row['z_bin'] = 0
        x_coord = int(row['x_bin'])
        y_coord = int(row['y_bin'])
        z_coord = int(row['z_bin'])
        #val = row['Fitness']
        arr[z_coord, x_coord, y_coord] = 0

    labels = ["hand", "stick_1", "stick_2", "magnet_1", "magnet_2", "magnet_3", "scratch_1", "scratch_2", "scratch_3", "cat", "dog", "static_1", "static_2", "static_3", "static_4s"]

    for k in range(15):

        fig, ax = plt.subplots(figsize=(12,10)) 

        #mynorm = mpl.colors.Normalize(vmin=-1.5, vmax=0, clip = 0.1)
        plt.imshow(arr[k, :, :], plt.cm.get_cmap('viridis'), interpolation='nearest')
        cbar = plt.colorbar()
        
        

        plt.savefig(new_dir + "/gen_"+str(i)+"_"+labels[k]+".png", dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format='png',
                transparent=False, bbox_inches="tight", pad_inches=0.1, metadata=None)
        filenames.append(new_dir + "/gen_"+str(i)+"_"+labels[k]+".png")


images = []
for i in filenames:
    images.append(imageio.imread(i))
    

imageio.mimwrite(new_dir + '/MAP_Robox2D.gif', images, fps = 3)
'''
