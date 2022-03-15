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

FILES = glob.glob("*.dat")
print ("dat files: ", FILES)

# create results directory
try:
    os.mkdir(new_dir)
except OSError as error:
    pass

values = {}

with open('values.json', 'r') as f:
    values = json.load(f)

# has form values[ object ] [ preset ] [ generation ] = number of solutions

for f in FILES:
    # extract generation from current file
    generation = int(f[-12:-4])
    #extract presets from current file
    preset = f[2:-13]
    #print(preset) 
    if "batch" in f:
        data = pd.read_csv(f, delim_whitespace=False, \
                names = [ "Obj", "x", "y"], index_col=False)

        data["obj_bin"] = pd.cut(x = data['Obj'],
                                            bins = [p/15 for p in range(16)],
                                            labels = labels)
        data['x_bin']  = pd.cut(x = data['x'],
                                bins = [p/100 for p in range(101)],
                                labels = [p for p in range(100)])
        data['y_bin']  = pd.cut(x = data['y'],
                                bins = [p/100 for p in range(101)],
                                labels = [p for p in range(100)])
        extractable = data["obj_bin"].value_counts()
        #print(extractable) 
        for column in columns:
            if ( column not in values):
                values[column]={}
            if ( preset not in values[column]):
                values[column][preset]=[[], [], []]
            if (generation in values[column][preset][1]): 
                ind = values[column][preset][1].index(generation) 
                values[column][preset][0][ind] += extractable[column]
                values[column][preset][2][ind] += 1
            else:
                # extract all data into a dictionary
                values[column][preset][0].append(extractable[column])
                values[column][preset][1].append(generation)
                values[column][preset][2].append(1)



    else:
        # load data from current file
        data = pd.read_csv(f, delim_whitespace=False, \
                    names = [ x+c for x in columns for c in ["_x", "_y"]])
        # create discrete bins 
        for column in columns:
            data[column + "_x_bin"] = pd.cut(x = data[column+"_x"],
                                            bins = [p/100 for p in range(101)],
                                            labels = [p for p in range(100)])
            data[column + "_y_bin"] = pd.cut(x = data[column+"_y"],
                                            bins = [p/100 for p in range(101)],
                                            labels = [p for p in range(100)])
            if (not column in values):
                values[column]={}
            if (not preset in values[column]):
                values[column][preset]=[[], [], []]
            if (generation in values[column][preset][1]): 
                ind = values[column][preset][1].index(generation) 
                values[column][preset][0][ind] += len(data.index) - data.duplicated([column+"_x_bin", column+"_y_bin"]).sum()
                values[column][preset][2][ind] += 1
            else:
                # extract all data into a dictionary
                values[column][preset][0].append(len(data.index) - data.duplicated([column+"_x_bin", column+"_y_bin"]).sum())
                values[column][preset][1].append(generation)
                values[column][preset][2].append(1)

for x in values:
    for y in values[x]:
        for i in range(len(values[x][y][0])):
            values[x][y][0][i] = values[x][y][0][i] / values[x][y][2][i]

with open('values.json', 'w') as f:
    json.dump(values, f)

for obj in values:

    fig, ax = plt.subplots()
    sns.reset_orig()
    NUM_COLORS = len(values[obj].keys())
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    i = 0
    clrs = sns.color_palette('husl', n_colors=NUM_COLORS)
    for pre in values[obj]:
        if (values["Magnetic Stick"][pre][0][-1] >= 2000):
            values[obj][pre][1], values[obj][pre][0] = zip(*sorted(zip(values[obj][pre][1], values[obj][pre][0])))
            lines = ax.plot(values[obj][pre][1], values[obj][pre][0], label = pre)
            lines[0].set_color(clrs[i])
            lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
            i+=1
    lg = ax.legend(bbox_to_anchor=(1.05,1.0), loc='upper left')
    plt.savefig(new_dir + "/" + obj + ".png", dpi=None, facecolor='w', edgecolor='w',
                orientation='landscape', papertype=None, format='png',
                transparent=False, bbox_inches="tight", pad_inches=0.1, metadata=None, bbox_extra_artists=(lg,))
    plt.close()

'''
filenames = []
try:
    os.mkdir(new_dir)
except OSError as error:
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
