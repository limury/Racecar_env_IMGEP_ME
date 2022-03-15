import sys
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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


new_dir = "./plots/"
tmp_dir = "./tmp/"
# create results directory
try:
    os.mkdir(new_dir)
except OSError as error:
    pass
    
try:
    os.mkdir(tmp_dir)
except OSError as error:
    print("failed to create tmp/")

ARCHIVE_PATH = "./" #"../results/"


FILES = glob.glob("data/*.dat")
FILES_GRID = glob.glob("data/0-*.dat")

mappings = {
    0.25: "Arm",
    0.75: "Car"
}

def centroid_to_grid(shape, values):
    arr = np.full(shape=shape, fill_value=np.nan)
    tmp = values.copy()
    tmp['x'] -= 0.5/shape[0]
    tmp['x'] *= shape[0]
    tmp['y'] -= 0.5/shape[1]
    tmp['y'] *= shape[1]
    for ind, row in tmp.iterrows():
        arr[ round(row['x']) ][ round(row['y']) ] = row['curiosity']

    arr[0][0] = -15
    arr[-1][-1] = 3
    return arr

def plot_grid(name):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(25,10))
    data = pd.read_csv(name, delim_whitespace=False, names=["obj","x","y","curiosity"], index_col=False).round(8)

    arm = centroid_to_grid( (50, 50), data.query('obj == 0.25') )
    car = centroid_to_grid( (50, 50), data.query('obj == 0.75') )

    ax0.set_title('Arm', fontsize=20)
    c = ax0.pcolor(arm, edgecolors='k', linewidths=1)
    fig.colorbar(c, ax=ax0)

    ax1.set_title('Car', fontsize=20)
    c = ax1.pcolor(car, edgecolors='k', linewidths=1)
    fig.colorbar(c, ax=ax1)
    
    fig.suptitle("Population map: " + name[7:-13], fontsize=30)
    plt.savefig(new_dir + name[5:-5] + ".png", dpi=None, facecolor='w', edgecolor='w',
                orientation='landscape', format='png')
    plt.close(fig)

# has form values[ object ] [ preset ] = [ [generation number], [number of indiv] ]
def get_population_data():
    values = {
        "Arm": {},
        "Car": {},
        "Total": {}
    }
    for f in FILES_GRID:
        filename = f[7:]
        # get list of files that belong to this exact preset and generation, this way we can average them.
        files = glob.glob("data/*" + filename)
        # extract generation from current file
        generation = int(f[-12:-4])
        #extract presets from current file
        preset = f[7:-13]

        sum_ = {
            "Arm": 0,
            "Car": 0,
            "Total": 0
        }
        count = 0
        # make it so sum_ contains sum of populations for each obj in this preset
        for fi in files:
            data = pd.read_csv(fi, delim_whitespace=False, names=["obj","x","y"], index_col=False).round(8)
                        
            count += 1

            for obj in mappings:
                name = mappings[obj]
                query = 'obj == '+str(obj)
                obj_data = data.query(query) 
                sum_[name] += len(obj_data)
            sum_["Total"] += len(data)

        # convert sum_ to average
        for key in sum_:
            sum_[key] /= count 

        for obj in values:
            if preset not in values[obj]:
                # first item is generation
                # second item is population
                values[obj][preset] = [[], []]
            values[obj][preset][0].append(generation)
            values[obj][preset][1].append(sum_[obj])

    for obj in values:
        for x in values[obj]:
            values[obj][x][0], values[obj][x][1] = zip(*sorted(zip(values[obj][x][0], values[obj][x][1])))
    return values


def plot_population( values: dict, name = "Maze map population" ):
    # plot graph
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    NUM_COLORS = len(values)
    TOTAL_COLORS = 0
    fig, ax = plt.subplots()
    sns.reset_orig()

    i = 0
    clrs = sns.color_palette('husl', n_colors=NUM_COLORS)
    for pre in values:
        lines = ax.plot(values[pre][0], values[pre][1], label = pre)
        lines[0].set_color(clrs[i])
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        i+=1
    
    lg = ax.legend(bbox_to_anchor=(1.05,1.0), loc='upper left')
    plt.title(name)
    plt.xlabel('Evaluation')
    plt.ylabel('Population')
    plt.savefig(new_dir + name + ".png", dpi=None, facecolor='w', edgecolor='w',
                orientation='landscape', papertype=None, format='png',
                transparent=False, bbox_inches="tight", pad_inches=0.1, metadata=None, bbox_extra_artists=(lg,))
    plt.close()


def cull_population( values ):
    for x in list(values['Car'].keys()):
        if values['Car'][x][1][-1] <= 25:
            del values['Car'][x]
    return values

if __name__ == "__main__":
    for x in FILES:
        plot_grid(x)

    values = get_population_data()
    values = cull_population(values)
    for obj in values:
        plot_population(values[obj], obj)


#for obj in values:
#
#    fig, ax = plt.subplots()
#    sns.reset_orig(){
#    NUM_COLORS = len(values[obj].keys())
#    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
#    NUM_STYLES = len(LINE_STYLES)
#    i = 0
#    clrs = sns.color_palette('husl', n_colors=NUM_COLORS)
#    for pre in values[obj]:
#        if (values["Magnetic Stick"][pre][0][-1] >= 2000):
#            values[obj][pre][1], values[obj][pre][0] = zip(*sorted(zip(values[obj][pre][1], values[obj][pre][0])))
#            lines = ax.plot(values[obj][pre][1], values[obj][pre][0], label = pre)
#            lines[0].set_color(clrs[i])
#            lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
#            i+=1
#    lg = ax.legend(bbox_to_anchor=(1.05,1.0), loc='upper left')
#    plt.savefig(new_dir + "/" + obj + ".png", dpi=None, facecolor='w', edgecolor='w',
#                orientation='landscape', papertype=None, format='png',
#                transparent=False, bbox_inches="tight", pad_inches=0.1, metadata=None, bbox_extra_artists=(lg,))
#    plt.close()

'''
filenames = []
try:
    os.mkdir(new_dir)
except OSError as error:
    pass

joints = [ "J_" + str(x) for x in range(20)]

lottable_values = {}
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
