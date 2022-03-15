import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import imageio
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns

ARCHIVE_PATH = "./" #"../results/"

filenames = []
new_dir = "./Cmaps_scatter"
try:
    os.mkdir(new_dir)
except OSError as error:
    pass

joints = [ "J_" + str(x) for x in range(20)]

for i in range(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])):
   
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
