import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from scipy.spatial.distance import cdist, squareform
from matplotlib.widgets import Button
from matplotlib import colors as mcolors

from plyfile import PlyData

N = 2048

def plot_trans(xyz, predicted_xyz):
    fig = plt.figure(facecolor='white', figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(elev=10, azim=25)
    ax.axis('equal')
    #ax.set_alpha(1)
    ax.set_facecolor('none')

    '''ax.set_zticks(np.arange(-8, 8, 0.1)*0.05)  
    ax.set_yticks(np.arange(-8, 8, 1)*0.05)
    ax.set_xticks(np.arange(-16, 16, 1)*0.05)   '''
    ax.scatter(xyz[:,0]-0.5, xyz[:,2], xyz[:,1], s = 80, marker = '.')
    ax.scatter(predicted_xyz[:,0]+0.5, predicted_xyz[:,2], predicted_xyz[:,1], s = 80, marker = '.')

    ax.axis('equal')
    delta = 0.05
    n = 512
    d = 2048//n
    for i in range(n):
        line = [xyz[i*d,:]-[0.5,0,0], predicted_xyz[i*d,:]+[0.5,0,0]]
        line = np.array(line)
        sub = xyz[i*d, :] - predicted_xyz[i*d,:]
        dist = np.sqrt(np.sum(sub*sub, -1))
        
        if dist > delta:
            
            ax.plot(line[:,0], line[:,2], line[:,1], c='grey', alpha=0.4)

    ax.axis('equal')


if __name__ == '__main__':

    Folder = './output_horse'
    #Folder = './output_horse_withLoss_0.02'
    Folder2 = 'output_horse_withoutTransLoss'
    #Folder = './output_octopus'
    #Folder2 = 'output_octopus_withoutTransLoss'
    fname2 = 'Ep240_predicted_X1/'
    fname1 = 'gt_surface/'
    fname3 = 'Ep300_predicted_X1/'

    source_path  = Folder + '/' + fname1 + 'mesh_0040.ply'
    predicted_path  = Folder + '/' + fname2 + 'mesh_0040_mesh_0045.ply'
    predicted_path2  = Folder2 + '/' + fname3 + 'mesh_0040_mesh_0045.ply'  ##withoutTransLoss

    '''source_path  = Folder + '/' + fname1 + 'mesh_0010.ply'
    predicted_path  = Folder + '/' + fname2 + 'mesh_0010_mesh_0075.ply'
    predicted_path2  = Folder2 + '/' + fname3 + 'mesh_0010_mesh_0075.ply' '''

    PlyData_Source = PlyData.read( source_path )
    #PlyData_Predicted = PlyData.read( './output_horse/gt_surface/mesh_0045.ply' )
    PlyData_Predicted = PlyData.read( predicted_path )
    #PlyData_Predicted2 = PlyData.read( predicted_path2 )

    xyz = np.zeros([2048, 3])
    predicted_xyz = np.zeros([2048, 3])
    predicted_xyz2 = np.zeros([2048, 3])

    xyz[ :, 0 ] = PlyData_Source['vertex']['x']
    xyz[ :, 1 ] = PlyData_Source['vertex']['y']
    xyz[ :, 2 ] = PlyData_Source['vertex']['z']

    predicted_xyz[ :, 0 ] = PlyData_Predicted['vertex']['x']
    predicted_xyz[ :, 1 ] = PlyData_Predicted['vertex']['y']
    predicted_xyz[ :, 2 ] = PlyData_Predicted['vertex']['z']

    '''predicted_xyz2[ :, 0 ] = PlyData_Predicted2['vertex']['x']
    predicted_xyz2[ :, 1 ] = PlyData_Predicted2['vertex']['y']
    predicted_xyz2[ :, 2 ] = PlyData_Predicted2['vertex']['z']'''

    plot_trans(xyz, predicted_xyz)
    #plot_trans(xyz, predicted_xyz2)

    plt.show()
