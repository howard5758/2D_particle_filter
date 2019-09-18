import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.collections import LineCollection


from Gridmap import Gridmap

class Visualization(object):
    # Visualization tools
    def __init__(self):

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal')

        plt.ion()
        plt.axis('off')
        plt.tight_layout()

        self.particle_handle = None
        self.true_pose_handle = None
        self.mean_pose_handle = None
        self.lidar_handle = None



    # Function to draw the gridmap
    #    gridmap:            An instance of the Gridmap class that specifies
    #                        an occupancy grid representation of the map
    #                        where 1: occupied and 0: free
    def drawGridmap(self, gridmap):

        (m,n) = gridmap.getShape()

        # Set the axis size
        self.ax.set_xlim(0, (n)*gridmap.xres, True, False)
        self.ax.set_ylim(0, (m)*gridmap.yres, True, False)

        for j in range(n):
            for i in range(m):
                # x0 = (j-1)*gridmap.xres
                # y0 = (i-1)*gridmap.yres
                x0 = (j)*gridmap.xres
                y0 = (i)*gridmap.yres
                #y0 = ((m-1) - i)*gridmap.yres

                if gridmap.inCollision(j,i, True):
                    self.ax.add_patch (
                        patches.Rectangle(
                            (x0, y0),
                            gridmap.xres,
                            gridmap.yres
                        ))

        self.fig.canvas.draw()


    # Function to draw particles
    #   particles:   An N x 3 array where each column is a particle
    #   weights:     An N x 1 array of particle weights
    def drawParticles (self, particles, weights=None):

        if self.particle_handle == None:
            self.particle_handle, = self.ax.plot(particles[0,:], particles[1,:], 'k.')
        else:
            self.particle_handle.set_xdata(particles[0,:])
            self.particle_handle.set_ydata(particles[1,:])

        self.fig.canvas.draw()



    # Function to draw ground-truth pose
    #   x, y, theta:   Position and orientation
    def drawGroundTruthPose (self, x, y, theta):

        if self.true_pose_handle == None:
            self.true_pose_handle, = self.ax.plot(x, y, 'r.')
        else:
            self.true_pose_handle.set_xdata(x)
            self.true_pose_handle.set_ydata(y)

        self.fig.canvas.draw()


    # Function to draw mean pose
    #   x, y, theta:   Position and orientation
    def drawMeanPose (self, x, y, theta):

        if self.mean_pose_handle == None:
            self.mean_pose_handle, = self.ax.plot(x, y, 'go')
        else:
            self.mean_pose_handle.set_xdata(x)
            self.mean_pose_handle.set_ydata(y)

        self.fig.canvas.draw()



    # Function to draw a LIDAR scan
    #   range:       Array of ranges
    #   bearing:     Array of bearings
    #   (x,y,theta)  Pose from which scan was acquired
    def drawLidar (self, range, bearing, x, y, theta):

        # Get the XY points corresponding to range and bearing in the LIDAR frame
        CosSin = np.vstack((np.cos(bearing[:]),np.sin(bearing[:])))
        XY_lidar = np.tile(range.transpose(),(2,1))*CosSin

        # Define the rotation matrix
        R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

        XY_robot = np.tile(np.array([[x],[y]]),(1,bearing.shape[0]))

        XY_world = R.dot(XY_lidar) + XY_robot #np.tile(np.array([[x],[y]]),(1,bearing.shape[0]))

        # Restructure the data to make it suitable for LineCollection
        # (a bit ugly, but it works)
        XY_worldT = XY_world.transpose()
        temp3 = XY_worldT.reshape(-1,1,2)

        XY_robotT = XY_robot.transpose()
        temp4 = XY_robotT.reshape(-1,1,2)
        lines = np.hstack((temp3,temp4))

        if self.lidar_handle == None:
            # for i in range(XY_world.shape[1]):
            #     self.line_handle,
            self.lidar_handle = LineCollection(lines, cmap=plt.cm.gist_ncar,linewidths=0.5, color='red')
            self.ax.add_collection(self.lidar_handle)
        else:
            self.lidar_handle.set_segments(lines)

        self.fig.canvas.draw()
