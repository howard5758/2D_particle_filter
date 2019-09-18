import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Gridmap import Gridmap
from Laser import Laser
from Visualization import Visualization
import time

#matplotlib.use("Agg")
import matplotlib.pyplot as plt



class PF(object):
    # Construct an PF instance with the following set of variables
    #    numParticles:       Number of particles
    #    Alpha:              Vector of 6 noise coefficients for the motion
    #                        model (See Table 5.3 in Probabilistic Robotics)
    #    laser:              Instance of the laser class that defines
    #                        LIDAR params, observation likelihood, and utils
    #    gridmap:            An instance of the Gridmap class that specifies
    #                        an occupancy grid representation of the map
    #                        where 1: occupied and 0: free
    #    visualize:          Boolean variable indicating whether to visualize
    #                        the particle filter
    def __init__(self, numParticles, Alpha, laser, gridmap, visualize = True):
        self.numParticles = numParticles
        self.Alpha = Alpha
        self.laser = laser
        self.gridmap = gridmap
        self.visualize = visualize

        # particles is a numParticles x 3 array, where each column denote a particle_handle
        # weights is a numParticles x 1 array of particle weights
        self.particles = None
        self.weights = None

        if self.visualize == True:
            self.vis = Visualization()
            self.vis.drawGridmap(self.gridmap)
        else:
            self.vis = None




    # Samples the set of particles according to a uniform distribution
    # and sets the weigts to 1/numParticles. Particles in collision are rejected
    def sampleParticlesUniform (self):

        (m,n) = self.gridmap.getShape()

        self.particles = np.empty([3,self.numParticles])

        for i in range(self.numParticles):
            theta = np.random.uniform(-np.pi,np.pi)
            inCollision = True
            while inCollision:
                x = np.random.uniform(0,(n-1)*self.gridmap.xres)
                y = np.random.uniform(0,(m-1)*self.gridmap.yres)

                inCollision = self.gridmap.inCollision(x, y)

            self.particles[:,i] = np.array([[x, y, theta]])

        self.weights = (1./self.numParticles) * np.ones((1, self.numParticles))


    # Samples the set of particles according to a Gaussian distribution
    # Orientation are sampled from a uniform distribution
    #    (x0, y0):    Mean position
    #    sigma:       Standard deviation
    def sampleParticlesGaussian (self, x0, y0, sigma):

        (m,n) = self.gridmap.getShape()

        self.particles = np.empty([3,self.numParticles])

        for i in range(self.numParticles):
            #theta = np.random.uniform(-np.pi,np.pi)
            inCollision = True
            while inCollision:
                x = np.random.normal(x0,sigma)
                y = np.random.normal(y0,sigma)
                theta = np.random.uniform(-np.pi, np.pi)

                inCollision = self.gridmap.inCollision(x, y)

            self.particles[:,i] = np.array([[x, y, theta]])

        self.weights = (1./self.numParticles) * np.ones((1, self.numParticles))



    # Returns desired particle (3 x 1 array) and weight
    def getParticle (self, k):

        if k < self.particles.shape[1]:
            return (self.particles[:,k], self.weights[:,k])
        else:
            print ('getParticle: Request for k=%d exceeds number of particles (%d)' % (k, self.particles.shape[1]))
            return (None, None)

    # Return an array of normalized weights. Does not normalize the weights
    # maintained in the PF instance
    #
    # Returns:
    #   weights:   Array of normalized weights
    def getNormalizedWeights (self):

        return self.weights/np.sum(self.weights)


    # Returns the particle filter mean
    def getMean(self):

        weights = self.getNormalizedWeights()
        return np.sum(np.tile(weights, (self.particles.shape[0], 1)) * self.particles, axis=1)


    # Visualize filter strategies
    #   ranges:  Array of LIDAR ranges
    #   deltat:  Step size
    #   XGT:     Array with ground-truth pose
    def render(self, ranges, deltat, XGT):
        self.vis.drawParticles(self.particles, self.weights)
        if XGT is not None:
            self.vis.drawLidar(ranges, self.laser.Angles, XGT[0], XGT[1], XGT[2])
            self.vis.drawGroundTruthPose (XGT[0], XGT[1], XGT[2])
        mean = self.getMean()
        self.vis.drawMeanPose(mean[0], mean[1], mean[2])
        plt.pause(deltat/10)


    # Sample a new pose from an initial pose (x, y, theta)
    # with inputs v (forward velocity) and w (angular velocity)
    # for deltat seconds
    #
    # This model corresponds to that in Table 5.3 in Probabilistic Robotics
    #
    # Returns:
    #   (xs, ys, thetas):   Position and heading for sample
    #   (u1, u2):           Control (velocity) inputs
    #   deltat:             Time increment
    def sampleMotion (self, x, y, theta, u1, u2, deltat):

        # Your code goes here: Implement the algorithm given in Table 5.3
        # Note that the "sample" function in the text assumes zero-mean
        # Gaussian noise. You can use the NumPy random.normal() function
        # Be sure to reject samples that are in collision
        # (see Gridmap.inCollision), and to unwrap orientation so that it
        # it is between -pi and pi.

        # Hint: Repeatedly calling np.random.normal() inside a for loop
        #       can consume a lot of time. You may want to consider drawing
        #       n (e.g., n=10) samples of each noise term at once
        #       (drawing n samples is faster than drawing 1 sample n times)
        #       and if none of the estimated poses are not in collision, assume
        #       that the robot doesn't move from t-1 to t.

        N = 20
        u1_samples = np.random.normal(0, self.Alpha[0]*u1*u1*3 + self.Alpha[1]*u2*u2*3, N*self.numParticles)
        u2_samples = np.random.normal(0, self.Alpha[2]*u1*u1*3 + self.Alpha[3]*u2*u2*3, N*self.numParticles)
        g_sample = np.random.normal(0, self.Alpha[4]*u1*u1*3 + self.Alpha[5]*u2*u2*3, self.numParticles)
        
        colli = True
        i = 0
        u1hat = u1 + u1_samples[i*self.numParticles:(i+1)*self.numParticles]
        u2hat = u2 + u2_samples[i*self.numParticles:(i+1)*self.numParticles]
        div = u1hat/u2hat
        new_x = x - div*np.sin(theta) + div*np.sin(theta+u2hat*deltat)
        new_y = y + div*np.cos(theta) - div*np.cos(theta+u2hat*deltat)
        new_theta = theta + (u2+u2_samples[i*self.numParticles:(i+1)*self.numParticles])*deltat + g_sample*deltat
        colli_result = self.gridmap.inCollision(new_x, new_y)
        while colli_result.any():
            i = i + 1
            if i == N:
                break
            colli_idx = np.where(colli_result)
            temp_u1_samples = u1_samples[i*self.numParticles:(i+1)*self.numParticles]
            temp_u2_samples = u2_samples[i*self.numParticles:(i+1)*self.numParticles]
            temp_u1hat = u1 + temp_u1_samples[colli_idx]
            temp_u2hat = u2 + temp_u2_samples[colli_idx]
            div = temp_u1hat/temp_u2hat
            new_x[colli_idx] = x[colli_idx] - div*np.sin(theta[colli_idx]) + div*np.sin(theta[colli_idx]+temp_u2hat*deltat)
            new_y[colli_idx] = y[colli_idx] + div*np.cos(theta[colli_idx]) - div*np.cos(theta[colli_idx]+temp_u2hat*deltat)
            new_theta = theta + (u2+u2_samples[i*self.numParticles:(i+1)*self.numParticles])*deltat + g_sample*deltat
            colli_result = self.gridmap.inCollision(new_x, new_y)
            colli_idx = np.where(colli_result)

        if i == N:
            print("colli")
            new_x[colli_idx] = x[colli_idx]
            new_y[colli_idx] = y[colli_idx]
            new_theta[colli_idx] = theta[colli_idx] + g_sample[colli_idx]*deltat - np.pi/8
        else:
            new_theta = theta + (u2+u2_samples[i*self.numParticles:(i+1)*self.numParticles])*deltat + g_sample*deltat

        new_theta = ((new_theta+np.pi)%(2*np.pi)) - np.pi
        self.particles[0, :] = new_x
        self.particles[1, :] = new_y
        self.particles[2, :] = new_theta


    # Function that performs resampling with replacement
    def resample (self):

        idx = np.random.choice(self.numParticles, self.numParticles, p=self.getNormalizedWeights())
        self.particles = self.particles[:, idx]

        # Your code goes here
        # The np.random.choice function may be useful


    # Perform the prediction step
    def prediction(self, u, deltat):

        self.sampleMotion(self.particles[0,:], self.particles[1,:], self.particles[2,:], u[0], u[1], deltat) 




    # Perform the measurement update step
    #   Ranges:   Array of ranges (Laser.Angles provides bearings)
    def update(self, Ranges):

        self.weights = self.laser.scanProbability(Ranges, self.particles, self.gridmap)


    # Runs the particle filter algorithm
    #   U:        Array of control inputs, one column per time step
    #   Ranges:   Array of LIDAR ranges for each time step
    #             The corresponding bearings are defined in Laser.angles
    #   deltat:   Number of seconds per time step
    #   X0:       Array indicating the initial pose (may be None)
    #   XGT:      Array of ground-truth poses (may be None)
    #   filename: Name of file for plot
    def run(self, U, Ranges, deltat, X0, XGT, filename):

        # Try different sampling strategies (including different values for sigma)
        sampleGaussian = True
        print(X0)
        if sampleGaussian and (X0 is not None):
            sigma = 0.5
            self.sampleParticlesGaussian(X0[0,0], X0[1,0], sigma)
        else:
            self.sampleParticlesUniform()

        # Iterate over the data
        for k in range(U.shape[1]):
        #for k in range(0, 2):

            u = U[:,k]
            ranges = Ranges[:,k+1][0]


            if self.visualize:
                if XGT is None:
                    self.render (ranges, deltat, None)
                else:
                    self.render (ranges, deltat, XGT[:,k])

            # Your code goes here
            
            self.prediction(u, deltat)
            self.update(ranges)
            self.resample()
        
        plt.savefig(filename)
