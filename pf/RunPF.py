import sys
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PF import PF
from Laser import Laser
from Gridmap import Gridmap


if __name__ == '__main__':

    # This function should be called with two arguments:
    #    sys.argv[1]: Pickle file defining problem setup
    #    sys.argv[2]: Number of particles (default=100)
    if (len(sys.argv) == 3):
        numParticles = int(sys.argv[2])
    elif (len(sys.argv) == 2):
        numParticles = 100
    else:
        print ("usage: RunPF.py Data.pickle numParticles (optional, default=100)")
        sys.exit(2)

    # Load data
    Data = pickle.load(open(sys.argv[1], 'rb'), encoding='latin1')
    deltat = Data['deltat']#[0,0]
    occupancy = Data['occupancy']
    U = Data['U']
    X0 = Data['X0']
    Ranges = Data['Ranges']
    XGT = Data['XGT']
    Alpha = Data['Alpha']
    sparsity = 5

    numBearings = Ranges[0,0].shape[0]
    Ranges = np.array(Ranges.tolist())[:,:,::sparsity]

    # Gridmap class
    gridmap = Gridmap(occupancy)

    # Laser class
    laser = Laser(numBearings, sparsity)

    print(Alpha)
    # Instantiate the PF class
    pf = PF(numParticles, Alpha, laser, gridmap, True)

    filename = os.path.basename(sys.argv[1]).split('.')[0] + '_Pn' + str(numParticles)
    pf.run(U, Ranges, deltat, X0, XGT, filename)
