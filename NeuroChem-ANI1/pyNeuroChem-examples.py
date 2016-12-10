uthor__ = 'jujuman'

# Import pyNeuroChem
import pyNeuroChem as pync
import numpy as np

# Set required files for pyNeuroChem
wkdir    = '/home/olexandr/ani/NeuroChem-ANI1/' # <----- !!!! Just set this to root folder !!!!
cnstfile = wkdir + 'ANI-1-ntwk/rHCNO-4.6A_32-3.1A_a8-8.params'
saefile  = wkdir + 'ANI-1-ntwk/sae_6-31gd.dat'
nnfdir   = wkdir + 'ANI-1-ntwk/networks/'

# Coordinates for 3 conformers of water
xyz = np.array([[[0.00000000,  0.07775414,  0.14191920],  [0.00000000,  0.87511754, -0.40200517],  [0.00000000, -0.64261788, -0.50009561]]
               ,[[0.00000000,  0.07782031,  0.14223997],  [0.00000000,  0.87485045, -0.40213168],  [0.00000000, -0.64241695, -0.50028986]]
               ,[[0.00000000,  0.07782238,  0.14223947],  [0.00000000,  0.87484127, -0.40213102],  [0.00000000, -0.64240986, -0.50029004]]
               ,[[0.00000000,  0.07782238,  0.14223947],  [0.00000000,  0.87484127, -0.40213102],  [0.00000000, -0.64240986, -0.50029004]]],dtype=np.float32)

# Coordinates for a single molecule example
# xyz2 = np.array([[0.00000000,  0.2775414,  0.14191920],  [0.00000000,  0.87511754, -0.40200517],  [0.00000000, -0.64261788, -0.50009561]],dtype=np.float32)

# List of atomic types
typ = ['O','H','H']

# Construct pyNeuroChem class
nc = pync.pyNeuroChem(cnstfile, saefile, nnfdir, 0)

# Set the conformers in NeuroChem XYZ is 3D array
nc.setConformers(confs=xyz,types=typ)

# For single molecule you can use:  -- xyz2 is 2D array
# nc.setMolecule(coords=xyz2,types=typ)

# Print some data from the NeuroChem
print( 'Number of Atoms Loaded: ' + str(nc.getNumAtoms()) )
print( 'Number of Confs Loaded: ' + str(nc.getNumConfs()) )

# Compute Energies of Conformations - Energies for each conformer returned 1D np.ndarray
E = nc.energy()

# Compute Forces of Conformations - Forces returned in np.ndarray
# If only a single molecule was supplied the returned ndarray
# will be 2D if multiple conformers are given it will be 3D.
# -- computeEnergies has to be called PRIOR to calling forces --
F = nc.force()

# You can do an MD loop by calling setMolecule() with new coords, 
# then energy(), then force(). Rinse and repeat.

# Access the atomic energies - ordered the same is the input typ
AE = nc.aenergies()

print ('-----------------ENERGIES (Ha)---------------')
print (E)

print ('-----------------FORCES (Ha/A)---------------')
print (F)

print ('-------------ATOMIC ENERGIES (Ha)------------')
print (AE)

