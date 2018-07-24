__author__ = 'jujuman'

# Import ANI ensemble loader
import sys
#sys.path.append('/home/olexandr/notebooks/ASE_ANI/lib')
from ase_interface import ANIENS
from ase_interface import aniensloader

import  ase
import time
from ase import units
from ase.io import read, write
from ase.optimize import BFGS, LBFGS

# Read molecule from xyz
mol = read('data/water.xyz')

# Current ANI model options are:
# '../ani_models/ani-1ccx_8x.info' Coupled cluster transfer learned model
# '../ani_models/ani-1x_8x.info'   Full ANI-1x wb97x/6-31g* dataset model
mol.set_calculator(ANIENS(aniensloader('../ani_models/ani-1ccx_8x.info')))

# Calculate energy
ei = mol.get_potential_energy()
print("Initial Energy: ",ei)

# Optimize molecule
print("Optimizing...")
start_time = time.time()
dyn = LBFGS(mol)
dyn.run(fmax=0.001)
print('[ANI Optimization - Total time:', time.time() - start_time, 'seconds]')

# Calculate energy
ef = mol.get_potential_energy()
print("Final Energy: ",ef)

