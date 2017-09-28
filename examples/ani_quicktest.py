__author__ = 'jujuman'

# Import pyNeuroChem
import sys
from ase_interface import ANI

import numpy as np
import  ase
import time
from ase import units
from ase.io import read, write
from ase.optimize import BFGS, LBFGS

import pyNeuroChem as pync

# Set required files for pyNeuroChem
#anipath  = '/home/jujuman/Dropbox/ChemSciencePaper.AER/ANI-c08e-ntwk'
#cnstfile = anipath + '/rHCNO-4.6A_16-3.1A_a4-8.params'
#saefile  = anipath + '/sae_6-31gd.dat'
#nnfdir   = anipath + '/networks/'

# Construct pyNeuroChem class
#nc = pync.molecule(cnstfile, saefile, nnfdir, 0)

mol = read('examples/data/water.xyz')

mol.set_calculator(ANI())
#mol.calc.setnc(nc)

ei = mol.get_potential_energy()
print("Initial Energy: ",ei)

foo = mol.calc.get_atomicenergies()
#foo = nc.aenergies()
print(foo.sum())

# O of Conformations
print("Optimizing...")
start_time = time.time()
dyn = LBFGS(mol)
dyn.run(fmax=0.00001)
print('[ANI Optimization - Total time:', time.time() - start_time, 'seconds]')

ef = mol.get_potential_energy()
print("Final Energy: ",ef)

# Write visualization of molecule
#f = open("optmol.xyz",'w')
#f.write('\n' + str(len(mol)) + '\n')
#for i in mol:
#    f.write(str(i.symbol) + ' ' + str(i.x) + ' ' + str(i.y) + ' ' + str(i.z) + '\n')
#f.close()
