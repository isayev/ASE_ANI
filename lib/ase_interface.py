import numpy as np
import os, sys
from math import sqrt
import math
import time

import types

from ase.units import Bohr
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList
try:
    from ased3._d3 import d3
    d3present = True
except ImportError:
    print('van Der Waals correction will be unavailable. Please install ased3')
    d3present = False
    pass

import pyNeuroChem as pync

import multiprocessing
from multiprocessing import Process, Value, Array, Queue

# ANI energy a.u. to eV conversion
global conv_au_ev
conv_au_ev = 27.21138505

#import numpy as np
#import matplotlib.mlab as mlab
#import matplotlib.pyplot as plt

class ANI(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    #implemented_properties = ['energy', 'forces']
    default_parameters = {'xc': 'ani'}

    nolabel = True
    def __init__(self, build=True,gpuid=0,reslist=[],**kwargs):
        Calculator.__init__(self, **kwargs)

        if build:
            anipath = os.path.dirname(__file__)
            cnstfile = anipath + '/../ani-1x_dft_x8ens/rHCNO-5.2R_16-3.5A_a4-8.params'
            saefile = anipath + '/../ani-1x_dft_x8ens/sae_linfit.dat'
            nnfdir = anipath + '/../ani-1x_dft_x8ens/train0/networks/'
            self.nc = pync.molecule(cnstfile, saefile, nnfdir, gpuid)

        self.Setup=True
        self.reslist=reslist

    def setnc(self,nc):
        self.nc = nc

    def calculate(self, atoms=None, properties=['energy'],
                   system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        #start_time2 = time.time()
        ## make up for stress
        ## TODO
        #stress_ani = np.zeros((1, 3))
        stress_ani = np.zeros(6)

        
        if self.Setup or self.nc.request_setup():
            #Setup molecule for MD
            natoms = len(self.atoms)
            atom_symbols = self.atoms.get_chemical_symbols()
            xyz = self.atoms.get_positions()
            self.nc.setMolecule(coords=xyz.astype(np.float32),types=atom_symbols)
            self.nc.setPBC(self.atoms.get_pbc()[0],self.atoms.get_pbc()[1],self.atoms.get_pbc()[2])

            self.Setup=False
        else:
            xyz = self.atoms.get_positions()
            # Set the conformers in NeuroChem
            self.nc.setCoordinates(coords=xyz.astype(np.float32))


            # TODO works only for 3D periodic. For 1,2D - update np.zeros((3,3)) part
            pbc_inv = (np.linalg.inv(self.atoms.get_cell())).astype(np.float32) if atoms.pbc.all() else np.zeros((3,3), dtype=np.float32)
            self.nc.setCell((self.atoms.get_cell()).astype(np.float32), pbc_inv)
            #self.nc.setCell((self.atoms.get_cell()).astype(np.float32),(np.linalg.inv(self.atoms.get_cell())).astype(np.float32))

        #start_time2 = time.time()
        self.results['energy'] = conv_au_ev*self.nc.energy()[0]
        if 'forces' in properties:
            forces = conv_au_ev*self.nc.force()

            # restrain atoms
            for i in self.reslist:
                forces[i] = 0.0

            self.results['forces'] = forces
        self.results['stress'] = conv_au_ev*stress_ani
        #end_time2 = time.time()
        #print('ANI Time:', end_time2 - start_time2)

    def __update_neighbors(self):
        #print('------------------------')
        #szs = []
        #an = self.atoms.get_atomic_numbers()
        for a in range(0,len(self.atoms)):
            indices,offsets = self.nlR.get_neighbors(a)
            #if an[a] == 8:
                #print(an[a])
                #szs.append(len(indices))
            self.nc.setNeighbors(ind=a,indices=indices.astype(np.int32),offsets=offsets.astype(np.float32))

        #indices, offsets = self.nlR.get_neighbors(302)
        #f = open('test2.xyz','w')
        #f.write(str(len(indices))+'\n')
        #f.write("   comment\n")
        #an = self.atoms.get_atomic_numbers()
        #for i, offset in zip(indices, offsets):
        #    xyz = self.atoms.positions[i]
        #    f.write(str(an[i]) + ' ' + str(xyz[0]) + ' ' + str(xyz[1]) + ' ' + str(xyz[2]) + '\n')

        #print(szs)
        #plt.hist(szs, max(szs)-min(szs), normed=1, facecolor='green', alpha=0.75)
        #plt.xlabel('Number of neighbors')
        #plt.ylabel('Count')
        #plt.show()
        #print('------------------------')

    def get_atomicenergies(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        ## make up for stress
        ## TODO
        stress_ani = np.zeros((1, 3))

        if self.Setup or self.nc.request_setup():
            # Setup molecule for MD
            natoms = len(self.atoms)
            atom_symbols = self.atoms.get_chemical_symbols()
            xyz = self.atoms.get_positions()
            self.nc.setMolecule(coords=xyz.astype(np.float32), types=atom_symbols)
            self.Setup = False
        else:
            xyz = self.atoms.get_positions()
            # Set the conformers in NeuroChem
            self.nc.setCoordinates(coords=xyz.astype(np.float32))

        self.nc.energy()

        return self.nc.aenergies(True) * conv_au_ev

##-------------------------------------
##      Molecule worker class
##--------------------------------------
def molecule_worker(task_queue, gpuid, net_list, energy, forces, net_dict):
    print('Building...')
    ncl = [pync.molecule(net_dict['cns'], net_dict['sae'], net_dict['nnf'] + str(i) + '/networks/', gpuid, False) for i
           in net_list]

    Nn = net_dict['Nn']

    if net_dict['epw']:
        for nc in ncl:
            nc.setPairWise()

    set_pbc = False

    Sp = []

    while True:
        next_task = task_queue.get()
        if next_task is None:
            # Poison pill means shutdown
            time.sleep(1)
            print ('Exiting')
            task_queue.task_done()
            break

        if not set_pbc:
            pbc = next_task['pbc']
            for i, netid in enumerate(net_list):
                #print('PBC:',pbc)
                ncl[i].setPBC(pbc[0], pbc[1], pbc[2])
            set_pbc=True

        # Atomic elements
        S = next_task['S']

        # Cell
        cell = next_task['cell']

        # pbc_inv
        pinv = next_task['pinv']

        # make it two-dimensional
        for i, netid in enumerate(net_list):
            # Set the molecule/coordinates

            if Sp == S:
                if 'bynet' in next_task:
                    ncl[i].setCoordinates(coords=next_task['X'][netid].astype(np.float32))
                else:
                    ncl[i].setCoordinates(coords=next_task['X'].astype(np.float32))
            else:
                if 'bynet' in next_task:
                    ncl[i].setMolecule(next_task['X'][netid].astype(np.float32), S)
                else:
                    ncl[i].setMolecule(next_task['X'].astype(np.float32), S)

            # Set the cell
            ncl[i].setCell(cell, pinv)
            #print('CELL:',cell)

            energy[netid] = ncl[i].energy().copy()
            forces[netid] = ncl[i].force().copy()
            #if netid == 0 and net_dict['epw']:
                #energy[netid] += ncl[i].pwenergy()
                #forces[netid] += ncl[i].pwforce()
                #print(ncl[i].pwenergy().copy())
                #print(-ncl[i].pwforce().copy()[0])

            if net_dict['epw']:
                energy[netid] +=  ncl[i].pwenergy()
                forces[netid] +=  ncl[i].pwforce()
        Sp = S

        task_queue.task_done()
    return

##-------------------------------------
## Class for ANI ensemble prediction
##--------------------------------------
class ensemblemolecule_multigpu(object):
    def __init__(self, cnstfile, saefile, nnfprefix, Nnet, gpu_list=0, sinet=False, forcesigma=False, enablepairwise=False):
        # Number of networks
        self.Nn = Nnet

        if not isinstance(gpu_list, list):
            gpu_list = [gpu_list]

        # Build network dict
        self.netdict = {'cns': cnstfile,
                        'sae': saefile,
                        'nnf': nnfprefix,
                        'Nn' : self.Nn, 
                        'epw': enablepairwise}

        # number of cores
        self.cores = len(gpu_list)

        # Build task list
        self.task_list = [multiprocessing.JoinableQueue() for i in range(self.cores)]

        # Construct working containers
        self.energy = Array('d', range(self.Nn))

        # Construct working containers
        self.manager = multiprocessing.Manager()

        self.forces = self.manager.list()
        self.forces[:] = [[] for i in range(self.Nn)]

        # Construct threads with pyNeuroChem molecule classes
        self.p_list = []
        for i in range(self.cores):
            self.net_list = [j + int(self.Nn / self.cores) * i for j in range(int(self.Nn / self.cores))]
            self.p_list.append(
                Process(target=molecule_worker, args=(self.task_list[i], gpu_list[i], self.net_list, self.energy, self.forces, self.netdict)))
            self.p_list[-1].start()

    def request_setup(self):
        return False

    def set_molecule(self, X, S):
        self.S = S
        self.set_coordinates(X)
        self.Na = len(S)

    def set_coordinates(self, X):
        self.X = X

    def set_pbc(self, pbc0, pbc1, pbc2):
        self.pbc = [pbc0, pbc1, pbc2]

    def set_cell(self, cell, pbc_inv):
        self.cell = cell
        self.pinv = pbc_inv

    def compute_props(self):
        data_dict = {'X': self.X,
                     'S': self.S,
                     'pbc' : self.pbc,
                     'cell' : self.cell,
                     'pinv' : self.pinv}

        self.forces[:] = [[] for i in range(self.Nn)]

        # Launch Jobs
        for i in range(self.cores):
            self.task_list[i].put(data_dict)

        # Wait for jobs
        for i in range(self.cores):
            self.task_list[i].join()

        self.E = np.array(self.energy[:])
        self.F = np.stack(self.forces[:])
        return self.E, self.F

    def compute_props_by_net(self, X, S):
        data_dict = {'X': X,
                     'S': S,
                     'pbc' : self.pbc,
                     'cell' : self.cell,
                     'pinv' : self.pinv,
                     'bynet': True}

        self.forces[:] = [[] for i in range(self.Nn)]

        # Launch Jobs
        for i in range(self.cores):
            self.task_list[i].put(data_dict)

        # Wait for jobs
        for i in range(self.cores):
            self.task_list[i].join()

        self.E = np.array(self.energy[:])
        self.F = np.stack(self.forces[:])
        return self.E, self.F

    def compute_mean_props(self):
        data_dict = {'X': self.X,
                     'S': self.S,
                     'pbc' : self.pbc,
                     'cell' : self.cell,
                     'pinv' : self.pinv}

        self.forces[:] = [[] for i in range(self.Nn)]

        # Launch Jobs
        for i in range(self.cores):
            self.task_list[i].put(data_dict)

        # Wait for jobs
        for i in range(self.cores):
            self.task_list[i].join()

        self.E = np.array(self.energy[:])
        self.F = np.stack(self.forces[:])

        # dF and C
        dF = self.F - np.mean(self.F, axis=0)[np.newaxis, :, :]

        # Compute var sqr
        v2 = np.sum(np.sum(np.power(dF, 2), axis=0) / (self.Nn * (self.Nn - 1)))

        # Store intermediates 
        self.intermediates = {'var_sqr': v2}

        return np.mean(self.E, axis=0), np.mean(self.F, axis=0), np.std(self.E, axis=0) / np.sqrt(float(self.Na)), np.mean(np.std(self.F, axis=0))

    def compute_sigma_bias_potential(self, X, S, Efunc, Ffunc, epsilon=0.001, disable_ani=False):
        if self.Nn < 2:
            print('Error: compute_sigma_bias_potential requires more than 1 network in the ensemble.')
            raise ValueError

        # Initial Force
        self.set_molecule(X, S)
        E, F = self.compute_props()

        # dF and C
        dF = F - np.mean(F, axis=0)[np.newaxis, :, :]

        # Forward calc
        E1, F1 = self.compute_props_by_net(X[np.newaxis, :, :] + epsilon * dF, S)

        # Backward calc
        E2, F2 = self.compute_props_by_net(X[np.newaxis, :, :] - epsilon * dF, S)

        # Computer ds/dx via central difference
        ds = np.sum((F1 - F2) / epsilon, axis=0) / (self.Nn * (self.Nn - 1))

        # Compute var sqr
        v2 = np.sum(np.sum(np.power(dF, 2), axis=0) / (self.Nn * (self.Nn - 1)))

        E_bias = Efunc(v2)
        F_bias = Ffunc(v2, ds)

        Eani = np.mean(E, axis=0)
        Fani = np.mean(F, axis=0)
        self.intermediates = {'Ebias': E_bias,'Eani': Eani,'Fbias': F_bias,'Fani': Fani,'var_sqr': v2}

        if not disable_ani:
            return E_bias+Eani, F_bias+Fani, np.std(E, axis=0) / np.sqrt(float(self.Na)), np.mean(np.std(F, axis=0))
        else:
            return E_bias, F_bias, np.std(E, axis=0) / np.sqrt(float(self.Na)), np.mean(np.std(F, axis=0))

    def cleanup(self):
        # Add a poison pill for each consumer
        for task, in zip(self.task_list):
            task.put(None)

        #for proc in zip(self.p_list):
        #    proc.join()

    def __del__(self):
        self.cleanup()

##-------------------------------------
## Class for ANI ensemble prediction
##--------------------------------------
class ensemblemolecule(object):
    def __init__(self, cnstfile, saefile, nnfprefix, Nnet, gpuid=0, sinet=False, forcesigma=False):
        # Number of networks
        self.Nn = Nnet

        # Construct pyNeuroChem molecule classes
        self.ncl = [pync.molecule(cnstfile, saefile, nnfprefix + str(i) + '/networks/', gpuid, sinet) for i in
                    range(self.Nn)]


    def set_molecule(self, X, S):
        for nc in self.ncl:
            nc.setMolecule(coords=np.array(X,dtype=np.float32), types=list(S))

        self.E = np.zeros((self.Nn), dtype=np.float64)
        self.F = np.zeros((self.Nn, X.shape[0], X.shape[1]), dtype=np.float32)

        self.Na = X.shape[0]

    def set_molecule_per_net(self, X, S):
        for nc,x in zip(self.ncl,X):
            nc.setMolecule(coords=np.array(x,dtype=np.float32), types=list(S))

        self.E = np.zeros((self.Nn), dtype=np.float64)
        self.F = np.zeros((self.Nn, X.shape[1], X.shape[2]), dtype=np.float32)

        self.Na = X.shape[0]

    def request_setup(self):
        return self.ncl[0].request_setup()

    def set_coordinates(self, X):
        for nc in self.ncl:
            nc.setCoordinates(coords=X.astype(np.float32))

    def set_pbc(self, pbc0, pbc1, pbc2):
        for nc in self.ncl:
            nc.setPBC(pbc0, pbc1, pbc2)

    def set_cell(self, cell, pbc_inv):
        for nc in self.ncl:
            nc.setCell(cell, pbc_inv)

    def compute_stddev_molecule(self, X):
        for i, nc in enumerate(self.ncl):
            nc.setCoordinates(coords=X)
            self.E[i] = nc.energy()[0]
        sigma = np.std(self.E, axis=0) / np.sqrt(float(self.Na))
        return sigma

    def compute_props(self):
        for i, nc in enumerate(self.ncl):
            self.E[i] = nc.energy().copy()
            self.F[i] = nc.force().copy()
        return self.E.copy(), self.F.copy()

    def compute_energies(self):
        for i, nc in enumerate(self.ncl):
            self.E[i] = nc.energy().copy()
        return self.E.copy()

    def compute_aenergies(self,sae):
        Ea = np.zeros((self.Nn,self.Na),dtype=np.float64)
        for i, nc in enumerate(self.ncl):
            Ea[i,:] = nc.aenergies(sae).copy()
        return Ea

    def compute_mean_props(self):
        for i, nc in enumerate(self.ncl):
            self.E[i] = nc.energy().copy()
            self.F[i] = nc.force().copy()

        # dF and C
        dF = self.F - np.mean(self.F, axis=0)[np.newaxis, :, :]

        # Compute var sqr
        v2 = np.sum(np.sum(np.power(dF, 2), axis=0) / (self.Nn * (self.Nn - 1)))

        # Store intermediates 
        self.intermediates = {'var_sqr': v2}

        # Return
        return np.mean(self.E, axis=0), np.mean(self.F, axis=0), np.std(self.E, axis=0) / np.sqrt(float(self.Na)), np.mean(np.std(self.F, axis=0))

    def compute_mean_energies(self):
        for i, nc in enumerate(self.ncl):
            self.E[i] = nc.energy().copy()
        return np.mean(self.E, axis=0), np.std(self.E, axis=0) / np.sqrt(float(self.Na))

    def compute_sigma_bias_potential(self, X, S, Efunc, Ffunc, epsilon=0.001, disable_ani=False):
        if self.Nn < 2:
            print('Error: compute_sigma_bias_potential requires more than 1 network in the ensemble.')
            raise ValueError

        # Initial Force
        self.set_molecule(X, S)
        E, F = self.compute_props()

        # dF and C
        dF = F - np.mean(F, axis=0)[np.newaxis, :, :]

        # Forward calc
        self.set_molecule_per_net(X[np.newaxis, :, :] + epsilon * dF, S)
        E1, F1 = self.compute_props()

        # Backward calc
        self.set_molecule_per_net(X[np.newaxis, :, :] - epsilon * dF, S)
        E2, F2 = self.compute_props()

        # Computer ds/dx via central difference
        ds = np.sum((F1 - F2) / epsilon, axis=0) / (self.Nn * (self.Nn - 1))

        # Compute var sqr
        v2 = np.sum(np.sum(np.power(dF, 2), axis=0) / (self.Nn * (self.Nn - 1)))

        E_bias = Efunc(v2)
        F_bias = Ffunc(v2, ds)

        Eani = np.mean(E, axis=0)
        Fani = np.mean(F, axis=0)
        self.intermediates = {'Ebias': E_bias,'Eani': Eani,'Fbias': F_bias,'Fani': Fani,'var_sqr': v2}

        if not disable_ani:
            return E_bias+Eani, F_bias+Fani, np.std(E, axis=0) / np.sqrt(float(self.Na)), np.mean(np.std(F, axis=0))
        else:
            return E_bias, F_bias, np.std(E, axis=0) / np.sqrt(float(self.Na)), np.mean(np.std(F, axis=0))

##-------------------------------------
##         Cos cutoff functions
##--------------------------------------
def coscut(Rmag, iRc, sRc):
    return 0.5 * np.cos(math.pi * (Rmag+sRc) * iRc) + 0.5

def dcoscut(Rmag, R, iRc, sRc):
    return -0.5 * math.pi * iRc * np.sin(math.pi * iRc * (Rmag+sRc)) * R/Rmag

def tanhcut(Rmag, Shf):
    return 0.5*np.tanh(-20*(Rmag-Shf)) + 0.5

def dtanhcut(Rmag, R, Shf):
    #return 0.5*np.tanh(-20*(Rmag-Shf)) + 0.5
    return -0.5*20*(1.0 - np.power(np.tanh(-20*(Rmag-Shf)),2)) * R/Rmag

##-------------------------------------
##     ANI Ensemble ASE interface
##--------------------------------------
class ANIENS(Calculator):
    implemented_properties = ['energy', 'forces', 'stress', 'dipole']
    default_parameters = {'xc': 'ani'}

    nolabel = True

    ### Constructor ###
    def __init__(self, aniens, sdmax=sys.float_info.max, energy_conversion=conv_au_ev, **kwargs):
        Calculator.__init__(self, **kwargs)

        self.nc = aniens
        self.energy_conversion = energy_conversion
        self.sdmax = sdmax
        self.Setup = True

        self.nc_time=0.0

        self.Epwise = None
        self.Emodel = None

        self.pwiter = True
        self.pairwise = False
        self.add_bias = False

        self.hipmodels = None

        # Tortional Restraint List
        self.tres = []

        self.Xn = np.zeros((0,0,3), dtype=np.float64)
        self.Dn = np.zeros((0,0,3), dtype=np.float64)

    ### Set the pairwise functions ###
    def set_pairwise(self, Efunc, Ffunc):
        self.Efunc = Efunc
        self.Ffunc = Ffunc
        self.pairwise = True

    def set_sigmabias(self,bias_Efunc, bias_Ffunc, epsilon=0.001, disable_ani=False):
        self.bias_Efunc = bias_Efunc
        self.bias_Ffunc = bias_Ffunc
        self.epsilon=epsilon
        self.disable_ani_in_bias = disable_ani
        self.add_bias=True

    ### Set HIPNN for dipole ###
    def set_hipnn_dipole_model(self, hipmodels):
        self.hipmodels = hipmodels

    def resize_XnDn(self,ne):
        if ne > self.Xn.shape[1]:
            self.Xn = np.pad(self.Xn,((0, 0), (0, ne - self.Xn.shape[1]), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            self.Dn = np.pad(self.Dn,((0, 0), (0, ne - self.Dn.shape[1]), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
        return self.Xn.shape[1]

    ### Adds the pairwise energies/forces to the calculated ###
    def add_pairwise(self, properties):
        positions = self.atoms.get_positions()

        mcut = 0.0
        lcut = 2.5

        if self.pwiter:
            N = positions.shape[0]
            self.nl = NeighborList(np.full(N, mcut/2.0), skin=0.25, self_interaction=False)
            self.Xn = np.zeros((N, 0, 3), dtype=np.float64)
            self.Dn = np.zeros((N, 0, 3), dtype=np.float64)
            self.pwiter = False

        #start_time = time.time()
        self.nl.update(self.atoms)

        Epairwise = 0.0
        if 'forces' in properties:
            Fpairwise = 0. * positions

        # loop over all atoms in the cell
        for ia, posa in enumerate(positions):
            indices, offsets = self.nl.get_neighbors(ia)
            #print('Neh/Dsp:', indices.shape, offsets.shape)

            #nposition = positions[indices]
            R = positions[indices] + np.dot(offsets, self.atoms.get_cell()) - posa[np.newaxis,:]
            Rmag = np.linalg.norm(R,axis=1)
            cidx = np.where( (Rmag >= lcut) & (Rmag < mcut)  )
            sidx = np.where(Rmag >= mcut)
            E = self.Efunc(Rmag)

            Ecut = coscut(Rmag[cidx], 1.0/(mcut-lcut), lcut)
            #Ecut = tanhcut(Rmag[cidx], lcut-0.1*lcut)
            E[cidx] = E[cidx]*Ecut
            E[sidx] = 0.0*E[sidx]

            Epairwise += E.sum() # Neighbors list supplies only neighbors once (NO DOUBLE COUNT)

            if 'forces' in properties:
                F = self.Ffunc(Rmag[:,np.newaxis],R)
                F[cidx] = (E[cidx][:,np.newaxis]*dcoscut(Rmag[cidx][:,np.newaxis], R[cidx], 1.0/(mcut-lcut), lcut)+Ecut[:,np.newaxis]*F[cidx])
                #F[cidx] = (E[cidx][:,np.newaxis]*dtanhcut(Rmag[cidx][:,np.newaxis], R[cidx], lcut-0.1*lcut)+Ecut[:,np.newaxis]*F[cidx])
                F[sidx] = 0.0 * F[sidx]
                Fpairwise[indices] += -F
                Fpairwise[ia] += np.sum(F,axis=0)

        self.results['energy'] += Epairwise
        self.Epwise = Epairwise
        if 'forces' in properties:
            #print(Fpairwise.shape,np.sum(Fpairwise, axis=1))
            #print(Fpairwise[0])
            self.results['forces'] += Fpairwise

    ### Calculate function require by ASE ###
    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        ## make up for stress
        ## TODO
        # stress_ani = np.zeros((1, 3))
        stress_ani = np.zeros(6)

        ## Check if models are initilized (first step)
        if self.Setup or self.nc.request_setup():
            # Setup molecule for MD
            natoms = len(self.atoms)
            atom_symbols = self.atoms.get_chemical_symbols()
            xyz = self.atoms.get_positions()
            self.nc.set_molecule(xyz.astype(np.float32), atom_symbols)
            self.nc.set_pbc(bool(self.atoms.get_pbc()[0]), bool(self.atoms.get_pbc()[1]), bool(self.atoms.get_pbc()[2]))
            # TODO works only for 3D periodic. For 1,2D - update np.zeros((3,3)) part
            pbc_inv = (np.linalg.inv(self.atoms.get_cell())).astype(np.float32) if atoms.pbc.all() else np.zeros(
                (3, 3), dtype=np.float32)
            self.nc.set_cell((self.atoms.get_cell()).astype(np.float32), pbc_inv)

            self.Setup = False
        ## Run this if models are initialized
        else:
            xyz = self.atoms.get_positions().astype(np.float32)
            # Set the conformers in NeuroChem
            self.nc.set_coordinates(xyz)

            # TODO works only for 3D periodic. For 1,2D - update np.zeros((3,3)) part
            pbc_inv = (np.linalg.inv(self.atoms.get_cell())).astype(np.float32) if atoms.pbc.all() else np.zeros(
                (3, 3), dtype=np.float32)
            self.nc.set_cell((self.atoms.get_cell()).astype(np.float32), pbc_inv)

        ## Compute the model properties (you can speed up ASE energy prediction by not doing force backprop unless needed.)
        start_time = time.time()
        if not self.add_bias:
            energy, force, stddev, Fstddev = self.nc.compute_mean_props()
            self.Fstddev = Fstddev
        else:
            energy, force, stddev, Fstddev = self.nc.compute_sigma_bias_potential(self.atoms.get_positions().astype(np.float32),
                                                                                  self.atoms.get_chemical_symbols(),
                                                                                  self.bias_Efunc, self.bias_Ffunc,
                                                                                  epsilon=self.epsilon,
                                                                                  disable_ani=self.disable_ani_in_bias)
            self.Fstddev=Fstddev
        self.nc_time+=time.time() - start_time

        ## convert std dev to correct units
        self.stddev = self.energy_conversion * stddev

        ## Store energies in ASE
        self.results['energy'] = energy
        self.Emodel = energy

        ## If forces are requested store forces 
        if 'forces' in properties:
            forces = force

            if len(self.tres) > 0:
                for res in self.tres:
                    forces = hard_restrain_tortion_force(res, xyz, forces)

            self.results['forces'] = forces

        ## Set stress tensor
        self.results['stress'] = self.energy_conversion * stress_ani

        ## If the HIP-NN model is set run this for dipoles
        if self.hipmodels is not None:
            Z = self.atoms.get_atomic_numbers()
            R = self.atoms.get_positions()

            R = R.reshape([1,len(Z),3])
            Z = Z.reshape([1, len(Z)]).astype(np.int32)

            dipole = np.zeros((3),dtype=np.float32)
            for hippy in self.hipmodels:
                output = hippy.pred_fn([Z, R], shape_output=True)
                dipole += output[1][0][0]
            self.results['dipole'] = dipole/len(self.hipmodels)

        ## Compute pairwise if requested
        if self.pairwise:
            self.add_pairwise(properties)
        
        ## Convert energies and forces to requested units
        self.results['energy'] = self.energy_conversion * self.results['energy']
        if 'forces' in properties:
            self.results['forces'] = self.energy_conversion * self.results['forces']

    ### NL Update function now handled internally ###
    def __update_neighbors(self):
        for a in range(0, len(self.atoms)):
            indices, offsets = self.nlR.get_neighbors(a)
            self.nc.setNeighbors(ind=a, indices=indices.astype(np.int32), offsets=offsets.astype(np.float32))

    ### Return the atomic energies (in eV) ###
    def get_atomicenergies(self, atoms=None, properties=['energy'],
                           system_changes=all_changes,sae=False):
        Calculator.calculate(self, atoms, properties, system_changes)

        ## make up for stress
        ## TODO
        stress_ani = np.zeros((1, 3))

        if self.Setup or self.nc.request_setup():
            # Setup molecule for MD
            natoms = len(self.atoms)
            atom_symbols = self.atoms.get_chemical_symbols()
            xyz = self.atoms.get_positions()
            self.nc.setMolecule(coords=xyz.astype(np.float32), types=atom_symbols)
            self.Setup = False
        else:
            xyz = self.atoms.get_positions()
            # Set the conformers in NeuroChem
            self.nc.set_coordinates(xyz.astype(np.float32))

        E = self.nc.compute_energies()

        return self.nc.compute_aenergies(sae) * conv_au_ev

##--------------------------------------------------------------
##        Easy loader for ANI networks in python interface
##--------------------------------------------------------------
def aniensloader(model, gpu=0, multigpu=False):
    # Set locations of all required network files
    wkdir = model.rsplit('/', 1)[0] + '/'  # Note the relative path

    data = np.loadtxt(model, dtype=str)

    cnstfile = wkdir + data[0]  # AEV parameters
    saefile = wkdir + data[1]  # Atomic shifts
    nnfdir = wkdir + data[2]  # network prefix
    Nn = int(data[3])  # Number of networks in the ensemble

    if multigpu:
        if not isinstance(gpu, list):
            raise ValueError('When running on multiple GPUs please supply the GPU indices in a list.')
        return ensemblemolecule_multigpu(cnstfile, saefile, nnfdir, Nn, gpu)
    else:
        return ensemblemolecule(cnstfile, saefile, nnfdir, Nn, gpu)


###
#   ANI with D3 correction
##

if d3present:

    d3_calc = d3.d3_calc

    alp = 14.

    # D3 damping parameters
    #                            (RS6  , S18   , RS18 , S6  )
    damp0 = {
        'ani': (1.281, 1.0940, 1.000, 1.00),
        'slater-dirac-exchange': (0.999, -1.957, 0.697, 1.00),
        'b-lyp': (1.094, 1.6820, 1.000, 1.00),
        'b-p': (1.139, 1.6830, 1.000, 1.00),
        'b97-d': (0.892, 0.9090, 1.000, 1.00),
        'revpbe': (0.923, 1.0100, 1.000, 1.00),
        'pbe': (1.217, 0.7220, 1.000, 1.00),
        'pbesol': (1.345, 0.6120, 1.000, 1.00),
        'rpw86-pbe': (1.224, 0.9010, 1.000, 1.00),
        'rpbe': (0.872, 0.5140, 1.000, 1.00),
        'tpss': (1.166, 1.1050, 1.000, 1.00),
        'b3-lyp': (1.261, 1.7030, 1.000, 1.00),
        'pbe0': (1.287, 0.9280, 1.000, 1.00),
        'hse06': (1.129, 0.1090, 1.000, 1.00),
        'revpbe38': (1.021, 0.8620, 1.000, 1.00),
        'pw6b95': (1.532, 0.8620, 1.000, 1.00),
        'tpss0': (1.252, 1.2420, 1.000, 1.00),
        'b2-plyp': (1.427, 1.0220, 1.000, 0.64),
        'pwpb95': (1.557, 0.7050, 1.000, 0.82),
        'b2gp-plyp': (1.586, 0.7600, 1.000, 0.56),
        'ptpss': (1.541, 0.8790, 1.000, 0.75),
        'hf': (1.158, 1.7460, 1.000, 1.00),
        'mpwlyp': (1.239, 1.0980, 1.000, 1.00),
        'bpbe': (1.087, 2.0330, 1.000, 1.00),
        'bh-lyp': (1.370, 1.4420, 1.000, 1.00),
        'tpssh': (1.223, 1.2190, 1.000, 1.00),
        'pwb6k': (1.660, 0.5500, 1.000, 1.00),
        'b1b95': (1.613, 1.8680, 1.000, 1.00),
        'bop': (0.929, 1.9750, 1.000, 1.00),
        'o-lyp': (0.806, 1.7640, 1.000, 1.00),
        'o-pbe': (0.837, 2.0550, 1.000, 1.00),
        'ssb': (1.215, 0.6630, 1.000, 1.00),
        'revssb': (1.221, 0.5600, 1.000, 1.00),
        'otpss': (1.128, 1.4940, 1.000, 1.00),
        'b3pw91': (1.176, 1.7750, 1.000, 1.00),
        'revpbe0': (0.949, 0.7920, 1.000, 1.00),
        'pbe38': (1.333, 0.9980, 1.000, 1.00),
        'mpw1b95': (1.605, 1.1180, 1.000, 1.00),
        'mpwb1k': (1.671, 1.0610, 1.000, 1.00),
        'bmk': (1.931, 2.1680, 1.000, 1.00),
        'cam-b3lyp': (1.378, 1.2170, 1.000, 1.00),
        'lc-wpbe': (1.355, 1.2790, 1.000, 1.00),
        'm05': (1.373, 0.5950, 1.000, 1.00),
        'm052x': (1.417, 0.0000, 1.000, 1.00),
        'm06l': (1.581, 0.0000, 1.000, 1.00),
        'm06': (1.325, 0.0000, 1.000, 1.00),
        'm062x': (1.619, 0.0000, 1.000, 1.00),
        'm06hf': (1.446, 0.0000, 1.000, 1.00),
        'dftb3': (1.235, 0.6730, 1.000, 1.00),
        'hcth120': (1.221, 1.2060, 1.000, 1.00),
    }

    # D3(BJ) damping parameters
    #                    (RS6    , S18    , RS18  , S6  )
    dampbj = {
        'ani': (0.48310, 2.00770, 4.5323, 1.00),
        'b-p': (0.39460, 3.28220, 4.8516, 1.00),
        'b-lyp': (0.42980, 2.69960, 4.2359, 1.00),
        'revpbe': (0.52380, 2.35500, 3.5016, 1.00),
        'rpbe': (0.18200, 0.83180, 4.0094, 1.00),
        'b97-d': (0.55450, 2.26090, 3.2297, 1.00),
        'pbe': (0.42890, 0.78750, 4.4407, 1.00),
        'rpw86-pbe': (0.46130, 1.38450, 4.5062, 1.00),
        'b3-lyp': (0.39810, 1.98890, 4.4211, 1.00),
        'tpss': (0.45350, 1.94350, 4.4752, 1.00),
        'hf': (0.33850, 0.91710, 2.8830, 1.00),
        'tpss0': (0.37680, 1.25760, 4.5865, 1.00),
        'pbe0': (0.41450, 1.21770, 4.8593, 1.00),
        'hse06': (0.38300, 2.31000, 5.6850, 1.00),
        'revpbe38': (0.43090, 1.47600, 3.9446, 1.00),
        'pw6b95': (0.20760, 0.72570, 6.3750, 1.00),
        'b2-plyp': (0.30650, 0.91470, 5.0570, 0.64),
        'dsd-blyp': (0.00000, 0.21300, 6.0519, 0.50),
        'dsd-blyp-fc': (0.00090, 0.21120, 5.9807, 0.50),
        'bop': (0.48700, 3.29500, 3.5043, 1.00),
        'mpwlyp': (0.48310, 2.00770, 4.5323, 1.00),
        'o-lyp': (0.52990, 2.62050, 2.8065, 1.00),
        'pbesol': (0.44660, 2.94910, 6.1742, 1.00),
        'bpbe': (0.45670, 4.07280, 4.3908, 1.00),
        'opbe': (0.55120, 3.38160, 2.9444, 1.00),
        'ssb': (-0.0952, -0.1744, 5.2170, 1.00),
        'revssb': (0.47200, 0.43890, 4.0986, 1.00),
        'otpss': (0.46340, 2.74950, 4.3153, 1.00),
        'b3pw91': (0.43120, 2.85240, 4.4693, 1.00),
        'bh-lyp': (0.27930, 1.03540, 4.9615, 1.00),
        'revpbe0': (0.46790, 1.75880, 3.7619, 1.00),
        'tpssh': (0.45290, 2.23820, 4.6550, 1.00),
        'mpw1b95': (0.19550, 1.05080, 6.4177, 1.00),
        'pwb6k': (0.18050, 0.93830, 7.7627, 1.00),
        'b1b95': (0.20920, 1.45070, 5.5545, 1.00),
        'bmk': (0.19400, 2.08600, 5.9197, 1.00),
        'cam-b3lyp': (0.37080, 2.06740, 5.4743, 1.00),
        'lc-wpbe': (0.39190, 1.85410, 5.0897, 1.00),
        'b2gp-plyp': (0.00000, 0.25970, 6.3332, 0.56),
        'ptpss': (0.00000, 0.28040, 6.5745, 0.75),
        'pwpb95': (0.00000, 0.29040, 7.3141, 0.82),
        'hf/mixed': (0.56070, 3.90270, 4.5622, 1.00),
        'hf/sv': (0.42490, 2.18490, 4.2783, 1.00),
        'hf/minis': (0.17020, 0.98410, 3.8506, 1.00),
        'b3-lyp/6-31gd': (0.50140, 4.06720, 4.8409, 1.00),
        'hcth120': (0.35630, 1.08210, 4.3359, 1.00),
        'dftb3': (0.74610, 3.20900, 4.1906, 1.00),
    }


    # Class of ANI + D3 energies
    
    class ANID3(Calculator):
        implemented_properties = ['energy', 'forces', 'stress']
    
        default_parameters = {'xc': 'ani',
                              'bj': True,
                              'threebody': True,
                              'rcut': 95 * Bohr,
                              'rcutcn': 40 * Bohr,
                              'rs6': None,
                              's18': None,
                              'rs18': None,
                              's6': None,
                              'calculator': None}
    
    
        nolabel = True
    
        def __init__(self, build=True, gpuid=0, reslist=[], **kwargs):
            Calculator.__init__(self, **kwargs)
    
            if build:
                anipath = os.path.dirname(__file__)
                cnstfile = anipath + '/../ANI-c08f-ntwk/rHCNO-4.6A_16-3.1A_a4-8.params'
                saefile = anipath + '/../ANI-c08f-ntwk/sae_6-31gd.dat'
                nnfdir = anipath + '/../ANI-c08f-ntwk/networks/'
                self.nc = pync.molecule(cnstfile, saefile, nnfdir, gpuid)
    
            self.Setup = True
            self.reslist = reslist
    
        def setnc(self, nc):
            self.nc = nc
    
        def calculate(self, atoms=None, properties=['energy'],
                      system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
    
            xc = self.parameters.xc.lower()
            bj = self.parameters.bj
            threebody = self.parameters.threebody
            rcut = self.parameters.rcut
            rcutcn = self.parameters.rcutcn
            calculator = self.parameters.calculator
    
            if bj:
                damp = dampbj
            else:
                damp = damp0
    
            rs6 = s18 = rs18 = s6 = None
    
            try:
                rs6, s18, rs18, s6 = damp[xc]
            except KeyError:
                unknown_functional = True
            else:
                unknown_functional = False
    
            if self.parameters.s6 is not None:
                s6 = self.parameters.s6
            if self.parameters.s18 is not None:
                s18 = self.parameters.s18
            if self.parameters.rs6 is not None:
                rs6 = self.parameters.rs6
            if self.parameters.rs18 is not None:
                rs18 = self.parameters.rs18
    
            if unknown_functional and None in (s6, s18, rs6, rs18):
                raise ValueError("Unknown functional {}! \
                        Please specify damping parameters.".format(xc))
    
            # D3 calculation part
            energy_d3, forces_d3, stress_d3 = d3_calc(
                self.atoms.get_atomic_numbers(),
                self.atoms.get_cell(),
                self.atoms.get_positions().T,
                rcut=rcut,
                rcutcn=rcutcn,
                s6=s6,
                s18=s18,
                rs6=rs6,
                rs18=rs18,
                alp6=alp,
                alp8=alp + 2,
                pbc=self.atoms.get_pbc(),
                bj=bj,
                threebody=threebody)
    
            ## make up for stress
            ## TODO
            stress_ani = np.zeros(6)
    
            if self.Setup or self.nc.request_setup():
                # Setup molecule for MD
                natoms = len(self.atoms)
                atom_symbols = self.atoms.get_chemical_symbols()
                xyz = self.atoms.get_positions()
                self.nc.setMolecule(coords=xyz.astype(np.float32), types=atom_symbols)
                self.nc.setPBC(self.atoms.get_pbc()[0], self.atoms.get_pbc()[1], self.atoms.get_pbc()[2])
    
                self.Setup = False
            else:
                xyz = self.atoms.get_positions()
                # Set the conformers in NeuroChem
                self.nc.setCoordinates(coords=xyz.astype(np.float32))
    
                # TODO works only for 3D periodic. For 1,2D - update np.zeros((3,3)) part
                pbc_inv = (np.linalg.inv(self.atoms.get_cell())).astype(np.float32) if atoms.pbc.all() else np.zeros((3, 3),
                                                                                                                     dtype=np.float32)
                self.nc.setCell((self.atoms.get_cell()).astype(np.float32), pbc_inv)
                # self.nc.setCell((self.atoms.get_cell()).astype(np.float32),(np.linalg.inv(self.atoms.get_cell())).astype(np.float32))
    
            # start_time2 = time.time()
            self.results['energy'] = conv_au_ev * self.nc.energy()[0] + energy_d3
            if 'forces' in properties:
                forces = -conv_au_ev * self.nc.force() + forces_d3.T
    
                # restrain atoms
                for i in self.reslist:
                    forces[i] = 0.0
    
                self.results['forces'] = forces
            self.results['stress'] = conv_au_ev * stress_ani + stress_d3.flat[[0, 4, 8, 5, 2, 1]]
            # end_time2 = time.time()
            # print('ANI Time:', end_time2 - start_time2)
    
        def __update_neighbors(self):
            # print('------------------------')
            # szs = []
            # an = self.atoms.get_atomic_numbers()
            for a in range(0, len(self.atoms)):
                indices, offsets = self.nlR.get_neighbors(a)
                # if an[a] == 8:
                # print(an[a])
                # szs.append(len(indices))
                self.nc.setNeighbors(ind=a, indices=indices.astype(np.int32), offsets=offsets.astype(np.float32))
    
                # indices, offsets = self.nlR.get_neighbors(302)
                # f = open('test2.xyz','w')
                # f.write(str(len(indices))+'\n')
                # f.write("   comment\n")
                # an = self.atoms.get_atomic_numbers()
                # for i, offset in zip(indices, offsets):
                #    xyz = self.atoms.positions[i]
                #    f.write(str(an[i]) + ' ' + str(xyz[0]) + ' ' + str(xyz[1]) + ' ' + str(xyz[2]) + '\n')
    
                # print(szs)
                # plt.hist(szs, max(szs)-min(szs), normed=1, facecolor='green', alpha=0.75)
                # plt.xlabel('Number of neighbors')
                # plt.ylabel('Count')
                # plt.show()
                # print('------------------------')
    
        def get_atomicenergies(self, atoms=None, properties=['energy'],
                               system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
    
            ## make up for stress
            ## TODO
            stress_ani = np.zeros((1, 3))
    
            if self.Setup or self.nc.request_setup():
                # Setup molecule for MD
                natoms = len(self.atoms)
                atom_symbols = self.atoms.get_chemical_symbols()
                xyz = self.atoms.get_positions()
                self.nc.setMolecule(coords=xyz.astype(np.float32), types=atom_symbols)
                self.Setup = False
            else:
                xyz = self.atoms.get_positions()
                # Set the conformers in NeuroChem
                self.nc.setCoordinates(coords=xyz.astype(np.float32))
    
            self.nc.energy()
    
            return self.nc.aenergies(True) * conv_au_ev


class D3(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters = {'xc': 'ani',
                          'bj': True,
                          'threebody': True,
                          'rcut': 95 * Bohr,
                          'rcutcn': 40 * Bohr,
                          'rs6': None,
                          's18': None,
                          'rs18': None,
                          's6': None,
                          'calculator': None}

    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        xc = self.parameters.xc.lower()
        bj = self.parameters.bj
        threebody = self.parameters.threebody
        rcut = self.parameters.rcut
        rcutcn = self.parameters.rcutcn
        calculator = self.parameters.calculator

        if bj:
            damp = dampbj
        else:
            damp = damp0

        rs6 = s18 = rs18 = s6 = None

        try:
            rs6, s18, rs18, s6 = damp[xc]
        except KeyError:
            unknown_functional = True
        else:
            unknown_functional = False

        if self.parameters.s6 is not None:
            s6 = self.parameters.s6
        if self.parameters.s18 is not None:
            s18 = self.parameters.s18
        if self.parameters.rs6 is not None:
            rs6 = self.parameters.rs6
        if self.parameters.rs18 is not None:
            rs18 = self.parameters.rs18

        if unknown_functional and None in (s6, s18, rs6, rs18):
            raise ValueError("Unknown functional {}! \
                    Please specify damping parameters.".format(xc))

        energy, forces, stress = d3_calc(
            self.atoms.get_atomic_numbers(),
            self.atoms.get_cell(),
            self.atoms.get_positions().T,
            rcut=rcut,
            rcutcn=rcutcn,
            s6=s6,
            s18=s18,
            rs6=rs6,
            rs18=rs18,
            alp6=alp,
            alp8=alp + 2,
            pbc=self.atoms.get_pbc(),
            bj=bj,
            threebody=threebody)

        self.results['energy'] = energy
        self.results['forces'] = forces.T
        self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]

        if calculator is not None:
            calculate(self.atoms)
            self.results['energy'] += calculator.results['energy']
            self.results['forces'] += calculator.results['forces']
            self.results['stress'] += calculator.results['stress']

