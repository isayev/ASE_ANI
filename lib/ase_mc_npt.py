"""Monte Carlo barostat (NPT) class."""

import numpy as np
import numpy.random as rand

global PCONV
global KB
PCONV = 6.241509E-7
KB = 8.6173303E-5

class MCBarostat:
    """Monte Carlo Barostat (constant N, P, T) molecular dynamics.

    Usage: MCBarostat(atoms, temperature, pressure)

    atoms
        The list of atoms.

    temperature
        The simulation temperature, in Kelvin. NOTE: this class only acts
        as a barostat, and thus temperature regulation is assumed to be
        performed by the main Molecular Dynamics algorithm in use.

    pressure
        The desired pressure, in bar.

    isotropic
        If True, each trial change to the unit cell volume is performed in a
        symmetric way to all three lattice directions. If False, only a single
        lattice direction is scaled at each trial step.

    dV_max_init
        Initial maximum change to fractional volume attempted at each step.

    dV_interval
        How often dV_max is checked for change (based on MC success rate)

    dV_scale
        How much dV_max is scaled by if it is changed.

    This class's step function is meant to be attached to a main dynamics
    class (e.g., Langevin) and thus called at an interval specified by the user."""

    def __init__(self, atoms, temperature, pressure,
                   isotropic=True, dV_max_init=0.01, dV_interval=10, dV_scale=1.1):
        self.atoms = atoms
        self.pres = pressure
        self.iso = isotropic

        self.temp = temperature
        self.kT = (self.temp * KB)

        self.natoms = self.atoms.get_number_of_atoms()

        self.dV_max = dV_max_init * self.atoms.get_volume()
        self.dV_interval = dV_interval
        self.dV_scale = dV_scale

        self.mcbar_attempts = 0
        self.mcbar_successes = 0

    def set_pressure(self, pressure):
        self.pres = pressure

    def set_temperature(self, temperature):
        self.temp = temperature
        self.kT = (self.temp * KB)

    def step(self):
        self.mcbar_attempts += 1
        E0 = self.atoms.get_potential_energy()
        V0 = self.atoms.get_volume()
        cell0 = self.atoms.get_cell()

        # dV = np.zeros(3)
        # if self.iso:
        #      dV[:] = (rand.random() - 0.5) * self.dV_max
        # else:
        #      dim = rand.random_integers(0,2)
        #      dV[dim] = 2.0 * (rand.random() - 0.5) * self.dV_max

        dV = 2.0 * (rand.random() - 0.5) * self.dV_max
        Vn = V0+dV

        lengthScale = np.power(Vn/V0,1.0/3.0)
        cell = cell0 * lengthScale
        self.atoms.set_cell(cell,scale_atoms=True)

        En = self.atoms.get_potential_energy()

        #w = En - E0 + self.pres * dV * PCONV - self.kT * self.natoms * np.log(Vn/V0)
        w = En - E0 + self.pres * dV * PCONV

        if w > 0 and rand.random() > np.exp(-w / self.kT):
            # Reject the step.
            # On failure, revert the system to previous volume
            self.atoms.set_cell(cell0,scale_atoms=True)
        else:
            self.mcbar_successes += 1

        # Check if we are succeeding too often or not often enough, and change dV_max if so
        if self.mcbar_attempts % self.dV_interval == 0:
            print('Acceptance Ratio:',self.mcbar_successes/self.mcbar_attempts)
            if self.mcbar_successes >= 0.75 * self.mcbar_attempts:
                self.dV_max *= self.dV_scale
                print("MC BAR INCREASE DVMAX",self.dV_max,self.mcbar_attempts,self.mcbar_successes)
                self.mcbar_attempts = 0
                self.mcbar_successes = 0
            elif self.mcbar_successes <= 0.25 * self.mcbar_attempts:
                self.dV_max /= self.dV_scale
                print("MC BAR DECREASE DVMAX:",self.dV_max,self.mcbar_attempts,self.mcbar_successes)
                self.mcbar_attempts = 0
                self.mcbar_successes = 0

    # def step(self):
    #     self.mcbar_attempts += 1
    #
    #     # Get energy, volume for current system configuration
    #     E0 = self.atoms.get_potential_energy()
    #     V0 = self.atoms.get_volume()
    #     cell0 = self.atoms.get_cell()
    #
    #     dV = np.zeros(3)
    #
    #     # Random change to fractional unit cell volume in range (-0.5*dV_max,0.5*dV_max)
    #     if self.iso:
    #         dV[:] = (rand.random() - 0.5) * self.dV_max
    #     else:
    #         dim = rand.random_integers(0,2)
    #         dV[dim] = (rand.random() - 0.5) * self.dV_max
    #
    #     rmu = (1. + dV) ** (1./3.)
    #     # Is this correct for non-rectangular cells?
    #     cell = cell0.copy()
    #     cell[0] *= rmu[0]
    #     cell[1] *= rmu[1]
    #     cell[2] *= rmu[2]
    #     # Scale system to new unit cell and get new energy, volume
    #     self.atoms.set_cell(cell,scale_atoms=True)
    #
    #     E = self.atoms.get_potential_energy()
    #     V = self.atoms.get_volume()
    #
    #     #print('V/V0:',V/V0,'rmu**3',rmu[0]*rmu[1]*rmu[2])
    #
    #     pv_work = self.pres * (V - V0) * PCONV
    #
    #     #print(np.exp(self.natoms * (np.log(rmu[0]*rmu[1]*rmu[2])-np.log(V/V0))))
    #     #print((E - E0 + pv_work) * self.beta,self.natoms * np.log(rmu[0]*rmu[1]*rmu[2]))
    #     mc_term = np.exp((E - E0 + pv_work) * self.beta + self.natoms * np.log(rmu[0]*rmu[1]*rmu[2]))
    #     #mc_term = np.exp((E - E0 + pv_work) * self.beta)
    #     mc_check = rand.random()
    #
    #     # Monte Carlo condition check
    #     if mc_check < mc_term:
    #         self.mcbar_successes += 1
    #     else:
    #         # On failure, revert the system to previous volume
    #         self.atoms.set_cell(cell0,scale_atoms=True)
    #
    #     # Check if we are succeeding too often or not often enough, and change dV_max if so
    #     if self.mcbar_attempts % self.dV_interval == 0:
    #         print('Acceptance Ratio:',self.mcbar_successes/self.mcbar_attempts)
    #         if self.mcbar_successes >= 0.75 * self.mcbar_attempts:
    #             self.dV_max *= self.dV_scale
    #             print("MC BAR INCREASE DVMAX",self.dV_max,self.mcbar_attempts,self.mcbar_successes)
    #             self.mcbar_attempts = 0
    #             self.mcbar_successes = 0
    #         elif self.mcbar_successes <= 0.25 * self.mcbar_attempts:
    #             self.dV_max /= self.dV_scale
    #             print("MC BAR DECREASE DVMAX:",self.dV_max,self.mcbar_attempts,self.mcbar_successes)
    #             self.mcbar_attempts = 0
    #             self.mcbar_successes = 0
