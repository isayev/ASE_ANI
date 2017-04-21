
import numpy as np
from itertools import product
from ase.units import Hartree, Bohr
import scipy.linalg as la


def vnorm(vector):
    """normalized vector
    """
    return vector / la.norm(vector)


def get_hessian(atoms, delta=0.005, mass_weighted=False):
    """
    Calculate (mass weighted) hessian using central diff formula.
    :param atoms: atoms object with defined calculator
    :param delta: step size for numeric differentiation
    :type atoms: ase.Atoms
    :type delta: float
    :return: numpy square symmetric array
    """
    # convert delta to Angs
    delta *= Bohr
    # allocate matrix
    l = len(atoms)
    H = np.zeros((3 * l, 3 * l), dtype=np.float64)
    r = 0
    # gradients matrix
    for i, j in product(range(l), range(3)):
        g = np.zeros((l, 3))
        for k in (-1, 1):
            atoms1 = atoms.copy()
            atoms1[i].position[j] += k * delta
            atoms1.set_calculator(atoms.get_calculator())
            g += - k * atoms1.get_forces()

        H[r] = 0.5 * g.flatten()
        r += 1
    # check symmetry assuming gradients computed with 10^-3 Hartree/Bohr precision
    gprec = 0.001 * Hartree
    assert np.max(np.abs(H - H.T)) < gprec, np.max(np.abs(H - H.T))
    # Hessian
    H /= delta
    # symmetrize
    H = 0.5 * (H + H.T)
    # mass weight
    if mass_weighted:
        v = np.sqrt(atoms.get_masses()).repeat(3).reshape(-1, 1)
        H /= np.dot(v, v.T)
    return H



