from ase.optimize.optimize import Optimizer
import numpy as np
import numpy.linalg as la
from hessian import get_hessian
from ase.io.trajectory import Trajectory


def vnorm(vector):
    """normalized vector
    """
    return vector / la.norm(vector)


def calc_angle(a, b, c):
    ba = a.flatten() - b.flatten()
    bc = c.flatten() - b.flatten()
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle


class IRC(Optimizer):
    class IRCError(Exception):
        pass

    _dump_attr = ('stride', 'mw', 'M', 'Mi', 'sv', 'forward', 'H', 'S')

    def __init__(self, atoms, stride=0.1, hessian=None, search_vector=None, forward=True, mw=True,
                 restart=None, logfile='-', trajectory=None, master=None):
        """IRC with Gonsalez-Schlegel method
        
        Parameters:
        
        atoms: Atoms object
        
        stride: step radius
        
        hessian: Hessian matrix in cartesian coordinates. Will be calculates if not given, 
                 or search_vector will be used instead. 
                 Will use first eigenvector of hessian, should be negative  
             
        search_vector: initial step direction, in cartesian coordinates
        
        forward: positive or negative direction along search vector
        
        mw: do search in mass-weighted (IRC) or cartesian (MEP) coordinates
        """

        self.stride = stride
        self.H = hessian
        self.sv = search_vector
        self.forward = forward
        self.maxstep = 0.3

        self.S = None
        self._s_prev = None
        self._g_prev = None
        self.is_converged = False

        # mass weighing
        if mw:
            self.M = np.sqrt(atoms.get_masses()).reshape(-1, 1)
        else:
            self.M = np.ones(len(atoms)).reshape(-1, 1)
        self.Mi = 1 / self.M

        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)

        self.trajectory = Trajectory(trajectory, 'w')


    def initialize(self):
        # path taken so far
        self.S = 0
        # search vector
        if self.sv is not None:
            assert len(self.sv.flat) == 3 * len(self.atoms), "search_vector is in bad shape"
            self.sv = self.sv * (1 if self.forward else -1) * self.M.repeat(3)
        else:
            self.H = self.H or get_hessian(self.atoms, mass_weighted=False)
            v = self.Mi.repeat(3).reshape(-1, 1)
            self.H *= v.dot(v.T)
            assert self.H.shape == (3 * len(self.atoms), 3 * len(self.atoms)), "hessian is in bad shape"
            eval, evec = la.eigh(self.H)
            assert (eval[0] < -0.1).all(), "Hessian doesn't have imaginary eigenvalues!"
            self.sv = evec[:, 0] * (1 if self.forward else -1)
        self.sv = vnorm(self.sv.reshape(-1, 3))

        # TODO: will use unit initial hessian until figure out how to project out rotations
        self.H = 70 * np.eye(3 * len(self.atoms))
        v = self.Mi.repeat(3).reshape(-1, 1)
        self.H *= v.dot(v.T)

    def todoct(self):
        d = Optimizer.todict()
        for attr in self._dump_attr:
            if hasattr(self, attr):
                d.update(attr, getattr(self, attr))
        return d

    def read(self):
        data = self.load()
        for attr in self._dump_attr:
            setattr(self, data[attr])

    def run(self, fmax=0.05, steps=1000, maxmicro=1000, yield_path=True):
        self.fmax = fmax
        step = 0
        pass_high_grad = False
        e0 = self.atoms.get_potential_energy()
        while step < steps:
            p0 = self.atoms.get_positions() * self.M
            p1 = self.do_step_forward(self.stride)
            conv = self.do_sg2_search(p1, maxmicro)
            if not conv:
                raise self.IRCError("SG2 iterations not converged! Terminating.")
            e1 = self.atoms.get_potential_energy()
            if e1 > e0:
                self.logfile.write(
                    'Energy is rising. Probably we just passed though a minimum. \n' +
                    'Signaling convergence now without storing last geometry!\n')
                self.is_converged = True
                break
            self.S += self.calc_path(p0, p1, self.atoms.get_positions() * self.M)
            self.trajectory.write(atoms)
            f = self.atoms.get_forces()
            angle = calc_angle(p0, p1, self.atoms.get_positions() * self.M)
            self.logfile.write('IRC: step {:3d} path {:.4f} energy {:.5f} unconstrained force {:5f} angle {:.2f}\n'.format(
                step, self.S, e1, la.norm(f), angle
            ))
            if yield_path:
                yield self.S, e1, la.norm(f), angle
            if not self.converged(f * 3):
                pass_high_grad = True
            elif pass_high_grad and self.converged(f / 3):
                self.logfile.write('Total gradient is below threshold. Signaling convergence now!\n')
                self.is_converged = True
                break

            e0 = e1
            self.sv = vnorm(self.atoms.get_positions() * self.M - p1)

            step += 1

    def do_step_forward(self, stride):
        """Initial guess for next IRC point
        Return circle center
        """
        p0 = self.atoms.get_positions() * self.M
        p1 = p0 + stride * self.sv
        p2 = p1 + stride * self.sv
        self.atoms.set_positions(p2 * self.Mi)
        return p1

    def do_sg2_search(self, c, maxstep):
        step = 0
        converged = False
        while step < maxstep:
            p = self.atoms.get_positions() * self.M
            r = p - c
            rn = vnorm(r)
            # calculate force
            f = self.atoms.get_forces() * self.Mi
            # update hessian
            # TODO: Something wrong with H update. skip for now
            # self.h_update_dfp(p, ft)
            # self.h_update_dfp(p, f)
            # tangent force
            ft = f - rn * np.vdot(f.reshape(-1, 1), rn.reshape(-1, 1))
            if self.converged(ft):
                converged = True
                break
            # scaled unconstrained step
            x = self.determine_step(self.nr_step(f) * self.Mi, self.stride / 9.0) * self.M
            # constrained displacement
            x = la.norm(r) * vnorm(r + x) - r

            # update coords
            self.atoms.set_positions((p + x) * self.Mi)

            self.logfile.write('SG2 step {:3d}: force {:.5f} displ {:.5f} angle {:.3f} energy {:.5f}\n'.format(
                step, np.sqrt((ft**2).sum(axis=1)).max(), np.sqrt((x**2).sum(axis=1)).max(),
                2 * np.rad2deg(np.arcsin(la.norm(x*0.5) / la.norm(r))),
                self.atoms.get_potential_energy()
            ))

            step += 1
            # self.trajectory.write(self.atoms)

        self.logfile.write('SG2 converged! force: {:.5f} energy {:.5f}\n'.format(
            np.sqrt((ft ** 2).sum(axis=1)).max(),  self.atoms.get_potential_energy()
        ))
        return converged

    @staticmethod
    def calc_path(a, b, c):
        theta = np.pi - calc_angle(a, b, c)
        chord = la.norm(a - c)
        r = chord / 2 / np.sin(theta / 2)
        s = theta * r
        return s

    def h_update_bfgs(self, positions, force):
        """ Do Hessian update with BFGS formula
        """
        positions = positions.reshape(-1, 1)
        grad = - force.reshape(-1, 1)
        if self._s_prev is not None and self._g_prev is not None:
            s = positions - self._s_prev
            g = grad - self._g_prev
            if (s < 1e-4).all() or (g < 1e-5).all():
                return None
            a = np.vdot(s, g)
            v = np.dot(self.H, s)
            b = np.dot(s.T, v)
            self.H += np.dot(g, g.T) / a - np.dot(v, v.T) / b
        self._s_prev = positions
        self._g_prev = grad

    def h_update_dfp(self, positions, force):
        """ Do Hessian update with DFP formula
        """
        positions = positions.reshape(-1, 1)
        grad = - force.reshape(-1, 1)
        if self._s_prev is not None and self._g_prev is not None:
            s = positions - self._s_prev
            g = grad - self._g_prev
            if (s < 1e-4).all() or (g < 1e-5).all():
                return None
            self.H += - self.H.dot(g).dot(g.T).dot(self.H) / g.T.dot(self.H).dot(g) + s.dot(s.T) / g.T.dot(s)
        self._s_prev = positions
        self._g_prev = grad

    def nr_step(self, f):
        f = f.flatten()
        val, vec = la.eigh(self.H)
        return np.dot(vec, np.dot(f, vec) / np.abs(val)).reshape(-1, 3)

    def determine_step(self, dr, maxstep):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        maxsteplength = np.sqrt((dr * dr).sum(axis=1)).max()
        if maxsteplength >= maxstep:
            dr *= maxstep / maxsteplength

        return dr

if __name__ == '__main__':
    import os
    import numpy as np
    import numpy.linalg as la
    from ase.calculators import mopac
    from ase import Atoms
    import pybel

    os.chdir('../test')

    MOPAC = os.path.join(os.getcwd(), 'MOPAC')
    os.environ['MOPAC_LICENSE'] = MOPAC
    os.environ['LD_LIBRARY_PATH'] = MOPAC

    mopac_calc = mopac.MOPAC()
    mopac_calc.command = 'MOPAC/MOPAC2016.exe PREFIX.mop 2> /dev/null'
    mopac_calc.set(method='pm3')

    mol = next(pybel.readfile('xyz', 'ts2.xyz'))
    atoms = Atoms(numbers=[a.atomicnum for a in mol.atoms],
                  positions=[a.coords for a in mol.atoms])
    atoms.set_positions(atoms.positions - atoms.get_center_of_mass())
    atoms.set_calculator(mopac_calc)

    irccalc = IRC(atoms, stride=0.15, mw=True, forward=True, trajectory='ts1.traj')
    irccalc.run(yield_path=False)

