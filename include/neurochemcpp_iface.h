#ifndef __NEUROCHEMCPP_IFACE__
#define __NEUROCHEMCPP_IFACE__

#include <vector>
#include <string>

// source version: /projects/ml4chem/Programs/NeuroChem/src-hdatomnnp/neurochemcpp_iface.h

namespace neurochem {

extern void instantiate_ani_ensemble(const std::string &nnf,
                                     const unsigned gpu_idx=0);

extern void set_cell(const std::vector<double> &cell,
                     const bool x,const bool y,const bool z);

extern double compute_ensemble_energy(const std::vector<double> &coordinates,
                                      const std::vector<std::string> &elements,
                                      const unsigned num_ghost=0);

// Double overload
extern std::vector<double> compute_ensemble_energy_lammps(const unsigned inum,
                                                          const int *ilist,
                                                          const int *numneigh,
                                                          const int **firstneigh,
                                                          const std::vector<double> &coordinates,
                                                          const std::vector<std::string> &elements,
                                                          bool sae_shift = true);

extern std::vector<double> get_atomic_energies(const bool sae = true);

extern std::vector<double> compute_ensemble_force();

extern std::vector<double> compute_ensemble_atomic_virial();

extern double get_radial_cutoff();

extern void cleanup_ani();
}
#endif
