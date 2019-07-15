#ifndef __NEUROCHEMCPP_IFACE__
#define __NEUROCHEMCPP_IFACE__

#include <vector>
#include <string>

// source version: /projects/ml4chem/Programs/NeuroChem/src-hdatomnnp/neurochemcpp_iface.h

namespace neurochem {

/*extern void instantiate_ani_ensemble(const std::string &cnst, // Constants file
                                     const std::string &saef, // Linear fit energy
                                     const std::string &prfx, // Network directory prefix
                                     const unsigned num_nets,
                                     const unsigned gpu_idx=0);*/


extern void instantiate_ani_ensemble(const std::string &nnf,
                                     const unsigned gpu_idx=0);

extern void set_cell(const std::vector<float> &cell,
                     const bool x,const bool y,const bool z);


extern double compute_ensemble_energy(const std::vector<float> &coordinates,
                                      const std::vector<std::string> &elements,
                                      const unsigned num_ghost=0);

extern std::vector<float> get_atomic_energies(const bool sae = true);

extern std::vector<float> compute_ensemble_force();

extern std::vector<float> compute_ensemble_atomic_virial();

extern float get_radial_cutoff();

extern void cleanup_ani();
}
#endif
