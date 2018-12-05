#ifndef __NEUROCHEMCPP_IFACE__
#define __NEUROCHEMCPP_IFACE__

#include <list>
#include <vector>
#include <string>

class moleculecpp;

namespace neurochem {

bool setup_molecule_cpp = true;
std::list<moleculecpp*> molecule_instances;

extern void instantiate_ani_ensemble(const std::string &cnst, // Constants file
                                     const std::string &saef, // Linear fit energy
                                     const std::string &prfx, // Network directory prefix
                                     const unsigned num_nets,
                                     const unsigned gpu_idx=0);

extern double compute_ensemble_energy(const std::vector<float> &coordinates,
                                      const std::vector<std::string> &elements);

extern std::vector<float> compute_ensemble_force(const unsigned num_atoms);
}
#endif
