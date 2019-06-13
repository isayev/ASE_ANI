#ifndef __NEUROCHEMCPP_IFACE__
#define __NEUROCHEMCPP_IFACE__

#include <list>
#include <vector>
#include <string>
#include <cuda.h>

class moleculecpp;

namespace neurochem {

CUcontext context;
bool setup_molecule_cpp = true;
std::list<moleculecpp*> molecule_instances;

extern void instantiate_ani_ensemble(const std::string &cnst, // Constants file
                                     const std::string &saef, // Linear fit energy
                                     const std::string &prfx, // Network directory prefix
                                     const unsigned num_nets,
                                     const unsigned gpu_idx=0);

extern void set_cell(const std::vector<float> &cell,
                     const bool x,const bool y,const bool z);


extern double compute_ensemble_energy(const std::vector<float> &coordinates,
                                      const std::vector<std::string> &elements,
                                      const unsigned num_ghost=0);

extern std::vector<float> compute_ensemble_force(const unsigned num_atoms);


extern void cleanup_ani();
}
#endif
