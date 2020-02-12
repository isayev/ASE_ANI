#include <vector>
#include <string>
#include <iostream>

#include "neurochemcpp_iface.h"

int main (int argc, char *argv[]) {
    std::cout << "Testing NeuroChem cpp interface!" << std::endl;

    // Network files
    std::string neuralnetfile = std::string("/home/jujuman/Gits/ASE_ANI/ani_models/ani-1ccx_8x.info");

    // Instantiate model
    std::cout << "Instantiating ensemble..." << std::endl;
    neurochem::instantiate_ani_ensemble(neuralnetfile);

    // Setup System
    std::vector<double> coords = {0.000000,  0.000000,  0.000000,
                                 1.000000,  0.000000,  0.000000,
                                 2.000000,  0.0000070, -0.000000,
                                 };

    std::vector<std::string> elements = {"O","C","N"};

    //unsigned inum = 3;
    //int ilist [] = { 0,1,2 };
    //int numneigh [] = { 1, 2, 1 };
    //int firstneigh[3][4] = {
    //    {1} ,
    //    {0, 2} ,
    //    {2}
    //    };

    /*std::vector<double> coords = {};
    std::vector<std::string> elements = {};*/

    // Example PBC usage -- set the periodic cell (use angstroms)
    //=================Code below=================
    /*std::vector<float> cell = {10.000000,  0.000000,  0.000000,
                                0.000000, 10.000000,  0.000000,
                                0.000000,  0.000000, 10.000000};
    neurochem::set_cell(cell, true, true, true);*/
    //=================Code above=================

    std::cout << "Radial Cutoff: " << std::endl;
    std::cout << neurochem::get_radial_cutoff() << std::endl;

    std::cout << "Computing energies..." << std::endl;
    // Compute energies for the ensemble
    double energy = neurochem::compute_ensemble_energy(coords, elements, 4);

    //std::vector<double> aeenergies = neurochem::compute_ensemble_energy_lammps(inum,&ilist,&numneigh,&firstneigh,coords,elements);

    //std::cout << "Compute atomic energies..." << std::endl;
    std::vector<double> aeenergies = neurochem::get_atomic_energies();

    std::cout << "Computing forces..." << std::endl;
    // Compute forces for the ensemble
    std::vector<double> forces = neurochem::compute_ensemble_force();


    std::cout << "Computing virials..." << std::endl;
    // Compute forces for the ensemble
    //std::vector<double> virials = neurochem::compute_ensemble_atomic_virial();

    // Output results
    std::cout.precision(16);
    //std::cout << "Ensemble energy: " << energy << std::endl;

    std::cout.precision(7);
    for (unsigned i = 0; i < aeenergies.size(); ++i) {
        std::cout << "ATOM(" << i << ")="<< aeenergies[i] << std::endl;
    }

    for (unsigned i = 0; i < elements.size(); ++i) {
        std::cout << i << ") " << forces[3*i+0] << " "  << forces[3*i+1] << " " << forces[3*i+2] << std::endl;
    }

    //for (unsigned i = 0; i < elements.size()-6; ++i) {
    //    std::cout << "Virial atom(" << i << ")" << std::endl;
    //    std::cout << virials[9*i+0] << " "  << virials[9*i+1] << " " << virials[9*i+2] << std::endl;
    //    std::cout << virials[9*i+3] << " "  << virials[9*i+4] << " " << virials[9*i+5] << std::endl;
    //    std::cout << virials[9*i+6] << " "  << virials[9*i+7] << " " << virials[9*i+8] << std::endl;
    //    std::cout << std::endl;
    //}

    std::cout << "Shutdown." << std::endl;
    // Cleanup instances (required to shut the classes down before the driver shuts down)
    neurochem::cleanup_ani();
}
