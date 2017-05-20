# ASE-ANI

### NOTICE: Python binaries built for python 3.5 and CUDA 8
### Works only under linux 

This is a prototype interface for ANI-1 neural net potential for The Atomic Simulation Environment (ASE) 

##REQUIREMENTS:
* Python 3.5 (we recommend [Anaconda](https://www.continuum.io/downloads) distribution)
* [CUDA 8.0](https://developer.nvidia.com/cuda-downloads)
* [ASE](https://wiki.fysik.dtu.dk/ase/index.html)
* Modified [ased3](https://github.com/isayev/ased3) for D3 van der Waals correction (Optional) 
* MOPAC2012 or MOPAC2016 for some examples to compare results (Optional) 

## Installation
Clone this repository into desired folder and add environmental variables from `bashrc_example.sh` to your `.bashrc`. 

For use cases please refer to examples folder with several iPython notebooks

## Citation
If you use this code, please cite:

Justin S. Smith, Olexandr Isayev, Adrian E. Roitberg. *ANI-1: An extensible neural network potential with DFT accuracy at force field computational cost*. Chemical Science, 2017, DOI: 10.1039/C6SC05720A 

