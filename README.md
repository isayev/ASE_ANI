# ASE-ANI

### NOTICE: Python binaries built for python 3.6 and CUDA 9
### Works only under Ubuntu variants of Linux 

This is a prototype interface for ANI-1 neural net potential for The Atomic Simulation Environment (ASE). Current ANI-1.1 potential implements CHNO elements and experimental support for S and F.

##REQUIREMENTS:
* Python 3.6 (we recommend [Anaconda](https://www.continuum.io/downloads) distribution)
* [CUDA 9.0](https://developer.nvidia.com/cuda-downloads)
* [ASE](https://wiki.fysik.dtu.dk/ase/index.html)
* Modified [ased3](https://github.com/isayev/ased3) for D3 van der Waals correction (Optional) 
* MOPAC2012 or MOPAC2016 for some examples to compare results (Optional) 

## Installation
Clone this repository into desired folder and add environmental variables from `bashrc_example.sh` to your `.bashrc`. 

For use cases please refer to examples folder with several iPython notebooks

## Citation
If you use this code, please cite:

Justin S. Smith, Olexandr Isayev, Adrian E. Roitberg. *ANI-1: An extensible neural network potential with DFT accuracy at force field computational cost*. Chemical Science, 2017, DOI: [10.1039/C6SC05720A](http://pubs.rsc.org/en/content/articlelanding/2017/sc/c6sc05720a)

