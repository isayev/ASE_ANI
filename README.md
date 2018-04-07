# ASE-ANI

### NOTICE: Python binaries built for python 3.6 and CUDA 9
### Works only under Ubuntu variants of Linux 

This is a prototype interface for ANI-1x neural network potential for The Atomic Simulation Environment (ASE). Current ANI-1x potential implements CHNO elements.

##REQUIREMENTS:
* Python 3.6 (we recommend [Anaconda](https://www.continuum.io/downloads) distribution)
* [CUDA 9.0](https://developer.nvidia.com/cuda-downloads)
* [ASE](https://wiki.fysik.dtu.dk/ase/index.html)
* Modified [ased3](https://github.com/isayev/ased3) for D3 van der Waals correction (Optional) 
* MOPAC2012 or MOPAC2016 for some examples to compare results (Optional) 

## Installation
Clone this repository into desired folder and add environmental variables from `bashrc_example.sh` to your `.bashrc`. 

For use cases please refer to examples folder with several iPython notebooks

## Cool stuff
### Teaser of the new ANI-2x (CHNOSFCl) potential in action! 
<a href="https://www.youtube.com/watch?v=37Ba9hxEnHI" target="_blank"><img src="http://img.youtube.com/vi/37Ba9hxEnHI/0.jpg" 
alt="MD simulation of Protein-ligand complex with deep learning potential ANI-1x" width="240" height="180" border="10" /></a>

### ANI-1x running 5ns MD on a box of C<sub>2</sub> at high temperature.
<a href="https://www.youtube.com/watch?v=DRVMH5u8EA0" target="_blank"><img src="http://img.youtube.com/vi/DRVMH5u8EA0/0.jpg" 
alt="Nucleation of carbon nanoparticles from hot vapor simulation with ANI-1 deep learning potential" width="240" height="180" border="10" /></a>

## ANI-1 dataset
https://github.com/isayev/ANI1_dataset

## COMP6 benchmark
https://github.com/isayev/COMP6

## Citation
If you use this code, please cite:

### ML Potential Method:
Justin S. Smith, Olexandr Isayev, Adrian E. Roitberg. *ANI-1: An extensible neural network potential with DFT accuracy at force field computational cost*. Chemical Science, 2017, DOI: [10.1039/C6SC05720A](http://pubs.rsc.org/en/content/articlelanding/2017/sc/c6sc05720a)

### Original data:
Justin S. Smith, Olexandr Isayev, Adrian E. Roitberg. ANI-1, A data set of 20 million calculated off-equilibrium conformations for organic molecules. Scientific Data, 4, Article number: 170193, DOI: 10.1038/sdata.2017.193 https://www.nature.com/articles/sdata2017193

### Active-learning based data:
Justin S. Smith, Ben Nebgen, Nicholas Lubbers, Olexandr Isayev, Adrian E. Roitberg. *Less is more: sampling chemical space with active learning*. arXiv, 2018, DOI: [arXiv:1801.09319] (https://arxiv.org/abs/1801.09319)

