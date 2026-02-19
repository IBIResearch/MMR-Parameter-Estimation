#  Parameter Estimation for Model-Based Sensing of Magneto-Mechanical Resonators
This repository contains code for estimating the required parameters to model the dynamics of magneto-mechanical resonators (MMRs). The torsion model is the focus of this code example.

The method corresponding to this code is described in the associated publication.

*To be announced.*

## Installation
In order to use this code, one first has to download [Julia](https://julialang.org/) (version 1.11 or later) and clone this repository.

Download the data from the MISSING and place it in the `data` directory.
You should end up with the following structure:
```
.
├── Experiment1
│   ├── MMRS
│   ├── MMRL
├── Experiment2
│   ├── MMRS
│   ├── MMRL
```

## Execution
After installation the example code can be executed by navigating to the folder, running `julia` and entering
```
include("example.jl")
```
to estimate the parameters and reconstruct the measured signal. The example script automatically activates the environment and installs all necessary packages. This will take several minutes before the actual code is run, since all packages are precompiled during installation.

## Citation
If you use this code in your research, please cite the following paper:

*To be announced.*
