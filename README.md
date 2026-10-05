#  Parameter Estimation for Model-Based Sensing of Magneto-Mechanical Resonators
This repository contains code for estimating the required parameters to model the dynamics of magneto-mechanical resonators (MMRs). The torsion model is the focus of this code example.

The method corresponding to this code is described in the associated publication (see below).

## Installation
In order to use this code, one first has to download [Julia](https://julialang.org/) (version 1.11 or later) and clone this repository.

Download the data from [here](https://doi.org/10.15480/882.16742) and place it in the `data` directory.
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
```bibtex
@article{reiss_parameter_2026,
	title = {Parameter {Estimation} for {Model}-{Based} {Sensing} of {Magneto}-{Mechanical} {Resonators}},
	volume = {9},
	issn = {2399-3650},
	doi = {10.1038/s42005-026-02884-1},
	journal = {Communications Physics},
	author = {Reiss, Sarah and Knopp, Tobias and Ackers, Justin and Faltinath, Jonas and Mohn, Fabian and Boberg, Marija and Timm, Nora and Möddel, Martin},
	month = oct,
	year = {2026},
}

```
