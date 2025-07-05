# 2D Heat Equation Solver 

This repository contains implementations of a numerical solver for the 2D transient heat equation, implemented using the finite difference method (FDM). Depending on which src folder you use, it can support single-core, OpenMP multithreaded, and MPI-based multicore execution.

## Build Instructions

Requires:
1. C++17
2. MPI (e.g., MPICH or OpenMPI)
3. OpenMP-compatible compiler (e.g., `g++`, `clang++`)

### Generating venv for python environment

I used windows WSL for this exercise and used the following instructions to create a python virtual environment

```
python -m venv {environment name}
source venv/bin/activate
pip install -r requirements.txt
```

I then selected the environment that has the name used ({environment name}) that the above instructions generated 
in a jupyter notebook instance when opening `analysis.ipynb` in VSCode.

### Building C++ code

```
src
├── MPI
    ├── *.cpp
    ├── *.H
    ├── Makefile
    ├── T_*.txt (output files)
├── open_MP
    ├── *.cpp
    ├── *.H
    ├── Makefile
    ├── T_*.txt (output files)
├── single_core
    ├── *.cpp
    ├── *.H
    ├── Makefile
    ├── T_*.txt (output files)
```

```bash
cd src/MPI # change this to whichever folder for the executable you want to build
make clean 
make
```

## Running the Solver

```bash
# Single-core run
./heat_solver

# Multi-core MPI run
mpirun -n 2 heat_solver
```

Built in check for whether CFL condition is met. If CFL < 0.1, the run will continue, else
the simulation will break and the value of C will be printed. 

## Input/Output

### Input

You can change the parameters of the system within the `main` function in `main.cpp` in the area demarcated
by comments `// INPUT PARAMETERS` stating what are input parameters a user can change. An example is shown below

```
// INPUT PARAMETERS
const int threads = 2; // only used in openMP implementation
int nx = 1024; // size of field in x direction
int ny = 1024; // size of field in y direction

double dt = 0.000025; // timestep length 
double timesteps = 10000; // number of timesteps
int dump_freq = 10000; // Timesteps between data dump

double alpha = 0.002; // Thermal diffusivity of system
double T_hot = 373.0; // Temperature in K of the hot point
double T_cold = 273.0; // Temperature in K of the cold point
// INPUT PARAMETERS
```

### Output

1. Temperature fields dumped every `dump_freq` steps to `T_XXXXXX.txt`
2. Reading is performed in `analysis.ipynb` for plotting (see example below)
    1. `src.FD_helper.visualize.read_data(path)`


## Contributions

Contributions, improvements, and performance tips are welcome! Feel free to open issues or submit pull requests.