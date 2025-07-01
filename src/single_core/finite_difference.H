#ifndef FINITE_DIFFERENCE_H
#define FINITE_DIFFERENCE_H

#include <vector>
#include <fstream>
#include <cmath>
#include <iostream>
#include <array>

void timestep(const std::vector<double>& grid_old,
              std::vector<double>& grid_new, 
              const std::array<int, 2> dims,
              const double dx, const double dy, const double dt, 
              const double alpha, const int halo = 1);

void boundary_condition_periodic(std::vector<double>& grid, 
                                 const std::array<int, 2> dims, const int halo = 1);

#endif