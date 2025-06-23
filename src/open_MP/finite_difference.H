#ifndef FINITE_DIFFERENCE_H
#define FINITE_DIFFERENCE_H

#include <vector>
#include <fstream>
#include <cmath>
#include <iostream>

void timestep(const std::vector<std::vector<double>>& grid_old,
              std::vector<std::vector<double>>& grid_new, 
              const double dx, const double dy, const double dt, 
              const double alpha, const int halo = 1);

void boundary_condition_periodic(std::vector<std::vector<double>>& grid, 
                                 const int halo = 1);

#endif