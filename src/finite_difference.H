#ifndef FINITE_DIFFERENCE_H
#define FINITE_DIFFERENCE_H

#include <vector>
#include <fstream>
#include <cmath>
#include <iostream>

void timestep(std::vector<std::vector<double>>& grid_old,
              std::vector<std::vector<double>>& grid_new, 
              double dx, double dy, double dt, 
              double alpha, int halo = 1);

void boundary_condition_periodic(std::vector<std::vector<double>>& grid, 
                                 int halo = 1);

#endif