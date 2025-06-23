#ifndef FINITE_DIFFERENCE_H
#define FINITE_DIFFERENCE_H

#include <vector>
#include <fstream>
#include <cmath>
#include <iostream>

std::vector<std::vector<double>> timestep(std::vector<std::vector<double>> grid, 
                                          double dx, double dy, double dt, 
                                          double alpha, int halo = 1);

std::vector<std::vector<double>> boundary_condition_periodic(std::vector<std::vector<double>> grid, 
                                                             int halo = 1);

#endif