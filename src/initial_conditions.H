#ifndef INITIAL_CONDITIONS_H
#define INITIAL_CONDITIONS_H

#include "finite_difference.H"

std::vector<std::vector<double>> init_hot_center(std::vector<std::vector<double>> grid, 
                                                 double hot = 100., double cold = 50., 
                                                 double radius = 5){
    const int nx = grid.size(); // includes halo
    const int ny = grid[0].size(); // includes halo

    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            double test = pow(i - nx/2., 2) + pow(j - ny/2., 2);
            if (test < pow(radius, 2)){
                grid[i][j] = hot;
            }
            else{
                grid[i][j] = cold;
            }
        }
    }
    return grid;
}

std::vector<std::vector<double>> init_vertical_hot_cylinder(std::vector<std::vector<double>> grid, 
                                                 double hot = 100., double cold = 50., 
                                                 double radius = 10, int height = 0){
    const int nx = grid.size(); // includes halo
    const int ny = grid[0].size(); // includes halo

    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            // double height_diff = j - height;
            if (abs(j - height) <= radius){
                grid[i][j] = hot;
            }
            else{
                grid[i][j] = cold;
            }
        }
    }
    return grid;
}

std::vector<std::vector<double>> init_horizontal_hot_cylinder(std::vector<std::vector<double>> grid, 
                                                 double hot = 100., double cold = 50., 
                                                 double radius = 10, int height = 0){
    const int nx = grid.size(); // includes halo
    const int ny = grid[0].size(); // includes halo

    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            // double height_diff = j - height;
            if (abs(i - height) <= radius){
                grid[i][j] = hot;
            }
            else{
                grid[i][j] = cold;
            }
        }
    }
    return grid;
}

#endif