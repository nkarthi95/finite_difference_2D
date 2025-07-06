#ifndef INITIAL_CONDITIONS_H
#define INITIAL_CONDITIONS_H

#include "finite_difference.H"
#include "misc_functions.H"

void init_hot_center(std::vector<double>& grid,
                     const std::array<int, 2> dims,
                     double hot = 100., double cold = 50., 
                     double radius = 5){
    int nx = dims[0]; // includes halo
    int ny = dims[1]; // includes halo
    int index_1d;
    bool check;

    #ifdef USE_OMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            index_1d = idx_2d_to_1d(i, j, ny);
            check = pow(i - nx/2., 2) + pow(j - ny/2., 2) < pow(radius, 2);
            
            if (check){
                grid[index_1d] = hot;
            }
            else{
                grid[index_1d] = cold;
            }
        }
    }
}

void init_vertical_hot_cylinder(std::vector<double>& grid, 
                                const std::array<int, 2> dims,
                                double hot = 100., double cold = 50., 
                                double radius = 10, int height = 0){
    int nx = dims[0]; // includes halo
    int ny = dims[1]; // includes halo
    int index_1d;
    bool check;

    #ifdef USE_OMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            index_1d = idx_2d_to_1d(i, j, ny);
            check = abs(j - height) <= radius;

            if (check){
                grid[index_1d] = hot;
            }
            else{
                grid[index_1d] = cold;
            }
        }
    }
}

void init_horizontal_hot_cylinder(std::vector<double>& grid, 
                                  const std::array<int, 2> dims,
                                  double hot = 100., double cold = 50., 
                                  double radius = 10, int height = 0){
    int nx = dims[0]; // includes halo
    int ny = dims[1]; // includes halo
    int index_1d;
    bool check;

    #ifdef USE_OMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            index_1d = idx_2d_to_1d(i, j, ny);
            check = abs(i - height) <= radius;

            if (check){
                grid[index_1d] = hot;
            }
            else{
                grid[index_1d] = cold;
            }
        }
    }
}

void init_hot_square(std::vector<double>& grid, 
                     const std::array<int, 2> dims,
                     double hot = 100., double cold = 50., 
                     int Lx = 10, int Ly = 10){

    int nx = dims[0]; // includes halo
    int ny = dims[1]; // includes halo
    int index_1d;
    bool checkx, checky;
    
    #ifdef USE_OMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            checkx = abs(i - nx/2) < Lx/2;
            checky = abs(j - ny/2) < Ly/2;
            index_1d = idx_2d_to_1d(i, j, ny);

            if (checkx && checky){grid[index_1d] = hot;}
            else{grid[index_1d] = cold;}
        }
    }
}


#endif