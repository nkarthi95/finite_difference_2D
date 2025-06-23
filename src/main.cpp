#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>  // for std::runtime_error
#include <algorithm>  // for std::min

#include "finite_difference.H"
#include "initial_conditions.H"

void check_CFL(double alpha, double dx, double dy, double dt) {
    double min_dxdy = std::min(dx, dy);
    double C = alpha*dt/(min_dxdy*min_dxdy); // CFL condition

    if (C > 0.1) {
        throw std::runtime_error("CFL condition not fulfilled: C = " + std::to_string(C));
    }
}

std::string zero_pad(int number, int width) {
    std::ostringstream oss;
    oss << std::setw(width) << std::setfill('0') << number;
    return oss.str();
}

void write_to_file(const std::string& filename, std::vector<std::vector<double>> grid, int halo){
    std::ofstream fout(filename);

    const int nx = grid.size(); // includes halo
    const int ny = grid[0].size(); // includes halo

    for (int i = halo; i < nx - halo; i++){
        for (int j = halo; j < ny - halo; j++){
            fout << grid[i][j] << ", ";
        }
        fout << "\n";
    }
    fout.close();
}

int main(){
    int nx = 512;
    int ny = 512;
    int halo = 1;

    double dx = 1/double(nx);
    double dy = 1/double(ny);
    double dt = 0.0001;
    double timesteps = 10000;
    int dump_freq = 100;

    double alpha = 0.002;
    double T_hot = 373.0;
    double T_cold = 273.0;

    check_CFL(alpha, dx, dy, dt);

    std::vector<std::vector<double>> grid_old(nx+2*halo, std::vector<double>(ny+2*halo, 0.0));
    std::vector<std::vector<double>> grid_new(nx+2*halo, std::vector<double>(ny+2*halo, 0.0));

    // grid_old = init_hot_center(grid_old, T_hot, T_cold, 32);
    // grid_old = init_vertical_hot_cylinder(grid_old, T_hot, T_cold, 32, nx/2);
    grid_old = init_hot_square(grid_old, T_hot, T_cold, nx/4, ny/4);

    for(int t = 0; t <= timesteps; t++){
        grid_old = boundary_condition_periodic(grid_old, halo);

        if (t%dump_freq == 0){
            const std::string filename = "T_" + zero_pad(t, 6) + ".txt";
            write_to_file(filename, grid_old, halo);
        }

        grid_new = timestep(grid_old, dx, dy, dt, alpha, halo);
        std::swap(grid_new, grid_old);
    }

    return 0;
}