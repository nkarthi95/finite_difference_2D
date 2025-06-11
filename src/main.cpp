#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

#include "finite_difference.H"
#include "initial_conditions.H"

std::string zero_pad(int number, int width) {
    std::ostringstream oss;
    oss << std::setw(width) << std::setfill('0') << number;
    return oss.str();
}

void write_to_file(const std::string& filename, std::vector<std::vector<double>> grid){
    std::ofstream fout(filename);

    const int nx = grid.size(); // includes halo
    const int ny = grid[0].size(); // includes halo

    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            fout << grid[i][j] << ", ";
        }
        fout << "\n";
    }
    fout.close();
}

int main(){
    int nx = 256;
    int ny = 256;
    int halo = 2;

    double dx = 0.01;
    double dy = 0.01;
    double dt = 0.0001;
    double timesteps = 1000;
    int dump_freq = 1000;

    double alpha = 0.1;
    double T_hot = 373.0;
    double T_cold = 273.0;

    std::vector<std::vector<double>> grid_old(nx+2*halo, std::vector<double>(ny+2*halo, 0.0));
    std::vector<std::vector<double>> grid_new(nx+2*halo, std::vector<double>(ny+2*halo, 0.0));


    // grid_old = init_hot_center(grid_old, T_hot, T_cold, 32);
    grid_old = init_vertical_hot_cylinder(grid_old, T_hot, T_cold, 32, nx/2);

    for(int t = 0; t <= timesteps; t++){
        grid_new = timestep(grid_old, dx, dy, dt, alpha);
        grid_new = boundary_condition_periodic(grid_new);
        std::swap(grid_new, grid_old);
        if (t%dump_freq == 0){
            const std::string filename = "T_" + zero_pad(t, 6) + ".txt";
            write_to_file(filename, grid_old);
        }
    }

    return 0;
}