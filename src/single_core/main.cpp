#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>

#include "finite_difference.H"
#include "initial_conditions.H"
#include "misc_functions.H"

std::string zero_pad(int number, int width) {
    std::ostringstream oss;
    oss << std::setw(width) << std::setfill('0') << number;
    return oss.str();
}

void write_to_file(const std::string& filename, std::vector<double> grid, 
                   const std::array<int, 2> dims, int halo){
    std::ofstream fout(filename);

    int nx = dims[0]; // includes halo
    int ny = dims[1]; // includes halo
    int curr_index;

    for (int i = halo; i < nx - halo; i++){
        for (int j = halo; j < ny - halo; j++){
            curr_index = idx_2d_to_1d(i, j, ny);
            fout << grid[curr_index] << ", ";
        }
        fout << "\n";
    }
    fout.close();
}

int main(){
    auto start = std::chrono::high_resolution_clock::now();

    int nx = 512;
    int ny = 512;
    const std::array<int, 2> dims = {nx, ny};
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

    std::vector<double> grid_old((nx+2*halo)*(ny+2*halo), 0.);
    std::vector<double> grid_new((nx+2*halo)*(ny+2*halo), 0.);

    init_hot_square(grid_old, dims, T_hot, T_cold, nx/4, ny/4);

    for(int t = 0; t <= timesteps; t++){

        boundary_condition_periodic(grid_old, dims, halo);

        if (t%dump_freq == 0){
            const std::string filename = "T_" + zero_pad(t, 6) + ".txt";
            write_to_file(filename, grid_old, dims, halo);
        }

        timestep(grid_old, grid_new, dims, dx, dy, dt, alpha, halo);
        std::swap(grid_new, grid_old);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Runtime: " << elapsed.count() << " seconds\n";
    return 0;
}