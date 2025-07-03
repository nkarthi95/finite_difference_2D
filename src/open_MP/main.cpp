#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <omp.h>
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
    // INPUT PARAMETERS
    const int threads = 2;
    int nx = 1024; // size of field in x direction
    int ny = 1024; // size of field in y direction

    double dt = 0.000025; // timestep length 
    double timesteps = 10000; // number of timesteps
    int dump_freq = 10000; // Timesteps between data dump

    double alpha = 0.002; // Thermal diffusivity of system
    double T_hot = 373.0; // Temperature in K of the hot point
    double T_cold = 273.0; // Temperature in K of the cold point
    // INPUT PARAMETERS

    omp_set_num_threads(threads); // Number of threads to use

    double dx = 1/double(nx);
    double dy = 1/double(ny);
    check_CFL(alpha, dx, dy, dt);

    double total_time = 0., IO_time = 0., step_time = 0., bc_time = 0.;
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;
    auto total_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
    #pragma omp critical
    std::cout << "Thread " << omp_get_thread_num() << " of " << omp_get_num_threads() << "\n";
    }

    const std::array<int, 2> dims = {nx, ny}; // Box dimensions
    int halo = 1; // Number of halo layers in the box

    std::vector<double> grid_old((nx+2*halo)*(ny+2*halo), 0.);
    std::vector<double> grid_new((nx+2*halo)*(ny+2*halo), 0.);

    init_hot_square(grid_old, dims, T_hot, T_cold, nx/4, ny/4);

    for(int t = 0; t <= timesteps; t++){

        start = std::chrono::high_resolution_clock::now();
        boundary_condition_periodic(grid_old, dims, halo);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        bc_time += elapsed.count();

        start = std::chrono::high_resolution_clock::now();
        if (t%dump_freq == 0){
            const std::string filename = "T_" + zero_pad(t, 6) + ".txt";
            write_to_file(filename, grid_old, dims, halo);
        }
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        IO_time += elapsed.count();

        start = std::chrono::high_resolution_clock::now();
        timestep(grid_old, grid_new, dims, dx, dy, dt, alpha, halo);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        step_time += elapsed.count();

        std::swap(grid_new, grid_old);
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end - total_start;
    total_time = total_elapsed.count();

    std::cout << "TS Runtime: " << step_time << " seconds\n";
    std::cout << "BC Runtime: " << bc_time << " seconds\n";
    std::cout << "IO Runtime: " << IO_time << " seconds\n";
    std::cout << "Total Runtime: " << total_time << " seconds\n";
    return 0;
}