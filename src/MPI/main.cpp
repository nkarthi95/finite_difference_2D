#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <mpi.h>

#include "finite_difference.H"
#include "initial_conditions.H"
#include "misc_functions.H"

std::string zero_pad(int number, int width) {
    std::ostringstream oss;
    oss << std::setw(width) << std::setfill('0') << number;
    return oss.str();
}

void write_to_file(const std::string& filename, std::vector<double>& grid, 
                   const std::array<int, 2> dims, int halo){
    std::ofstream fout(filename);

    int nx = dims[0]; // includes halo
    int ny = dims[1]; // includes halo
    int curr_index;

    // for (int i = halo; i < nx - halo; i++){
    //     for (int j = halo; j < ny - halo; j++){
    //         curr_index = idx_2d_to_1d(i, j, ny);
    //         fout << grid[curr_index] << ", ";
    //     }
    //     fout << "\n";
    // }
    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            curr_index = idx_2d_to_1d(i, j, ny);
            fout << grid[curr_index] << ", ";
        }
        fout << "\n";
    }
    fout.close();
}

void halo_exchange(std::vector<double>& grid, int nx, int ny, int local_nx, int halo, int rank, int size) {
    int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    MPI_Sendrecv(&grid[halo * ny], ny, MPI_DOUBLE, up, 0,
                 &grid[(local_nx + halo) * ny], ny, MPI_DOUBLE, down, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&grid[(local_nx - 1 + halo) * ny], ny, MPI_DOUBLE, down, 1,
                 &grid[0], ny, MPI_DOUBLE, up, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    // initialize grids without halo
    // std::vector<double> grid_old((nx + 2*halo)*(ny + 2*halo), 0.);
    // std::vector<double> grid_new((nx + 2*halo)*(ny + 2*halo), 0.);
    std::vector<double> grid_old((nx)*(ny), 0.);
    std::vector<double> grid_new((nx)*(ny), 0.);

    if (rank == 0){init_hot_square(grid_old, dims, T_hot, T_cold, nx/4, ny/4);}
    // init_hot_square(grid_old, dims, T_hot, T_cold, nx/4, ny/4);

    // MPI STUFF
    const int local_nx = nx / size;               // split x-dimension
    const int padded_ny = ny;                     // local y-size = global y-size (no y-halos)
    const int padded_nx = local_nx + 2 * halo;    // add top/bottom halos
    std::array<int, 2> local_dims = {padded_nx, padded_ny};

    std::vector<double> local_old(padded_nx * padded_ny, 0.0);
    std::vector<double> local_new(padded_nx * padded_ny, 0.0);

    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r]  = local_nx * ny;               // send only interior rows per rank
        displs[r]  = r * local_nx * ny;
    }
    // MPI STUFF

    MPI_Scatterv(rank == 0 ? grid_old.data() : nullptr,
            counts.data(), displs.data(), MPI_DOUBLE,
            &local_old[halo*padded_ny],  // start at first interior row
            local_nx*padded_ny, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

    for(int t = 0; t <= timesteps; t++){
        halo_exchange(local_old, padded_nx, padded_ny, local_nx, halo, rank, size);

        // boundary_condition_periodic(grid_old, dims, halo);
        boundary_condition_periodic(local_old, local_dims, halo);

        if (t%dump_freq == 0){
            MPI_Gatherv(&local_old[halo*padded_ny], local_nx*padded_ny, MPI_DOUBLE,
            rank == 0 ? grid_old.data() : nullptr,
            counts.data(), displs.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD);
            
            if(rank == 0){
            const std::string filename = "T_" + zero_pad(t, 6) + ".txt";
            write_to_file(filename, grid_old, dims, halo);
            }
        }

        // timestep(grid_old, grid_new, dims, dx, dy, dt, alpha, halo);
        timestep(local_old, local_new, local_dims, dx, dy, dt, alpha, halo);
        // std::swap(grid_new, grid_old);
        std::swap(local_new, local_old);
    }

    MPI_Finalize();
    return 0;
}