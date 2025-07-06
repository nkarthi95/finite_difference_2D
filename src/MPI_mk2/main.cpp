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

    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            curr_index = idx_2d_to_1d(i, j, ny);
            fout << grid[curr_index] << ", ";
        }
        fout << "\n";
    }
    fout.close();
}

#ifdef USE_SENDV_1D
void halo_exchange(std::vector<double>& grid, int nx, int ny, int local_nx, int halo, int rank, int size) {
    // std::cout << "sendrecv\n";
    int up = (rank == 0) ? size - 1 : rank - 1;
    int down = (rank == size - 1) ? 0 : rank + 1;

    MPI_Sendrecv(&grid[halo*ny], ny, MPI_DOUBLE, up, 0,
                 &grid[(local_nx+halo)*ny], ny, MPI_DOUBLE, down, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // std::cout << "Rank " << rank << ": sending row " << halo << " to rank: " << up << " recieving row " << local_nx + halo << "\n";

    MPI_Sendrecv(&grid[(local_nx-1+halo)*ny], ny, MPI_DOUBLE, down, 1,
                 &grid[0], ny, MPI_DOUBLE, up, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // std::cout << "Rank " << rank << ": sending row " << local_nx + halo - 1 << " to rank: " << down << " recieving row " << 0 << "\n";
}
// #ifdef USE_ISEND_1D
#else
void halo_exchange(std::vector<double>& grid, int nx, int ny, int local_nx, int halo, int rank, int size) {
    // std::cout << "irecv\n";
    int up = (rank == 0) ? size - 1 : rank - 1;
    int down = (rank == size - 1) ? 0 : rank + 1;

    MPI_Request requests[4];

    // Non-blocking receives
    MPI_Irecv(&grid[0], ny, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, &requests[0]);                         // from up into top halo
    MPI_Irecv(&grid[(local_nx + halo) * ny], ny, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &requests[1]);  // from down into bottom halo

    // Non-blocking sends
    MPI_Isend(&grid[halo * ny], ny, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &requests[2]);                 // send top interior to up
    MPI_Isend(&grid[(local_nx + halo - 1) * ny], ny, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &requests[3]); // send bottom interior to down

    // Wait for all to complete
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
}
#endif

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    #ifdef USE_OMP
    omp_set_num_threads(threads); // Number of threads to use
    #endif

    // INPUT PARAMETERS
    int nx = 256; // size of field in x direction
    int ny = 256; // size of field in y direction

    double dt = 0.000025; // timestep length 
    double timesteps = 100000; // number of timesteps
    int dump_freq = 100000;  // Timesteps between data dump

    double alpha = 0.002; // Thermal diffusivity of system
    double T_hot = 373.0; // Temperature in K of the hot point
    double T_cold = 273.0; // Temperature in K of the cold point
    // INPUT PARAMETERS

    double total_time = 0., halo_time = 0., IO_time = 0., step_time = 0., bc_time = 0., start = 0., end = 0.;
    double total_start = MPI_Wtime();

    double dx = 1/double(nx);
    double dy = 1/double(ny);
    check_CFL(alpha, dx, dy, dt);

    const std::array<int, 2> dims = {nx, ny};
    int halo = 1;
    // initialize grids without halo unlike OMP and single core versions
    std::vector<double> grid_old((nx)*(ny), 0.);
    std::vector<double> grid_new((nx)*(ny), 0.);

    if (rank == 0){init_hot_square(grid_old, dims, T_hot, T_cold, nx/4, ny/4);}

    // MPI STUFF
    const int local_nx = nx/size;               // split x-dimension
    const int padded_ny = ny;                     // local y-size = global y-size (no y-halos)
    const int padded_nx = local_nx + 2*halo;    // add top/bottom halos
    std::array<int, 2> local_dims = {padded_nx, padded_ny};

    std::vector<double> local_old(padded_nx*padded_ny, 0.0);
    std::vector<double> local_new(padded_nx*padded_ny, 0.0);

    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r]  = local_nx*ny;               // send only interior rows per rank
        displs[r]  = r*local_nx*ny;
    }
    // MPI STUFF

    MPI_Scatterv(rank == 0 ? grid_old.data() : nullptr,
            counts.data(), displs.data(), MPI_DOUBLE,
            &local_old[halo*padded_ny],  // start at first interior row
            local_nx*padded_ny, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

    for(int t = 0; t <= timesteps; t++){
        start = MPI_Wtime();
        halo_exchange(local_old, padded_nx, padded_ny, local_nx, halo, rank, size);
        end = MPI_Wtime();
        halo_time += end - start;

        start = MPI_Wtime();
        boundary_condition_periodic(local_old, local_dims, halo);
        end = MPI_Wtime();
        bc_time += end - start;

        start = MPI_Wtime();
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
        end = MPI_Wtime();
        IO_time += end - start;

        start = MPI_Wtime();
        timestep(local_old, local_new, local_dims, dx, dy, dt, alpha, halo);
        end = MPI_Wtime();
        step_time += end - start;

        std::swap(local_new, local_old);
    }

    double total_end = MPI_Wtime();
    total_time = total_end - total_start;

    if (rank == 0){
        // double total_time = 0., halo_time = 0., IO_time = 0., step_time = 0., bc_time = 0.;
        std::cout << "TS Runtime: " << step_time << " seconds\n";
        std::cout << "MP Runtime: " << halo_time << " seconds\n";
        std::cout << "BC Runtime: " << bc_time << " seconds\n";
        std::cout << "IO Runtime: " << IO_time << " seconds\n";
        std::cout << "Total Runtime: " << total_time << " seconds\n";
    }
    MPI_Finalize();
    return 0;
}