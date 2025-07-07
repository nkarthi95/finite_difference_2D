#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <mpi.h>
#include <omp.h>

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

#ifdef USE_OLDMPI
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
#else
void halo_exchange(std::vector<double>& grid,
                   int padded_nx, int padded_ny,
                   int local_nx, int local_ny,
                   int halo,
                   int north, int south, int west, int east,
                   MPI_Comm cart_comm)
{
    MPI_Request requests[8];

    // === Exchange rows ===
    // Receive north halo row
    MPI_Irecv(&grid[idx_2d_to_1d(0, halo, padded_ny)],
              local_ny, MPI_DOUBLE, north, 0, cart_comm, &requests[0]);

    // Receive south halo row
    MPI_Irecv(&grid[idx_2d_to_1d(halo + local_nx, halo, padded_ny)],
              local_ny, MPI_DOUBLE, south, 1, cart_comm, &requests[1]);

    // Send top interior row to north
    MPI_Isend(&grid[idx_2d_to_1d(halo, halo, padded_ny)],
              local_ny, MPI_DOUBLE, north, 1, cart_comm, &requests[2]);

    // Send bottom interior row to south
    MPI_Isend(&grid[idx_2d_to_1d(halo + local_nx - 1, halo, padded_ny)],
              local_ny, MPI_DOUBLE, south, 0, cart_comm, &requests[3]);

    // === Exchange columns ===
    std::vector<double> send_west(local_nx), send_east(local_nx);
    std::vector<double> recv_west(local_nx), recv_east(local_nx);

    for (int i = 0; i < local_nx; ++i) {
        send_west[i] = grid[idx_2d_to_1d(i + halo, halo, padded_ny)];                   // first interior col
        send_east[i] = grid[idx_2d_to_1d(i + halo, halo + local_ny - 1, padded_ny)];    // last interior col
    }

    MPI_Irecv(recv_west.data(), local_nx, MPI_DOUBLE, west, 2, cart_comm, &requests[4]);
    MPI_Irecv(recv_east.data(), local_nx, MPI_DOUBLE, east, 3, cart_comm, &requests[5]);
    MPI_Isend(send_west.data(), local_nx, MPI_DOUBLE, west, 3, cart_comm, &requests[6]);
    MPI_Isend(send_east.data(), local_nx, MPI_DOUBLE, east, 2, cart_comm, &requests[7]);

    // Wait for all
    MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);

    // Unpack column halos
    for (int i = 0; i < local_nx; ++i) {
        grid[idx_2d_to_1d(i + halo, 0, padded_ny)]              = recv_west[i];  // left halo
        grid[idx_2d_to_1d(i + halo, halo + local_ny, padded_ny)] = recv_east[i]; // right halo
    }
}

void scatter_global_grid(const std::vector<double>& global_grid,
                         std::vector<double>& local_grid,
                         int global_nx, int global_ny,
                         int local_nx, int local_ny,
                         int padded_nx, int padded_ny,
                         int halo,
                         MPI_Comm cart_comm)
{
    int rank, size;
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Comm_size(cart_comm, &size);

    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            int coords[2];
            MPI_Cart_coords(cart_comm, r, 2, coords);
            int i0 = coords[0] * local_nx;
            int j0 = coords[1] * local_ny;

            std::vector<double> sendbuf(local_nx * local_ny);
            for (int i = 0; i < local_nx; ++i) {
                for (int j = 0; j < local_ny; ++j) {
                    sendbuf[idx_2d_to_1d(i, j, local_ny)] = global_grid[idx_2d_to_1d(i0 + i, j0 + j, global_ny)];
                }
            }

            if (r == 0) {
                for (int i = 0; i < local_nx; ++i) {
                    for (int j = 0; j < local_ny; ++j) {
                        local_grid[idx_2d_to_1d(i + halo, j + halo, padded_ny)] = sendbuf[idx_2d_to_1d(i, j, local_ny)];
                    }
                }
            } else {
                MPI_Send(sendbuf.data(), local_nx * local_ny, MPI_DOUBLE, r, 0, cart_comm);
            }
        }
    } else {
        std::vector<double> recvbuf(local_nx * local_ny);
        MPI_Recv(recvbuf.data(), local_nx * local_ny, MPI_DOUBLE, 0, 0, cart_comm, MPI_STATUS_IGNORE);

        for (int i = 0; i < local_nx; ++i) {
            for (int j = 0; j < local_ny; ++j) {
                local_grid[idx_2d_to_1d(i + halo, j + halo, padded_ny)] = recvbuf[idx_2d_to_1d(i, j, local_ny)];
            }
        }
    }
}

void gather_global_grid(std::vector<double>& global_grid,
                        const std::vector<double>& local_grid,
                        int global_nx, int global_ny,
                        int local_nx, int local_ny,
                        int padded_nx, int padded_ny,
                        int halo,
                        MPI_Comm cart_comm)
{
    int rank, size;
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Comm_size(cart_comm, &size);

    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            int coords[2];
            MPI_Cart_coords(cart_comm, r, 2, coords);
            int i0 = coords[0] * local_nx;
            int j0 = coords[1] * local_ny;

            std::vector<double> recvbuf(local_nx * local_ny);

            if (r == 0) {
                for (int i = 0; i < local_nx; ++i) {
                    for (int j = 0; j < local_ny; ++j) {
                        recvbuf[idx_2d_to_1d(i, j, local_ny)] =
                            local_grid[idx_2d_to_1d(i + halo, j + halo, padded_ny)];
                    }
                }
            } else {
                MPI_Recv(recvbuf.data(), local_nx * local_ny, MPI_DOUBLE, r, 0, cart_comm, MPI_STATUS_IGNORE);
            }

            for (int i = 0; i < local_nx; ++i) {
                for (int j = 0; j < local_ny; ++j) {
                    global_grid[idx_2d_to_1d(i0 + i, j0 + j, global_ny)] =
                        recvbuf[idx_2d_to_1d(i, j, local_ny)];
                }
            }
        }
    } else {
        std::vector<double> sendbuf(local_nx * local_ny);
        for (int i = 0; i < local_nx; ++i) {
            for (int j = 0; j < local_ny; ++j) {
                sendbuf[idx_2d_to_1d(i, j, local_ny)] =
                    local_grid[idx_2d_to_1d(i + halo, j + halo, padded_ny)];
            }
        }
        MPI_Send(sendbuf.data(), local_nx * local_ny, MPI_DOUBLE, 0, 0, cart_comm);
    }
}
#endif

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    #ifdef USE_OMP
    omp_set_num_threads(2); // Number of threads to use
    // #pragma omp parallel
    // {
    // #pragma omp critical
    // std::cout << "Thread " << omp_get_thread_num() << " of " << omp_get_num_threads() << "\n";
    // }
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

    #ifdef USE_OLDMPI
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

    MPI_Scatterv(rank == 0 ? grid_old.data() : nullptr,
        counts.data(), displs.data(), MPI_DOUBLE,
        &local_old[halo*padded_ny],  // start at first interior row
        local_nx*padded_ny, MPI_DOUBLE,
        0, MPI_COMM_WORLD);
    // MPI STUFF
    #else
    // Create 2D Cartesian grid (e.g., px × py)
    int mpi_dims[2] = {0, 0};
    MPI_Dims_create(size, 2, mpi_dims);  // fills dims[0] and dims[1]
    int periods[2] = {1, 1};         // periodic in both directions
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, mpi_dims, periods, 0, &cart_comm);

    // Get this rank's 2D coords and neighbors
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    int north, south, west, east;
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south); // vertical
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);   // horizontal

    // Local grid sizes
    int local_nx = nx/mpi_dims[0]; // rows per proc
    int local_ny = ny/mpi_dims[1]; // cols per proc

    int padded_nx = local_nx + 2*halo;
    int padded_ny = local_ny + 2*halo;

    std::array<int, 2> local_dims = {padded_nx, padded_ny};
    std::vector<double> local_old(padded_nx * padded_ny, 0.0);
    std::vector<double> local_new(padded_nx * padded_ny, 0.0);

    scatter_global_grid(grid_old, local_old,
                         nx, ny, local_nx, local_ny,
                         padded_nx, padded_ny,  
                         halo, cart_comm);
    #endif

    for(int t = 0; t <= timesteps; t++){
        #ifdef USE_OLDMPI
        start = MPI_Wtime();
        halo_exchange(local_old, padded_nx, padded_ny, local_nx, halo, rank, size);
        end = MPI_Wtime();
        halo_time += end - start;

        start = MPI_Wtime();
        boundary_condition_periodic(local_old, local_dims, halo);
        end = MPI_Wtime();
        bc_time += end - start;
        #else
        // Neighbor ranks from MPI_Cart_shift
        // int north, south, west, east;
        start = MPI_Wtime();
        MPI_Cart_shift(cart_comm, 0, 1, &north, &south); // row shift
        MPI_Cart_shift(cart_comm, 1, 1, &west, &east);   // col shift

        halo_exchange(local_old, padded_nx, padded_ny,
                      local_nx, local_ny, halo,
                      north, south, west, east,
                      cart_comm);
        end = MPI_Wtime();
        halo_time += end - start;
        bc_time += end - start;
        #endif

        start = MPI_Wtime();
        if (t%dump_freq == 0){
            #ifdef USE_OLDMPI
            MPI_Gatherv(&local_old[halo*padded_ny], local_nx*padded_ny, MPI_DOUBLE,
            rank == 0 ? grid_old.data() : nullptr,
            counts.data(), displs.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD);
            #else
            gather_global_grid(grid_old, local_old,
                               nx, ny,
                               local_nx, local_ny,
                               padded_nx, padded_ny,
                               halo, cart_comm);
            #endif
            
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