#include "finite_difference.H"
#include "misc_functions.H"

void timestep(const std::vector<double>& grid_old,
              std::vector<double>& grid_new, 
              const std::array<int, 2> dims,
              const double dx, const double dy, const double dt, 
              const double alpha, const int halo){
    int nx = dims[0]; // includes halo
    int ny = dims[1]; // includes halo
    
    int index_left, index_center, index_right;
    double l, c, r, d2x, d2y;
    
    #ifdef USE_OMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int i = halo; i < nx - halo; i++){
        for (int j = halo; j < ny - halo; j++){
            // center
            index_center = idx_2d_to_1d(i, j, ny);
            c = grid_old[index_center];

            // x direction
            index_left =   idx_2d_to_1d(i-1, j, ny);
            index_right =  idx_2d_to_1d(i+1, j, ny);

            r = grid_old[index_right];
            l = grid_old[index_left];

            d2x = (r - 2.*c + l)/(dx*dx);

            // y direction
            index_left =   idx_2d_to_1d(i, j-1, ny);
            index_right =  idx_2d_to_1d(i, j+1, ny);

            r = grid_old[index_right];
            l = grid_old[index_left];

            d2y = (r - 2.*c + l)/(dy*dy);

            // timestepping
            grid_new[index_center] = grid_old[index_center] + alpha*dt*(d2x + d2y);
        }
    }
}

void boundary_condition_periodic(std::vector<double>& grid, 
                                 const std::array<int, 2> dims, const int halo){
    int nx = dims[0]; // includes halo at index 0 and nx - 1
    int ny = dims[1]; // includes halo at index 0 and ny - 1
    
    int ghost_index, interior_index;
    // // Periodic in Y (top/bottom)
    // for (int j = halo; j < ny - halo; ++j) {
    //     for (int h = 0; h < halo; ++h) {
    //         ghost_index = idx_2d_to_1d(h, j, ny);
    //         interior_index  = idx_2d_to_1d(nx-2*halo+h, j, ny);
    //         grid[ghost_index] = grid[interior_index];             // top ghost = bottom interior
            
    //         ghost_index = idx_2d_to_1d(nx-halo+h, j, ny);
    //         interior_index  = idx_2d_to_1d(halo+h, j, ny);
    //         grid[ghost_index] = grid[interior_index];          // bottom ghost = top interior
    //     }
    // }
    #ifdef USE_OMP
    #pragma omp parallel for collapse(2)
    #endif
    // Periodic in X (left/right)
    for (int i = 0; i < nx; ++i) {
        for (int h = 0; h < halo; ++h) {
            ghost_index = idx_2d_to_1d(i, h, ny);
            interior_index  = idx_2d_to_1d(i, ny-2*halo+h, ny);
            grid[ghost_index] = grid[interior_index];             // left ghost = right interior

            ghost_index = idx_2d_to_1d(i, ny-halo+h, ny);
            interior_index  = idx_2d_to_1d(i, halo+h, ny);
            grid[ghost_index] = grid[interior_index];             // right ghost = left interior
        }
    }
}