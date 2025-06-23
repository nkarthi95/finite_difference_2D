#include "finite_difference.H"

void timestep(const std::vector<std::vector<double>>& grid_old,
              std::vector<std::vector<double>>& grid_new, 
              const double dx, const double dy, const double dt, 
              const double alpha, const int halo){
    const int nx = grid_old.size(); // includes halo
    const int ny = grid_old[0].size(); // includes halo
    // std::vector<std::vector<double>> out(nx, std::vector<double>(ny, 0.0));

    double l, c, r, d2x, d2y;
    
    #pragma omp parallel for collapse(2)
    for (int i = halo; i < nx - halo; i++){
        for (int j = halo; j < ny - halo; j++){
            // x direction
            r = grid_old[i+1][j];
            c = grid_old[i][j];
            l = grid_old[i-1][j];

            d2x = (r - 2.*c + l)/(dx*dx);

            // y direction
            r = grid_old[i][j+1];
            c = grid_old[i][j];
            l = grid_old[i][j-1];

            d2y = (r - 2.*c + l)/(dy*dy);

            // timestepping
            grid_new[i][j] = grid_old[i][j] + alpha*dt*(d2x + d2y);
        }
    }
}

void boundary_condition_periodic(std::vector<std::vector<double>>& grid, const int halo){
    const int nx = grid.size(); // includes halo at index 0 and nx - 1
    const int ny = grid[0].size(); // includes halo at index 0 and ny - 1
    
    #pragma omp parallel for
    // Periodic in Y (top/bottom)
    for (int j = halo; j < ny - halo; ++j) {
        for (int h = 0; h < halo; ++h) {
            grid[h][j] = grid[nx - 2 * halo + h][j];             // top ghost = bottom interior
            grid[nx - halo + h][j] = grid[halo + h][j];          // bottom ghost = top interior
        }
    }

    #pragma omp parallel for
    // Periodic in X (left/right)
    for (int i = 0; i < nx; ++i) {
        for (int h = 0; h < halo; ++h) {
            grid[i][h] = grid[i][ny - 2 * halo + h];             // left ghost = right interior
            grid[i][ny - halo + h] = grid[i][halo + h];          // right ghost = left interior
        }
    }
}