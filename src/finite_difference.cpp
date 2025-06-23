#include "finite_difference.H"

std::vector<std::vector<double>> timestep(std::vector<std::vector<double>> grid, 
                                          double dx, double dy, double dt, 
                                          double alpha, int halo){
    const int nx = grid.size(); // includes halo
    const int ny = grid[0].size(); // includes halo
    std::vector<std::vector<double>> out(nx, std::vector<double>(ny, 0.0));

    double l, c, r, d2x, d2y;
                                        
    for (int i = halo; i < nx - halo; i++){
        for (int j = halo; j < ny - halo; j++){
            // x direction
            r = grid[i+1][j];
            c = grid[i][j];
            l = grid[i-1][j];

            d2x = (r - 2.*c + l)/(dx*dx);

            // y direction
            r = grid[i][j+1];
            c = grid[i][j];
            l = grid[i][j-1];

            d2y = (r - 2.*c + l)/(dy*dy);

            // timestepping
            out[i][j] = grid[i][j] + alpha*dt*(d2x + d2y);
        }
    }
    return out;
}

std::vector<std::vector<double>> boundary_condition_periodic(std::vector<std::vector<double>> grid, int halo){
    const int nx = grid.size(); // includes halo at index 0 and nx - 1
    const int ny = grid[0].size(); // includes halo at index 0 and ny - 1
    
    // Periodic in Y (top/bottom)
    for (int j = halo; j < ny - halo; ++j) {
        for (int h = 0; h < halo; ++h) {
            grid[h][j] = grid[nx - 2 * halo + h][j];             // top ghost = bottom interior
            grid[nx - halo + h][j] = grid[halo + h][j];          // bottom ghost = top interior
        }
    }

    // Periodic in X (left/right)
    for (int i = 0; i < nx; ++i) {
        for (int h = 0; h < halo; ++h) {
            grid[i][h] = grid[i][ny - 2 * halo + h];             // left ghost = right interior
            grid[i][ny - halo + h] = grid[i][halo + h];          // right ghost = left interior
        }
    }

    // // periodic in Y (top & bottom)
    // for (int i = 0; i < halo; i++){
    //     for (int j = 0; j < ny; j++){
    //         grid[i][j] = grid[nx-2-i][j]; // top ghost = bottom interior
    //         grid[nx-1-i][j] = grid[i+1][j]; // bottom ghost = top interior
    //     }
    // }

    // // periodic in X (left & right)
    // for (int i = 0; i < nx; i++){
    //     for (int j = 0; j < halo; j++){
    //         grid[i][j] = grid[i][ny-2-j]; // left ghost = right interior
    //         grid[i][ny-1-j] = grid[i][j+1]; // right ghost = left interior
    //     }
    // }
    return grid;
}