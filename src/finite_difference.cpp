#include "finite_difference.H"

std::vector<std::vector<double>> timestep(std::vector<std::vector<double>> grid, 
                                          double dx, double dy, double dt, 
                                          double alpha, int halo){
    const int nx = grid.size(); // includes halo
    const int ny = grid[0].size(); // includes halo
    std::vector<std::vector<double>> out(nx, std::vector<double>(ny, 0.0));
    
    for (int i = halo; i < nx - halo; i++){
        for (int j = halo; j < ny - halo; j++){
            out[i][j] = grid[i][j] + alpha*dt*(
                                               (grid[i+1][j] - 2*grid[i][j] + grid[i-1][j]) / (dx*dx) +
                                               (grid[i][j+1] - 2*grid[i][j] + grid[i][j-1]) / (dy*dy)
                                              );
        }
    }
    return out;
}

std::vector<std::vector<double>> boundary_condition_periodic(std::vector<std::vector<double>> grid, int halo){
    const int nx = grid.size(); // includes halo
    const int ny = grid[0].size(); // includes halo
    
    // Periodic in Y (top/bottom)
    for (int j = halo; j < ny - halo; ++j) {
        for (int h = 0; h < halo; ++h) {
            grid[h][j] = grid[nx - 2 * halo + h][j];             // top halo
            grid[nx - halo + h][j] = grid[halo + h][j];          // bottom halo
        }
    }

    // Periodic in X (left/right)
    for (int i = 0; i < nx; ++i) {
        for (int h = 0; h < halo; ++h) {
            grid[i][h] = grid[i][ny - 2 * halo + h];             // left halo
            grid[i][ny - halo + h] = grid[i][halo + h];          // right halo
        }
    }
    
    return grid;
}