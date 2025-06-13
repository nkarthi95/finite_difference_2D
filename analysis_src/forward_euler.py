import numpy as np

def FD_timestep(grid, dx, dy, dt, alpha, halo):
    new_grid = grid.copy()
    nx, ny = grid.shape # current shape includes the halo points on the edge of the grid

    slc = np.s_[halo:nx-halo, halo:ny-halo] # array slicing for non edge slices
    
    # central difference scheme for laplacian in x direction for spatial integration
    central_difference_x = (grid[halo+1:nx-halo+1, halo:ny-halo] - 2*grid[halo:nx-halo, halo:ny-halo] + grid[halo-1:nx-halo-1, halo:ny-halo])/dx**2
    # central difference scheme for laplacian in y direction for spatial integration
    central_difference_y = (grid[halo:nx-halo, halo+1:ny-halo+1] - 2*grid[halo:nx-halo, halo:nx-halo] + grid[halo:nx-halo, halo-1:nx-halo-1])/dy**2
    
    # performing forward euler for time integration for non halo regions
    new_grid[slc] = grid[slc] + alpha*dt*(central_difference_x + central_difference_y)

    return new_grid

def periodic_boundary(grid, halo):
    nx, ny = grid.shape # current shape includes the halo points on the edge of the grid
    new_grid = grid.copy()

    # periodic boundary for y direction copies the values in the other side of the 'real' region to the halo
    new_grid[halo:halo+halo,:] = grid[nx-halo*halo:nx-halo*halo+halo, :]
    new_grid[nx-halo*halo:nx-halo*halo+halo,:] = grid[halo:halo+halo, :]

    # periodic boundary for x direction copies the values in the other side of the 'real' region to the halo
    new_grid[:,halo:halo+halo] = grid[:,ny-halo*halo:ny-halo*halo+halo]
    new_grid[:,ny-halo*halo:ny-halo*halo+halo] = grid[:,halo:halo+halo]

    return new_grid