import numpy as np
from enum import Enum

class BoundaryType(Enum):
    PERIODIC = 1
    NO_FLUX = 2
    DIRICHLET = 3

def make_system(nx, L, T_hot, T_cold):
    rod = np.ones(nx)*T_cold
    rod[nx//2-L//2:nx//2+L//2] = T_hot
    return rod

def FD_timestep(grid, dx, dt, alpha, halo):
    new_grid = grid.copy()
    nx = grid.size

    right = grid[halo+1:nx+1-halo]
    center = grid[halo:nx-halo]
    left = grid[halo-1:nx-halo-1]

    d2_x = (right - 2*center + left)/dx**2
    new_grid[halo:nx-halo] = grid[halo:nx-halo] + alpha*dt*(d2_x)

    return new_grid

def no_flux_boundary(grid, halo):
    nx = grid.size
    grid[halo - 1] = grid[halo]
    grid[nx-halo] = grid[nx-halo-1]
    return grid

def periodic_boundary(grid, halo):
    nx = grid.size
    grid[halo - 1] = grid[nx - 1 - halo]      # left ghost = last physical
    grid[nx - 1] = grid[halo]      # right ghost = first physical
    return grid

def constant_value_boundary(grid, T_left, T_right, halo):
    nx = grid.size
    grid[halo - 1] = T_left       # left boundary value
    grid[nx - 1] = T_right     # right boundary value
    return grid

def apply_boundary(grid, bc_type, T_left=None, T_right=None, halo=1):
    if bc_type == BoundaryType.PERIODIC:
        return periodic_boundary(grid, halo)
    elif bc_type == BoundaryType.NO_FLUX:
        return no_flux_boundary(grid, halo)
    elif bc_type == BoundaryType.DIRICHLET:
        if T_left is None or T_right is None:
            raise ValueError("T_left and T_right must be specified for Dirichlet BC")
        return constant_value_boundary(grid, T_left, T_right)
    else:
        raise ValueError("Unknown boundary condition type")