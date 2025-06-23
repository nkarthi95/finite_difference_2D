import numpy as np
from enum import Enum

class BoundaryType(Enum):
    PERIODIC = 1
    NO_FLUX = 2
    DIRICHLET = 3

def make_system_circle(nx, ny, R, T_hot = 373, T_cold = 273):
    X, Y = np.meshgrid(*[np.arange(0, nx), np.arange(0, ny)])
    out = np.where((X - nx/2)**2 + (Y - ny/2)**2 <= R**2, T_hot, T_cold)
    return out

def make_system_square(nx, ny, Lx, Ly, T_hot = 373, T_cold = 273):
    out = np.ones((nx, ny))*T_cold
    out[nx//2-Lx//2:nx//2+Lx//2, ny//2-Ly//2:ny//2+Ly//2] = T_hot

    return out

def FD_timestep(grid, dx, dy, dt, alpha, halo):
    new_grid = grid.copy()
    nx, ny = grid.shape # current shape includes the halo points on the edge of the grid

    slc = np.s_[halo:nx-halo, halo:ny-halo] # array slicing for non edge slices

    # x direction
    r = grid[halo+1:nx, halo:ny-halo]
    c = grid[halo:nx-1, halo:ny-halo]
    l = grid[halo-1:nx-2, halo:ny-halo]

    d2_x = (r - 2*c + l)/dx**2

    # y direction
    r = grid[halo:nx-halo, halo+1:ny]
    c = grid[halo:nx-halo, halo:ny-1]
    l = grid[halo:nx-halo, halo-1:ny-1-halo]

    d2_y = (r - 2*c + l)/dy**2
    
    # performing forward euler for time integration for non halo regions
    new_grid[slc] = grid[slc] + alpha*dt*(d2_x + d2_y)

    return new_grid

def periodic_boundary(grid, halo):
    # Wrap top and bottom
    grid[:halo, :] = grid[-2 * halo:-halo, :]        # top ghost = bottom interior
    grid[-halo:, :] = grid[halo:2 * halo, :]         # bottom ghost = top interior

    # Wrap left and right
    grid[:, :halo] = grid[:, -2 * halo:-halo]        # left ghost = right interior
    grid[:, -halo:] = grid[:, halo:2 * halo]         # right ghost = left interior

    return grid

def no_flux_boundary(grid, halo):
    nx, ny = grid.shape

    # x directios
    grid[0, :] = grid[1, :]
    grid[nx-halo, :] = grid[nx-halo-1, :]

    # y directios
    grid[:, 0] = grid[:, 1]
    grid[:, ny-halo] = grid[:, ny-halo-1]

    return grid

def constant_value_boundary(grid, T_l, T_r, T_t, T_b, halo):
    nx, ny = grid.shape
    grid[halo - 1, :] = T_l # left boundary value
    grid[nx - 1, :] = T_r   # right boundary value

    grid[:, halo - 1] = T_b # bottom boundary value
    grid[:, ny - 1] = T_t   # top boundary value
    return grid

def apply_boundary(grid, bc_type, T_left=None, T_right=None, T_top = None, T_bottom = None, halo=1):
    if bc_type == BoundaryType.PERIODIC:
        return periodic_boundary(grid, halo)
    elif bc_type == BoundaryType.NO_FLUX:
        return no_flux_boundary(grid, halo)
    elif bc_type == BoundaryType.DIRICHLET:
        if T_left is None or T_right is None or T_top is None or T_bottom is None:
            raise ValueError("Wall temperature must be specified for Dirichlet BC")
        return constant_value_boundary(grid, T_left, T_right, T_top, T_bottom, halo)
    else:
        raise ValueError("Unknown boundary condition type")